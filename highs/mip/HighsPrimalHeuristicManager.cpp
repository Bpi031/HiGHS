/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "mip/HighsPrimalHeuristicManager.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include "lp_data/HConst.h"

constexpr size_t HighsPrimalOpportunityTrace::kCapacity;

namespace {
int64_t addWork(int64_t left, int64_t right) {
  right = std::max<int64_t>(0, right);
  return left > std::numeric_limits<int64_t>::max() - right
      ? std::numeric_limits<int64_t>::max() : left + right;
}
}

void HighsPrimalHeuristicManager::initialise(
    const HighsPrimalHeuristicManagerConfig& config) {
  std::lock_guard<std::mutex> lock(mutex_);
  config_ = config;
  accounts_ = {};
  last_rins_node_ = -1;
  last_rins_incumbent_sequence_ = -1;
  last_rins_signature_ = 0;
  rins_reservations_.clear();
  reserved_lp_iterations_ = 0;
  portfolio_reservations_.clear();
  portfolio_restrictions_.clear();
  portfolio_calls_.clear();
  portfolio_charged_lp_ = 0;
  slot_selection_ordinal_ = 0;
  opportunity_account_ = {};
  partition_accounts_ = {};
  last_portfolio_proof_ = last_portfolio_sequence_ = -1;
  late_proof_baseline_ = -1;
  late_released_ = false;
  denial_started_ = -1;
  integrity_errors_ = 0;
  burst_ = {};
  deferred_ = {};
  burst_account_ = {};
  // Do not recycle owned call IDs across reinitialization.
  trace_.reset();
  opportunity_trace_.reset();
  if (config_.trace_decisions)
    trace_.reset(new HighsPrimalHeuristicTrace());
  if (config_.trace_decisions && portfolioEnabled())
    opportunity_trace_.reset(new HighsPrimalOpportunityTrace());
}

int64_t HighsPrimalHeuristicManager::nextSlotSelectionOrdinal() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!portfolioEnabled() || slot_selection_ordinal_ == std::numeric_limits<int64_t>::max())
    return -1;
  return slot_selection_ordinal_++;
}

HighsPrimalHeuristicDecision HighsPrimalHeuristicManager::before(
    const HighsPrimalHeuristicContext& context) {
  std::lock_guard<std::mutex> lock(mutex_);
  HighsPrimalHeuristicDecision decision;
  decision.budget = config_.rins_budget;
  HighsPrimalHeuristicAccount& method_account = accounts_[methodIndex(context.method)];
  ++method_account.considered;

  if (config_.schedule == HighsPrimalHeuristicSchedule::kOff) {
    decision.run = true;
    decision.reason = HighsPrimalHeuristicDecisionReason::kManagerOff;
    ++method_account.decisions[static_cast<size_t>(decision.reason)];
    observeDecision(context, decision);
    return decision;
  }

  // Compatibility mode records decisions but cannot alter the existing call
  // schedule or work limits.
  if (portfolioEnabled()) {
    beforePortfolio(context, decision);
    ++method_account.decisions[static_cast<size_t>(decision.reason)];
    observeDecision(context, decision);
    return decision;
  }
  if (config_.schedule == HighsPrimalHeuristicSchedule::kCompatibility ||
      context.method != HighsPrimalHeuristicMethod::kRins) {
    ++method_account.scheduled;
    ++method_account.decisions[static_cast<size_t>(decision.reason)];
    observeDecision(context, decision);
    return decision;
  }

  const auto deny = [&](HighsPrimalHeuristicDecisionReason reason) {
    decision.run = false;
    decision.reason = reason;
  };
  if (!context.has_incumbent)
    deny(HighsPrimalHeuristicDecisionReason::kNoIncumbent);
  else if (!context.has_lp_solution)
    deny(HighsPrimalHeuristicDecisionReason::kNoLpSolution);
  else if (context.recursion_depth != 0)
    deny(HighsPrimalHeuristicDecisionReason::kRecursiveSubMip);
  else if (config_.rins_max_calls >= 0 &&
           method_account.scheduled >= config_.rins_max_calls)
    deny(HighsPrimalHeuristicDecisionReason::kCallLimit);
  if (decision.run && config_.rins_budget.max_lp_iterations >= 0) {
    const int64_t remaining_lp_iterations =
        config_.rins_budget.max_lp_iterations - method_account.lp_iterations -
        reserved_lp_iterations_;
    if (remaining_lp_iterations <= 0)
      deny(HighsPrimalHeuristicDecisionReason::kLpWorkLimit);
    else
      decision.budget.max_lp_iterations = std::min(
          remaining_lp_iterations, std::max<int64_t>(1, config_.rins_lp_iterations_per_call));
  }
  if (decision.run && rins_reservations_.count(context.worker_id))
    deny(HighsPrimalHeuristicDecisionReason::kLpWorkLimit);
  if (decision.run && proofReserveReached(context))
    deny(HighsPrimalHeuristicDecisionReason::kProofReserve);
  if (decision.run && last_rins_node_ >= 0 &&
      context.incumbent_sequence == last_rins_incumbent_sequence_ &&
      context.node - last_rins_node_ < config_.rins_cooldown_nodes)
    deny(HighsPrimalHeuristicDecisionReason::kCooldown);
  if (decision.run && context.neighbourhood_signature != 0 &&
      context.incumbent_sequence == last_rins_incumbent_sequence_ &&
      context.neighbourhood_signature == last_rins_signature_)
    deny(HighsPrimalHeuristicDecisionReason::kDuplicateNeighbourhood);

  if (decision.run) {
    ++method_account.scheduled;
    last_rins_node_ = context.node;
    last_rins_incumbent_sequence_ = context.incumbent_sequence;
    last_rins_signature_ = context.neighbourhood_signature;
    if (decision.budget.max_lp_iterations >= 0) {
      rins_reservations_.emplace(context.worker_id,
                                 decision.budget.max_lp_iterations);
      reserved_lp_iterations_ += decision.budget.max_lp_iterations;
    }
  }
  ++method_account.decisions[static_cast<size_t>(decision.reason)];
  observeDecision(context, decision);
  return decision;
}

void HighsPrimalHeuristicManager::beforePortfolio(
    const HighsPrimalHeuristicContext& context,
    HighsPrimalHeuristicDecision& decision) {
  const auto deny = [&](HighsPrimalHeuristicDecisionReason reason) {
    decision.run = false;
    decision.reason = reason;
  };
  if (context.recursion_depth != 0)
    return deny(HighsPrimalHeuristicDecisionReason::kRecursiveSubMip);
  if (context.method != HighsPrimalHeuristicMethod::kRens && !context.has_incumbent)
    return deny(HighsPrimalHeuristicDecisionReason::kNoIncumbent);
  if (context.method != HighsPrimalHeuristicMethod::kLocalBranching && !context.has_lp_solution)
    return deny(HighsPrimalHeuristicDecisionReason::kNoLpSolution);
  if (context.method == HighsPrimalHeuristicMethod::kRootReducedCost ||
      context.restriction_identity.empty() ||
      std::any_of(context.restriction_identity.begin(), context.restriction_identity.end(),
                  [](double value) { return std::isnan(value); }) ||
      config_.portfolio_max_calls <= 0 || config_.portfolio_lp_cap <= 0 ||
      config_.portfolio_bootstrap_lp < 0 || config_.portfolio_lp_per_call <= 0 ||
      config_.portfolio_proof_divisor <= 0 ||
      !std::isfinite(config_.portfolio_time_per_call) || config_.portfolio_time_per_call <= 0)
    return deny(HighsPrimalHeuristicDecisionReason::kInvalidRestriction);
  if (!portfolio_reservations_.empty())
    return deny(HighsPrimalHeuristicDecisionReason::kPortfolioBusy);
  if (adaptiveEnabled()) {
    if (config_.portfolio_solve_calls <= 0 || config_.portfolio_cooldown_lp < 0 ||
        config_.portfolio_construction_lp_per_call <= 0 ||
        opportunity_account_.reserved_calls >= config_.portfolio_solve_calls)
      return deny(HighsPrimalHeuristicDecisionReason::kCallLimit);
    if (config_.portfolio_late_reserve_calls < 0 ||
        config_.portfolio_late_reserve_calls >= config_.portfolio_solve_calls ||
        (config_.portfolio_late_reserve_calls > 0 && config_.portfolio_late_proof_progress <= 0))
      return deny(HighsPrimalHeuristicDecisionReason::kInvalidRestriction);
    if (!lateOpportunityReady(context.main_lp_iterations))
      return deny(HighsPrimalHeuristicDecisionReason::kProofReserve);
    if (context.method == HighsPrimalHeuristicMethod::kLocalBranching &&
        (context.partition_variant < 0 || context.partition_variant >= int(partitionVariantCount()) ||
         (context.partition_variant < 3 ? context.partition_groups <= 0 : context.partition_groups != 0)))
      return deny(HighsPrimalHeuristicDecisionReason::kInvalidRestriction);
    if (context.method != HighsPrimalHeuristicMethod::kLocalBranching &&
        (context.partition_variant != -1 || context.partition_groups != 0))
      return deny(HighsPrimalHeuristicDecisionReason::kInvalidRestriction);
    if (composedEnabled() && context.method == HighsPrimalHeuristicMethod::kLocalBranching &&
        context.neighbourhood != (context.partition_variant < 3 ? HighsPrimalNeighbourhoodKind::kStaffPartition :
          context.partition_variant == 3 ? HighsPrimalNeighbourhoodKind::kAssignmentLocalBranching :
          HighsPrimalNeighbourhoodKind::kReducedLocalBranching))
      return deny(HighsPrimalHeuristicDecisionReason::kInvalidRestriction);
    if (last_portfolio_proof_ >= 0 &&
        context.incumbent_sequence == last_portfolio_sequence_ &&
        std::max<int64_t>(0, context.main_lp_iterations) - last_portfolio_proof_ <
            config_.portfolio_cooldown_lp)
      return deny(HighsPrimalHeuristicDecisionReason::kCooldown);
  }
  if (burstEnabled() &&
      (!context.burst_id || context.burst_id != burst_.id ||
       context.worker_id != burst_.worker || context.restart != burst_.restart))
    return deny(HighsPrimalHeuristicDecisionReason::kPortfolioBusy);
  if (context.burst_id && (!std::isfinite(context.time) ||
      context.time >= burst_.deadline || burst_.calls >= config_.portfolio_burst_calls ||
      burst_.lp >= config_.portfolio_burst_lp))
    return deny(HighsPrimalHeuristicDecisionReason::kLpWorkLimit);
  if (context.burst_id && burst_.calls > 0 &&
      (!burst_.continuation_authorized ||
       context.incumbent_sequence != burst_.authorized_sequence ||
       context.seed_upper_bound != burst_.last.admission.after ||
       (composedEnabled() && context.checked_seed != burst_.authorized_seed)))
    return deny(HighsPrimalHeuristicDecisionReason::kNoIncumbent);
  const bool deferred_start = context.burst_id && burst_.deferred_start && burst_.calls == 0;
  if (deferred_start && ((!composedEnabled() && context.method != HighsPrimalHeuristicMethod::kLocalBranching) ||
      context.seed_upper_bound != burst_.deferred_upper ||
      context.checked_seed != burst_.deferred_seed))
    return deny(HighsPrimalHeuristicDecisionReason::kNoIncumbent);
  if (!adaptiveEnabled() && !portfolio_restrictions_.empty() &&
      portfolio_restrictions_.back().restart != context.restart) {
    portfolio_restrictions_.clear();
    portfolio_calls_.clear();
  }
  const bool extra = !adaptiveEnabled() && (deferred_start ||
      portfolio_calls_[context.restart] >= config_.portfolio_max_calls);
  if (extra) {
    // Extra work can only spend an active burst's single-use own-gain permit.
    // The solve-scoped count is not reset when restrictions/restarts change.
    if ((!deferredEnabled() &&
         config_.schedule != HighsPrimalHeuristicSchedule::kPortfolioContinuation) ||
        !context.burst_id || (!deferred_start &&
         (!burst_.continuation_authorized || !std::isfinite(burst_.last.admission.before))))
      return deny(HighsPrimalHeuristicDecisionReason::kCallLimit);
    if (burst_account_.extra_reserved >= config_.portfolio_extra_continuations) {
      ++burst_account_.extra_limit_denials;
      return deny(HighsPrimalHeuristicDecisionReason::kCallLimit);
    }
  }
  for (const auto& prior : portfolio_restrictions_)
    if ((prior.restart == context.restart || adaptiveEnabled()) && prior.method == context.method &&
        (!adaptiveEnabled() || (prior.checked_seed == context.checked_seed &&
         prior.partition_variant == context.partition_variant && prior.neighbourhood == context.neighbourhood)) &&
        prior.neighbourhood_signature == context.neighbourhood_signature &&
        prior.restriction_identity == context.restriction_identity)
      return deny(HighsPrimalHeuristicDecisionReason::kDuplicateNeighbourhood);
  const int64_t proof = std::max<int64_t>(0, context.main_lp_iterations);
  const int64_t allowance = std::min(config_.portfolio_lp_cap,
      addWork(config_.portfolio_bootstrap_lp, proof / config_.portfolio_proof_divisor));
  if (portfolio_charged_lp_ >= allowance)
    return deny(HighsPrimalHeuristicDecisionReason::kLpWorkLimit);
  if (next_reservation_ == std::numeric_limits<uint64_t>::max())
    return deny(HighsPrimalHeuristicDecisionReason::kCallLimit);
  decision.budget.max_lp_iterations = std::min(
      adaptiveEnabled() && !context.has_incumbent ? config_.portfolio_construction_lp_per_call :
      config_.portfolio_lp_per_call, allowance - portfolio_charged_lp_);
  decision.budget.max_time = config_.portfolio_time_per_call;
  if (context.burst_id) {
    decision.budget.max_lp_iterations = std::min(decision.budget.max_lp_iterations,
                                                config_.portfolio_burst_lp - burst_.lp);
    decision.budget.max_time = std::min(decision.budget.max_time,
                                       burst_.deadline - context.time);
    ++burst_.calls;
    ++burst_account_.calls;
    burst_.completed_call = false;
    burst_.continuation_authorized = false;
  }
  decision.budget.max_nodes = 200;
  decision.budget.max_leaves = 500;
  decision.budget.max_stall_nodes = 12;
  decision.reservation_id = ++next_reservation_;
  decision.productive_continuation = extra;
  decision.deferred_continuation = deferred_start;
  portfolio_reservations_.emplace(decision.reservation_id,
      Reservation{context, decision.budget.max_lp_iterations, extra, deferred_start});
  opportunity_account_.reserved_calls = addWork(opportunity_account_.reserved_calls, 1);
  if (adaptiveEnabled()) {
    last_portfolio_proof_ = std::max<int64_t>(0, context.main_lp_iterations);
    last_portfolio_sequence_ = context.incumbent_sequence;
    if (context.partition_variant >= 0) {
      auto& variant = partition_accounts_[context.partition_variant];
      ++variant.scheduled;
    }
    if (config_.portfolio_late_reserve_calls > 0 && late_proof_baseline_ < 0 &&
        opportunity_account_.reserved_calls >= config_.portfolio_solve_calls - config_.portfolio_late_reserve_calls)
      late_proof_baseline_ = std::max<int64_t>(0, context.main_lp_iterations);
  }
  burst_account_.extra_reserved += extra;
  burst_account_.deferred_reserved += deferred_start;
  portfolio_restrictions_.push_back(context);
  ++portfolio_calls_[context.restart];
  auto& account = accounts_[methodIndex(context.method)];
  ++account.scheduled;
  if (selectionEnabled() &&
      (context.portfolio_choice == HighsPrimalPortfolioChoice::kLeastWork ||
       context.portfolio_choice == HighsPrimalPortfolioChoice::kOwnGain ||
       context.portfolio_choice == HighsPrimalPortfolioChoice::kExplore ||
       context.portfolio_choice == HighsPrimalPortfolioChoice::kUcb)) {
    account.selection_reuses = addWork(account.selection_reuses, 1);
    if (context.portfolio_choice == HighsPrimalPortfolioChoice::kExplore)
      account.selection_explorations = addWork(account.selection_explorations, 1);
  }
}

HighsPrimalHeuristicMethod HighsPrimalHeuristicManager::selectPortfolio(
    const std::array<bool, kHighsPrimalHeuristicMethodCount>& eligible,
    HighsPrimalPortfolioChoice* choice) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (choice) *choice = HighsPrimalPortfolioChoice::kUnspecified;
  HighsPrimalHeuristicMethod best = HighsPrimalHeuristicMethod::kCount;
  HighsPrimalHeuristicMethod productive = HighsPrimalHeuristicMethod::kCount;
  double best_score = 0;
  size_t candidates = 0;
  int64_t reuses = 0;
  long double trials = 0;
  for (const auto& account : accounts_)
    reuses = addWork(reuses, account.selection_reuses);
  // First sweep precedes reuse; then spend the least measured work first.
  for (auto method : {HighsPrimalHeuristicMethod::kRens,
                      HighsPrimalHeuristicMethod::kRins,
                      HighsPrimalHeuristicMethod::kLocalBranching}) {
    const size_t index = methodIndex(method);
    if (!eligible[index]) continue;
    ++candidates;
    const auto& account = accounts_[index];
    if (((ucbSelectionEnabled() || adaptiveEnabled()) ? account.selection_samples : account.scheduled) == 0) {
      if (choice) *choice = HighsPrimalPortfolioChoice::kFirstSweep;
      return method;
    }
    trials += account.selection_samples;
    if (best == HighsPrimalHeuristicMethod::kCount ||
        accounts_[index].lp_iterations < accounts_[methodIndex(best)].lp_iterations ||
        (accounts_[index].lp_iterations == accounts_[methodIndex(best)].lp_iterations &&
         accounts_[index].scheduled < accounts_[methodIndex(best)].scheduled))
      best = method;
    // Only the separately qualified own-gain account influences this mode.
    const double score = account.selection_samples > 0 &&
                                 std::isfinite(account.selection_reward) &&
                                 account.selection_reward >= 0
                             ? account.selection_reward / account.selection_samples
                             : 0;
    if (score > best_score) {
      best_score = score;
      productive = method;
    }
  }
  if (best == HighsPrimalHeuristicMethod::kCount) return best;
  if (!selectionEnabled()) return best;
  if (candidates == 1) {
    if (choice) *choice = HighsPrimalPortfolioChoice::kForced;
    return best;
  }
  if (ucbSelectionEnabled()) {
    // Public UCB1, Auer et al. (2002). Rewards are bounded checked own gains;
    // only eligible completed trials contribute to this local selection.
    double maximum = -1;
    for (auto method : {HighsPrimalHeuristicMethod::kRens,
                        HighsPrimalHeuristicMethod::kRins,
                        HighsPrimalHeuristicMethod::kLocalBranching}) {
      const auto& account = accounts_[methodIndex(method)];
      if (!eligible[methodIndex(method)]) continue;
      const double count = static_cast<double>(account.selection_samples);
      const double score = account.selection_reward / count +
          std::sqrt(2 * std::log(static_cast<double>(trials)) / count);
      // Strict comparison gives deterministic method-order ties.
      if (std::isfinite(score) && score > maximum) {
        maximum = score;
        best = method;
      }
    }
    if (choice) *choice = HighsPrimalPortfolioChoice::kUcb;
    return best;
  }
  // Independent diagnostic cadence: queries/denials/forced continuations do
  // not advance it. Only a successfully reserved competitive reuse does.
  if (reuses % 4 == 3) {
    if (choice) *choice = HighsPrimalPortfolioChoice::kExplore;
    return best;
  }
  if (productive != HighsPrimalHeuristicMethod::kCount) {
    if (choice) *choice = HighsPrimalPortfolioChoice::kOwnGain;
    return productive;
  }
  if (choice) *choice = HighsPrimalPortfolioChoice::kLeastWork;
  return best;
}

bool HighsPrimalHeuristicManager::portfolioOpportunityAvailable(
    int64_t proof_lp, int64_t restart, double time, int64_t sequence) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto reason = HighsPrimalHeuristicDecisionReason::kAllowed;
  const auto calls = portfolio_calls_.find(restart);
  if (!portfolio_reservations_.empty() || burst_.id)
    reason = HighsPrimalHeuristicDecisionReason::kPortfolioBusy;
  else if ((adaptiveEnabled() &&
            opportunity_account_.reserved_calls >= config_.portfolio_solve_calls) ||
           (!adaptiveEnabled() && calls != portfolio_calls_.end() &&
            calls->second >= config_.portfolio_max_calls &&
            !(deferredEnabled() && deferred_.source_burst && deferred_.restart == restart)))
    reason = HighsPrimalHeuristicDecisionReason::kCallLimit;
  else if (adaptiveEnabled() && (config_.portfolio_late_reserve_calls < 0 ||
           config_.portfolio_late_reserve_calls >= config_.portfolio_solve_calls ||
           (config_.portfolio_late_reserve_calls > 0 && config_.portfolio_late_proof_progress <= 0)))
    reason = HighsPrimalHeuristicDecisionReason::kInvalidRestriction;
  else if (adaptiveEnabled() && !lateOpportunityReady(proof_lp))
    reason = HighsPrimalHeuristicDecisionReason::kProofReserve;
  else if (config_.portfolio_proof_divisor <= 0)
    reason = HighsPrimalHeuristicDecisionReason::kInvalidRestriction;
  else if (portfolio_charged_lp_ >= std::min(config_.portfolio_lp_cap,
      addWork(config_.portfolio_bootstrap_lp,
              std::max<int64_t>(0, proof_lp) / config_.portfolio_proof_divisor)))
    reason = HighsPrimalHeuristicDecisionReason::kLpWorkLimit;
  else if (adaptiveEnabled() && last_portfolio_proof_ >= 0 &&
           sequence == last_portfolio_sequence_ &&
           std::max<int64_t>(0, proof_lp) - last_portfolio_proof_ < config_.portfolio_cooldown_lp)
    reason = HighsPrimalHeuristicDecisionReason::kCooldown;
  opportunity_account_.considered = addWork(opportunity_account_.considered, 1);
  auto& count = opportunity_account_.reasons[static_cast<size_t>(reason)];
  count = addWork(count, 1);
  const bool allowed = reason == HighsPrimalHeuristicDecisionReason::kAllowed;
  opportunity_account_.allowed = addWork(opportunity_account_.allowed, allowed);
  if (std::isfinite(time) && time >= 0) {
    if (!allowed && denial_started_ < 0) denial_started_ = time;
    if (denial_started_ >= 0)
      opportunity_account_.longest_sampled_denial = std::max(
          opportunity_account_.longest_sampled_denial, std::max(0.0, time - denial_started_));
    if (allowed) {
      opportunity_account_.last_allowed_time = time;
      denial_started_ = -1;
    }
  }
  if (opportunity_trace_) {
    opportunity_trace_->considered = opportunity_account_.considered;
    if (opportunity_trace_->retained < HighsPrimalOpportunityTrace::kCapacity) {
      auto& row = opportunity_trace_->observations[opportunity_trace_->retained++];
      row.id = opportunity_account_.considered;
      row.restart = restart;
      row.sequence = sequence;
      row.proof_lp = std::max<int64_t>(0, proof_lp);
      row.charged_lp = portfolio_charged_lp_;
      row.solve_scoped_calls = adaptiveEnabled();
      const int64_t call_limit = adaptiveEnabled()
          ? config_.portfolio_solve_calls : config_.portfolio_max_calls;
      const int64_t used_calls = adaptiveEnabled()
          ? opportunity_account_.reserved_calls
          : calls == portfolio_calls_.end() ? 0 : calls->second;
      row.remaining_calls = call_limit > used_calls ? call_limit - used_calls : 0;
      const int64_t affordable = config_.portfolio_proof_divisor > 0
          ? std::max<int64_t>(0, std::min(config_.portfolio_lp_cap,
                addWork(config_.portfolio_bootstrap_lp,
                        row.proof_lp / config_.portfolio_proof_divisor))) : 0;
      row.remaining_lp = affordable > portfolio_charged_lp_
          ? affordable - portfolio_charged_lp_ : 0;
      row.outstanding = portfolio_reservations_.size();
      row.time = std::isfinite(time) && time >= 0 ? time : -1;
      if (burst_.id) {
        row.remaining_burst_calls = config_.portfolio_burst_calls > burst_.calls
            ? config_.portfolio_burst_calls - burst_.calls : 0;
        row.remaining_burst_lp = config_.portfolio_burst_lp > burst_.lp
            ? config_.portfolio_burst_lp - burst_.lp : 0;
        if (row.time >= 0)
          row.remaining_burst_seconds = std::max(0.0, burst_.deadline - row.time);
      }
      row.reason = reason;
      if (adaptiveEnabled() && !late_released_ && config_.portfolio_late_reserve_calls > 0) {
        row.late_calls_held = std::min(row.remaining_calls, config_.portfolio_late_reserve_calls);
        if (late_proof_baseline_ >= 0) {
          const int64_t progress = row.proof_lp >= late_proof_baseline_ ? row.proof_lp - late_proof_baseline_ : 0;
          row.late_proof_needed = config_.portfolio_late_proof_progress > progress
              ? config_.portfolio_late_proof_progress - progress : 0;
        }
      }
    }
  }
  return allowed;
}

bool HighsPrimalHeuristicManager::lateOpportunityReady(int64_t proof_lp) {
  if (!adaptiveEnabled() || config_.portfolio_late_reserve_calls == 0 || late_released_ ||
      opportunity_account_.reserved_calls < config_.portfolio_solve_calls - config_.portfolio_late_reserve_calls)
    return true;
  const int64_t proof = std::max<int64_t>(0, proof_lp);
  if (late_proof_baseline_ >= 0 && proof >= late_proof_baseline_ &&
      proof - late_proof_baseline_ >= config_.portfolio_late_proof_progress)
    late_released_ = true;
  return late_released_;
}

HighsPrimalOpportunityTrace HighsPrimalHeuristicManager::opportunityTrace() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return opportunity_trace_ ? *opportunity_trace_ : HighsPrimalOpportunityTrace{};
}

HighsPrimalOpportunityAccount HighsPrimalHeuristicManager::opportunityAccount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto result = opportunity_account_;
  result.charged_lp = portfolio_charged_lp_;
  return result;
}

std::array<HighsPrimalPartitionAccount, kHighsPrimalPartitionVariantCount>
HighsPrimalHeuristicManager::partitionAccounts() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return partition_accounts_;
}

int HighsPrimalHeuristicManager::selectPartitionVariant() const {
  std::lock_guard<std::mutex> lock(mutex_);
  int best = 0;
  double reward = -1;
  int64_t trials = 0;
  int least_work = 0;
  int oldest = 0;
  for (int i = 0; i < int(partitionVariantCount()); ++i) {
    const auto& account = partition_accounts_[i];
    if (!account.scheduled) return i;
    trials = addWork(trials, account.scheduled);
    if (account.lp_iterations < partition_accounts_[least_work].lp_iterations)
      least_work = i;
    if (account.last_exercised_call < partition_accounts_[oldest].last_exercised_call)
      oldest = i;
    const double score = account.completed ? account.reward / account.completed : 0;
    if (score > reward || (score == reward &&
        account.lp_iterations < partition_accounts_[best].lp_iterations)) {
      reward = score;
      best = i;
    }
  }
  return (trials - int(partitionVariantCount())) % 4 == 3
      ? (opportunityCoverageEnabled() ? oldest : least_work) : best;
}

uint64_t HighsPrimalHeuristicManager::beginPortfolioBurst(
    int64_t worker, int64_t restart, double time) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!burstEnabled() ||
      burst_.id || !portfolio_reservations_.empty() ||
      (deferredEnabled() && deferred_.source_burst) ||
      config_.portfolio_burst_calls <= 0 || config_.portfolio_burst_lp <= 0 ||
      !std::isfinite(time) || !std::isfinite(config_.portfolio_burst_time) ||
      config_.portfolio_burst_time <= 0 ||
      !std::isfinite(time + config_.portfolio_burst_time) ||
      next_burst_ == std::numeric_limits<uint64_t>::max()) return 0;
  burst_ = {};
  burst_.id = ++next_burst_;
  burst_.worker = worker;
  burst_.restart = restart;
  burst_.deadline = time + config_.portfolio_burst_time;
  ++burst_account_.started;
  return burst_.id;
}

bool HighsPrimalHeuristicManager::continuePortfolioBurst(
    uint64_t id, int64_t worker, int64_t restart, int64_t sequence,
    double upper, double time, const std::vector<double>& seed) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!id || burst_.id != id || burst_.worker != worker ||
      burst_.restart != restart || !portfolio_reservations_.empty()) return false;
  const auto& outcome = burst_.last;
  const auto& receipt = outcome.admission;
  // Only this child's checked publication can authorize another child. Peer
  // improvements and an accepted-but-not-improving seed are not a reward.
  if (!burst_.completed_call || !outcome.started || !outcome.candidate_accepted ||
      !receipt.measured || !receipt.improved ||
      !std::isfinite(receipt.after) || !std::isfinite(receipt.candidate) ||
      !(receipt.after < receipt.before) || receipt.candidate != receipt.after ||
      (outcome.preparation_stop_mask & (kHighsHeuristicStopCheckpoint |
                                       kHighsHeuristicStopUnfinishedLp))) {
    ++burst_account_.no_gain;
    return false;
  }
  if (sequence <= burst_.sequence || !std::isfinite(upper) || upper != receipt.after ||
      (composedEnabled() && (seed.empty() || seed != receipt.checked_seed))) {
    ++burst_account_.stale_seed;
    return false;
  }
  if (!std::isfinite(time) || time >= burst_.deadline ||
      burst_.calls >= config_.portfolio_burst_calls || burst_.lp >= config_.portfolio_burst_lp) {
    ++burst_account_.budget_exhausted;
    return false;
  }
  if (!adaptiveEnabled() && (config_.schedule == HighsPrimalHeuristicSchedule::kPortfolioContinuation || deferredEnabled()) &&
      portfolio_calls_[restart] >= config_.portfolio_max_calls &&
      (!std::isfinite(receipt.before) ||
       burst_account_.extra_reserved >= config_.portfolio_extra_continuations)) {
    ++burst_account_.extra_limit_denials;
    return false;
  }
  burst_.completed_call = false;
  burst_.continuation_authorized = true;
  burst_.authorized_sequence = sequence;
  if (composedEnabled()) burst_.authorized_seed = seed;
  ++burst_account_.continuations;
  return true;
}

bool HighsPrimalHeuristicManager::deferPortfolioContinuation(
    uint64_t id, int64_t worker, int64_t restart, int64_t sequence,
    double upper, double time, const std::vector<double>& seed, bool checkpoint_boundary) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!deferredEnabled() || !id || burst_.id != id || burst_.worker != worker ||
      burst_.restart != restart || !portfolio_reservations_.empty() ||
      burst_.deferred_issued || deferred_.source_burst ||
      (adaptiveEnabled() ? opportunity_account_.reserved_calls >= config_.portfolio_solve_calls :
       (portfolio_calls_[restart] < config_.portfolio_max_calls ||
        burst_account_.extra_reserved >= config_.portfolio_extra_continuations)) ||
      portfolio_charged_lp_ >= config_.portfolio_lp_cap) return false;
  const auto& outcome = burst_.last;
  const auto& receipt = outcome.admission;
  if (!burst_.completed_call || !outcome.started || !outcome.completed ||
      !outcome.candidate_accepted || !receipt.measured || !receipt.improved ||
      !std::isfinite(receipt.before) || !std::isfinite(receipt.after) ||
      !std::isfinite(receipt.candidate) || !(receipt.after < receipt.before) ||
      receipt.candidate != receipt.after || upper != receipt.after ||
      sequence <= burst_.sequence || !std::isfinite(time) ||
      (time < burst_.deadline && burst_.calls < config_.portfolio_burst_calls &&
       burst_.lp < config_.portfolio_burst_lp && !(composedEnabled() && checkpoint_boundary)) || seed.empty() ||
      (composedEnabled() && seed != receipt.checked_seed) ||
      !std::all_of(seed.begin(), seed.end(), [](double v) { return std::isfinite(v); }) ||
      (outcome.preparation_stop_mask & (kHighsHeuristicStopCheckpoint |
                                       kHighsHeuristicStopUnfinishedLp))) return false;
  deferred_.source_burst = id;
  deferred_.worker = worker;
  deferred_.restart = restart;
  deferred_.upper = upper;
  deferred_.seed = seed;
  burst_.deferred_issued = true;
  ++burst_account_.deferred_issued;
  return true;
}

uint64_t HighsPrimalHeuristicManager::beginDeferredPortfolioBurst(
    int64_t worker, int64_t restart, double time, double upper,
    const std::vector<double>& seed) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!deferredEnabled() || !deferred_.source_burst || burst_.id ||
      !portfolio_reservations_.empty()) return 0;
  if (deferred_.restart != restart) {
    deferred_ = {};
    ++burst_account_.deferred_invalidated;
    return 0;
  }
  if (deferred_.worker != worker) return 0;
  if (deferred_.restart != restart || deferred_.upper != upper || deferred_.seed != seed ||
      !std::isfinite(time) || !std::isfinite(config_.portfolio_burst_time) ||
      config_.portfolio_burst_time <= 0 || !std::isfinite(time + config_.portfolio_burst_time) ||
      config_.portfolio_burst_calls <= 0 || config_.portfolio_burst_lp <= 0 ||
      next_burst_ == std::numeric_limits<uint64_t>::max() ||
      (adaptiveEnabled() ? opportunity_account_.reserved_calls >= config_.portfolio_solve_calls :
       burst_account_.extra_reserved >= config_.portfolio_extra_continuations) ||
      portfolio_charged_lp_ >= config_.portfolio_lp_cap) {
    deferred_ = {};
    ++burst_account_.deferred_invalidated;
    return 0;
  }
  // Move a single witness across the scheduling boundary, never work credit.
  burst_ = {};
  burst_.id = ++next_burst_;
  burst_.worker = worker;
  burst_.restart = restart;
  burst_.deadline = time + config_.portfolio_burst_time;
  burst_.deferred_start = true;
  burst_.deferred_upper = upper;
  burst_.deferred_seed = std::move(deferred_.seed);
  deferred_ = {};
  ++burst_account_.started;
  ++burst_account_.deferred_claimed;
  return burst_.id;
}

void HighsPrimalHeuristicManager::endPortfolioBurst(uint64_t id) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!id || burst_.id != id || !portfolio_reservations_.empty()) {
    ++integrity_errors_;
    return;
  }
  burst_ = {};
}

HighsPrimalPortfolioBurstAccount HighsPrimalHeuristicManager::burstAccount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return burst_account_;
}

void HighsPrimalHeuristicManager::observeDecision(
    const HighsPrimalHeuristicContext& context,
    HighsPrimalHeuristicDecision& decision) {
  if (!trace_) return;
  decision.observation_id = ++trace_->considered;
  if (trace_->retained == kHighsPrimalHeuristicTraceCapacity) {
    ++trace_->dropped;
    return;
  }
  auto& observation = trace_->observations[trace_->retained++];
  observation.id = decision.observation_id;
  observation.context = context;
  observation.context.restriction_identity.clear();
  observation.context.checked_seed.clear();
  observation.decision = decision;
}

void HighsPrimalHeuristicManager::record(
    const HighsPrimalHeuristicContext& context,
    const HighsPrimalHeuristicOutcome& outcome,
    uint64_t observation_id, uint64_t reservation_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (portfolioEnabled()) {
    const auto reservation = portfolio_reservations_.find(reservation_id);
    if (reservation == portfolio_reservations_.end() ||
        reservation->second.context.method != context.method ||
        reservation->second.context.worker_id != context.worker_id ||
        reservation->second.context.restart != context.restart ||
        reservation->second.context.incumbent_sequence != context.incumbent_sequence ||
        reservation->second.context.burst_id != context.burst_id ||
        reservation->second.context.portfolio_choice != context.portfolio_choice ||
        reservation->second.context.seed_upper_bound != context.seed_upper_bound ||
        reservation->second.context.checked_seed != context.checked_seed ||
        reservation->second.context.partition_variant != context.partition_variant ||
        reservation->second.context.partition_groups != context.partition_groups ||
        reservation->second.context.neighbourhood != context.neighbourhood ||
        reservation->second.context.at_root != context.at_root ||
        reservation->second.context.restriction_identity != context.restriction_identity) {
      ++integrity_errors_;
      return;
    }
    if (outcome.lp_iterations < 0 || !std::isfinite(outcome.elapsed) || outcome.elapsed < 0) {
      portfolio_charged_lp_ = addWork(portfolio_charged_lp_, reservation->second.lp);
      if (reservation->second.productive_continuation)
        burst_account_.extra_lp_iterations = addWork(
            burst_account_.extra_lp_iterations, reservation->second.lp);
      if (reservation->second.deferred_continuation)
        burst_account_.deferred_lp_iterations = addWork(
            burst_account_.deferred_lp_iterations, reservation->second.lp);
      if (context.burst_id == burst_.id && context.burst_id) {
        burst_.lp = addWork(burst_.lp, reservation->second.lp);
        burst_.completed_call = false;
      }
      portfolio_reservations_.erase(reservation);
      ++integrity_errors_;
      return;
    }
    portfolio_charged_lp_ = addWork(portfolio_charged_lp_, outcome.lp_iterations);
    if (reservation->second.productive_continuation) {
      burst_account_.extra_started += outcome.started;
      burst_account_.extra_completed += outcome.completed;
      burst_account_.extra_lp_iterations = addWork(
          burst_account_.extra_lp_iterations, outcome.lp_iterations);
    }
    if (reservation->second.deferred_continuation) {
      burst_account_.deferred_started += outcome.started;
      burst_account_.deferred_completed += outcome.completed;
      burst_account_.deferred_lp_iterations = addWork(
          burst_account_.deferred_lp_iterations, outcome.lp_iterations);
    }
    if (context.burst_id == burst_.id && context.burst_id) {
      burst_.lp = addWork(burst_.lp, outcome.lp_iterations);
      burst_.sequence = context.incumbent_sequence;
      burst_.last = outcome;
      burst_.completed_call = true;
    }
    portfolio_reservations_.erase(reservation);
  }
  if (context.method == HighsPrimalHeuristicMethod::kRins) {
    const auto reservation = rins_reservations_.find(context.worker_id);
    if (reservation != rins_reservations_.end()) {
      reserved_lp_iterations_ -= reservation->second;
      rins_reservations_.erase(reservation);
    }
  }
  HighsPrimalHeuristicAccount& method_account = accounts_[methodIndex(context.method)];
  method_account.started += outcome.started ? 1 : 0;
  method_account.completed += outcome.completed ? 1 : 0;
  method_account.proposed += outcome.candidate_proposed ? 1 : 0;
  method_account.accepted += outcome.candidate_accepted ? 1 : 0;
  method_account.improved += outcome.candidate_improved ? 1 : 0;
  method_account.nodes = addWork(method_account.nodes, outcome.nodes);
  method_account.leaves = addWork(method_account.leaves, outcome.leaves);
  method_account.lp_iterations = addWork(method_account.lp_iterations, outcome.lp_iterations);
  method_account.elapsed += std::max(0.0, outcome.elapsed);
  method_account.attributed_improvements += outcome.admission.measured && outcome.admission.improved;
  const double reward = attributedReward(outcome);
  if (reward >= 0) {
    ++method_account.reward_observations;
    method_account.reward += reward;
  }
  if (selectionEnabled() &&
      (!adaptiveEnabled() || (context.has_incumbent &&
                             context.portfolio_choice != HighsPrimalPortfolioChoice::kForced)) &&
      ((!ucbSelectionEnabled() && !adaptiveEnabled()) || (outcome.started && outcome.completed))) {
    method_account.selection_samples = addWork(method_account.selection_samples, 1);
    const auto& receipt = outcome.admission;
    if (outcome.started && outcome.completed && outcome.candidate_proposed &&
        outcome.candidate_accepted && outcome.candidate_improved &&
        receipt.measured && receipt.improved && receipt.after < receipt.before &&
        receipt.candidate == receipt.after && reward > 0 && std::isfinite(reward) &&
        !(outcome.preparation_stop_mask & (kHighsHeuristicStopCheckpoint |
                                          kHighsHeuristicStopUnfinishedLp)))
      method_account.selection_reward +=
          config_.schedule == HighsPrimalHeuristicSchedule::kPortfolioUcbSuccess ? 1 : reward;
  }
  if (adaptiveEnabled() && context.partition_variant >= 0 && context.partition_variant < int(partitionVariantCount())) {
    auto& variant = partition_accounts_[context.partition_variant];
    if (outcome.started) variant.last_exercised_call = opportunity_account_.reserved_calls;
    variant.completed = addWork(variant.completed, outcome.completed);
    variant.lp_iterations = addWork(variant.lp_iterations, outcome.lp_iterations);
    variant.elapsed += outcome.elapsed;
    if (reward > 0 && std::isfinite(reward) && outcome.started && outcome.completed &&
        outcome.candidate_proposed && outcome.candidate_accepted && outcome.candidate_improved &&
        outcome.admission.measured && outcome.admission.improved &&
        outcome.admission.after < outcome.admission.before &&
        outcome.admission.candidate == outcome.admission.after &&
        !(outcome.preparation_stop_mask & (kHighsHeuristicStopCheckpoint |
                                          kHighsHeuristicStopUnfinishedLp))) {
      variant.improved = addWork(variant.improved, 1);
      variant.reward += reward;
    }
    // Public ALNS-style feedback. A limited child is not an exhausted neighborhood.
    if (outcome.children.retained == 1 && outcome.completed &&
        (!outcome.candidate_proposed || outcome.candidate_accepted)) {
      const int status = outcome.children.items[0].status;
      if (!outcome.candidate_improved && context.partition_variant >= 3 &&
          status == static_cast<int>(HighsModelStatus::kOptimal))
        variant.radius = std::min<int64_t>(8, variant.radius * 2);
      else if (!outcome.candidate_improved && context.partition_variant >= 3 &&
          (status == static_cast<int>(HighsModelStatus::kTimeLimit) ||
           status == static_cast<int>(HighsModelStatus::kIterationLimit)))
        variant.radius = std::max<int64_t>(2, variant.radius / 2);
      else if (!outcome.candidate_improved && context.partition_variant < 3 &&
          (status == static_cast<int>(HighsModelStatus::kOptimal) ||
           status == static_cast<int>(HighsModelStatus::kInfeasible)))
        variant.free_groups = std::min<int64_t>(8, context.partition_groups + 1);
      else if (!outcome.candidate_improved && context.partition_variant < 3 &&
          (status == static_cast<int>(HighsModelStatus::kTimeLimit) ||
           status == static_cast<int>(HighsModelStatus::kIterationLimit) ||
           status == static_cast<int>(HighsModelStatus::kSolutionLimit)))
        variant.free_groups = std::max<int64_t>(1, context.partition_groups - 1);
    }
  }
  if (outcome.phase_measured) {
    ++method_account.phase_measured_calls;
    method_account.preparation_lp_iterations += std::max<int64_t>(0, outcome.preparation_lp_iterations);
    method_account.child_lp_iterations += std::max<int64_t>(0, outcome.child_lp_iterations);
    method_account.child_calls += std::max<int64_t>(0, outcome.child_calls);
    method_account.preparation_exhaustions += std::max<int64_t>(0, outcome.preparation_exhaustions);
    method_account.preparation_seconds += std::max(0.0, outcome.preparation_seconds);
    method_account.child_seconds += std::max(0.0, outcome.child_seconds);
  }
  if (trace_ && observation_id != 0) {
    for (size_t i = 0; i < trace_->retained; ++i) {
      auto& observation = trace_->observations[i];
      if (observation.id != observation_id || !observation.decision.run ||
          observation.context.worker_id != context.worker_id ||
          observation.context.method != context.method ||
          observation.context.restart != context.restart ||
          observation.context.incumbent_sequence != context.incumbent_sequence ||
          observation.outcome_recorded)
        continue;
      observation.outcome = outcome;
      observation.outcome_recorded = true;
      break;
    }
  }
}

uint64_t HighsPrimalHeuristicManager::integrityErrors() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return integrity_errors_;
}

double HighsPrimalHeuristicManager::attributedReward(
    const HighsPrimalHeuristicOutcome& outcome) {
  const auto& receipt = outcome.admission;
  if (!receipt.measured || !std::isfinite(receipt.before) ||
      !std::isfinite(receipt.after) || !std::isfinite(receipt.candidate) ||
      !std::isfinite(outcome.parent_lower_bound) ||
      outcome.parent_lower_bound >= receipt.before ||
      outcome.parent_lower_bound > receipt.after || outcome.lp_iterations < 0 ||
      !std::isfinite(outcome.elapsed) || outcome.elapsed < 0) return -1;
  if (!receipt.improved) return 0;
  const long double gain = std::max(0.0L, std::min(
      (long double)receipt.before - receipt.after,
      (long double)receipt.before - receipt.candidate));
  const long double gap = (long double)receipt.before - outcome.parent_lower_bound;
  const long double cost = std::max((long double)outcome.lp_iterations / 100000,
                                   (long double)outcome.elapsed / 20);
  if (!std::isfinite(gain) || !std::isfinite(gap) || !std::isfinite(cost)) return -1;
  return static_cast<double>(std::min(1.0L, gain / gap) / (1 + cost));
}

HighsPrimalHeuristicTrace HighsPrimalHeuristicManager::decisionTrace() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return trace_ ? *trace_ : HighsPrimalHeuristicTrace{};
}

HighsPrimalHeuristicAccount HighsPrimalHeuristicManager::account(
    HighsPrimalHeuristicMethod method) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return accounts_[methodIndex(method)];
}

bool HighsPrimalHeuristicManager::proofReserveReached(
    const HighsPrimalHeuristicContext& context) const {
  if (config_.proof_work_reserve <= 0.0) return false;
  if (config_.proof_work_reserve >= 1.0) return true;
  const double heuristic_work =
      static_cast<double>(std::max<int64_t>(
          std::max<int64_t>(0, context.heuristic_lp_iterations),
          accounts_[methodIndex(HighsPrimalHeuristicMethod::kRins)].lp_iterations)) +
      static_cast<double>(reserved_lp_iterations_);
  const double total = std::max(
      1.0, static_cast<double>(std::max<int64_t>(0, context.main_lp_iterations)) +
               heuristic_work);
  const double heuristic_share =
      heuristic_work / total;
  return heuristic_share >= 1.0 - config_.proof_work_reserve;
}

size_t HighsPrimalHeuristicManager::methodIndex(
    HighsPrimalHeuristicMethod method) {
  const size_t index = static_cast<size_t>(method);
  assert(index < kHighsPrimalHeuristicMethodCount);
  return index;
}

const char* HighsPrimalHeuristicManager::methodName(
    HighsPrimalHeuristicMethod method) {
  switch (method) {
    case HighsPrimalHeuristicMethod::kRootReducedCost:
      return "root_reduced_cost";
    case HighsPrimalHeuristicMethod::kRens:
      return "rens";
    case HighsPrimalHeuristicMethod::kRins:
      return "rins";
    case HighsPrimalHeuristicMethod::kLocalBranching:
      return "local_branching";
    case HighsPrimalHeuristicMethod::kCount:
      break;
  }
  return "unknown";
}

const char* HighsPrimalHeuristicManager::scheduleName(
    HighsPrimalHeuristicSchedule schedule) {
  switch (schedule) {
    case HighsPrimalHeuristicSchedule::kOff:
      return "off";
    case HighsPrimalHeuristicSchedule::kCompatibility:
      return "compatibility";
    case HighsPrimalHeuristicSchedule::kManagedRins:
      return "managed-rins";
    case HighsPrimalHeuristicSchedule::kPortfolio:
      return "portfolio";
    case HighsPrimalHeuristicSchedule::kPortfolioBurst:
      return "portfolio-burst";
    case HighsPrimalHeuristicSchedule::kPortfolioContinuation:
      return "portfolio-continuation";
    case HighsPrimalHeuristicSchedule::kPortfolioDeferredContinuation:
      return "portfolio-deferred-continuation";
    case HighsPrimalHeuristicSchedule::kPortfolioCostAware:
      return "portfolio-cost-aware";
    case HighsPrimalHeuristicSchedule::kPortfolioUcbSuccess:
      return "portfolio-ucb-success";
    case HighsPrimalHeuristicSchedule::kPortfolioUcbCost:
      return "portfolio-ucb-cost";
    case HighsPrimalHeuristicSchedule::kPortfolioAssignmentAdaptive:
      return "portfolio-assignment-adaptive";
    case HighsPrimalHeuristicSchedule::kPortfolioAssignmentComposed:
      return "portfolio-assignment-composed";
    case HighsPrimalHeuristicSchedule::kPortfolioAssignmentOpportunities:
      return "portfolio-assignment-opportunities";
    case HighsPrimalHeuristicSchedule::kPortfolioAssignmentProgressive:
      return "portfolio-assignment-progressive";
  }
  return "unknown";
}

const char* HighsPrimalHeuristicManager::neighbourhoodName(HighsPrimalNeighbourhoodKind kind) {
  switch (kind) {
    case HighsPrimalNeighbourhoodKind::kRens: return "rens_box";
    case HighsPrimalNeighbourhoodKind::kRins: return "rins_agreement";
    case HighsPrimalNeighbourhoodKind::kStaffPartition: return "staff_partition";
    case HighsPrimalNeighbourhoodKind::kCoupledSlots: return "coupled_slots";
    case HighsPrimalNeighbourhoodKind::kAssignmentLocalBranching: return "assignment_distance";
    case HighsPrimalNeighbourhoodKind::kReducedLocalBranching: return "reduced_distance";
    case HighsPrimalNeighbourhoodKind::kUnspecified: return "unspecified";
  }
  return "unknown";
}

const char* HighsPrimalHeuristicManager::reasonName(
    HighsPrimalHeuristicDecisionReason reason) {
  switch (reason) {
    case HighsPrimalHeuristicDecisionReason::kAllowed:
      return "allowed";
    case HighsPrimalHeuristicDecisionReason::kManagerOff:
      return "manager_off";
    case HighsPrimalHeuristicDecisionReason::kNoIncumbent:
      return "no_incumbent";
    case HighsPrimalHeuristicDecisionReason::kNoLpSolution:
      return "no_lp_solution";
    case HighsPrimalHeuristicDecisionReason::kRecursiveSubMip:
      return "recursive_submip";
    case HighsPrimalHeuristicDecisionReason::kCallLimit:
      return "call_limit";
    case HighsPrimalHeuristicDecisionReason::kCooldown:
      return "cooldown";
    case HighsPrimalHeuristicDecisionReason::kDuplicateNeighbourhood:
      return "duplicate_neighbourhood";
    case HighsPrimalHeuristicDecisionReason::kLpWorkLimit:
      return "lp_work_limit";
    case HighsPrimalHeuristicDecisionReason::kProofReserve:
      return "proof_reserve";
    case HighsPrimalHeuristicDecisionReason::kPortfolioBusy:
      return "portfolio_busy";
    case HighsPrimalHeuristicDecisionReason::kInvalidRestriction:
      return "invalid_restriction";
    case HighsPrimalHeuristicDecisionReason::kCount:
      break;
  }
  return "unknown";
}
