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

void HighsPrimalHeuristicManager::initialise(
    const HighsPrimalHeuristicManagerConfig& config) {
  config_ = config;
  accounts_ = {};
  last_rins_node_ = -1;
  last_rins_incumbent_sequence_ = -1;
  last_rins_signature_ = 0;
}

HighsPrimalHeuristicDecision HighsPrimalHeuristicManager::before(
    const HighsPrimalHeuristicContext& context) {
  HighsPrimalHeuristicDecision decision;
  decision.budget = config_.rins_budget;
  HighsPrimalHeuristicAccount& method_account = accounts_[methodIndex(context.method)];
  ++method_account.considered;

  if (config_.schedule == HighsPrimalHeuristicSchedule::kOff) {
    decision.run = true;
    decision.reason = HighsPrimalHeuristicDecisionReason::kManagerOff;
    ++method_account.decisions[static_cast<size_t>(decision.reason)];
    return decision;
  }

  // Compatibility mode records decisions but cannot alter the existing call
  // schedule or work limits.
  if (config_.schedule == HighsPrimalHeuristicSchedule::kCompatibility ||
      context.method != HighsPrimalHeuristicMethod::kRins) {
    ++method_account.scheduled;
    ++method_account.decisions[static_cast<size_t>(decision.reason)];
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
        config_.rins_budget.max_lp_iterations - method_account.lp_iterations;
    if (remaining_lp_iterations <= 0)
      deny(HighsPrimalHeuristicDecisionReason::kLpWorkLimit);
    else
      decision.budget.max_lp_iterations = remaining_lp_iterations;
  }
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
  }
  ++method_account.decisions[static_cast<size_t>(decision.reason)];
  return decision;
}

void HighsPrimalHeuristicManager::record(
    const HighsPrimalHeuristicContext& context,
    const HighsPrimalHeuristicOutcome& outcome) {
  HighsPrimalHeuristicAccount& method_account = accounts_[methodIndex(context.method)];
  method_account.started += outcome.started ? 1 : 0;
  method_account.completed += outcome.completed ? 1 : 0;
  method_account.proposed += outcome.candidate_proposed ? 1 : 0;
  method_account.accepted += outcome.candidate_accepted ? 1 : 0;
  method_account.improved += outcome.candidate_improved ? 1 : 0;
  method_account.nodes += std::max<int64_t>(0, outcome.nodes);
  method_account.leaves += std::max<int64_t>(0, outcome.leaves);
  method_account.lp_iterations += std::max<int64_t>(0, outcome.lp_iterations);
  method_account.elapsed += std::max(0.0, outcome.elapsed);
}

const HighsPrimalHeuristicAccount& HighsPrimalHeuristicManager::account(
    HighsPrimalHeuristicMethod method) const {
  return accounts_[methodIndex(method)];
}

bool HighsPrimalHeuristicManager::proofReserveReached(
    const HighsPrimalHeuristicContext& context) const {
  if (config_.proof_work_reserve <= 0.0) return false;
  if (config_.proof_work_reserve >= 1.0) return true;
  const int64_t total = std::max<int64_t>(
      1, context.main_lp_iterations + context.heuristic_lp_iterations);
  const double heuristic_share =
      static_cast<double>(context.heuristic_lp_iterations) /
      static_cast<double>(total);
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
    case HighsPrimalHeuristicDecisionReason::kCount:
      break;
  }
  return "unknown";
}
