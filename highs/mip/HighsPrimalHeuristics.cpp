/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "mip/HighsPrimalHeuristics.h"

#include <numeric>
#include <map>
#include <unordered_set>

#include "../extern/pdqsort/pdqsort.h"
#include "io/HighsIO.h"
#include "lp_data/HConst.h"
#include "lp_data/HighsLpUtils.h"
#include "mip/HighsCutGeneration.h"
#include "mip/HighsCandidateVerifier.h"
#include "mip/HighsChildProposalSink.h"
#include "mip/HighsDomainChange.h"
#include "mip/HighsLpRelaxation.h"
#include "mip/HighsMipSolverData.h"
#include "mip/HighsReplayCapture.h"
#include "mip/MipTimer.h"
#include "util/HighsHash.h"
#include "util/HighsIntegers.h"

// GCC floating point errors are well-known for 32-bit architectures;
// see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=323.
// An easy workaround is to add the "volatile" keyword to avoid
// problematic GCC optimizations that impact precision.
#ifdef __i386__
#define FP_32BIT_VOLATILE volatile
#else
#define FP_32BIT_VOLATILE
#endif

HighsPrimalHeuristics::HighsPrimalHeuristics(HighsMipSolver& mipsolver)
    : mipsolver(mipsolver),
      successObservations(0.0),
      numSuccessObservations(0),
      infeasObservations(0.0),
      numInfeasObservations(0),
      randgen(mipsolver.options_mip_->random_seed) {}

const char* HighsPrimalHeuristics::subMipMethodName(HighsSubMipOrigin origin) {
  switch (origin) {
    case HighsSubMipOrigin::kRootReducedCost:
      return "root_reduced_cost";
    case HighsSubMipOrigin::kRootRens:
    case HighsSubMipOrigin::kTreeRens:
      return "rens";
    case HighsSubMipOrigin::kTreeRins:
      return "rins";
    case HighsSubMipOrigin::kTreeLocalBranching:
      return "local_branching";
  }
  return "unknown";
}

void HighsPrimalHeuristics::configureSubMipOptions(
    HighsOptions& options, HighsSubMipOrigin /*origin*/) {
  // Every neighborhood helper is terminal. A helper may use cheap local
  // repair, but it must never launch another sub-MIP or a parallel portfolio.
  options.mip_heuristic_run_rens = false;
  options.mip_heuristic_run_rins = false;
  options.mip_heuristic_run_root_reduced_cost = false;
  options.mip_incumbent_exchange = "off";
  options.mip_primal_heuristic_schedule = "off";
  options.mip_primal_partition_file.clear();
  options.parallel = kHighsOffString;
  options.threads = 1;
  options.mip_detect_symmetry = false;
  options.mip_heuristic_effort = 0.05;
}

double HighsPrimalHeuristics::remainingHeuristicTime(double started,
                                                     double now,
                                                     double allowance) {
  if (!std::isfinite(started) || !std::isfinite(now) ||
      std::isnan(allowance) || allowance <= 0)
    return 0.0;
  if (allowance == kHighsInf) return kHighsInf;
  return std::max(0.0, allowance - std::max(0.0, now - started));
}

bool HighsPrimalHeuristics::buildRinsNeighbourhood(
    const HighsLp& model, const std::vector<double>& relaxation,
    const std::vector<double>& incumbent, double tolerance,
    double min_fixing_rate, std::vector<double>& lower,
    std::vector<double>& upper, double& fixing_rate) {
  fixing_rate = 0.0;
  const size_t n = static_cast<size_t>(std::max(HighsInt{0}, model.num_col_));
  if (model.num_col_ < 0 || relaxation.size() != n || incumbent.size() != n ||
      lower.size() != n || upper.size() != n || model.integrality_.size() != n ||
      !std::isfinite(tolerance) || tolerance < 0 ||
      !std::isfinite(min_fixing_rate) || min_fixing_rate < 0 || min_fixing_rate > 1)
    return false;
  std::vector<double> restricted_lower = lower, restricted_upper = upper;
  size_t eligible = 0, fixed = 0;
  for (size_t col = 0; col < n; ++col) {
    const auto type = model.integrality_[col];
    if ((type != HighsVarType::kContinuous && type != HighsVarType::kInteger &&
         type != HighsVarType::kImplicitInteger) ||
        !std::isfinite(relaxation[col]) || !std::isfinite(incumbent[col]) ||
        std::isnan(lower[col]) || std::isnan(upper[col]) || lower[col] > upper[col] ||
        incumbent[col] < lower[col] - tolerance ||
        incumbent[col] > upper[col] + tolerance)
      return false;
    if (type != HighsVarType::kInteger) continue;
    const double value = std::round(incumbent[col]);
    if (std::abs(value - incumbent[col]) > tolerance) return false;
    if (lower[col] == upper[col]) continue;
    ++eligible;
    if (std::abs(relaxation[col] - incumbent[col]) > tolerance ||
        value < lower[col] || value > upper[col])
      continue;
    restricted_lower[col] = restricted_upper[col] = value;
    ++fixed;
  }
  fixing_rate = eligible == 0 ? 0.0 : double(fixed) / double(eligible);
  if (fixed == 0 || fixed == eligible || fixing_rate < min_fixing_rate)
    return false;
  lower.swap(restricted_lower);
  upper.swap(restricted_upper);
  return true;
}

bool HighsPrimalHeuristics::buildRensNeighbourhood(
    const HighsLp& model, const std::vector<double>& relaxation,
    double tolerance, std::vector<double>& lower, std::vector<double>& upper,
    double& fixing_rate) {
  fixing_rate = 0;
  const size_t n = static_cast<size_t>(std::max(HighsInt{0}, model.num_col_));
  if (model.num_col_ < 0 || relaxation.size() != n || lower.size() != n ||
      upper.size() != n || model.integrality_.size() != n ||
      !std::isfinite(tolerance) || tolerance < 0) return false;
  auto lo = lower, hi = upper;
  size_t changed = 0, eligible = 0, fixed = 0;
  for (size_t j = 0; j < n; ++j) {
    const auto type = model.integrality_[j];
    if ((type != HighsVarType::kContinuous && type != HighsVarType::kInteger &&
         type != HighsVarType::kImplicitInteger) ||
        !std::isfinite(relaxation[j]) || std::isnan(lo[j]) || std::isnan(hi[j]) ||
        lo[j] > hi[j] || relaxation[j] < lo[j] - tolerance ||
        relaxation[j] > hi[j] + tolerance) return false;
    if (type != HighsVarType::kInteger || lo[j] == hi[j]) continue;
    ++eligible;
    lo[j] = std::max(std::ceil(lo[j]), std::floor(relaxation[j] + tolerance));
    hi[j] = std::min(std::floor(hi[j]), std::ceil(relaxation[j] - tolerance));
    if (lo[j] > hi[j]) return false;
    changed += lo[j] != lower[j] || hi[j] != upper[j];
    fixed += lo[j] == hi[j];
  }
  if (!changed) return false;
  fixing_rate = eligible ? double(fixed) / eligible : 0;
  lower.swap(lo);
  upper.swap(hi);
  return true;
}

bool HighsPrimalHeuristics::buildLocalBranchingRow(
    const HighsLp& model, const std::vector<double>& incumbent,
    double tolerance, HighsInt radius, HighsLocalBranchingRow& row) {
  const size_t n = static_cast<size_t>(std::max(HighsInt{0}, model.num_col_));
  if (model.num_col_ < 0 || incumbent.size() != n || model.integrality_.size() != n ||
      model.col_lower_.size() != n || model.col_upper_.size() != n || radius < 0 ||
      !std::isfinite(tolerance) || tolerance < 0) return false;
  HighsLocalBranchingRow built;
  built.upper = radius;
  for (size_t j = 0; j < n; ++j) {
    const auto type = model.integrality_[j];
    if ((type != HighsVarType::kContinuous && type != HighsVarType::kInteger &&
         type != HighsVarType::kImplicitInteger) || !std::isfinite(incumbent[j]) ||
        std::isnan(model.col_lower_[j]) || std::isnan(model.col_upper_[j]) ||
        model.col_lower_[j] > model.col_upper_[j] ||
        incumbent[j] < model.col_lower_[j] - tolerance ||
        incumbent[j] > model.col_upper_[j] + tolerance) return false;
    if (type != HighsVarType::kInteger || model.col_lower_[j] < 0 ||
        model.col_upper_[j] > 1) continue;
    const double value = std::round(incumbent[j]);
    if ((value != 0 && value != 1) || std::abs(value - incumbent[j]) > tolerance ||
        value < model.col_lower_[j] || value > model.col_upper_[j]) return false;
    built.index.push_back(static_cast<HighsInt>(j));
    built.value.push_back(value == 0 ? 1 : -1);
    built.upper -= value;
  }
  if (built.index.empty()) return false;
  row = std::move(built);
  return true;
}

bool HighsPrimalHeuristics::buildAssignmentLocalBranchingRow(
    const HighsLp& model, const std::vector<double>& incumbent,
    const std::vector<HighsInt>& groups, double tolerance, HighsInt radius,
    HighsLocalBranchingRow& row) {
  if (groups.size() != size_t(std::max(HighsInt{0}, model.num_col_))) return false;
  HighsLocalBranchingRow all, built;
  if (!buildLocalBranchingRow(model, incumbent, tolerance, radius, all)) return false;
  for (size_t col = 0; col < groups.size(); ++col)
    if (groups[col] < 0 || groups[col] > model.num_col_ ||
        (groups[col] && (model.integrality_[col] != HighsVarType::kInteger ||
         model.col_lower_[col] < 0 || model.col_upper_[col] > 1))) return false;
  built.upper = radius;
  for (size_t i = 0; i < all.index.size(); ++i) {
    if (!groups[all.index[i]]) continue;
    built.index.push_back(all.index[i]);
    built.value.push_back(all.value[i]);
    if (all.value[i] < 0) --built.upper;
  }
  if (built.index.empty()) return false;
  row = std::move(built);
  return true;
}

bool HighsPrimalHeuristics::buildBinaryPartitionNeighbourhood(
    const HighsLp& model, const std::vector<double>& incumbent,
    const std::vector<HighsInt>& support,
    const std::vector<HighsInt>& free_support, double tolerance,
    std::vector<double>& lower, std::vector<double>& upper,
    double& fixing_rate) {
  fixing_rate = 0;
  const size_t n = static_cast<size_t>(std::max(HighsInt{0}, model.num_col_));
  if (model.num_col_ < 0 || incumbent.size() != n || lower.size() != n ||
      upper.size() != n || model.integrality_.size() != n ||
      model.col_lower_.size() != n || model.col_upper_.size() != n ||
      &lower == &upper || support.empty() || free_support.empty() ||
      !std::isfinite(tolerance) || tolerance < 0)
    return false;
  for (size_t col = 0; col < n; ++col) {
    const auto type = model.integrality_[col];
    if ((type != HighsVarType::kContinuous && type != HighsVarType::kInteger &&
         type != HighsVarType::kImplicitInteger) ||
        !std::isfinite(incumbent[col]) || std::isnan(lower[col]) ||
        std::isnan(upper[col]) || std::isnan(model.col_lower_[col]) ||
        std::isnan(model.col_upper_[col]) || lower[col] > upper[col] ||
        lower[col] < model.col_lower_[col] || upper[col] > model.col_upper_[col] ||
        incumbent[col] < lower[col] - tolerance ||
        incumbent[col] > upper[col] + tolerance ||
        (type == HighsVarType::kInteger &&
         std::abs(incumbent[col] - std::round(incumbent[col])) > tolerance))
      return false;
  }
  std::vector<unsigned char> membership(n, 0);
  for (HighsInt col : support) {
    if (col < 0 || static_cast<size_t>(col) >= n || membership[col] ||
        model.integrality_[col] != HighsVarType::kInteger ||
        model.col_lower_[col] < 0 || model.col_upper_[col] > 1)
      return false;
    const double bit = std::round(incumbent[col]);
    if ((bit != 0 && bit != 1) || bit < lower[col] || bit > upper[col])
      return false;
    membership[col] = 1;
  }
  for (HighsInt col : free_support) {
    if (col < 0 || static_cast<size_t>(col) >= n || membership[col] != 1)
      return false;
    membership[col] = 2;
  }
  size_t movable = 0, fixed = 0;
  for (HighsInt col : support) {
    if (lower[col] <= 0 && upper[col] >= 1) {
      ++movable;
      fixed += membership[col] == 1;
    }
  }
  if (!fixed || fixed == movable) return false;
  auto restricted_lower = lower, restricted_upper = upper;
  for (HighsInt col : support) {
    if (membership[col] == 1)
      restricted_lower[col] = restricted_upper[col] = std::round(incumbent[col]);
  }
  fixing_rate = double(fixed) / double(movable);
  lower.swap(restricted_lower);
  upper.swap(restricted_upper);
  return true;
}

bool HighsPrimalHeuristics::readBinaryPartitionContract(std::istream& input,
    const HighsLp& model, std::vector<HighsInt>& groups, HighsInt& free_groups) {
  std::string version;
  HighsInt cols, count, free_count, support_count;
  if (!(input >> version >> cols >> count >> free_count >> support_count) ||
      version != "ROST_BINARY_PARTITION_V1" || cols != model.num_col_ ||
      count <= 1 || count > cols || free_count <= 0 || free_count >= count ||
      support_count < count || support_count > cols ||
      model.sense_ != ObjSense::kMinimize ||
      model.col_names_.size() != size_t(cols) ||
      model.col_lower_.size() != size_t(cols) || model.col_upper_.size() != size_t(cols) ||
      model.integrality_.size() != size_t(cols)) return false;
  std::vector<HighsInt> parsed(cols, 0), sizes(count + 1, 0);
  for (HighsInt record = 0; record < support_count; ++record) {
    HighsInt col, group;
    std::string name;
    if (!(input >> col >> group >> name) || col < 0 || col >= cols ||
        group <= 0 || group > count || parsed[col] ||
        name != model.col_names_[col] ||
        model.integrality_[col] != HighsVarType::kInteger ||
        !(0 <= model.col_lower_[col] &&
          model.col_lower_[col] <= model.col_upper_[col] &&
          model.col_upper_[col] <= 1)) return false;
    parsed[col] = group;
    ++sizes[group];
  }
  std::string extra;
  if (input >> extra || !input.eof()) return false;
  for (HighsInt group = 1; group <= count; ++group)
    if (!sizes[group]) return false;
  groups.swap(parsed);
  free_groups = free_count;
  return true;
}

bool HighsPrimalHeuristics::selectBinaryPartition(const HighsLp& model,
    const std::vector<double>& incumbent, const std::vector<HighsInt>& groups,
    HighsInt free_groups, int64_t ordinal, std::vector<HighsInt>& support,
    std::vector<HighsInt>& free_support, int variant) {
  if (groups.size() != size_t(model.num_col_) || incumbent.size() != groups.size() ||
      model.col_cost_.size() != groups.size() || model.col_lower_.size() != groups.size() ||
      model.col_upper_.size() != groups.size() || free_groups <= 0 || ordinal < 0 ||
      variant < 0 || variant > 2)
    return false;
  std::map<HighsInt, HighsCDouble> scores;
  std::vector<HighsInt> declared;
  for (HighsInt col = 0; col < model.num_col_; ++col) {
    if (groups[col] < 0 || groups[col] > model.num_col_ ||
        !std::isfinite(incumbent[col])) return false;
    if (!groups[col]) continue;
    declared.push_back(col);
    if (model.col_lower_[col] <= 0 && model.col_upper_[col] >= 1)
      scores.emplace(groups[col], 0);
  }
  if (scores.size() <= size_t(free_groups)) return false;
  auto rows = model.a_matrix_;
  rows.ensureRowwise();
  for (HighsInt row = 0; row < model.num_row_; ++row) {
    std::unordered_set<HighsInt> owners, penalties;
    for (HighsInt k = rows.start_[row]; k < rows.start_[row + 1]; ++k) {
      if (!rows.value_[k]) continue;
      const HighsInt col = rows.index_[k];
      if (groups[col] && scores.count(groups[col])) owners.insert(groups[col]);
      if (!groups[col] && model.col_cost_[col] > 0 && incumbent[col] > 0)
        penalties.insert(col);
    }
    if (owners.empty() || (variant == 0 && owners.size() > size_t(free_groups))) continue;
    HighsCDouble penalty = 0;
    for (HighsInt col : penalties) penalty += HighsCDouble(model.col_cost_[col]) * incumbent[col];
    if (!std::isfinite(double(penalty))) return false;
    for (HighsInt group : owners) scores[group] += penalty / double(owners.size());
  }
  std::vector<HighsInt> ranked;
  for (const auto& score : scores) {
    if (!std::isfinite(double(score.second))) return false;
    ranked.push_back(score.first);
  }
  std::sort(ranked.begin(), ranked.end(), [&](HighsInt a, HighsInt b) {
    return scores[a] != scores[b] ? scores[a] > scores[b] : a < b;
  });
  if (variant == 1) std::sort(ranked.begin(), ranked.end());
  if (variant == 2) {
    const HighsInt anchor = ranked[size_t(ordinal) % ranked.size()];
    std::map<HighsInt, HighsCDouble> affinity;
    for (HighsInt row = 0; row < model.num_row_; ++row) {
      std::vector<HighsInt> owners;
      bool touches = false;
      for (HighsInt k = rows.start_[row]; k < rows.start_[row + 1]; ++k) {
        if (!rows.value_[k]) continue;
        const HighsInt group = groups[rows.index_[k]];
        if (group && scores.count(group)) owners.push_back(group);
        touches |= group == anchor;
      }
      if (!touches) continue;
      std::sort(owners.begin(), owners.end());
      owners.erase(std::unique(owners.begin(), owners.end()), owners.end());
      if (owners.size() < 2) continue;
      for (HighsInt group : owners) affinity[group] += 1.0 / double(owners.size());
    }
    for (HighsInt group : ranked) affinity.emplace(group, 0);
    std::sort(ranked.begin(), ranked.end(), [&](HighsInt a, HighsInt b) {
      if (a == anchor || b == anchor) return a == anchor && b != anchor;
      if (affinity[a] != affinity[b]) return affinity[a] > affinity[b];
      return scores[a] != scores[b] ? scores[a] > scores[b] : a < b;
    });
  }
  const size_t offset = variant == 2 ? 0 :
      (size_t(ordinal % int64_t(ranked.size())) * size_t(free_groups)) % ranked.size();
  std::unordered_set<HighsInt> selected;
  for (HighsInt i = 0; i < free_groups; ++i)
    selected.insert(ranked[(offset + size_t(i)) % ranked.size()]);
  std::vector<HighsInt> freed;
  for (HighsInt col : declared) if (selected.count(groups[col])) freed.push_back(col);
  support.swap(declared);
  free_support.swap(freed);
  return true;
}

void HighsPrimalHeuristics::portfolio(HighsMipWorker& worker,
                                     const std::vector<double>& relaxation, bool at_root) {
  auto& data = *mipsolver.mipdata_;
  auto& manager = data.heuristic_manager;
  if (mipsolver.submip || worker.getGlobalDomain().infeasible()) return;
  if (manager.deferredEnabled()) worker.importPrimalSnapshot();
  if (!manager.portfolioOpportunityAvailable(
      std::max<int64_t>(0, data.total_lp_iterations - data.heuristic_lp_iterations),
      data.numRestarts, mipsolver.timer_.read(), worker.repairIncumbentSequence())) return;
  const bool burst_mode = manager.burstEnabled();
  const uint64_t deferred = manager.deferredEnabled() ? manager.beginDeferredPortfolioBurst(
      worker.workerId(), data.numRestarts, mipsolver.timer_.read(),
      worker.upper_bound, worker.repairIncumbent()) : 0;
  HighsPrimalPortfolioBurstScope burst(manager, worker.workerId(), data.numRestarts,
                                     mipsolver.timer_.read(), deferred);
  if (burst_mode && !burst.id()) return;
  bool followup = deferred != 0;
  for (int64_t call = 0; call < (burst_mode ? manager.config().portfolio_burst_calls : 1); ++call) {
    worker.importPrimalSnapshot();
    if (!portfolioCall(worker, relaxation, burst.id(), followup, at_root) || !burst_mode) return;
    worker.importPrimalSnapshot();
    if (manager.composedEnabled() && worker.search_work.checkpointRequested()) {
      manager.deferPortfolioContinuation(burst.id(), worker.workerId(), data.numRestarts,
          worker.repairIncumbentSequence(), worker.upper_bound, mipsolver.timer_.read(),
          worker.repairIncumbent(), true);
      return;
    }
    if (!manager.continuePortfolioBurst(burst.id(), worker.workerId(), data.numRestarts,
        worker.repairIncumbentSequence(), worker.upper_bound, mipsolver.timer_.read(), worker.repairIncumbent())) {
      if (manager.deferredEnabled())
        manager.deferPortfolioContinuation(burst.id(), worker.workerId(), data.numRestarts,
            worker.repairIncumbentSequence(), worker.upper_bound, mipsolver.timer_.read(),
            worker.repairIncumbent());
      return;
    }
    followup = true;
  }
}

bool HighsPrimalHeuristics::portfolioCall(HighsMipWorker& worker,
    const std::vector<double>& relaxation, uint64_t burst_id, bool followup, bool at_root) {
  auto& data = *mipsolver.mipdata_;
  auto& manager = data.heuristic_manager;
  worker.importPrimalSnapshot();
  std::array<bool, kHighsPrimalHeuristicMethodCount> eligible{};
  eligible[static_cast<size_t>(HighsPrimalHeuristicMethod::kRens)] =
      mipsolver.options_mip_->mip_heuristic_run_rens;
  eligible[static_cast<size_t>(HighsPrimalHeuristicMethod::kRins)] =
      mipsolver.options_mip_->mip_heuristic_run_rins && !worker.repairIncumbent().empty();
  eligible[static_cast<size_t>(HighsPrimalHeuristicMethod::kLocalBranching)] =
      !worker.repairIncumbent().empty();
  // Continue with a fresh binary neighborhood of the checked improving seed,
  // not a second RENS box around the unchanged relaxation.
  if (followup && !manager.composedEnabled()) {
    eligible[static_cast<size_t>(HighsPrimalHeuristicMethod::kRens)] = false;
    eligible[static_cast<size_t>(HighsPrimalHeuristicMethod::kRins)] = false;
  }
  for (int attempted = 0; attempted < 3; ++attempted) {
    HighsPrimalPortfolioChoice choice;
    const auto method = manager.selectPortfolio(eligible, &choice);
    if (method == HighsPrimalHeuristicMethod::kCount) return false;
    eligible[static_cast<size_t>(method)] = false;
    const double started = mipsolver.timer_.read();
    const auto incumbent = worker.repairIncumbent();
    std::vector<double> lower = worker.getGlobalDomain().col_lower_;
    std::vector<double> upper = worker.getGlobalDomain().col_upper_;
    const char* capture_method = method == HighsPrimalHeuristicMethod::kRens
        ? "rens" : method == HighsPrimalHeuristicMethod::kRins ? "rins" : "local_branching";
    const bool capture_enabled = HighsReplayCapture::enabled("sub_mip", capture_method);
    // Retain the exact builder inputs only for an explicitly selected capture.
    const auto capture_lower = capture_enabled ? lower : std::vector<double>{};
    const auto capture_upper = capture_enabled ? upper : std::vector<double>{};
    HighsLocalBranchingRow row;
    const bool has_partition = method == HighsPrimalHeuristicMethod::kLocalBranching &&
        !mipsolver.primal_partition_groups.empty();
    const int variant = has_partition && manager.adaptiveEnabled()
        ? manager.selectPartitionVariant() : -1;
    const bool partition = has_partition && variant < 3;
    const bool assignment_distance = has_partition && variant == 3;
    const bool coupled_slots = method == HighsPrimalHeuristicMethod::kLocalBranching &&
        !mipsolver.primal_assignment_slots.column_to_slot.empty();
    const bool original_space = partition || assignment_distance || coupled_slots;
    HighsAssignmentSupportResult slot_support;
    const auto kind = method == HighsPrimalHeuristicMethod::kRens ? HighsPrimalNeighbourhoodKind::kRens :
        method == HighsPrimalHeuristicMethod::kRins ? HighsPrimalNeighbourhoodKind::kRins :
        coupled_slots ? HighsPrimalNeighbourhoodKind::kCoupledSlots :
        partition ? HighsPrimalNeighbourhoodKind::kStaffPartition :
        assignment_distance ? HighsPrimalNeighbourhoodKind::kAssignmentLocalBranching :
        HighsPrimalNeighbourhoodKind::kReducedLocalBranching;
    HighsInt radius = variant >= 3 ? HighsInt(manager.partitionAccounts()[variant].radius) : 0;
    HighsInt free_groups = mipsolver.primal_partition_free_groups;
    int64_t partition_ordinal = manager.account(method).scheduled;
    if (variant >= 0 && variant < 3) {
      const auto variants = manager.partitionAccounts();
      free_groups = HighsInt(variants[variant].free_groups);
      partition_ordinal = variants[variant].scheduled;
      std::unordered_set<HighsInt> movable_groups;
      const auto& original = *mipsolver.orig_model_;
      for (HighsInt j = 0; j < original.num_col_; ++j)
        if (mipsolver.primal_partition_groups[j] &&
            original.col_lower_[j] <= 0 && original.col_upper_[j] >= 1)
          movable_groups.insert(mipsolver.primal_partition_groups[j]);
      free_groups = std::min(free_groups, HighsInt(movable_groups.size()) - 1);
    }
    std::vector<double> original_seed;
    std::vector<HighsInt> partition_support, partition_free;
    double fixing_rate = 0;
    bool built = false;
    if (method == HighsPrimalHeuristicMethod::kRins)
      built = buildRinsNeighbourhood(*mipsolver.model_, relaxation, incumbent,
          data.feastol, manager.config().rins_budget.min_fixing_rate,
          lower, upper, fixing_rate);
    else if (method == HighsPrimalHeuristicMethod::kRens)
      built = buildRensNeighbourhood(*mipsolver.model_, relaxation, data.feastol,
                                    lower, upper, fixing_rate);
    else if (original_space) {
      // Own the decoding cursor: neither preparation nor forward mapping may
      // mutate the proof owner's shared postsolve stack.
      auto mapping = data.postSolveStack;
      HighsSolution restored;
      restored.value_valid = true;
      restored.col_value = incumbent;
      mapping.undoPrimal(*mipsolver.options_mip_, restored, -1, true);
      original_seed = std::move(restored.col_value);
      const auto& original = *mipsolver.orig_model_;
      lower = original.col_lower_;
      upper = original.col_upper_;
      if (highsVerifyCandidate(original, original_seed, nullptr, data.feastol).feasible) {
        if (coupled_slots) {
          partition_ordinal = manager.nextSlotSelectionOrdinal();
          built = HighsAssignmentSupport::select(original, original_seed,
              mipsolver.primal_assignment_slots, partition_ordinal, slot_support) &&
              buildBinaryPartitionNeighbourhood(original, original_seed, slot_support.support,
                  slot_support.free_columns, data.feastol, lower, upper, fixing_rate);
          // Exact fixes can expose a seed's original-row residual: reject rather
          // than weakening tolerance or launching from an unchecked start.
          if (built) {
            HighsLp restricted = original;
            restricted.col_lower_ = lower;
            restricted.col_upper_ = upper;
            built = highsVerifyCandidate(restricted, original_seed, nullptr, data.feastol).feasible;
          }
        } else if (assignment_distance)
          built = buildAssignmentLocalBranchingRow(original, original_seed,
              mipsolver.primal_partition_groups, data.feastol, radius, row);
        else
          built = selectBinaryPartition(original, original_seed, mipsolver.primal_partition_groups,
              free_groups, partition_ordinal, partition_support, partition_free,
              variant < 0 ? 0 : variant) &&
              buildBinaryPartitionNeighbourhood(original, original_seed, partition_support,
                  partition_free, data.feastol, lower, upper, fixing_rate);
      }
    } else {
      const int64_t calls = manager.account(method).scheduled;
      if (variant < 3) radius = calls == 0 ? 2 : calls == 1 ? 4 : 8;
      built = buildLocalBranchingRow(*mipsolver.model_, incumbent, data.feastol, radius, row);
    }
    HighsPrimalHeuristicContext context;
    context.method = method;
    context.at_root = at_root;
    context.partition_variant = variant;
    context.partition_groups = variant >= 0 && variant < 3 ? free_groups : 0;
    context.neighbourhood = kind;
    context.portfolio_choice = choice;
    context.burst_id = burst_id;
    context.seed_upper_bound = worker.upper_bound;
    context.time = started;
    context.restart = data.numRestarts;
    context.worker_id = worker.workerId();
    context.node = data.num_nodes + worker.search_ptr_->nnodes;
    context.main_lp_iterations = std::max<int64_t>(0,
        data.total_lp_iterations - data.heuristic_lp_iterations);
    context.heuristic_lp_iterations = data.heuristic_lp_iterations;
    context.incumbent_sequence = worker.repairIncumbentSequence();
    if (manager.deferredEnabled() || manager.adaptiveEnabled()) context.checked_seed = incumbent;
    context.recursion_depth = mipsolver.submip_level;
    context.has_incumbent = !incumbent.empty();
    context.has_lp_solution = relaxation.size() == static_cast<size_t>(mipsolver.numCol());
    if (built) {
      context.restriction_identity = lower;
      context.restriction_identity.insert(context.restriction_identity.end(), upper.begin(), upper.end());
      if (method == HighsPrimalHeuristicMethod::kLocalBranching && !partition && !coupled_slots) {
        context.restriction_identity.push_back(row.upper);
        for (size_t j = 0; j < row.index.size(); ++j) {
          context.restriction_identity.push_back(row.index[j]);
          context.restriction_identity.push_back(row.value[j]);
        }
      }
      context.neighbourhood_signature = 1;
      for (double value : context.restriction_identity)
        context.neighbourhood_signature = HighsHashHelpers::hash(
            context.neighbourhood_signature ^ std::hash<double>{}(value));
    }
    const auto decision = manager.before(context);
    if (coupled_slots)
      highsLogUser(mipsolver.options_mip_->log_options, HighsLogType::kInfo,
          "MIP-SlotSelection: version=1 decision=%llu worker=%lld restart=%lld "
          "seed_sequence=%lld ordinal=%lld free_columns=%lld run=%d reason=%s\n",
          static_cast<unsigned long long>(decision.observation_id),
          static_cast<long long>(context.worker_id), static_cast<long long>(context.restart),
          static_cast<long long>(context.incumbent_sequence), static_cast<long long>(partition_ordinal),
          static_cast<long long>(slot_support.free_columns.size()), int(decision.run),
          HighsPrimalHeuristicManager::reasonName(decision.reason));
    if (!decision.run) {
      // The existing three-attempt loop bounds duplicate support searches. A
      // rejected support consumes no child credit, but must advance the anchor.
      if (coupled_slots && decision.reason == HighsPrimalHeuristicDecisionReason::kDuplicateNeighbourhood)
        eligible[static_cast<size_t>(method)] = true;
      continue;
    }
    if (manager.adaptiveEnabled())
      highsLogUser(mipsolver.options_mip_->log_options, HighsLogType::kInfo,
          "MIP-Adaptive-Call: version=1 decision=%llu method=%s root=%d "
          "has_incumbent=%d variant=%d seed_sequence=%lld time=%.17g lp_budget=%lld\n",
          static_cast<unsigned long long>(decision.observation_id), capture_method,
          int(at_root), int(context.has_incumbent), variant,
          static_cast<long long>(context.incumbent_sequence), started,
          static_cast<long long>(decision.budget.max_lp_iterations));
    if (manager.composedEnabled())
      highsLogUser(mipsolver.options_mip_->log_options, HighsLogType::kInfo,
          "MIP-Neighbourhood: version=1 decision=%llu kind=%s variant=%d "
          "radius=%d coordinates=%s\n",
          static_cast<unsigned long long>(decision.observation_id),
          HighsPrimalHeuristicManager::neighbourhoodName(kind), variant, int(radius),
          original_space ? "original" : "reduced");
    HighsPrimalHeuristicReservation reservation(manager, context, decision);
    HighsPrimalHeuristicOutcome outcome;
    outcome.fixing_rate = fixing_rate;
    outcome.fixing_rate_measured = true;
    HighsSubMipCallBudget remaining = decision.budget;
    remaining.max_time = std::min(
        remainingHeuristicTime(started, mipsolver.timer_.read(), remaining.max_time),
        std::max(0.0, mipsolver.options_mip_->time_limit - mipsolver.timer_.read()));
    const double child_started = mipsolver.timer_.read();
    HighsRepairReservationScope repair(worker.search_work, remaining.max_lp_iterations,
        at_root && manager.adaptiveEnabled(), child_started);
    if (repair.granted() && remaining.max_time > 0) {
      const auto origin = method == HighsPrimalHeuristicMethod::kRins
          ? HighsSubMipOrigin::kTreeRins : method == HighsPrimalHeuristicMethod::kRens
          ? (at_root ? HighsSubMipOrigin::kRootRens : HighsSubMipOrigin::kTreeRens)
          : HighsSubMipOrigin::kTreeLocalBranching;
      const auto& lp = worker.getLpRelaxation();
      HighsReplayContext replay;
      replay.decision = decision.observation_id;
      replay.worker = worker.workerId();
      replay.restart = data.numRestarts;
      replay.relaxation = original_space ? nullptr : &relaxation;
      replay.base_lower = original_space ? nullptr : &capture_lower;
      replay.base_upper = original_space ? nullptr : &capture_upper;
      replay.checked_seed = original_space ? &original_seed : &incumbent;
      replay.original_space = original_space;
      if (original_space) {
        replay.base_lower = &mipsolver.orig_model_->col_lower_;
        replay.base_upper = &mipsolver.orig_model_->col_upper_;
      }
      replay.lp_solution_status = original_space ? kSolutionStatusNone :
          lp.getLpSolver().getInfo().primal_solution_status;
      if (partition)
        highsLogUser(mipsolver.options_mip_->log_options, HighsLogType::kInfo,
            "MIP-Partition: version=1 decision=%llu worker=%lld ordinal=%lld "
            "support=%lld free_columns=%lld free_groups=%lld coordinates=original\n",
            static_cast<unsigned long long>(decision.observation_id),
            static_cast<long long>(worker.workerId()),
            static_cast<long long>(manager.account(method).scheduled - 1),
            static_cast<long long>(partition_support.size()),
            static_cast<long long>(partition_free.size()),
            static_cast<long long>(free_groups));
      if (variant >= 0 && variant < 3)
        highsLogUser(mipsolver.options_mip_->log_options, HighsLogType::kInfo,
            "MIP-Partition-Variant: version=1 decision=%llu variant=%d "
            "free_groups=%lld ordinal=%lld seed_sequence=%lld\n",
            static_cast<unsigned long long>(decision.observation_id), variant,
            static_cast<long long>(free_groups), static_cast<long long>(partition_ordinal),
            static_cast<long long>(context.incumbent_sequence));
      solveSubMip(worker, original_space ? *mipsolver.orig_model_ : lp.getLp(),
          lp.getLpSolver().getBasis(), fixing_rate,
          std::move(lower), std::move(upper), 500, 200, 12, origin,
          remaining.max_time, &remaining, &outcome,
          original_space ? &original_seed : incumbent.empty() ? nullptr : &incumbent,
          method == HighsPrimalHeuristicMethod::kLocalBranching && !partition && !coupled_slots ? &row : nullptr,
          &replay, original_space);
      if (coupled_slots) {
        const uint64_t invocation = outcome.children.retained == 1 ? outcome.children.items[0].invocation : 0;
        HighsInt parent_fixed = 0, free_movable = 0, builder_fixed = 0;
        const auto& original = *mipsolver.orig_model_;
        for (HighsInt col : slot_support.support) {
          if (!(original.col_lower_[col] <= 0 && original.col_upper_[col] >= 1)) ++parent_fixed;
          else if (std::binary_search(slot_support.free_columns.begin(), slot_support.free_columns.end(), col)) ++free_movable;
          else ++builder_fixed;
        }
        highsLogUser(mipsolver.options_mip_->log_options, HighsLogType::kInfo,
            "MIP-SlotChild: version=2 decision=%llu invocation=%llu worker=%lld restart=%lld "
            "seed_sequence=%lld ordinal=%lld coordinates=original support=%lld free_columns=%lld free_slots=%lld "
            "parent_fixed=%lld free_movable=%lld builder_fixed=%lld auxiliary_fixed=0 "
            "visited_rows=%lld visited_edges=%lld skipped_dense_rows=%lld support_cap=%d work_cap=%d "
            "first_lp_seconds=%.17g first_lp_iterations=%lld started=%d accepted=%d own_gain=%d\n",
            static_cast<unsigned long long>(decision.observation_id), static_cast<unsigned long long>(invocation),
            static_cast<long long>(worker.workerId()), static_cast<long long>(data.numRestarts),
            static_cast<long long>(context.incumbent_sequence), static_cast<long long>(partition_ordinal),
            static_cast<long long>(slot_support.support.size()),
            static_cast<long long>(slot_support.free_columns.size()), static_cast<long long>(slot_support.free_slots),
            static_cast<long long>(parent_fixed), static_cast<long long>(free_movable), static_cast<long long>(builder_fixed),
            static_cast<long long>(slot_support.visited_rows), static_cast<long long>(slot_support.visited_edges),
            static_cast<long long>(slot_support.skipped_dense_rows), int(slot_support.support_cap_reached),
            int(slot_support.work_cap_reached), outcome.children.retained == 1 ? outcome.children.items[0].first_lp_seconds : -1.0,
            static_cast<long long>(outcome.children.retained == 1 ? outcome.children.items[0].first_lp_iterations : 0),
            int(outcome.started), int(outcome.candidate_accepted), int(outcome.admission.measured && outcome.admission.improved));
      }
      if (manager.composedEnabled() && outcome.children.retained == 1)
        highsLogUser(mipsolver.options_mip_->log_options, HighsLogType::kInfo,
            "MIP-Neighbourhood-Child: version=1 decision=%llu invocation=%llu\n",
            static_cast<unsigned long long>(decision.observation_id),
            static_cast<unsigned long long>(outcome.children.items[0].invocation));
    } else {
      outcome.reservation_denied = !repair.granted();
      outcome.preparation_exhaustions = 1;
      outcome.preparation_stop_mask = remaining.max_time <= 0
          ? kHighsHeuristicStopDeadline : kHighsHeuristicStopCheckpoint;
    }
    outcome.phase_measured = outcome.fixing_rate_measured = true;
    outcome.child_lp_iterations = std::max<int64_t>(0,
        outcome.lp_iterations - outcome.completion_lp_iterations);
    outcome.child_calls = outcome.started ? 1 : 0;
    outcome.child_seconds = std::max(0.0, mipsolver.timer_.read() - child_started);
    outcome.elapsed = std::max(0.0, mipsolver.timer_.read() - started);
    outcome.preparation_seconds = std::max(0.0, outcome.elapsed - outcome.child_seconds);
    outcome.finished_time = mipsolver.timer_.read();
    if (worker.search_work.checkpointRequested() &&
        (!manager.composedEnabled() || (outcome.children.retained == 1 &&
         outcome.children.items[0].status == int(HighsModelStatus::kInterrupt))))
      outcome.preparation_stop_mask |= kHighsHeuristicStopCheckpoint;
    reservation.finish(outcome);
    if (manager.adaptiveEnabled())
      highsLogUser(mipsolver.options_mip_->log_options, HighsLogType::kInfo,
          "MIP-Adaptive-Outcome: version=1 decision=%llu started=%d completed=%d "
          "proposed=%d accepted=%d improved=%d lp_iterations=%lld "
          "setup_seconds=%.17g child_seconds=%.17g elapsed=%.17g finished_time=%.17g\n",
          static_cast<unsigned long long>(decision.observation_id), int(outcome.started),
          int(outcome.completed), int(outcome.candidate_proposed), int(outcome.candidate_accepted),
          int(outcome.candidate_improved), static_cast<long long>(outcome.lp_iterations),
          outcome.preparation_seconds, outcome.child_seconds, outcome.elapsed, outcome.finished_time);
    return true;
  }
  return false;
}

const char* HighsPrimalHeuristics::subMipParentPhaseName(
    HighsSubMipOrigin origin) {
  switch (origin) {
    case HighsSubMipOrigin::kRootReducedCost:
    case HighsSubMipOrigin::kRootRens:
      return "root";
    case HighsSubMipOrigin::kTreeRens:
    case HighsSubMipOrigin::kTreeRins:
    case HighsSubMipOrigin::kTreeLocalBranching:
      return "tree";
  }
  return "unknown";
}

static int subMipSolutionSource(HighsSubMipOrigin origin) {
  switch (origin) {
    case HighsSubMipOrigin::kRootReducedCost:
      return kSolutionSourceRootReducedCost;
    case HighsSubMipOrigin::kRootRens:
    case HighsSubMipOrigin::kTreeRens:
      return kSolutionSourceRens;
    case HighsSubMipOrigin::kTreeRins:
      return kSolutionSourceRins;
    case HighsSubMipOrigin::kTreeLocalBranching:
      return kSolutionSourceSubMip;
  }
  return kSolutionSourceSubMip;
}

void HighsPrimalHeuristics::setupIntCols() {
  intcols = mipsolver.mipdata_->integer_cols;

  pdqsort(intcols.begin(), intcols.end(), [&](HighsInt c1, HighsInt c2) {
    const FP_32BIT_VOLATILE double lockScore1 =
        (mipsolver.mipdata_->feastol + mipsolver.mipdata_->uplocks[c1]) *
        (mipsolver.mipdata_->feastol + mipsolver.mipdata_->downlocks[c1]);

    const FP_32BIT_VOLATILE double lockScore2 =
        (mipsolver.mipdata_->feastol + mipsolver.mipdata_->uplocks[c2]) *
        (mipsolver.mipdata_->feastol + mipsolver.mipdata_->downlocks[c2]);

    if (lockScore1 > lockScore2) return true;
    if (lockScore2 > lockScore1) return false;

    const FP_32BIT_VOLATILE double cliqueScore1 =
        (mipsolver.mipdata_->feastol +
         mipsolver.mipdata_->cliquetable.getNumImplications(c1, 1)) *
        (mipsolver.mipdata_->feastol +
         mipsolver.mipdata_->cliquetable.getNumImplications(c1, 0));

    const FP_32BIT_VOLATILE double cliqueScore2 =
        (mipsolver.mipdata_->feastol +
         mipsolver.mipdata_->cliquetable.getNumImplications(c2, 1)) *
        (mipsolver.mipdata_->feastol +
         mipsolver.mipdata_->cliquetable.getNumImplications(c2, 0));

    return std::make_tuple(cliqueScore1, HighsHashHelpers::hash(uint64_t(c1)),
                           c1) >
           std::make_tuple(cliqueScore2, HighsHashHelpers::hash(uint64_t(c2)),
                           c2);
  });
}

bool HighsPrimalHeuristics::solveSubMip(
    HighsMipWorker& worker, const HighsLp& lp, const HighsBasis& basis,
    double fixingRate, std::vector<double> colLower,
    std::vector<double> colUpper, HighsInt maxleaves, HighsInt maxnodes,
    HighsInt stallnodes, HighsSubMipOrigin origin, double max_time,
    const HighsSubMipCallBudget* call_budget,
    HighsPrimalHeuristicOutcome* outcome,
    const std::vector<double>* starting_solution,
    const HighsLocalBranchingRow* local_branching,
    const HighsReplayContext* replay_context, bool original_space) {
  const double call_started = mipsolver.timer_.read();
  if (!subMipLaunchAllowed(mipsolver.submip, mipsolver.submip_level)) {
    highsLogUser(
        mipsolver.options_mip_->log_options, HighsLogType::kError,
        "Nested heuristic sub-MIP rejected at recursion depth %lld\n",
        static_cast<long long>(mipsolver.submip_level));
    worker.setHeurTerminationStatus(HighsModelStatus::kSolveError);
    if (outcome != nullptr) *outcome = HighsPrimalHeuristicOutcome{};
    return false;
  }
  if (outcome != nullptr) {
    *outcome = HighsPrimalHeuristicOutcome{};
    outcome->fixing_rate = fixingRate;
  }
  if (call_budget != nullptr &&
      (call_budget->max_time <= 0 || call_budget->max_lp_iterations == 0 ||
       call_budget->max_nodes == 0 || call_budget->max_leaves == 0 ||
       call_budget->max_stall_nodes == 0))
    return false;
  if (local_branching) {
    if (lp.num_row_ == std::numeric_limits<HighsInt>::max() ||
        local_branching->index.empty() ||
        local_branching->index.size() != local_branching->value.size() ||
        !std::isfinite(local_branching->upper) ||
        colLower.size() != static_cast<size_t>(lp.num_col_) ||
        colUpper.size() != static_cast<size_t>(lp.num_col_)) return false;
    std::unordered_set<HighsInt> indices;
    for (size_t j = 0; j < local_branching->index.size(); ++j) {
      const HighsInt col = local_branching->index[j];
      if (col < 0 || col >= lp.num_col_ ||
          (local_branching->value[j] != 1 && local_branching->value[j] != -1) ||
          !indices.insert(col).second) return false;
    }
  }
  HighsOptions submipoptions = *mipsolver.options_mip_;
  HighsLp submip = lp;

  // set bounds and restore integrality of the lp relaxation copy
  submip.col_lower_ = std::move(colLower);
  submip.col_upper_ = std::move(colUpper);
  submip.integrality_ = original_space ? mipsolver.orig_model_->integrality_ : mipsolver.model_->integrality_;
  if (!original_space) submip.offset_ = 0;
  if (local_branching) {
    HighsSparseMatrix added;
    added.format_ = MatrixFormat::kRowwise;
    added.num_row_ = 1;
    added.num_col_ = submip.num_col_;
    added.start_ = {0, static_cast<HighsInt>(local_branching->index.size())};
    added.index_ = local_branching->index;
    added.value_ = local_branching->value;
    submip.a_matrix_.addRows(added);
    ++submip.num_row_;
    submip.row_lower_.push_back(-kHighsInf);
    submip.row_upper_.push_back(local_branching->upper);
    if (!submip.row_names_.empty()) submip.row_names_.push_back("portfolio_local_branching");
  }

  // set limits
  if (call_budget != nullptr) {
    if (call_budget->max_leaves >= 0)
      maxleaves = std::min<int64_t>(maxleaves, call_budget->max_leaves);
    if (call_budget->max_nodes >= 0)
      maxnodes = std::min<int64_t>(maxnodes, call_budget->max_nodes);
    if (call_budget->max_stall_nodes >= 0)
      stallnodes = std::min<int64_t>(stallnodes,
                                     call_budget->max_stall_nodes);
    if (std::isfinite(call_budget->max_time))
      max_time = std::min(max_time, call_budget->max_time);
  }
  submipoptions.mip_max_leaves = maxleaves;
  submipoptions.output_flag = false;

  const bool allow_submip_log = true;
  if (allow_submip_log && lp.num_col_ == -54 && lp.num_row_ == -172) {
    submipoptions.output_flag = true;
    if (mipsolver.profiling_->sub_solver_)
      printf(
          "HighsPrimalHeuristics::solveSubMip (%d, %d) with output_flag = %s\n",
          int(lp.num_col_), int(lp.num_row_),
          highsBoolToString(submipoptions.output_flag).c_str());
  }

  submipoptions.mip_max_nodes = maxnodes;
  submipoptions.mip_max_stall_nodes = stallnodes;
  if (call_budget != nullptr && call_budget->max_lp_iterations >= 0)
    submipoptions.simplex_iteration_limit = std::min<int64_t>(
        submipoptions.simplex_iteration_limit,
        call_budget->max_lp_iterations);
  submipoptions.mip_pscost_minreliable = 0;
  submipoptions.time_limit -= mipsolver.timer_.read();
  if (max_time >= 0.0)
    submipoptions.time_limit = std::min(submipoptions.time_limit, max_time);
  submipoptions.time_limit = std::max(0.0, submipoptions.time_limit);
  submipoptions.objective_bound = original_space
      ? worker.upper_limit + mipsolver.model_->offset_ : worker.upper_limit;

  const bool log_provenance =
      !mipsolver.submip && mipsolver.options_mip_->mip_root_work_budget;
  if (!mipsolver.submip) {
    double curr_abs_gap = worker.upper_limit - mipsolver.mipdata_->lower_bound;

    if (curr_abs_gap == kHighsInf) {
      curr_abs_gap = fabs(mipsolver.mipdata_->lower_bound);
      if (curr_abs_gap == kHighsInf) curr_abs_gap = 0.0;
    }

    submipoptions.mip_rel_gap = 0.0;
    submipoptions.mip_abs_gap =
        mipsolver.mipdata_->feastol * std::max(curr_abs_gap, 1000.0);
  }

  // check if only root presolve is allowed
  if (submipoptions.mip_root_presolve_only)
    submipoptions.presolve = kHighsOffString;
  else
    submipoptions.presolve = kHighsOnString;
  configureSubMipOptions(submipoptions, origin);
  // setup solver and run it

  HighsSolution solution;
  solution.value_valid = false;
  solution.dual_valid = false;
  if (starting_solution != nullptr &&
      highsVerifyCandidate(submip, *starting_solution, nullptr,
                           submipoptions.mip_feasibility_tolerance).feasible &&
      calculateRowValuesQuad(submip, *starting_solution, solution.row_value) ==
          HighsStatus::kOk) {
    solution.col_value = *starting_solution;
    solution.value_valid = true;
  }
  submipoptions.time_limit = std::min(
      submipoptions.time_limit,
      remainingHeuristicTime(call_started, mipsolver.timer_.read(), max_time));
  if (submipoptions.time_limit <= 0) return false;
  // Create HighsMipSolver instance for sub-MIP
  HighsReplayContext capture_context = replay_context ? *replay_context : HighsReplayContext{};
  capture_context.method = subMipMethodName(origin);
  capture_context.parent_phase = subMipParentPhaseName(origin);
  capture_context.parent_time = mipsolver.timer_.read();
  capture_context.options = &submipoptions;
  capture_context.helper_lp_limit = worker.search_work.remaining();
  if (call_budget && call_budget->max_lp_iterations >= 0)
    capture_context.helper_lp_limit = std::min<int64_t>(
        capture_context.helper_lp_limit, call_budget->max_lp_iterations);
  const double capture_started = mipsolver.timer_.read();
  const bool captured = HighsReplayCapture::capture(
      submip, "sub_mip", mipsolver.submip_level,
      local_branching || original_space ? nullptr : &basis, &capture_context);
  const double capture_elapsed = mipsolver.timer_.read() - capture_started;
  HighsMipSolver submipsolver(*mipsolver.callback_, submipoptions, submip,
                              solution, true, mipsolver.submip_level + 1);
  submipsolver.helper_lp_limit = worker.search_work.remaining();
  submipsolver.parent_candidate_checkpoint = worker.search_work.checkpoint();
  submipsolver.parent_candidate_sequence = worker.search_work.batchSequence();
  if (mipsolver.parent_candidate_checkpoint) {
    submipsolver.parent_candidate_checkpoint = mipsolver.parent_candidate_checkpoint;
    submipsolver.parent_candidate_sequence = mipsolver.parent_candidate_sequence;
  }
  if (call_budget != nullptr && call_budget->max_lp_iterations >= 0)
    submipsolver.helper_lp_limit = std::min<int64_t>(
        submipsolver.helper_lp_limit, call_budget->max_lp_iterations);
  // Initialise termination_status_ and propagate any terminator to
  // the sub-MIP
  submipsolver.initialiseTerminator(mipsolver);
  // A newly appended row invalidates the copied basis extent. Use the native
  // cold child basis rather than changing the parent's proof LP or basis.
  submipsolver.rootbasis = local_branching || original_space ? nullptr : &basis;
  HighsPseudocostInitialization pscostinit(worker.getPseudocost(), 1);
  if (!original_space) {
    submipsolver.pscostinit = &pscostinit;
    submipsolver.clqtableinit = &mipsolver.mipdata_->cliquetable;
    submipsolver.implicinit = &mipsolver.mipdata_->implications;
  }
  submipoptions.time_limit = std::min(
      submipoptions.time_limit,
      remainingHeuristicTime(call_started, mipsolver.timer_.read(), max_time));
  submipoptions.time_limit = std::min(
      submipoptions.time_limit,
      std::max(0.0, mipsolver.options_mip_->time_limit - mipsolver.timer_.read()));
  if (submipoptions.time_limit <= 0) return false;
  const bool trace_child = outcome &&
      mipsolver.mipdata_->heuristic_manager.config().trace_decisions;
  const int64_t invocation = (log_provenance || trace_child ||
      mipsolver.mipdata_->heuristic_manager.progressiveEnabled())
      ? nextSubMipInvocation.fetch_add(1, std::memory_order_relaxed) + 1 : 0;
  const double invocation_started = call_started;
  std::unique_ptr<HighsChildProposalSink> progressive;
  HighsPrimalAdmissionReceipt progressive_admission;
  std::vector<double> progressive_reduced;
  struct ProgressivePublication {
    int64_t sequence, epoch;
    double discovery, publication;
    bool own_gain;
  };
  std::vector<ProgressivePublication> progressive_publications;
  double first_publication = -1;
  // Reset before the owned sink is destroyed, including an exceptional exit.
  struct ProposalLinkReset {
    HighsMipSolver& child;
    ~ProposalLinkReset() { child.child_proposal_sink = nullptr; }
  } proposal_link_reset{submipsolver};
  // Keep an independent mapping cursor. Parent transformations may be shared
  // by sibling workers; getReducedPrimalSolution mutates its stack cursor.
  if (!mipsolver.submip && outcome &&
      mipsolver.mipdata_->heuristic_manager.progressiveEnabled()) {
    const int64_t generation = mipsolver.callback_->data_in.admission_generation;
    auto mapping = std::make_shared<presolve::HighsPostsolveStack>(
        mipsolver.mipdata_->postSolveStack);
    progressive.reset(new HighsChildProposalSink(submip,
        mipsolver.mipdata_->feastol,
        [&, mapping, generation](const std::vector<double>& values) {
          if (generation != mipsolver.callback_->data_in.admission_generation)
            return false;
          std::vector<double> reduced = original_space
              ? mapping->getReducedPrimalSolution(values) : values;
          const auto parent_check = highsVerifyCandidate(*mipsolver.model_,
              reduced, nullptr, mipsolver.mipdata_->feastol);
          if (!parent_check.feasible ||
              static_cast<double>(parent_check.objective) - mipsolver.model_->offset_ >=
                  worker.upper_bound)
            return false;
          if (original_space) {
            HighsSolution restored;
            restored.value_valid = true;
            restored.col_value = reduced;
            mapping->undoPrimal(*mipsolver.options_mip_, restored, -1, true);
            if (!highsVerifyCandidate(submip, restored.col_value, nullptr,
                                     mipsolver.mipdata_->feastol).feasible)
              return false;
          }
          HighsPrimalAdmissionReceipt receipt;
          const int64_t sequence = worker.nextPublicationSequence();
          // Worker-local checks/mailbox only, including the single-worker case.
          // Never recurse into parent addIncumbent or public callbacks here.
          if (!worker.trySolution(reduced, subMipSolutionSource(origin), &receipt))
            return false;
          if (worker.nextPublicationSequence() != sequence + 1) return false;
          progressive_publications.push_back({sequence, worker.workEpoch(),
              progressive->current_discovery_s, mipsolver.timer_.read(),
              receipt.measured && receipt.improved});
          progressive_reduced = std::move(reduced);
          if (receipt.measured && receipt.improved &&
              (!progressive_admission.improved ||
               receipt.candidate < progressive_admission.candidate))
            progressive_admission = std::move(receipt);
          if (first_publication < 0)
            first_publication = progressive_publications.back().publication;
          return true;
        }, 8, [&]() { return mipsolver.timer_.read(); }));
    submipsolver.child_proposal_sink = progressive.get();
  }
  if (progressive) {
    const double prepared_at = mipsolver.timer_.read();
    submipoptions.time_limit = std::min(submipoptions.time_limit,
        remainingHeuristicTime(call_started, prepared_at, max_time));
    submipoptions.time_limit = std::min(submipoptions.time_limit,
        std::max(0.0, mipsolver.options_mip_->time_limit - prepared_at));
    if (submipoptions.time_limit <= 0) return false;
  }
  if (captured && log_provenance)
    highsLogUser(mipsolver.options_mip_->log_options, HighsLogType::kInfo,
        "MIP-ReplayCapture: version=1 method=%s decision=%llu invocation=%lld "
        "worker=%lld restart=%lld elapsed=%.17g scope=restricted_model_only\n",
        capture_context.method, static_cast<unsigned long long>(capture_context.decision),
        static_cast<long long>(invocation), static_cast<long long>(capture_context.worker),
        static_cast<long long>(capture_context.restart), capture_elapsed);
  if (log_provenance)
    highsLogUser(
        mipsolver.options_mip_->log_options, HighsLogType::kInfo,
        "MIP-Incumbent: schema=rost/highs-incumbent/v1 event=submip_start "
        "invocation=%lld method=%s parent_phase=%s recursion_depth=%lld "
        "fixing_rate=%.17g max_leaves=%lld max_nodes=%lld stall_nodes=%lld "
        "max_time=%.17g primal_before=%.17g\n",
        static_cast<long long>(invocation), subMipMethodName(origin),
        subMipParentPhaseName(origin),
        static_cast<long long>(mipsolver.submip_level), fixingRate,
        static_cast<long long>(maxleaves), static_cast<long long>(maxnodes),
        static_cast<long long>(stallnodes), submipoptions.time_limit,
        worker.upper_limit);
  if (!mipsolver.submip && !mipsolver.mipdata_->parallelLockActive())
    mipsolver.profiling_->start(kMipClockSubMipSolve);
  mipsolver.profiling_->solveCall("MIP", mipsolver.submip);
  // Solve the sub-MIP
  //
  // Copy the pointer to global sub-solver data into the sub-MIP
  // solver
  submipsolver.setProfiling(mipsolver.profiling_);
  // Stop the solve timer so that presolve/solve/postsolve for the
  // sub-MIP are timed independently
  const bool was_running_solve = mipsolver.profiling_->running(kSolveTime);
  if (was_running_solve) mipsolver.profiling_->stop(kSolveTime);
  // Only start timing the submip if the calling MIP isn't a sub-MIP.
  // Do not write directly to stdout here: output_flag=false is a public
  // embedding contract and callers retain structured provenance separately.
  if (!mipsolver.submip) mipsolver.profiling_->start(kSubSolverSubMip);
  // Ensure that sub-solver call time data accumulate in the sub-MIP record
  mipsolver.profiling_->setSubMip(true);
  const double child_run_started = mipsolver.timer_.read();
  {
    HighsTaskPhaseScope phase(worker.search_work, HighsTaskPhase::kChildMip);
    if (outcome != nullptr) outcome->started = true;
    submipsolver.run();
  }
  const double child_run_returned = mipsolver.timer_.read();
  submipsolver.child_proposal_sink = nullptr;
  for (const auto& publication : progressive_publications)
    highsLogUser(mipsolver.options_mip_->log_options, HighsLogType::kInfo,
        "MIP-Progressive-Publication: version=1 invocation=%lld worker=%lld "
        "epoch=%lld worker_sequence=%lld discovery_s=%.17g publication_s=%.17g "
        "own_gain=%d authority=worker_mailbox\n", static_cast<long long>(invocation),
        static_cast<long long>(worker.workerId()), static_cast<long long>(publication.epoch),
        static_cast<long long>(publication.sequence), publication.discovery,
        publication.publication, int(publication.own_gain));
  if (progressive)
    highsLogUser(mipsolver.options_mip_->log_options, HighsLogType::kInfo,
        "MIP-Progressive-Child: version=1 invocation=%lld events=%llu proposed=%llu "
        "published=%llu rejected=%llu duplicates=%llu non_improving=%llu "
        "capacity_denied=%llu first_discovery_s=%.17g first_publication_s=%.17g "
        "check_seconds=%.17g returned_s=%.17g "
        "authority=worker_mailbox\n", static_cast<long long>(invocation),
        static_cast<unsigned long long>(progressive->events),
        static_cast<unsigned long long>(progressive->proposed),
        static_cast<unsigned long long>(progressive->published),
        static_cast<unsigned long long>(progressive->rejected),
        static_cast<unsigned long long>(progressive->duplicates),
        static_cast<unsigned long long>(progressive->non_improving),
        static_cast<unsigned long long>(progressive->capacity_denied),
        progressive->first_discovery_s, first_publication,
        progressive->check_seconds, child_run_returned);
  // Ensure that further sub-solver call time data accumulate in the
  // MIP or sub-MIP record, according to whether the calling MIP is a
  // sub-MIP
  mipsolver.profiling_->setSubMip(mipsolver.submip);
  if (!mipsolver.submip) mipsolver.profiling_->stop(kSubSolverSubMip);
  worker.updateHeurStatsMaxSubMipLevel(submipsolver.max_submip_level + 1);
  if (!mipsolver.submip && !mipsolver.mipdata_->parallelLockActive()) {
    // Only stop timing the submip if the calling MIP isn't a sub-MIP
    mipsolver.profiling_->stop(kMipClockSubMipSolve);
  }
  if (was_running_solve) {
    // Re-start the solve timer now that presolve/solve/postsolve for the
    // sub-MIP have been timed independently
    const bool restart = true;
    mipsolver.profiling_->start(kSolveTime, restart);
  }
  // 22/07/25: Seems impossible for submipsolver.mipdata_ to be a null
  // pointer after calling HighsMipSolver::run(), and assert isn't
  // triggered for anything in ctest, but use direct test of
  // submipsolver.termination_status_, rather than
  // submipsolver.mipdata_.terminatorTerminated()
  if (!submipsolver.mipdata_) {
    printf(
        "HighsPrimalHeuristics::solveSubMip: submipsolver.mipdata_ is "
        "nullptr\n");
    assert(submipsolver.mipdata_);
  }
  const auto log_submip_end = [&](bool candidate_proposed,
                                  bool candidate_accepted,
                                  bool candidate_improved) {
    const int64_t lp_iterations =
        submipsolver.mipdata_ ? submipsolver.mipdata_->total_lp_iterations : 0;
    const int64_t completion_lp = outcome ? outcome->completion_lp_iterations : 0;
    worker.search_work.chargeSubMip(lp_iterations + completion_lp);
    if (outcome != nullptr) {
      outcome->completed = true;
      outcome->candidate_proposed = candidate_proposed;
      outcome->candidate_accepted = candidate_accepted;
      outcome->candidate_improved = candidate_improved;
      outcome->elapsed = mipsolver.timer_.read() - invocation_started;
      outcome->nodes = submipsolver.node_count_;
      outcome->leaves = submipsolver.mipdata_
                            ? submipsolver.mipdata_->num_leaves
                            : 0;
      outcome->lp_iterations = lp_iterations + completion_lp;
      if (trace_child) {
        HighsPrimalChildReceipt receipt;
        receipt.invocation = static_cast<uint64_t>(invocation);
        receipt.status = static_cast<int>(submipsolver.modelstatus_);
        receipt.input_rows = submip.num_row_;
        receipt.input_columns = submip.num_col_;
        receipt.input_nonzeros = submip.a_matrix_.numNz();
        receipt.working_rows = submipsolver.numRow();
        receipt.working_columns = submipsolver.numCol();
        receipt.working_nonzeros = submipsolver.numNonzero();
        receipt.lp_iterations = lp_iterations;
        receipt.nodes = submipsolver.node_count_;
        receipt.fixing_rate = fixingRate;
        receipt.started_time = child_run_started;
        receipt.returned_time = child_run_returned;
        receipt.setup_seconds = std::max(0.0, child_run_started - call_started);
        receipt.first_lp_seconds = submipsolver.child_first_lp_seconds;
        receipt.first_lp_iterations = submipsolver.child_first_lp_iterations;
        receipt.proposed = candidate_proposed;
        receipt.accepted = candidate_accepted;
        receipt.admission_measured = outcome->admission.measured;
        receipt.own_gain = outcome->admission.measured && outcome->admission.improved;
        outcome->children.append(receipt);
      }
    }
    if (!log_provenance) return;
    highsLogUser(
        mipsolver.options_mip_->log_options, HighsLogType::kInfo,
        "MIP-Incumbent: schema=rost/highs-incumbent/v1 event=submip_end "
        "invocation=%lld method=%s parent_phase=%s recursion_depth=%lld "
        "status=%d elapsed=%.17g nodes=%lld lp_iterations=%lld "
        "candidate_proposed=%d candidate_accepted=%d candidate_improved=%d "
        "primal_before=%.17g primal_after=%.17g\n",
        static_cast<long long>(invocation), subMipMethodName(origin),
        subMipParentPhaseName(origin),
        static_cast<long long>(mipsolver.submip_level),
        static_cast<int>(submipsolver.modelstatus_),
        mipsolver.timer_.read() - invocation_started,
        static_cast<long long>(submipsolver.node_count_),
        static_cast<long long>(lp_iterations), candidate_proposed ? 1 : 0,
        candidate_accepted ? 1 : 0, candidate_improved ? 1 : 0,
        submipoptions.objective_bound, worker.upper_limit);
  };
  if (submipsolver.termination_status_ != HighsModelStatus::kNotset) {
    worker.setHeurTerminationStatus(submipsolver.termination_status_);
    if (outcome) outcome->admission = progressive_admission;
    log_submip_end(progressive && progressive->proposed > 0,
                   progressive && progressive->published > 0,
                   progressive_admission.improved);
    return false;
  }
  if (submipsolver.mipdata_) {
    double numUnfixed = mipsolver.mipdata_->integral_cols.size() +
                        mipsolver.mipdata_->continuous_cols.size();
    double adjustmentfactor =
        ((1 - fixingRate) * mipsolver.mipdata_->integral_cols.size() +
         mipsolver.mipdata_->continuous_cols.size()) /
        std::max(1.0, numUnfixed);
    // (double)mipsolver.orig_model_->a_matrix_.value_.size();
    int64_t adjusted_lp_iterations =
        (size_t)(adjustmentfactor * submipsolver.mipdata_->total_lp_iterations);
    if (original_space) adjusted_lp_iterations = submipsolver.mipdata_->total_lp_iterations;
    worker.updateHeurStatsLpIters(
        adjusted_lp_iterations, submipsolver.mipdata_->total_repair_lp,
        submipsolver.mipdata_->total_repair_lp_feasible,
        submipsolver.mipdata_->total_repair_lp_iterations);
    // Warning: This will not be deterministic if sub-mips are run in parallel
    if (mipsolver.submip)
      mipsolver.mipdata_->num_nodes += std::max(
          int64_t{1}, int64_t(adjustmentfactor * submipsolver.node_count_));
  }

  if (submipsolver.modelstatus_ == HighsModelStatus::kInfeasible) {
    worker.updateHeurStatsInfeasObservations(fixingRate);
  }
  if (submipsolver.node_count_ <= 1 &&
      submipsolver.modelstatus_ == HighsModelStatus::kInfeasible) {
    log_submip_end(false, false, false);
    return false;
  }
  double oldUpperLimit = worker.upper_limit;
  bool candidate_proposed = progressive && progressive->proposed > 0;
  bool candidate_accepted = progressive && progressive->published > 0;
  HighsPrimalAdmissionReceipt admission = progressive_admission;
  const int64_t completion_before = mipsolver.mipdata_->parallelLockActive()
      ? 0 : mipsolver.mipdata_->total_repair_lp_iterations;
  if (submipsolver.modelstatus_ != HighsModelStatus::kInfeasible &&
      !submipsolver.solution_.empty()) {
    candidate_proposed = true;
    if (original_space) {
      const bool checked = highsVerifyCandidate(submip, submipsolver.solution_,
          nullptr, mipsolver.mipdata_->feastol).feasible;
      bool mapped = false;
      if (checked) {
        auto mapping = mipsolver.mipdata_->postSolveStack;
        const auto reduced = mapping.getReducedPrimalSolution(submipsolver.solution_);
        mapped = highsVerifyCandidate(*mipsolver.model_, reduced, nullptr,
                                     mipsolver.mipdata_->feastol).feasible;
        if (mapped) {
          HighsSolution restored;
          restored.value_valid = true;
          restored.col_value = reduced;
          mapping.undoPrimal(*mipsolver.options_mip_, restored, -1, true);
          mapped = highsVerifyCandidate(submip, restored.col_value, nullptr,
                                        mipsolver.mipdata_->feastol).feasible;
        }
        if (mapped && !(mipsolver.mipdata_->parallelLockActive() &&
                        reduced == progressive_reduced)) {
          HighsPrimalAdmissionReceipt final_admission;
          candidate_accepted |= trySolution(reduced, subMipSolutionSource(origin), worker, &final_admission);
          if (!admission.measured || (final_admission.improved &&
              (!admission.improved || final_admission.candidate < admission.candidate)))
            admission = std::move(final_admission);
        }
      }
      highsLogUser(mipsolver.options_mip_->log_options, HighsLogType::kInfo,
          "MIP-Partition-Admission: version=1 invocation=%lld checked=%d mapped=%d accepted=%d\n",
          static_cast<long long>(invocation), checked ? 1 : 0, mapped ? 1 : 0,
          candidate_accepted ? 1 : 0);
    } else {
      if (!(mipsolver.mipdata_->parallelLockActive() &&
            submipsolver.solution_ == progressive_reduced)) {
        HighsPrimalAdmissionReceipt final_admission;
        candidate_accepted |= trySolution(submipsolver.solution_,
                                         subMipSolutionSource(origin), worker, &final_admission);
        if (!admission.measured || (final_admission.improved &&
            (!admission.improved || final_admission.candidate < admission.candidate)))
          admission = std::move(final_admission);
      }
    }
  }

  const bool portfolio_call = mipsolver.mipdata_->heuristic_manager.portfolioEnabled();
  const bool candidate_improved = portfolio_call
      ? admission.measured && admission.improved : worker.upper_limit < oldUpperLimit;
  if (outcome) {
    outcome->admission = admission;
    outcome->parent_lower_bound = mipsolver.mipdata_->lower_bound;
    outcome->completion_lp_iterations = mipsolver.mipdata_->parallelLockActive()
        ? 0 : std::max<int64_t>(0, mipsolver.mipdata_->total_repair_lp_iterations - completion_before);
  }
  if (candidate_improved) {
    // remember fixing rate as good
    worker.updateHeurStatsSuccessObservations(fixingRate);
  }

  log_submip_end(candidate_proposed, candidate_accepted, candidate_improved);

  return true;
}

double HighsPrimalHeuristics::determineTargetFixingRate(
    HighsMipWorker& worker) {
  double lowFixingRate = 0.6;
  double highFixingRate = 0.6;

  HighsRandom& randgen =
      mipsolver.mipdata_->parallelLockActive() ? worker.randgen : this->randgen;

  if (getNumInfeasObservations(worker) != 0) {
    double infeasRate =
        getInfeasObservations(worker) / getNumInfeasObservations(worker);
    highFixingRate = 0.9 * infeasRate;
    lowFixingRate = std::min(lowFixingRate, highFixingRate);
  }

  if (getNumSuccessObservations(worker) != 0) {
    double successFixingRate =
        getSuccessObservations(worker) / getNumSuccessObservations(worker);
    lowFixingRate = std::min(lowFixingRate, 0.9 * successFixingRate);
    highFixingRate = std::max(successFixingRate * 1.1, highFixingRate);
  }

  double fixingRate = randgen.real(lowFixingRate, highFixingRate);
  // if (!mipsolver.submip) printf("fixing rate: %.2f\n", 100.0 * fixingRate);
  return fixingRate;
}

class HeuristicNeighbourhood {
  HighsDomain& localdom;
  HighsInt numFixed;
  HighsHashTable<HighsInt> fixedCols;
  size_t startCheckedChanges;
  size_t nCheckedChanges;
  HighsInt numTotal;

 public:
  HeuristicNeighbourhood(const HighsMipSolver& mipsolver, HighsDomain& localdom)
      : localdom(localdom),
        numFixed(0),
        startCheckedChanges(localdom.getDomainChangeStack().size()),
        nCheckedChanges(startCheckedChanges) {
    for (HighsInt i : mipsolver.mipdata_->integral_cols)
      if (localdom.col_lower_[i] == localdom.col_upper_[i]) ++numFixed;

    numTotal = mipsolver.mipdata_->integral_cols.size() - numFixed;
  }

  double getFixingRate() {
    while (nCheckedChanges < localdom.getDomainChangeStack().size()) {
      HighsInt col = localdom.getDomainChangeStack()[nCheckedChanges++].column;
      if (localdom.variableType(col) == HighsVarType::kContinuous) continue;
      if (localdom.isFixed(col)) fixedCols.insert(col);
    }

    return numTotal ? static_cast<double>(fixedCols.size()) /
                          static_cast<double>(numTotal)
                    : 0.0;
  }

  void backtracked() {
    nCheckedChanges = startCheckedChanges;
    if (fixedCols.size()) fixedCols.clear();
  }
};

void HighsPrimalHeuristics::rootReducedCost(HighsMipWorker& worker,
                                            double max_submip_time) {
  const double heuristic_started = mipsolver.timer_.read();
  const auto remaining_time = [&]() {
    return remainingHeuristicTime(heuristic_started, mipsolver.timer_.read(),
                                  max_submip_time);
  };
  std::vector<std::pair<double, HighsDomainChange>> lurkingBounds =
      mipsolver.mipdata_->redcostfixing.getLurkingBounds(
          mipsolver, worker.getGlobalDomain());
  if (10 * lurkingBounds.size() < mipsolver.mipdata_->integral_cols.size())
    return;
  pdqsort(lurkingBounds.begin(), lurkingBounds.end(),
          [](const std::pair<double, HighsDomainChange>& a,
             const std::pair<double, HighsDomainChange>& b) {
            return a.first > b.first;
          });

  HighsDomain localdom = worker.getGlobalDomain();

  HeuristicNeighbourhood neighbourhood(mipsolver, localdom);

  double currCutoff = kHighsInf;
  double lower_bound =
      mipsolver.mipdata_->lower_bound + mipsolver.mipdata_->feastol;

  for (const std::pair<double, HighsDomainChange>& domchg : lurkingBounds) {
    if (remaining_time() <= 0.0) return;
    currCutoff = domchg.first;

    if (currCutoff <= lower_bound) break;

    if (localdom.isActive(domchg.second)) continue;
    localdom.changeBound(domchg.second);

    while (true) {
      localdom.propagate();
      if (localdom.infeasible()) {
        localdom.conflictAnalysis(worker.getConflictPool(),
                                  worker.getGlobalDomain(),
                                  worker.getPseudocost());

        mipsolver.mipdata_->updateLowerBound(
            std::max(mipsolver.mipdata_->lower_bound, currCutoff));

        localdom.backtrack();
        if (localdom.getBranchDepth() == 0) break;
        neighbourhood.backtracked();
        continue;
      }
      break;
    }
    double fixingRate = neighbourhood.getFixingRate();
    if (fixingRate >= 0.5) break;
    // double gap = (currCutoff - mipsolver.mipdata_->lower_bound) /
    //             std::max(std::abs(mipsolver.mipdata_->lower_bound), 1.0);
    // if (gap < 0.001) break;
  }

  double fixingRate = neighbourhood.getFixingRate();
  if (fixingRate < 0.3) return;

  const double submip_time = remaining_time();
  if (submip_time <= 0.0) return;

  solveSubMip(worker, *mipsolver.model_, mipsolver.mipdata_->firstrootbasis,
              fixingRate, localdom.col_lower_, localdom.col_upper_,
              500,  // std::max(50, int(0.05 *
                    // (mipsolver.mipdata_->num_leaves))),
              200 + static_cast<HighsInt>(mipsolver.mipdata_->num_nodes / 20),
              12, HighsSubMipOrigin::kRootReducedCost, submip_time);
}

static double calcFixVal(double rootchange, double fracval, double cost) {
  // reinforce direction of this solution away from root
  // solution if the change is at least 0.4
  // otherwise take the direction where the objective gets worse
  // if objective is zero round to nearest integer
  if (rootchange >= 0.4)
    return std::ceil(fracval);
  else if (rootchange <= -0.4)
    return std::floor(fracval);
  else if (cost > 0.0)
    return std::ceil(fracval);
  else if (cost < 0.0)
    return std::floor(fracval);
  else
    return std::floor(fracval + 0.5);
}

void HighsPrimalHeuristics::RENS(HighsMipWorker& worker,
                                 const std::vector<double>& tmp,
                                 double max_submip_time,
                                 HighsSubMipOrigin origin) {
  const double heuristic_started = mipsolver.timer_.read();
  const auto remaining_time = [&]() {
    return remainingHeuristicTime(heuristic_started, mipsolver.timer_.read(),
                                  max_submip_time);
  };
  // return if domain is infeasible
  if (worker.getGlobalDomain().infeasible() || remaining_time() <= 0.0) return;

  HighsPseudocost pscost(worker.getPseudocost());
  HighsSearch heur(worker, pscost);

  HighsDomain& localdom = heur.getLocalDomain();
  heur.setHeuristic(true);

  std::vector<HighsInt> intcols_;
  if (mipsolver.mipdata_->parallelLockActive()) {
    intcols_ = intcols;
  }
  std::vector<HighsInt>& intcols =
      mipsolver.mipdata_->parallelLockActive() ? intcols_ : this->intcols;
  intcols.erase(std::remove_if(intcols.begin(), intcols.end(),
                               [&](HighsInt i) {
                                 return worker.getGlobalDomain().isFixed(i);
                               }),
                intcols.end());

  // LP relaxation instantiation
  HighsTaskPhaseScope preparation_phase(worker.search_work,
                                        HighsTaskPhase::kHeuristicPreparation);
  HighsLpRelaxation heurlp(worker.getLpRelaxation());
  heurlp.setHelperDeadline(heuristic_started + max_submip_time);
  heurlp.setMipWorker(worker);
  heurlp.setCandidateInterruptible();
  heurlp.setProfiling(mipsolver.profiling_);
  // only use the global upper limit as LP limit so that dual proofs are valid
  heurlp.setObjectiveLimit(worker.upper_limit);
  heurlp.setAdjustSymmetricBranchingCol(false);
  heur.setLpRelaxation(&heurlp);

  heurlp.getLpSolver().changeColsBounds(0, mipsolver.numCol() - 1,
                                        localdom.col_lower_.data(),
                                        localdom.col_upper_.data());
  localdom.clearChangedCols();
  heur.createNewNode();

  // determine the initial number of unfixed variables fixing rate to decide if
  // the problem is restricted enough to be considered for solving a submip
  double maxfixingrate = determineTargetFixingRate(worker);
  double fixingrate = 0.0;
  bool stop = false;
  // heurlp.setIterationLimit(2 * mipsolver.mipdata_->maxrootlpiters);
  // printf("iterlimit: %" HIGHSINT_FORMAT "\n",
  //       heurlp.getLpSolver().getOptions().simplex_iteration_limit);
  HighsInt targetdepth = 1;
  HighsInt nbacktracks = -1;
  HeuristicNeighbourhood neighbourhood(mipsolver, localdom);
retry:
  if (remaining_time() <= 0.0 || worker.search_work.checkpointRequested() ||
      mipsolver.parentCheckpointRequested()) {
    worker.getHeurLpIterations() += heur.getLocalLpIterations();
    return;
  }
  ++nbacktracks;
  neighbourhood.backtracked();
  // printf("current depth : %" HIGHSINT_FORMAT
  //        "   target depth : %" HIGHSINT_FORMAT "\n",
  //        heur.getCurrentDepth(), targetdepth);
  if (heur.getCurrentDepth() > targetdepth) {
    if (!heur.backtrackUntilDepth(targetdepth)) {
      worker.getHeurLpIterations() += heur.getLocalLpIterations();
      return;
    }
  }

  // printf("fixingrate before loop is %g\n", fixingrate);
  assert(heur.hasNode());
  while (true) {
    if (remaining_time() <= 0.0 || worker.search_work.checkpointRequested() ||
        mipsolver.parentCheckpointRequested()) {
      worker.getHeurLpIterations() += heur.getLocalLpIterations();
      return;
    }
    // printf("evaluating node\n");
    const auto evaluation = heur.evaluateNode();
    if (evaluation == HighsSearch::NodeResult::kLpIterationLimit ||
        evaluation == HighsSearch::NodeResult::kCandidateCheckpoint) {
      worker.getHeurLpIterations() += heur.getLocalLpIterations();
      return;
    }
    // printf("done evaluating node\n");
    if (heur.currentNodePruned()) {
      ++nbacktracks;
      if (worker.getGlobalDomain().infeasible()) {
        worker.getHeurLpIterations() += heur.getLocalLpIterations();
        return;
      }

      if (!heur.backtrack()) break;
      neighbourhood.backtracked();
      continue;
    }

    fixingrate = neighbourhood.getFixingRate();
    // printf("after evaluating node current fixingrate is %g\n", fixingrate);
    if (fixingrate >= maxfixingrate) break;
    if (stop) break;
    if (nbacktracks >= 10) break;

    HighsInt numBranched = 0;
    double stopFixingRate = std::min(
        1.0 - (1.0 - neighbourhood.getFixingRate()) * 0.9, maxfixingrate);
    const auto& relaxationsol = heurlp.getSolution().col_value;
    for (HighsInt i : intcols) {
      if (localdom.col_lower_[i] == localdom.col_upper_[i]) continue;

      double downval =
          std::floor(relaxationsol[i] + mipsolver.mipdata_->feastol);
      double upval = std::ceil(relaxationsol[i] - mipsolver.mipdata_->feastol);

      downval = std::min(downval, localdom.col_upper_[i]);
      upval = std::max(upval, localdom.col_lower_[i]);
      if (localdom.col_lower_[i] < downval) {
        ++numBranched;
        heur.branchUpwards(i, downval, downval - 0.5);
        localdom.propagate();
        if (localdom.infeasible()) {
          localdom.conflictAnalysis(worker.getConflictPool(),
                                    worker.getGlobalDomain(),
                                    worker.getPseudocost());
          break;
        }
      }
      if (localdom.col_upper_[i] > upval) {
        ++numBranched;
        heur.branchDownwards(i, upval, upval + 0.5);
        localdom.propagate();
        if (localdom.infeasible()) {
          localdom.conflictAnalysis(worker.getConflictPool(),
                                    worker.getGlobalDomain(),
                                    worker.getPseudocost());
          break;
        }
      }

      if (neighbourhood.getFixingRate() >= stopFixingRate) break;
    }

    if (numBranched == 0) {
      auto getFixVal = [&](HighsInt col, double fracval) {
        // reinforce direction of this solution away from root
        // solution if the change is at least 0.4
        // otherwise take the direction where the objective gets worse
        // if objective is zero round to nearest integer
        double fixval =
            calcFixVal(mipsolver.mipdata_->rootlpsol.empty()
                           ? 0.0
                           : fracval - mipsolver.mipdata_->rootlpsol[col],
                       fracval, mipsolver.model_->col_cost_[col]);
        // make sure we do not set an infeasible domain
        fixval = std::min(localdom.col_upper_[col], fixval);
        fixval = std::max(localdom.col_lower_[col], fixval);
        return fixval;
      };

      pdqsort(heurlp.getFractionalIntegers().begin(),
              heurlp.getFractionalIntegers().end(),
              [&](const std::pair<HighsInt, double>& a,
                  const std::pair<HighsInt, double>& b) {
                return std::make_pair(
                           std::abs(getFixVal(a.first, a.second) - a.second),
                           HighsHashHelpers::hash(
                               (uint64_t(a.first) << 32) +
                               heurlp.getFractionalIntegers().size())) <
                       std::make_pair(
                           std::abs(getFixVal(b.first, b.second) - b.second),
                           HighsHashHelpers::hash(
                               (uint64_t(b.first) << 32) +
                               heurlp.getFractionalIntegers().size()));
              });

      double change = 0.0;
      // select a set of fractional variables to fix
      for (auto fracint : heurlp.getFractionalIntegers()) {
        double fixval = getFixVal(fracint.first, fracint.second);

        if (localdom.col_lower_[fracint.first] < fixval) {
          ++numBranched;
          heur.branchUpwards(fracint.first, fixval, fracint.second);
          localdom.propagate();
          if (localdom.infeasible()) {
            localdom.conflictAnalysis(worker.getConflictPool(),
                                      worker.getGlobalDomain(),
                                      worker.getPseudocost());
            break;
          }

          fixingrate = neighbourhood.getFixingRate();
        }

        if (localdom.col_upper_[fracint.first] > fixval) {
          ++numBranched;
          heur.branchDownwards(fracint.first, fixval, fracint.second);
          localdom.propagate();
          if (localdom.infeasible()) {
            localdom.conflictAnalysis(worker.getConflictPool(),
                                      worker.getGlobalDomain(),
                                      worker.getPseudocost());
            break;
          }

          fixingrate = neighbourhood.getFixingRate();
        }

        if (fixingrate >= maxfixingrate) break;

        change += std::abs(fixval - fracint.second);
        if (change >= 0.5) break;
      }
    }

    if (numBranched == 0) break;
    heurlp.flushDomain(localdom);
  }

  // printf("stopped heur dive with fixing rate %g\n", fixingrate);
  // if there is no node left it means we backtracked to the global domain and
  // the subproblem was solved with the dive
  if (!heur.hasNode()) {
    worker.getHeurLpIterations() += heur.getLocalLpIterations();
    return;
  }
  // determine the fixing rate to decide if the problem is restricted enough to
  // be considered for solving a submip

  fixingrate = neighbourhood.getFixingRate();
  // printf("fixing rate is %g\n", fixingrate);
  if (fixingrate < 0.1 ||
      (mipsolver.submip && mipsolver.mipdata_->numImprovingSols != 0)) {
    // heur.childselrule = ChildSelectionRule::kBestCost;
    heur.setMinReliable(0);
    heur.solveDepthFirst(10);
    worker.getHeurLpIterations() += heur.getLocalLpIterations();
    if (mipsolver.submip) mipsolver.mipdata_->num_nodes += heur.getLocalNodes();
    // lpiterations += heur.lpiterations;
    // pseudocost = heur.pseudocost;
    return;
  }

  heurlp.removeObsoleteRows(false, mipsolver.opportunisticIncumbentExchange()
      ? HighsLpRelaxation::CutRemovalMode::kModelAndBasisOnly
      : HighsLpRelaxation::CutRemovalMode::kReoptimize);
  HighsInt node_reduction_factor =
      mipsolver.mipdata_->parallelLockActive()
          ? std::max(
                HighsInt{1},
                static_cast<HighsInt>(mipsolver.mipdata_->workers.size()) / 4)
          : 1;
  const double submip_time = remaining_time();
  if (submip_time <= 0.0) {
    worker.getHeurLpIterations() += heur.getLocalLpIterations();
    return;
  }
  const bool solve_sub_mip_return = solveSubMip(
      worker, heurlp.getLp(), heurlp.getLpSolver().getBasis(), fixingrate,
      localdom.col_lower_, localdom.col_upper_,
      500,  // std::max(50, int(0.05 *
      // (mipsolver.mipdata_->num_leaves))),
      200 + mipsolver.mipdata_->num_nodes / (node_reduction_factor * 20), 12,
      origin, submip_time);
  if (worker.terminatorTerminated()) return;
  if (!solve_sub_mip_return) {
    int64_t new_lp_iterations =
        worker.getHeurLpIterations() + heur.getLocalLpIterations();
    if (new_lp_iterations + mipsolver.mipdata_->heuristic_lp_iterations >
        100000 + ((mipsolver.mipdata_->total_lp_iterations -
                   mipsolver.mipdata_->heuristic_lp_iterations -
                   mipsolver.mipdata_->sb_lp_iterations) >>
                  1)) {
      worker.getHeurLpIterations() = new_lp_iterations;
      return;
    }

    targetdepth = heur.getCurrentDepth() / 2;
    if (targetdepth <= 1 || (!mipsolver.mipdata_->parallelLockActive() &&
                             mipsolver.mipdata_->checkLimits())) {
      worker.getHeurLpIterations() = new_lp_iterations;
      return;
    }
    maxfixingrate = fixingrate * 0.5;
    // printf("infeasible in root node, trying with lower fixing rate %g\n",
    //        maxfixingrate);
    goto retry;
  }

  worker.getHeurLpIterations() += heur.getLocalLpIterations();
}

void HighsPrimalHeuristics::RINS(
    HighsMipWorker& worker, const std::vector<double>& relaxationsol,
    const HighsSubMipCallBudget* call_budget,
    HighsPrimalHeuristicOutcome* outcome) {
  if (outcome != nullptr) *outcome = HighsPrimalHeuristicOutcome{};
  const double call_started = mipsolver.timer_.read();
  double call_deadline = call_budget == nullptr
      ? kHighsInf : call_started + call_budget->max_time;
  // return if domain is infeasible
  if (worker.getGlobalDomain().infeasible()) return;

  if (relaxationsol.size() != static_cast<size_t>(mipsolver.numCol())) return;

  // A private seed remains stable even if evaluateNode imports a newer peer
  // snapshot. The master incumbent need not wait for an unrelated worker LP.
  worker.importPrimalSnapshot();
  const std::vector<double> incumbent = worker.repairIncumbent();
  if (incumbent.size() != relaxationsol.size()) return;

  if (call_budget != nullptr) {
    // Basic RINS uses the already available LP/incumbent agreement. Re-solving
    // a dive LP first can consume the entire managed reservation before repair.
    if (outcome != nullptr) outcome->phase_measured = true;
    const auto finish = [&](double child_seconds) {
      if (outcome == nullptr) return;
      outcome->elapsed = std::max(0.0, mipsolver.timer_.read() - call_started);
      outcome->preparation_seconds = std::max(0.0, outcome->elapsed - child_seconds);
      outcome->child_seconds = child_seconds;
      outcome->child_lp_iterations = outcome->lp_iterations;
      outcome->child_calls = outcome->started ? 1 : 0;
    };
    if (worker.search_work.checkpointRequested() || mipsolver.parentCheckpointRequested() ||
        mipsolver.timer_.read() >= call_deadline) {
      if (outcome != nullptr) {
        outcome->preparation_exhaustions = 1;
        outcome->preparation_stop_mask = mipsolver.timer_.read() >= call_deadline
            ? kHighsHeuristicStopDeadline : kHighsHeuristicStopCheckpoint;
      }
      finish(0.0);
      return;
    }
    std::vector<double> lower = worker.getGlobalDomain().col_lower_;
    std::vector<double> upper = worker.getGlobalDomain().col_upper_;
    double fixing_rate;
    const bool restricted = buildRinsNeighbourhood(
        *mipsolver.model_, relaxationsol, incumbent, mipsolver.mipdata_->feastol,
        call_budget->min_fixing_rate, lower, upper, fixing_rate);
    if (outcome != nullptr) {
      outcome->fixing_rate = fixing_rate;
      outcome->fixing_rate_measured = true;
    }
    if (!restricted) {
      finish(0.0);
      return;
    }
    HighsSubMipCallBudget remaining = *call_budget;
    remaining.max_time = remainingHeuristicTime(
        call_started, mipsolver.timer_.read(), call_budget->max_time);
    if (remaining.max_time <= 0 || remaining.max_nodes == 0 ||
        remaining.max_lp_iterations == 0) {
      if (outcome != nullptr) {
        outcome->preparation_exhaustions = 1;
        if (remaining.max_time <= 0) outcome->preparation_stop_mask |= kHighsHeuristicStopDeadline;
        if (remaining.max_nodes == 0) outcome->preparation_stop_mask |= kHighsHeuristicStopNodes;
        if (remaining.max_lp_iterations == 0) outcome->preparation_stop_mask |= kHighsHeuristicStopLpWork;
      }
      finish(0.0);
      return;
    }
    const double child_started = mipsolver.timer_.read();
    HighsPrimalHeuristicOutcome child;
    const auto& lp = worker.getLpRelaxation();
    solveSubMip(worker, lp.getLp(), lp.getLpSolver().getBasis(), fixing_rate,
                std::move(lower), std::move(upper), 500, 200, 12,
                HighsSubMipOrigin::kTreeRins, kHighsInf, &remaining, &child, &incumbent);
    if (outcome != nullptr) {
      *outcome = child;
      outcome->phase_measured = outcome->fixing_rate_measured = true;
    }
    finish(std::max(0.0, mipsolver.timer_.read() - child_started));
    return;
  }

  std::vector<HighsInt> intcols_;
  if (mipsolver.mipdata_->parallelLockActive()) {
    intcols_ = intcols;
  }
  std::vector<HighsInt>& intcols =
      mipsolver.mipdata_->parallelLockActive() ? intcols_ : this->intcols;
  intcols.erase(std::remove_if(intcols.begin(), intcols.end(),
                               [&](HighsInt i) {
                                 return worker.getGlobalDomain().isFixed(i);
                               }),
                intcols.end());

  HighsPseudocost pscost(worker.getPseudocost());
  HighsSearch heur(worker, pscost);

  struct RinsWorkLedger {
    HighsSearch& search;
    const HighsMipSolver& mip;
    HighsPrimalHeuristicOutcome* outcome;
    double started;
    double child_seconds = 0.0;
    int64_t child_calls = 0;
    int64_t preparation_exhaustions = 0;
    int64_t child_lp = 0;
    int64_t child_nodes = 0;
    double fixing_rate = 0.0;
    bool fixing_rate_measured = false;
    uint8_t stop_mask = 0;
    RinsWorkLedger(HighsSearch& search_, const HighsMipSolver& mip_,
                   HighsPrimalHeuristicOutcome* outcome_, double started_)
        : search(search_), mip(mip_), outcome(outcome_), started(started_) {}
    ~RinsWorkLedger() {
      if (outcome) {
        outcome->lp_iterations = child_lp + search.getLocalLpIterations();
        outcome->nodes = child_nodes + search.getLocalNodes();
        outcome->phase_measured = true;
        outcome->preparation_lp_iterations = search.getLocalLpIterations();
        outcome->child_lp_iterations = child_lp;
        outcome->child_calls = child_calls;
        outcome->preparation_exhaustions = preparation_exhaustions;
        outcome->child_seconds = child_seconds;
        outcome->preparation_seconds = std::max(0.0, mip.timer_.read() - started - child_seconds);
        outcome->fixing_rate = fixing_rate;
        outcome->fixing_rate_measured = fixing_rate_measured;
        outcome->preparation_stop_mask = stop_mask;
      }
    }
  } work(heur, mipsolver, outcome, call_started);

  HighsDomain& localdom = heur.getLocalDomain();
  heur.setHeuristic(true);

  // LP relaxation instantiation
  HighsTaskPhaseScope preparation_phase(worker.search_work,
                                        HighsTaskPhase::kHeuristicPreparation);
  HighsLpRelaxation heurlp(worker.getLpRelaxation());
  heurlp.setHelperDeadline(call_deadline);
  heurlp.setMipWorker(worker);
  heurlp.setCandidateInterruptible();
  heurlp.setProfiling(mipsolver.profiling_);
  // only use the global upper limit as LP limit so that dual proofs are valid
  heurlp.setObjectiveLimit(worker.upper_limit);
  heurlp.setAdjustSymmetricBranchingCol(false);
  heur.setLpRelaxation(&heurlp);

  heurlp.getLpSolver().changeColsBounds(0, mipsolver.numCol() - 1,
                                        localdom.col_lower_.data(),
                                        localdom.col_upper_.data());
  localdom.clearChangedCols();
  heur.createNewNode();

  // determine the initial number of unfixed variables fixing rate to decide if
  // the problem is restricted enough to be considered for solving a submip
  double maxfixingrate = determineTargetFixingRate(worker);
  double minfixingrate =
      call_budget == nullptr ? 0.25 : call_budget->min_fixing_rate;
  double fixingrate = 0.0;
  bool stop = false;
  HighsInt nbacktracks = -1;
  HighsInt targetdepth = 1;
  HeuristicNeighbourhood neighbourhood(mipsolver, localdom);
retry:
  if (mipsolver.timer_.read() >= call_deadline ||
      worker.search_work.checkpointRequested() || mipsolver.parentCheckpointRequested()) {
    if (mipsolver.timer_.read() >= call_deadline)
      work.stop_mask |= kHighsHeuristicStopDeadline;
    if (worker.search_work.checkpointRequested() || mipsolver.parentCheckpointRequested())
      work.stop_mask |= kHighsHeuristicStopCheckpoint;
    ++work.preparation_exhaustions;
    worker.getHeurLpIterations() += heur.getLocalLpIterations();
    return;
  }
  ++nbacktracks;
  neighbourhood.backtracked();
  // printf("current depth : %" HIGHSINT_FORMAT "   target depth : %"
  // HIGHSINT_FORMAT "\n", heur.getCurrentDepth(),
  //       targetdepth);
  if (heur.getCurrentDepth() > targetdepth) {
    if (!heur.backtrackUntilDepth(targetdepth)) {
      worker.getHeurLpIterations() += heur.getLocalLpIterations();
      return;
    }
  }

  assert(heur.hasNode());

  while (true) {
    if (worker.search_work.checkpointRequested() || mipsolver.parentCheckpointRequested()) {
      work.stop_mask |= kHighsHeuristicStopCheckpoint;
      ++work.preparation_exhaustions;
      worker.getHeurLpIterations() += heur.getLocalLpIterations();
      return;
    }
    if (call_budget != nullptr ||
        worker.search_work.remaining() != std::numeric_limits<int64_t>::max()) {
      const int64_t requested = call_budget == nullptr || call_budget->max_lp_iterations < 0
          ? kHighsIInf
          : call_budget->max_lp_iterations - work.child_lp - heur.getLocalLpIterations();
      const int64_t remaining = std::min<int64_t>(requested, worker.search_work.remaining());
      const bool nodes_exhausted = call_budget != nullptr && call_budget->max_nodes >= 0 &&
          work.child_nodes + heur.getLocalNodes() >= call_budget->max_nodes;
      if (remaining <= 0 || nodes_exhausted || mipsolver.timer_.read() >= call_deadline) {
        if (remaining <= 0) work.stop_mask |= kHighsHeuristicStopLpWork;
        if (nodes_exhausted) work.stop_mask |= kHighsHeuristicStopNodes;
        if (mipsolver.timer_.read() >= call_deadline)
          work.stop_mask |= kHighsHeuristicStopDeadline;
        ++work.preparation_exhaustions;
        worker.getHeurLpIterations() += heur.getLocalLpIterations();
        return;
      }
      // A managed reservation bounds the whole call. A second 10000-pivot
      // ceiling can abandon its very first preparation LP despite available
      // work; retain that legacy ceiling only for unmanaged preparation.
      heurlp.setHelperIterationLimit(static_cast<HighsInt>(std::min<int64_t>(
          remaining, call_budget == nullptr ? 10000 : kHighsIInf)));
    }
    const auto evaluation = heur.evaluateNode();
    if (evaluation == HighsSearch::NodeResult::kLpIterationLimit ||
        evaluation == HighsSearch::NodeResult::kCandidateCheckpoint) {
      if (evaluation == HighsSearch::NodeResult::kCandidateCheckpoint)
        work.stop_mask |= kHighsHeuristicStopCheckpoint;
      else {
        work.stop_mask |= kHighsHeuristicStopUnfinishedLp;
        if (heurlp.getLpSolver().getModelStatus() == HighsModelStatus::kIterationLimit)
          work.stop_mask |= kHighsHeuristicStopLpWork;
        if (worker.search_work.checkpointRequested() || mipsolver.parentCheckpointRequested())
          work.stop_mask |= kHighsHeuristicStopCheckpoint;
      }
      if (mipsolver.timer_.read() >= call_deadline)
        work.stop_mask |= kHighsHeuristicStopDeadline;
      ++work.preparation_exhaustions;
      worker.getHeurLpIterations() += heur.getLocalLpIterations();
      return;
    }
    if (heur.currentNodePruned()) {
      ++nbacktracks;
      // printf("backtrack1\n");
      if (worker.getGlobalDomain().infeasible()) {
        worker.getHeurLpIterations() += heur.getLocalLpIterations();
        return;
      }

      if (!heur.backtrack()) break;
      neighbourhood.backtracked();
      continue;
    }

    fixingrate = neighbourhood.getFixingRate();
    work.fixing_rate = fixingrate;
    work.fixing_rate_measured = true;

    if (stop) break;
    if (fixingrate >= maxfixingrate) break;
    if (nbacktracks >= 10) break;

    std::vector<std::pair<HighsInt, double>>::iterator fixcandend;

    // partition the fractional variables to consider which ones should we fix
    // in this dive first if there is an incumbent, we dive towards the RINS
    // neighbourhood
    fixcandend = std::partition(
        heurlp.getFractionalIntegers().begin(),
        heurlp.getFractionalIntegers().end(),
        [&](const std::pair<HighsInt, double>& fracvar) {
          return std::abs(relaxationsol[fracvar.first] -
                          incumbent[fracvar.first]) <=
                 mipsolver.mipdata_->feastol;
        });

    bool fixtolpsol = true;

    auto getFixVal = [&](HighsInt col, double fracval) {
      double fixval;
      if (fixtolpsol) {
        // RINS neighbourhood (with extension)
        fixval = std::floor(relaxationsol[col] + 0.5);
      } else {
        // reinforce direction of this solution away from root
        // solution if the change is at least 0.4
        // otherwise take the direction where the objective gets worse
        // if objective is zero round to nearest integer
        fixval = calcFixVal(fracval - mipsolver.mipdata_->rootlpsol[col],
                            fracval, mipsolver.model_->col_cost_[col]);
      }
      // make sure we do not set an infeasible domain
      fixval = std::min(localdom.col_upper_[col], fixval);
      fixval = std::max(localdom.col_lower_[col], fixval);
      return fixval;
    };

    // no candidates left to fix for getting to the neighbourhood, therefore we
    // switch to a different diving strategy until the minimal fixing rate is
    // reached
    HighsInt numBranched = 0;
    if (heurlp.getFractionalIntegers().begin() == fixcandend) {
      fixingrate = neighbourhood.getFixingRate();
      double stopFixingRate =
          std::min(maxfixingrate, 1.0 - (1.0 - fixingrate) * 0.9);
      const auto& currlpsol = heurlp.getSolution().col_value;
      for (HighsInt i : intcols) {
        if (localdom.col_lower_[i] == localdom.col_upper_[i]) continue;

        if (std::abs(currlpsol[i] - incumbent[i]) <=
            mipsolver.mipdata_->feastol) {
          double fixval = HighsIntegers::nearestInteger(currlpsol[i]);
          if (localdom.col_lower_[i] < fixval) {
            ++numBranched;
            heur.branchUpwards(i, fixval, fixval - 0.5);
            localdom.propagate();
            if (localdom.infeasible()) {
              localdom.conflictAnalysis(worker.getConflictPool(),
                                        worker.getGlobalDomain(),
                                        worker.getPseudocost());
              break;
            }

            fixingrate = neighbourhood.getFixingRate();
          }
          if (localdom.col_upper_[i] > fixval) {
            ++numBranched;
            heur.branchDownwards(i, fixval, fixval + 0.5);
            localdom.propagate();
            if (localdom.infeasible()) {
              localdom.conflictAnalysis(worker.getConflictPool(),
                                        worker.getGlobalDomain(),
                                        worker.getPseudocost());
              break;
            }

            fixingrate = neighbourhood.getFixingRate();
          }

          if (fixingrate >= stopFixingRate) break;
        }
      }

      if (numBranched != 0) {
        // printf(
        //    "fixed %" HIGHSINT_FORMAT " additional cols, old fixing rate:
        //    %.2f%%, new fixing " "rate: %.2f%%\n", numBranched, fixingrate,
        //    getFixingRate());
        heurlp.flushDomain(localdom);
        continue;
      }

      if (fixingrate >= minfixingrate)
        break;  // if the RINS neighbourhood achieved a high enough fixing rate
                // by itself we stop here
      fixcandend = heurlp.getFractionalIntegers().end();
      // now sort the variables by their distance towards the value they will
      // be fixed to
      fixtolpsol = false;
    }

    // now sort the variables by their distance towards the value they will be
    // fixed to
    pdqsort(heurlp.getFractionalIntegers().begin(), fixcandend,
            [&](const std::pair<HighsInt, double>& a,
                const std::pair<HighsInt, double>& b) {
              return std::make_pair(
                         std::abs(getFixVal(a.first, a.second) - a.second),
                         HighsHashHelpers::hash(
                             (uint64_t(a.first) << 32) +
                             heurlp.getFractionalIntegers().size())) <
                     std::make_pair(
                         std::abs(getFixVal(b.first, b.second) - b.second),
                         HighsHashHelpers::hash(
                             (uint64_t(b.first) << 32) +
                             heurlp.getFractionalIntegers().size()));
            });

    double change = 0.0;
    // select a set of fractional variables to fix
    for (auto fracint = heurlp.getFractionalIntegers().begin();
         fracint != fixcandend; ++fracint) {
      double fixval = getFixVal(fracint->first, fracint->second);

      if (localdom.col_lower_[fracint->first] < fixval) {
        ++numBranched;
        heur.branchUpwards(fracint->first, fixval, fracint->second);
        if (localdom.infeasible()) {
          localdom.conflictAnalysis(worker.getConflictPool(),
                                    worker.getGlobalDomain(),
                                    worker.getPseudocost());
          break;
        }

        fixingrate = neighbourhood.getFixingRate();
        work.fixing_rate = fixingrate;
      }

      if (localdom.col_upper_[fracint->first] > fixval) {
        ++numBranched;
        heur.branchDownwards(fracint->first, fixval, fracint->second);
        if (localdom.infeasible()) {
          localdom.conflictAnalysis(worker.getConflictPool(),
                                    worker.getGlobalDomain(),
                                    worker.getPseudocost());
          break;
        }

        fixingrate = neighbourhood.getFixingRate();
        work.fixing_rate = fixingrate;
      }

      if (fixingrate >= maxfixingrate) break;

      change += std::abs(fixval - fracint->second);
      if (change >= 0.5) break;
    }

    if (numBranched == 0) break;

    heurlp.flushDomain(localdom);

    // printf("%" HIGHSINT_FORMAT "/%" HIGHSINT_FORMAT " fixed, fixingrate is
    // %g\n", nfixed, ntotal, fixingrate);
  }

  // if there is no node left it means we backtracked to the global domain and
  // the subproblem was solved with the dive
  if (!heur.hasNode()) {
    worker.getHeurLpIterations() += heur.getLocalLpIterations();
    return;
  }
  // determine the fixing rate to decide if the problem is restricted enough
  // to be considered for solving a submip

  // printf("fixing rate is %g\n", fixingrate);
  fixingrate = neighbourhood.getFixingRate();
  work.fixing_rate = fixingrate;
  work.fixing_rate_measured = true;
  if (fixingrate < 0.1 ||
      (mipsolver.submip && mipsolver.mipdata_->numImprovingSols != 0)) {
    // This depth-first fallback predates managed whole-call reservations and
    // has no cumulative LP-work cap. Skip it in the bounded helper only.
    if (call_budget != nullptr) {
      worker.getHeurLpIterations() += heur.getLocalLpIterations();
      return;
    }
    // heur.childselrule = ChildSelectionRule::kBestCost;
    heur.setMinReliable(0);
    heur.solveDepthFirst(10);
    worker.getHeurLpIterations() += heur.getLocalLpIterations();
    if (mipsolver.submip) mipsolver.mipdata_->num_nodes += heur.getLocalNodes();
    // lpiterations += heur.lpiterations;
    // pseudocost = heur.pseudocost;
    return;
  }

  heurlp.removeObsoleteRows(false, mipsolver.opportunisticIncumbentExchange()
      ? HighsLpRelaxation::CutRemovalMode::kModelAndBasisOnly
      : HighsLpRelaxation::CutRemovalMode::kReoptimize);
  HighsInt node_reduction_factor =
      mipsolver.mipdata_->parallelLockActive()
          ? std::max(
                HighsInt{1},
                static_cast<HighsInt>(mipsolver.mipdata_->workers.size()) / 4)
          : 1;
  HighsSubMipCallBudget remaining_budget;
  if (call_budget != nullptr) {
    remaining_budget = *call_budget;
    remaining_budget.max_time = std::max(0.0, call_deadline - mipsolver.timer_.read());
    if (remaining_budget.max_lp_iterations >= 0)
      remaining_budget.max_lp_iterations = std::max<int64_t>(
          0, remaining_budget.max_lp_iterations - work.child_lp - heur.getLocalLpIterations());
    if (remaining_budget.max_nodes >= 0)
      remaining_budget.max_nodes = std::max<int64_t>(0, remaining_budget.max_nodes - work.child_nodes - heur.getLocalNodes());
    if (remaining_budget.max_time <= 0 || remaining_budget.max_lp_iterations == 0 || remaining_budget.max_nodes == 0) {
      if (remaining_budget.max_time <= 0)
        work.stop_mask |= kHighsHeuristicStopDeadline;
      if (remaining_budget.max_lp_iterations == 0)
        work.stop_mask |= kHighsHeuristicStopLpWork;
      if (remaining_budget.max_nodes == 0)
        work.stop_mask |= kHighsHeuristicStopNodes;
      ++work.preparation_exhaustions;
      worker.getHeurLpIterations() += heur.getLocalLpIterations();
      return;
    }
  }
  HighsPrimalHeuristicOutcome child_outcome;
  const double child_started = mipsolver.timer_.read();
  const bool solve_sub_mip_return = solveSubMip(
      worker, heurlp.getLp(), heurlp.getLpSolver().getBasis(), fixingrate,
      localdom.col_lower_, localdom.col_upper_,
      500,  // std::max(50, int(0.05 *
      // (mipsolver.mipdata_->num_leaves))),
      200 + mipsolver.mipdata_->num_nodes / (node_reduction_factor * 20), 12,
      HighsSubMipOrigin::kTreeRins, kHighsInf,
      call_budget == nullptr ? nullptr : &remaining_budget,
      &child_outcome);
  work.child_seconds += std::max(0.0, mipsolver.timer_.read() - child_started);
  ++work.child_calls;
  {
    work.child_lp += std::max<int64_t>(0, child_outcome.lp_iterations);
    work.child_nodes += std::max<int64_t>(0, child_outcome.nodes);
    if (outcome != nullptr) {
      HighsPrimalChildReceipts children = outcome->children;
      children.append(child_outcome.children);
      child_outcome.children = std::move(children);
      child_outcome.started |= outcome->started;
      child_outcome.completed |= outcome->completed;
      child_outcome.candidate_proposed |= outcome->candidate_proposed;
      child_outcome.candidate_accepted |= outcome->candidate_accepted;
      child_outcome.candidate_improved |= outcome->candidate_improved;
      *outcome = child_outcome;
    }
  }
  if (worker.terminatorTerminated()) return;
  if (!solve_sub_mip_return) {
    int64_t new_lp_iterations =
        worker.getHeurLpIterations() + heur.getLocalLpIterations();
    if (new_lp_iterations + mipsolver.mipdata_->heuristic_lp_iterations >
        100000 + ((mipsolver.mipdata_->total_lp_iterations -
                   mipsolver.mipdata_->heuristic_lp_iterations -
                   mipsolver.mipdata_->sb_lp_iterations) >>
                  1)) {
      worker.getHeurLpIterations() = new_lp_iterations;
      return;
    }

    targetdepth = heur.getCurrentDepth() / 2;
    if (targetdepth <= 1 || (!mipsolver.mipdata_->parallelLockActive() &&
                             mipsolver.mipdata_->checkLimits())) {
      worker.getHeurLpIterations() = new_lp_iterations;
      return;
    }
    // printf("infeasible in root node, trying with lower fixing rate\n");
    maxfixingrate = fixingrate * 0.5;
    goto retry;
  }

  worker.getHeurLpIterations() += heur.getLocalLpIterations();
}

bool HighsPrimalHeuristics::tryRoundedPoint(HighsMipWorker& worker,
                                            const std::vector<double>& point,
                                            const int solution_source) {
  HighsDomain localdom = worker.getGlobalDomain();
  bool integerFeasible = true;

  HighsInt numintcols = intcols.size();
  for (HighsInt i = 0; i != numintcols; ++i) {
    HighsInt col = intcols[i];
    double intval = point[col];
    double rounded;
    // check if solution value of integer-constrained variable is actually
    // integral
    bool feasible =
        fractionality(intval, &rounded) <= mipsolver.mipdata_->feastol;
    integerFeasible = integerFeasible && feasible;
    if (!feasible) continue;
    // use rounded solution value and check against bounds
    intval = rounded;
    intval = std::min(localdom.col_upper_[col], intval);
    intval = std::max(localdom.col_lower_[col], intval);

    localdom.fixCol(col, intval, HighsDomain::Reason::branching());
    if (localdom.infeasible()) {
      localdom.conflictAnalysis(worker.getConflictPool(),
                                worker.getGlobalDomain(),
                                worker.getPseudocost());
      return false;
    }
    localdom.propagate();
    if (localdom.infeasible()) {
      localdom.conflictAnalysis(worker.getConflictPool(),
                                worker.getGlobalDomain(),
                                worker.getPseudocost());
      return false;
    }
  }

  if (numintcols != mipsolver.numCol()) {
    // LP relaxation instantiation
    HighsLpRelaxation lprelax(mipsolver);
    lprelax.setMipWorker(worker);
    lprelax.setProfiling(mipsolver.profiling_);
    lprelax.loadModel();
    lprelax.setIterationLimit(
        std::max(int64_t{10000}, 2 * mipsolver.mipdata_->firstrootlpiters));
    lprelax.getLpSolver().changeColsBounds(0, mipsolver.numCol() - 1,
                                           localdom.col_lower_.data(),
                                           localdom.col_upper_.data());

    // check if only root presolve is allowed
    if (mipsolver.options_mip_->mip_root_presolve_only)
      lprelax.getLpSolver().setOptionValue("presolve", kHighsOffString);
    if (!mipsolver.options_mip_->mip_root_presolve_only &&
        (5 * numintcols) / mipsolver.numCol() >= 1)
      lprelax.getLpSolver().setOptionValue("presolve", kHighsOnString);
    else
      lprelax.getLpSolver().setBasis(mipsolver.mipdata_->firstrootbasis,
                                     "HighsPrimalHeuristics::tryRoundedPoint");

    HighsLpRelaxation::Status st = lprelax.resolveLp();

    if (st == HighsLpRelaxation::Status::kInfeasible) {
      std::vector<HighsInt> inds;
      std::vector<double> vals;
      double rhs;
      if (lprelax.computeDualInfProof(worker.getGlobalDomain(), inds, vals,
                                      rhs)) {
        HighsCutGeneration cutGen(lprelax, worker.getCutPool(),
                                  HighsCutOrigin::kHeuristic);
        cutGen.generateConflict(localdom, worker.getGlobalDomain(), inds, vals,
                                rhs);
      }
      return false;
    } else if (lprelax.unscaledPrimalFeasible(st)) {
      const auto& lpsol = lprelax.getLpSolver().getSolution().col_value;
      if (!integerFeasible) {
        // there may be fractional integer variables -> try ziRound heuristic
        ziRound(worker, lpsol);
        return trySolution(lpsol, solution_source, worker);
      } else {
        // all integer variables are fixed -> add incumbent
        addIncumbent(lpsol, lprelax.getObjective(), solution_source, worker);
        return true;
      }
    }
  }

  return trySolution(localdom.col_lower_, solution_source, worker);
}

bool HighsPrimalHeuristics::linesearchRounding(
    HighsMipWorker& worker, const std::vector<double>& point1,
    const std::vector<double>& point2, const int solution_source) {
  std::vector<double> roundedpoint;

  HighsInt numintcols = intcols.size();
  roundedpoint.resize(mipsolver.numCol());

  double alpha = 0.0;
  assert(int(mipsolver.mipdata_->uplocks.size()) == mipsolver.numCol());
  assert(int(point1.size()) == mipsolver.numCol());
  assert(int(point2.size()) == mipsolver.numCol());

  while (alpha < 1.0) {
    double nextalpha = 1.0;
    bool reachedpoint2 = true;
    // printf("trying alpha = %g\n", alpha);
    for (HighsInt i = 0; i != numintcols; ++i) {
      HighsInt col = intcols[i];
      assert(col >= 0);
      assert(col < mipsolver.numCol());
      if (mipsolver.mipdata_->uplocks[col] == 0) {
        roundedpoint[col] = std::ceil(std::max(point1[col], point2[col]) -
                                      mipsolver.mipdata_->feastol);
        continue;
      }

      if (mipsolver.mipdata_->downlocks[col] == 0) {
        roundedpoint[col] = std::floor(std::min(point1[col], point2[col]) +
                                       mipsolver.mipdata_->feastol);
        continue;
      }

      double convexcomb = (1.0 - alpha) * point1[col] + alpha * point2[col];
      double intpoint2 = std::floor(point2[col] + 0.5);
      roundedpoint[col] = std::floor(convexcomb + 0.5);

      if (roundedpoint[col] == intpoint2) continue;

      reachedpoint2 = false;
      double tmpalpha = (roundedpoint[col] + 0.5 + mipsolver.mipdata_->feastol -
                         point1[col]) /
                        std::abs(point2[col] - point1[col]);
      if (tmpalpha < nextalpha && tmpalpha > alpha + 1e-2) nextalpha = tmpalpha;
    }

    if (tryRoundedPoint(worker, roundedpoint, solution_source)) return true;

    if (reachedpoint2) return false;

    alpha = nextalpha;
  }

  return false;
}

void HighsPrimalHeuristics::randomizedRounding(
    HighsMipWorker& worker, const std::vector<double>& relaxationsol) {
  if (relaxationsol.size() != static_cast<size_t>(mipsolver.numCol())) return;

  HighsDomain localdom = worker.getGlobalDomain();
  HighsRandom& randgen =
      mipsolver.mipdata_->parallelLockActive() ? worker.randgen : this->randgen;

  for (HighsInt i : intcols) {
    double intval;
    if (mipsolver.mipdata_->uplocks[i] == 0)
      intval = std::ceil(relaxationsol[i] - mipsolver.mipdata_->feastol);
    else if (mipsolver.mipdata_->downlocks[i] == 0)
      intval = std::floor(relaxationsol[i] + mipsolver.mipdata_->feastol);
    else
      intval = std::floor(relaxationsol[i] + randgen.real(0.1, 0.9));

    intval = std::min(localdom.col_upper_[i], intval);
    intval = std::max(localdom.col_lower_[i], intval);

    localdom.fixCol(i, intval, HighsDomain::Reason::branching());
    if (localdom.infeasible()) {
      localdom.conflictAnalysis(worker.getConflictPool(),
                                worker.getGlobalDomain(),
                                worker.getPseudocost());
      return;
    }
    localdom.propagate();
    if (localdom.infeasible()) {
      localdom.conflictAnalysis(worker.getConflictPool(),
                                worker.getGlobalDomain(),
                                worker.getPseudocost());
      return;
    }
  }

  if (mipsolver.mipdata_->integer_cols.size() !=
      static_cast<size_t>(mipsolver.numCol())) {
    // LP relaxation instantiation
    HighsLpRelaxation lprelax(mipsolver);
    lprelax.setMipWorker(worker);
    lprelax.setProfiling(mipsolver.profiling_);
    lprelax.loadModel();
    lprelax.setIterationLimit(
        std::max(int64_t{10000}, 2 * mipsolver.mipdata_->firstrootlpiters));
    lprelax.getLpSolver().changeColsBounds(0, mipsolver.numCol() - 1,
                                           localdom.col_lower_.data(),
                                           localdom.col_upper_.data());

    // check if only root presolve is allowed
    if (mipsolver.options_mip_->mip_root_presolve_only)
      lprelax.getLpSolver().setOptionValue("presolve", kHighsOffString);

    if (!mipsolver.options_mip_->mip_root_presolve_only &&
        (5 * intcols.size()) / mipsolver.numCol() >= 1) {
      // LP to solve is very much smaller, so use presolve rather than
      // the root basis
      lprelax.getLpSolver().setOptionValue("presolve", kHighsOnString);
    } else {
      lprelax.getLpSolver().setBasis(
          mipsolver.mipdata_->firstrootbasis,
          "HighsPrimalHeuristics::randomizedRounding");
    }

    HighsLpRelaxation::Status st = lprelax.resolveLp();

    if (st == HighsLpRelaxation::Status::kInfeasible) {
      std::vector<HighsInt> inds;
      std::vector<double> vals;
      double rhs;
      if (lprelax.computeDualInfProof(worker.getGlobalDomain(), inds, vals,
                                      rhs)) {
        HighsCutGeneration cutGen(lprelax, worker.getCutPool(),
                                  HighsCutOrigin::kHeuristic);
        cutGen.generateConflict(localdom, worker.getGlobalDomain(), inds, vals,
                                rhs);
      }

    } else if (HighsLpRelaxation::unscaledPrimalFeasible(st)) {
      addIncumbent(lprelax.getLpSolver().getSolution().col_value,
                   lprelax.getObjective(), kSolutionSourceRandomizedRounding,
                   worker);
    }
  } else {
    trySolution(localdom.col_lower_, kSolutionSourceRandomizedRounding, worker);
  }
}

void HighsPrimalHeuristics::shifting(HighsMipWorker& worker,
                                     const std::vector<double>& relaxationsol) {
  if (relaxationsol.size() != static_cast<size_t>(mipsolver.numCol())) return;

  std::vector<double> current_relax_solution = relaxationsol;
  HighsInt t = 0;
  const HighsLp& currentLp = *mipsolver.model_;
  // LP relaxation instantiation
  HighsLpRelaxation lprelax(worker.getLpRelaxation());
  lprelax.setMipWorker(worker);
  lprelax.setProfiling(mipsolver.profiling_);
  HighsRandom& randgen =
      mipsolver.mipdata_->parallelLockActive() ? worker.randgen : this->randgen;
  std::vector<std::pair<HighsInt, double>> current_fractional_integers =
      lprelax.getFractionalIntegers();
  std::vector<std::tuple<HighsInt, HighsInt, double>> current_infeasible_rows =
      mipsolver.mipdata_->getInfeasibleRows(current_relax_solution);
  size_t previous_infeasible_rows_size = current_infeasible_rows.size();
  bool hasInfeasibleConstraints = current_infeasible_rows.size() != 0;
  HighsInt iterationsWithoutReductions = 0;
  HighsInt maxIterationsWithoutReductions = 5;
  std::unordered_map<HighsInt, std::vector<HighsInt>> shift_iterations_set;
  std::vector<HighsInt> shifts;

  auto findPairByIndex = [](std::vector<std::pair<HighsInt, double>>& vec,
                            HighsInt k) {
    return std::find_if(
        vec.begin(), vec.end(),
        [k](const std::pair<HighsInt, double>& p) { return p.first == k; });
  };

  auto findShiftsByIndex =
      [](const std::unordered_map<HighsInt, std::vector<HighsInt>>& shifts,
         HighsInt k) -> std::vector<HighsInt> {
    auto it = shifts.find(k);
    if (it != shifts.end()) {
      return it->second;
    }
    return {};
  };

  while ((current_fractional_integers.size() > 0 || hasInfeasibleConstraints) &&
         iterationsWithoutReductions <= maxIterationsWithoutReductions &&
         t <= static_cast<HighsInt>(mipsolver.mipdata_->integer_cols.size())) {
    t++;
    bool fractionalIntegersReduced = false;
    iterationsWithoutReductions++;
    if (hasInfeasibleConstraints) {
      // find an infeasible row that has a non-zero coefficient a(row,col) where
      // col in currentFractInt
      bool fractionalIntegerFound = false;
      HighsInt rIndex = 0;
      while (!fractionalIntegerFound &&
             rIndex != static_cast<HighsInt>(current_infeasible_rows.size())) {
        HighsInt r = std::get<0>(current_infeasible_rows[rIndex]);
        HighsInt start = mipsolver.mipdata_->ARstart_[r];
        HighsInt end = mipsolver.mipdata_->ARstart_[r + 1];
        for (HighsInt jInd = start; jInd != end; ++jInd) {
          auto it = findPairByIndex(current_fractional_integers,
                                    mipsolver.mipdata_->ARindex_[jInd]);
          fractionalIntegerFound = it != current_fractional_integers.end();
          if (fractionalIntegerFound) break;
        }
        rIndex++;
      }
      HighsInt row_index = rIndex - 1;
      if (!fractionalIntegerFound) {
        // otherwise select a random infeasible row
        row_index = randgen.integer(current_infeasible_rows.size());
      }

      HighsInt row = std::get<0>(current_infeasible_rows[row_index]);
      HighsInt row_sense = std::get<1>(current_infeasible_rows[row_index]);
      double infeasibility = std::get<2>(current_infeasible_rows[row_index]);
      double score_min = kHighsInf;
      HighsInt j_min = std::numeric_limits<HighsInt>::max();
      double x_j_min = kHighsInf;
      double aij_min = 0.0;
      bool moveValueUp = false;
      HighsInt start = mipsolver.mipdata_->ARstart_[row];
      HighsInt end = mipsolver.mipdata_->ARstart_[row + 1];
      for (HighsInt jInd = start; jInd != end; ++jInd) {
        HighsInt j = mipsolver.mipdata_->ARindex_[jInd];

        // skip fixed variables
        if (currentLp.col_lower_[j] == currentLp.col_upper_[j]) continue;

        // lambda for finding best shift
        auto repair = [&findPairByIndex, &current_fractional_integers,
                       &findShiftsByIndex, &shift_iterations_set, &t,
                       &score_min, &j_min, &aij_min, &x_j_min,
                       &current_relax_solution, &moveValueUp](
                          HighsInt col, HighsInt direction, HighsInt row_sense,
                          HighsInt numLocks, double coef, double cost,
                          bool isMaximization, bool isInteger, bool isAtBound) {
          if ((row_sense < 0 || direction * coef > 0) &&
              (row_sense > 0 || direction * coef < 0))
            return;

          // skip variables at bounds
          if (isAtBound) return;

          // search for column
          auto it = findPairByIndex(current_fractional_integers, col);

          // add data
          bool found = it != current_fractional_integers.end();

          double score = kHighsInf;
          if (found) {
            score = -1.0 + 1.0 / static_cast<double>(numLocks + 1);
          } else {
            const auto& shifts = findShiftsByIndex(shift_iterations_set, col);
            if (shifts.empty())
              score = direction * (isMaximization ? -cost : cost);
            else {
              score = 0.0;
              for (double shift : shifts) {
                if (direction * shift > 0)
                  score += pow(1.1, direction * shift - t);
              }
            }
            if (isInteger) score += 1;
          }

          if (score < score_min) {
            score_min = score;
            j_min = col;
            aij_min = coef;
            x_j_min = current_relax_solution[col];
            moveValueUp = direction > 0;
          }
        };

        // repair with up-rounding
        repair(j, HighsInt{1}, row_sense, mipsolver.mipdata_->downlocks[j],
               mipsolver.mipdata_->ARvalue_[jInd], currentLp.col_cost_[j],
               mipsolver.orig_model_->sense_ == ObjSense::kMaximize,
               currentLp.integrality_[j] == HighsVarType::kInteger,
               std::abs(currentLp.col_upper_[j] - current_relax_solution[j]) <=
                   mipsolver.mipdata_->feastol);

        // repair with down-rounding
        repair(j, HighsInt{-1}, row_sense, mipsolver.mipdata_->uplocks[j],
               mipsolver.mipdata_->ARvalue_[jInd], currentLp.col_cost_[j],
               mipsolver.orig_model_->sense_ == ObjSense::kMaximize,
               currentLp.integrality_[j] == HighsVarType::kInteger,
               std::abs(current_relax_solution[j] - currentLp.col_lower_[j]) <=
                   mipsolver.mipdata_->feastol);
      }

      if (j_min != std::numeric_limits<HighsInt>::max()) {
        // Update current_fractional_integers
        auto it = findPairByIndex(current_fractional_integers, j_min);
        if (it != current_fractional_integers.end()) {
          current_fractional_integers.erase(it);
          fractionalIntegersReduced = true;
        }
        // Update current_relax_solution and shift_iterations_set (for not
        // fractional integers)
        if (moveValueUp) {
          if (fractionalIntegersReduced) {
            current_relax_solution[j_min] =
                std::ceil(x_j_min - mipsolver.mipdata_->feastol);
          } else {
            if (currentLp.integrality_[j_min] == HighsVarType::kInteger) {
              // variable is integer and not at the upper bound, so increment
              // by 1.
              current_relax_solution[j_min] = x_j_min + 1.0;
            } else {
              current_relax_solution[j_min] = std::min(
                  x_j_min + infeasibility / std::abs(aij_min),
                  currentLp.col_upper_[j_min] + mipsolver.mipdata_->feastol);
            }
            shift_iterations_set[j_min].push_back(t);
          }
        } else {
          if (fractionalIntegersReduced) {
            current_relax_solution[j_min] =
                std::floor(x_j_min + mipsolver.mipdata_->feastol);
          } else {
            if (currentLp.integrality_[j_min] == HighsVarType::kInteger) {
              // variable is integer and not at the lower bound, so decrement
              // by 1.
              current_relax_solution[j_min] = x_j_min - 1.0;
            } else {
              current_relax_solution[j_min] = std::max(
                  x_j_min - infeasibility / std::abs(aij_min),
                  currentLp.col_lower_[j_min] - mipsolver.mipdata_->feastol);
            }

            shift_iterations_set[j_min].push_back(-t);
          }
        }
      }
    } else {
      double xi_max = -1;
      double delta_c_min = kHighsInf;
      HighsInt pind_j_min = std::numeric_limits<HighsInt>::max();
      HighsInt j_min = std::numeric_limits<HighsInt>::max();
      double x_j_min = kHighsInf;
      HighsInt sigma = 0;
      for (HighsInt i = 0;
           i != static_cast<HighsInt>(current_fractional_integers.size());
           ++i) {
        std::pair<HighsInt, double> it = current_fractional_integers[i];
        HighsInt col = it.first;
        assert(col >= 0);
        assert(col < mipsolver.numCol());

        auto isBetter = [&currentLp, &it, &xi_max, &delta_c_min, &pind_j_min,
                         &j_min, &x_j_min, &sigma,
                         &i](double col, double xi, double roundedval,
                             HighsInt direction) {
          double c_min = currentLp.col_cost_[col] * (roundedval - it.second);
          if (xi > xi_max || (xi == xi_max && c_min < delta_c_min)) {
            xi_max = xi;
            delta_c_min = c_min;
            pind_j_min = i;
            j_min = col;
            x_j_min = roundedval;
            sigma = direction;
          }
        };

        isBetter(col, mipsolver.mipdata_->uplocks[col],
                 std::floor(it.second + mipsolver.mipdata_->feastol),
                 HighsInt{-1});
        isBetter(col, mipsolver.mipdata_->downlocks[col],
                 std::ceil(it.second - mipsolver.mipdata_->feastol),
                 HighsInt{1});
      }
      if (sigma != 0) {
        current_relax_solution[j_min] = x_j_min;
      }
      if (pind_j_min != std::numeric_limits<HighsInt>::max()) {
        current_fractional_integers.erase(current_fractional_integers.begin() +
                                          pind_j_min);
        fractionalIntegersReduced = true;
      }
    }
    current_infeasible_rows =
        mipsolver.mipdata_->getInfeasibleRows(current_relax_solution);
    hasInfeasibleConstraints = current_infeasible_rows.size() != 0;
    if (current_infeasible_rows.size() < previous_infeasible_rows_size ||
        fractionalIntegersReduced)
      iterationsWithoutReductions = 0;
    previous_infeasible_rows_size = current_infeasible_rows.size();
  }
  // re-check for feasibility and add incumbent
  if (hasInfeasibleConstraints) {
    tryRoundedPoint(worker, current_relax_solution, kSolutionSourceShifting);
  } else {
    if (current_fractional_integers.size() > 0) {
      ziRound(worker, current_relax_solution);
    } else {
      trySolution(current_relax_solution, kSolutionSourceShifting, worker);
    }
  }
}

void HighsPrimalHeuristics::ziRound(HighsMipWorker& worker,
                                    const std::vector<double>& relaxationsol) {
  // if (mipsolver.submip) return;
  if (relaxationsol.size() != static_cast<size_t>(mipsolver.numCol())) return;

  std::vector<double> current_relax_solution = relaxationsol;

  auto zi = [this](double x) {
    return std::min(std::ceil(x - mipsolver.mipdata_->feastol) - x,
                    x - std::floor(x + mipsolver.mipdata_->feastol));
  };

  // auto localdom = mipsolver.mipdata_->getDomain();

  HighsCDouble zi_total = 0.0;
  for (HighsInt i : intcols) {
    zi_total += zi(current_relax_solution[i]);
  }

  if (zi_total <= mipsolver.mipdata_->feastol) return;

  const HighsLp& currentLp = *mipsolver.model_;

  std::vector<double> rowActivities;
  std::vector<double> XrowLower;
  std::vector<double> XrowUpper;
  rowActivities.resize(currentLp.num_row_);
  XrowLower.resize(currentLp.num_row_);
  XrowUpper.resize(currentLp.num_row_);

  HighsInt loop_count = 0;
  HighsInt max_loop_count = 5;
  HighsCDouble previous_zi_total;
  HighsCDouble improvement_in_feasibility = kHighsInf;

  while (zi_total > mipsolver.mipdata_->feastol &&
         improvement_in_feasibility > mipsolver.mipdata_->feastol &&
         loop_count <= max_loop_count) {
    previous_zi_total = zi_total;
    loop_count++;

    if (currentLp.num_row_ > 0)
      getLpRowBounds(currentLp, 0, currentLp.num_row_ - 1, XrowLower.data(),
                     XrowUpper.data());

    for (HighsInt j : intcols) {
      double relax_solution = current_relax_solution[j];
      if (fractionality(relax_solution) <= mipsolver.mipdata_->feastol)
        continue;

      rowActivities.assign(currentLp.num_row_, 0.0);
      calculateRowValuesQuad(currentLp, current_relax_solution, rowActivities);

      double min_row_ratio_for_upper = kHighsInf;
      double min_row_ratio_for_lower = kHighsInf;

      for (HighsInt el = currentLp.a_matrix_.start_[j];
           el < currentLp.a_matrix_.start_[j + 1]; el++) {
        HighsInt i = currentLp.a_matrix_.index_[el];
        double aij = currentLp.a_matrix_.value_[el];

        double slack_upper = XrowUpper[i] - rowActivities[i];
        double slack_lower = rowActivities[i] - XrowLower[i];
        min_row_ratio_for_upper =
            std::min(min_row_ratio_for_upper,
                     (aij > 0 ? slack_upper : -slack_lower) / aij);
        min_row_ratio_for_lower =
            std::min(min_row_ratio_for_lower,
                     (aij > 0 ? slack_lower : -slack_upper) / aij);
      }

      double upper_bound = std::min(currentLp.col_upper_[j] - relax_solution,
                                    min_row_ratio_for_upper);
      double lower_bound = std::min(relax_solution - currentLp.col_lower_[j],
                                    min_row_ratio_for_lower);

      auto performUpdates = [&](HighsInt col, double change) {
        double old_relax_solution = current_relax_solution[col];
        current_relax_solution[col] += change;
        zi_total =
            zi_total - zi(old_relax_solution) + zi(current_relax_solution[col]);
      };

      if (std::abs(zi(relax_solution + upper_bound) -
                   zi(relax_solution - lower_bound)) <=
              mipsolver.mipdata_->feastol &&
          zi(relax_solution + upper_bound) < zi(relax_solution)) {
        double XcolCost = currentLp.col_cost_[j];
        bool ubObjChangeSmaller = XcolCost * (relax_solution + upper_bound) <=
                                  XcolCost * (relax_solution - lower_bound);
        bool isMinimization = currentLp.sense_ == ObjSense::kMinimize;
        if ((isMinimization && ubObjChangeSmaller) ||
            (!isMinimization && !ubObjChangeSmaller)) {
          performUpdates(j, upper_bound);
        } else {
          performUpdates(j, -lower_bound);
        }
      } else if (zi(relax_solution + upper_bound) <
                     zi(relax_solution - lower_bound) &&
                 zi(relax_solution + upper_bound) < zi(relax_solution)) {
        performUpdates(j, upper_bound);
      } else if (zi(relax_solution + upper_bound) >
                     zi(relax_solution - lower_bound) &&
                 zi(relax_solution - lower_bound) < zi(relax_solution)) {
        performUpdates(j, -lower_bound);
      }
    }
    improvement_in_feasibility = previous_zi_total - zi_total;
  }
  // re-check for feasibility and add incumbent
  trySolution(current_relax_solution, kSolutionSourceZiRound, worker);
}

void HighsPrimalHeuristics::feasibilityPump(HighsMipWorker& worker) {
  // LP relaxation instantiation
  HighsLpRelaxation lprelax(worker.getLpRelaxation());
  lprelax.setMipWorker(worker);
  lprelax.setProfiling(mipsolver.profiling_);
  std::unordered_set<std::vector<HighsInt>, HighsVectorHasher, HighsVectorEqual>
      referencepoints;
  std::vector<double> roundedsol;
  HighsLpRelaxation::Status status = lprelax.resolveLp();
  worker.getHeurLpIterations() += lprelax.getNumLpIterations();

  HighsRandom& randgen =
      mipsolver.mipdata_->parallelLockActive() ? worker.randgen : this->randgen;

  std::vector<double> fracintcost;
  std::vector<HighsInt> fracintset;

  std::vector<HighsInt> mask(mipsolver.numCol(), 1);
  std::vector<double> cost(mipsolver.numCol(), 0.0);

  lprelax.getLpSolver().setOptionValue("simplex_strategy",
                                       kSimplexStrategyPrimal);
  lprelax.setObjectiveLimit();
  lprelax.getLpSolver().setOptionValue(
      "primal_simplex_bound_perturbation_multiplier", 0.0);

  lprelax.setIterationLimit(5 * mipsolver.mipdata_->avgrootlpiters);

  while (!lprelax.getFractionalIntegers().empty()) {
    const auto& lpsol = lprelax.getLpSolver().getSolution().col_value;
    roundedsol = lprelax.getLpSolver().getSolution().col_value;

    std::vector<HighsInt> referencepoint;
    referencepoint.reserve(mipsolver.mipdata_->integer_cols.size());

    HighsDomain localdom = worker.getGlobalDomain();
    for (HighsInt i : mipsolver.mipdata_->integer_cols) {
      assert(mipsolver.isColInteger(i));
      double intval = std::floor(roundedsol[i] + randgen.real(0.4, 0.6));
      intval = std::max(intval, localdom.col_lower_[i]);
      intval = std::min(intval, localdom.col_upper_[i]);
      roundedsol[i] = intval;
      referencepoint.push_back((HighsInt)intval);
      if (!localdom.infeasible()) {
        localdom.fixCol(i, intval, HighsDomain::Reason::branching());
        if (localdom.infeasible()) {
          localdom.conflictAnalysis(worker.getConflictPool(),
                                    worker.getGlobalDomain(),
                                    worker.getPseudocost());
          continue;
        }
        localdom.propagate();
        if (localdom.infeasible()) {
          localdom.conflictAnalysis(worker.getConflictPool(),
                                    worker.getGlobalDomain(),
                                    worker.getPseudocost());
          continue;
        }
      }
    }

    bool havecycle = !referencepoints.emplace(referencepoint).second;
    for (HighsInt k = 0; havecycle && k < 2; ++k) {
      for (HighsInt i = 0; i != 10; ++i) {
        HighsInt flippos =
            randgen.integer(mipsolver.mipdata_->integer_cols.size());
        HighsInt col = mipsolver.mipdata_->integer_cols[flippos];
        if (roundedsol[col] > lpsol[col])
          roundedsol[col] = (HighsInt)std::floor(lpsol[col]);
        else if (roundedsol[col] < lpsol[col])
          roundedsol[col] = (HighsInt)std::ceil(lpsol[col]);
        else if (roundedsol[col] < worker.getGlobalDomain().col_upper_[col])
          roundedsol[col] = worker.getGlobalDomain().col_upper_[col];
        else
          roundedsol[col] = worker.getGlobalDomain().col_lower_[col];

        referencepoint[flippos] = (HighsInt)roundedsol[col];
      }
      havecycle = !referencepoints.emplace(referencepoint).second;
    }

    if (havecycle) return;

    if (linesearchRounding(worker, lpsol, roundedsol,
                           kSolutionSourceFeasibilityPump))
      return;

    if (lprelax.getNumLpIterations() >=
        1000 + mipsolver.mipdata_->avgrootlpiters * 5)
      break;

    for (HighsInt i : mipsolver.mipdata_->integer_cols) {
      assert(mipsolver.isColInteger(i));

      if (mipsolver.mipdata_->uplocks[i] == 0 ||
          mipsolver.mipdata_->downlocks[i] == 0)
        cost[i] = 0.0;
      else if (lpsol[i] > roundedsol[i] - mipsolver.mipdata_->feastol)
        cost[i] = -1.0 + randgen.real(-1e-4, 1e-4);
      else
        cost[i] = 1.0 + randgen.real(-1e-4, 1e-4);
    }

    lprelax.getLpSolver().changeColsCost(mask.data(), cost.data());
    int64_t niters = -lprelax.getNumLpIterations();
    status = lprelax.resolveLp();
    niters += lprelax.getNumLpIterations();
    if (niters == 0) break;
    worker.getHeurLpIterations() += niters;
  }

  if (lprelax.getFractionalIntegers().empty() &&
      HighsLpRelaxation::unscaledPrimalFeasible(status)) {
    addIncumbent(lprelax.getLpSolver().getSolution().col_value,
                 lprelax.getObjective(), kSolutionSourceFeasibilityPump,
                 worker);
  }
}

void HighsPrimalHeuristics::centralRounding(HighsMipWorker& worker) {
  if (mipsolver.mipdata_->analyticCenter.size() !=
      static_cast<size_t>(mipsolver.numCol())) {
    // The root portfolio may deliberately omit analytic-centre work. Keep
    // the bounded completion pass useful with already-owned root LP points.
    const auto& first = mipsolver.mipdata_->firstlpsol;
    const auto& root = mipsolver.mipdata_->rootlpsol;
    const size_t columns = static_cast<size_t>(mipsolver.numCol());
    const bool have_first = first.size() == columns;
    const bool have_root = root.size() == columns;
    if (have_first || have_root)
      linesearchRounding(worker, have_first ? first : root,
                         have_root ? root : first,
                         kSolutionSourceRandomizedRounding);
    return;
  }

  if (!mipsolver.mipdata_->firstlpsol.empty())
    linesearchRounding(worker, mipsolver.mipdata_->firstlpsol,
                       mipsolver.mipdata_->analyticCenter,
                       kSolutionSourceCentralRounding);
  else if (!mipsolver.mipdata_->rootlpsol.empty())
    linesearchRounding(worker, mipsolver.mipdata_->rootlpsol,
                       mipsolver.mipdata_->analyticCenter,
                       kSolutionSourceCentralRounding);
  else
    linesearchRounding(worker, mipsolver.mipdata_->analyticCenter,
                       mipsolver.mipdata_->analyticCenter,
                       kSolutionSourceCentralRounding);
}

#if 0
void HighsPrimalHeuristics::clique() {
  HighsHashTable<HighsInt, double> entries;
  double offset = 0.0;

  HighsDomain& globaldom = mipsolver.mipdata_->getDomain();
  for (HighsInt j = 0; j != mipsolver.numCol(); ++j) {
    HighsInt col = j;
    double val = mipsolver.colCost(col);
    if (val == 0.0) continue;

    if (!globaldom.isBinary(col)) {
      offset += val * globaldom.col_lower_[col];
      continue;
    }

    mipsolver.mipdata_->cliquetable.resolveSubstitution(col, val, offset);
    entries[col] += val;
  }

  std::vector<double> profits;
  std::vector<HighsCliqueTable::CliqueVar> objvars;

  for (const auto& entry : entries) {
    double objprofit = -entry.value();
    if (objprofit < 0) {
      offset += objprofit;
      profits.push_back(-objprofit);
      objvars.emplace_back(entry.key(), 0);
    } else {
      profits.push_back(objprofit);
      objvars.emplace_back(entry.key(), 1);
    }
  }

  std::vector<double> solution(mipsolver.numCol());

  HighsInt nobjvars = profits.size();
  for (HighsInt i = 0; i != nobjvars; ++i) solution[objvars[i].col] = objvars[i].val;

  std::vector<std::vector<HighsCliqueTable::CliqueVar>> cliques;
  double bestviol;
  HighsInt bestviolpos;
  HighsInt numcliques;

  cliques = mipsolver.mipdata_->cliquetable.separateCliques(
      solution, mipsolver.mipdata_->getDomain(), mipsolver.mipdata_->feastol);
  numcliques = cliques.size();
  while (numcliques != 0) {
    bestviol = 0.5;
    bestviolpos = -1;

    for (HighsInt c = 0; c != numcliques; ++c) {
      double viol = -1.0;
      for (HighsCliqueTable::CliqueVar clqvar : cliques[c])
        viol += clqvar.weight(solution);

      if (viol > bestviolpos) {
        bestviolpos = c;
        bestviol = viol;
      }
    }

    cliques = mipsolver.mipdata_->cliquetable.separateCliques(
        solution, mipsolver.mipdata_->getDomain(), mipsolver.mipdata_->feastol);
    numcliques = cliques.size();
  }
}
#endif

bool HighsPrimalHeuristics::addIncumbent(const std::vector<double>& sol,
                                         double solobj,
                                         const int solution_source,
                                         HighsMipWorker& worker) {
  if (mipsolver.mipdata_->parallelLockActive()) {
    return worker.addIncumbent(sol, solobj, solution_source);
  } else {
    return mipsolver.mipdata_->addIncumbent(sol, solobj, solution_source);
  }
}

bool HighsPrimalHeuristics::trySolution(const std::vector<double>& solution,
                                        const int solution_source,
                                        HighsMipWorker& worker,
                                        HighsPrimalAdmissionReceipt* receipt) {
  if (receipt) *receipt = {};
  if (mipsolver.mipdata_->parallelLockActive()) {
    return worker.trySolution(solution, solution_source, receipt);
  } else {
    const double before = mipsolver.mipdata_->upper_bound;
    const bool accepted = mipsolver.mipdata_->trySolution(solution, solution_source);
    if (accepted && receipt) {
      receipt->measured = true;
      receipt->before = before;
      receipt->after = mipsolver.mipdata_->upper_bound;
      receipt->improved = receipt->after < before;
      // Only an improved master witness is known to be original-space checked.
      receipt->candidate = receipt->improved ? receipt->after : kHighsInf;
      if (receipt->improved) receipt->checked_seed = mipsolver.mipdata_->incumbent;
    }
    return accepted;
  }
}

HighsInt HighsPrimalHeuristics::getNumSuccessObservations(
    HighsMipWorker& worker) const {
  return numSuccessObservations + worker.getHeurNumSuccessObservations();
}

HighsInt HighsPrimalHeuristics::getNumInfeasObservations(
    HighsMipWorker& worker) const {
  return numInfeasObservations + worker.getHeurNumInfeasObservations();
}

double HighsPrimalHeuristics::getSuccessObservations(
    HighsMipWorker& worker) const {
  return successObservations + worker.getHeurSuccessObservations();
}

double HighsPrimalHeuristics::getInfeasObservations(
    HighsMipWorker& worker) const {
  return infeasObservations + worker.getHeurInfeasObservations();
}

void HighsPrimalHeuristics::flushStatistics(HighsMipSolver& mipsolver,
                                            HighsMipWorker& worker) {
  int64_t total_repair_lp;
  int64_t total_repair_lp_feasible;
  int64_t total_repair_lp_iterations;
  int64_t lp_iterations;
  double successObservations;
  HighsInt numSuccessObservations;
  double infeasObservations;
  HighsInt numInfeasObservations;
  HighsInt max_submip_level;
  HighsModelStatus termination_status;
  worker.getHeurStatsValues(total_repair_lp, total_repair_lp_feasible,
                            total_repair_lp_iterations, lp_iterations,
                            successObservations, numSuccessObservations,
                            infeasObservations, numInfeasObservations,
                            max_submip_level, termination_status);

  mipsolver.mipdata_->total_repair_lp += total_repair_lp;
  mipsolver.mipdata_->total_repair_lp_feasible += total_repair_lp_feasible;
  mipsolver.mipdata_->total_repair_lp_iterations += total_repair_lp_iterations;
  mipsolver.mipdata_->heuristic_lp_iterations += lp_iterations;
  mipsolver.mipdata_->total_lp_iterations += lp_iterations;
  mipsolver.max_submip_level =
      std::max(mipsolver.max_submip_level, max_submip_level);
  if (termination_status != HighsModelStatus::kNotset &&
      mipsolver.termination_status_ == HighsModelStatus::kNotset) {
    mipsolver.termination_status_ = termination_status;
  }
  this->successObservations += successObservations;
  this->numSuccessObservations += numSuccessObservations;
  this->infeasObservations += infeasObservations;
  this->numInfeasObservations += numInfeasObservations;
  worker.resetHeurStats();
}
