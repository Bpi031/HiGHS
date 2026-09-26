/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "mip/HighsRootWork.h"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace {
// Experimental root allocation remains opt-in. These conservative constants
// stop separation only after two measured low-value rounds and at least three
// completed rounds. The score charges elapsed time, LP iterations, and matrix
// growth, so a large cut LP must earn proportionate dual-bound progress.
constexpr int64_t kMinSeparationRoundsBeforeMarginalStop = 3;
constexpr int64_t kLowValueRoundsBeforeMarginalStop = 2;
constexpr double kMinimumSeparationMarginalValue = 1e-9;
constexpr double kLpIterationsPerCostUnit = 100000.0;
constexpr double kRowsPerCostUnit = 1000.0;
constexpr double kNonzerosPerCostUnit = 100000.0;
}  // namespace

void HighsRootWorkController::beginStage(const HighsRootWorkConfig& config,
                                         double now, double absolute_deadline) {
  config_ = config;
  snapshot_ = HighsRootWorkSnapshot{};
  activities_ = {};
  snapshot_.stage_start = now;
  snapshot_.absolute_deadline = absolute_deadline;
}

void HighsRootWorkController::beginEpoch() { ++snapshot_.epoch; }

HighsRootWorkDecision HighsRootWorkController::beforeOptional(
    HighsRootWorkPhase phase, double now) const {
  if (!config_.enabled)
    return {HighsRootWorkAction::kContinue, HighsRootWorkReason::kDisabled};

  if (atGlobalLimit(now))
    return {HighsRootWorkAction::kStopGlobal,
            HighsRootWorkReason::kGlobalLimit};

  if (phase == HighsRootWorkPhase::kSeparation && snapshot_.separation_stopped)
    return {HighsRootWorkAction::kStopSeparation,
            HighsRootWorkReason::kSeparationLowMarginalValue};

  if (atTreeReserve(now)) {
    if (phase == HighsRootWorkPhase::kSeparation)
      return {HighsRootWorkAction::kStopSeparation,
              HighsRootWorkReason::kTreeReserve};
    return {HighsRootWorkAction::kYieldToTree,
            HighsRootWorkReason::kTreeReserve};
  }

  if (phase == HighsRootWorkPhase::kSeparation && separationLimited()) {
    const HighsRootWorkReason reason =
        config_.max_separation_rounds >= 0 &&
                snapshot_.separation_rounds >= config_.max_separation_rounds
            ? HighsRootWorkReason::kSeparationRoundLimit
            : HighsRootWorkReason::kSeparationTimeLimit;
    return {HighsRootWorkAction::kStopSeparation, reason};
  }

  if (isHeuristic(phase) && heuristicLimited())
    return {HighsRootWorkAction::kSkipOptional,
            HighsRootWorkReason::kHeuristicTimeLimit};

  return {HighsRootWorkAction::kContinue, HighsRootWorkReason::kNone};
}

double HighsRootWorkController::optionalAllowance(HighsRootWorkPhase phase,
                                                  double now) const {
  if (!config_.enabled) return std::numeric_limits<double>::infinity();

  const HighsRootWorkDecision decision = beforeOptional(phase, now);
  if (decision.action != HighsRootWorkAction::kContinue) return 0.0;

  double allowance = std::numeric_limits<double>::infinity();
  if (std::isfinite(snapshot_.absolute_deadline)) {
    double deadline = snapshot_.absolute_deadline;
    if (config_.tree_reserve_time >= 0.0) deadline -= config_.tree_reserve_time;
    allowance = std::max(0.0, deadline - now);
  }
  if (config_.max_single_helper_time >= 0.0)
    allowance = std::min(allowance, config_.max_single_helper_time);
  if (phase == HighsRootWorkPhase::kSeparation &&
      config_.max_separation_time >= 0.0)
    allowance = std::min(
        allowance,
        std::max(0.0, config_.max_separation_time - snapshot_.separation_time));
  if (isHeuristic(phase) && config_.max_heuristic_time >= 0.0)
    allowance = std::min(
        allowance,
        std::max(0.0, config_.max_heuristic_time - snapshot_.heuristic_time));
  return allowance;
}

void HighsRootWorkController::recordCompleted(HighsRootWorkPhase phase,
                                              double elapsed,
                                              bool produced_separation_round) {
  if (!config_.enabled) return;
  const double accounted =
      std::isfinite(elapsed) ? std::max(0.0, elapsed) : 0.0;
  HighsRootWorkActivityAccount& account = activities_[phaseIndex(phase)];
  account.wall_time += accounted;
  ++account.calls;
  if (phase == HighsRootWorkPhase::kSeparation) {
    snapshot_.separation_time += accounted;
    if (produced_separation_round) ++snapshot_.separation_rounds;
  } else if (isHeuristic(phase)) {
    snapshot_.heuristic_time += accounted;
  }
}

void HighsRootWorkController::recordCompleted(
    HighsRootWorkPhase phase, const HighsRootWorkObservation& before,
    const HighsRootWorkObservation& after, int64_t cuts_generated,
    bool produced_separation_round) {
  if (!config_.enabled) return;

  const double elapsed = after.time - before.time;
  recordCompleted(phase, elapsed, produced_separation_round);

  HighsRootWorkActivityAccount& account = activities_[phaseIndex(phase)];
  const int64_t lp_iterations =
      std::max<int64_t>(0, after.lp_iterations - before.lp_iterations);
  const int64_t accepted_incumbents = std::max<int64_t>(
      0, after.improving_solutions - before.improving_solutions);
  const int64_t cut_pool_rows_added =
      std::max<int64_t>(0, after.cut_pool_rows - before.cut_pool_rows);
  const int64_t lp_rows_added =
      std::max<int64_t>(0, after.active_lp_rows - before.active_lp_rows);
  const int64_t lp_nonzeros_added = std::max<int64_t>(
      0, after.active_lp_nonzeros - before.active_lp_nonzeros);

  double primal_gain = 0.0;
  if (std::isfinite(before.incumbent) && std::isfinite(after.incumbent))
    primal_gain = std::max(0.0, before.incumbent - after.incumbent);

  double dual_gain = 0.0;
  if (std::isfinite(before.dual_bound) && std::isfinite(after.dual_bound))
    dual_gain = std::max(0.0, after.dual_bound - before.dual_bound);

  account.lp_iterations += lp_iterations;
  account.accepted_incumbents += accepted_incumbents;
  account.primal_gain += primal_gain;
  account.dual_gain += dual_gain;
  account.cuts_generated += std::max<int64_t>(0, cuts_generated);
  account.cut_pool_rows_added += cut_pool_rows_added;
  account.lp_rows_added += lp_rows_added;
  account.lp_nonzeros_added += lp_nonzeros_added;
  account.last_fractional_integers = after.fractional_integers;
  if (accepted_incumbents > 0 || primal_gain > 0.0 || dual_gain > 0.0 ||
      cuts_generated > 0 || cut_pool_rows_added > 0 || lp_rows_added > 0)
    ++account.successes;

  if (phase == HighsRootWorkPhase::kSeparation && produced_separation_round)
    updateSeparationMarginalValue(before, after);
}

void HighsRootWorkController::updateSeparationMarginalValue(
    const HighsRootWorkObservation& before,
    const HighsRootWorkObservation& after) {
  const double elapsed = std::max(0.0, std::isfinite(after.time - before.time)
                                           ? after.time - before.time
                                           : 0.0);
  const double dual_gain =
      std::isfinite(before.dual_bound) && std::isfinite(after.dual_bound)
          ? std::max(0.0, after.dual_bound - before.dual_bound)
          : 0.0;
  double reference_scale = 1.0;
  if (std::isfinite(before.dual_bound))
    reference_scale = std::max(reference_scale, std::abs(before.dual_bound));
  if (std::isfinite(before.incumbent))
    reference_scale = std::max(reference_scale, std::abs(before.incumbent));

  const int64_t lp_iterations =
      std::max<int64_t>(0, after.lp_iterations - before.lp_iterations);
  const int64_t rows_added =
      std::max<int64_t>(0, after.active_lp_rows - before.active_lp_rows);
  const int64_t nonzeros_added = std::max<int64_t>(
      0, after.active_lp_nonzeros - before.active_lp_nonzeros);
  const double cost =
      std::max(1e-9, elapsed + lp_iterations / kLpIterationsPerCostUnit +
                         rows_added / kRowsPerCostUnit +
                         nonzeros_added / kNonzerosPerCostUnit);
  snapshot_.last_separation_marginal_value =
      (dual_gain / reference_scale) / cost;

  if (snapshot_.separation_rounds < kMinSeparationRoundsBeforeMarginalStop) {
    snapshot_.consecutive_low_value_separation_rounds = 0;
    return;
  }

  if (snapshot_.last_separation_marginal_value <
      kMinimumSeparationMarginalValue)
    ++snapshot_.consecutive_low_value_separation_rounds;
  else
    snapshot_.consecutive_low_value_separation_rounds = 0;

  if (snapshot_.consecutive_low_value_separation_rounds >=
      kLowValueRoundsBeforeMarginalStop)
    snapshot_.separation_stopped = true;
}

void HighsRootWorkController::recordTreeEntry() {
  snapshot_.tree_entered = true;
}

bool HighsRootWorkController::atGlobalLimit(double now) const {
  return std::isfinite(snapshot_.absolute_deadline) &&
         now >= snapshot_.absolute_deadline;
}

bool HighsRootWorkController::atTreeReserve(double now) const {
  return config_.tree_reserve_time >= 0.0 &&
         std::isfinite(snapshot_.absolute_deadline) &&
         snapshot_.absolute_deadline - now <= config_.tree_reserve_time;
}

bool HighsRootWorkController::separationLimited() const {
  return (config_.max_separation_rounds >= 0 &&
          snapshot_.separation_rounds >= config_.max_separation_rounds) ||
         (config_.max_separation_time >= 0.0 &&
          snapshot_.separation_time >= config_.max_separation_time);
}

bool HighsRootWorkController::heuristicLimited() const {
  return config_.max_heuristic_time >= 0.0 &&
         snapshot_.heuristic_time >= config_.max_heuristic_time;
}

bool HighsRootWorkController::isHeuristic(HighsRootWorkPhase phase) {
  switch (phase) {
    case HighsRootWorkPhase::kRounding:
    case HighsRootWorkPhase::kReducedCostHeuristic:
    case HighsRootWorkPhase::kIncumbentCompletion:
    case HighsRootWorkPhase::kRens:
    case HighsRootWorkPhase::kFeasibilityPump:
      return true;
    case HighsRootWorkPhase::kInitialLp:
    case HighsRootWorkPhase::kSeparation:
    case HighsRootWorkPhase::kAnalyticCenter:
    case HighsRootWorkPhase::kLpReoptimization:
    case HighsRootWorkPhase::kSymmetry:
    case HighsRootWorkPhase::kRestart:
    case HighsRootWorkPhase::kCleanup:
    case HighsRootWorkPhase::kTree:
    case HighsRootWorkPhase::kCount:
      return false;
  }
  return false;
}

size_t HighsRootWorkController::phaseIndex(HighsRootWorkPhase phase) {
  const size_t index = static_cast<size_t>(phase);
  assert(index < kHighsRootWorkPhaseCount);
  return index;
}

const HighsRootWorkActivityAccount& HighsRootWorkController::activity(
    HighsRootWorkPhase phase) const {
  return activities_[phaseIndex(phase)];
}

const char* HighsRootWorkController::phaseName(HighsRootWorkPhase phase) {
  switch (phase) {
    case HighsRootWorkPhase::kInitialLp:
      return "initial_lp";
    case HighsRootWorkPhase::kSeparation:
      return "separation";
    case HighsRootWorkPhase::kAnalyticCenter:
      return "analytic_center";
    case HighsRootWorkPhase::kRounding:
      return "rounding";
    case HighsRootWorkPhase::kReducedCostHeuristic:
      return "reduced_cost_heuristic";
    case HighsRootWorkPhase::kIncumbentCompletion:
      return "incumbent_completion";
    case HighsRootWorkPhase::kRens:
      return "rens";
    case HighsRootWorkPhase::kFeasibilityPump:
      return "feasibility_pump";
    case HighsRootWorkPhase::kLpReoptimization:
      return "lp_reoptimization";
    case HighsRootWorkPhase::kSymmetry:
      return "symmetry";
    case HighsRootWorkPhase::kRestart:
      return "restart";
    case HighsRootWorkPhase::kCleanup:
      return "cleanup";
    case HighsRootWorkPhase::kTree:
      return "tree";
    case HighsRootWorkPhase::kCount:
      return "unknown";
  }
  return "unknown";
}

const char* HighsRootWorkController::actionName(HighsRootWorkAction action) {
  switch (action) {
    case HighsRootWorkAction::kContinue:
      return "continue";
    case HighsRootWorkAction::kSkipOptional:
      return "skip_optional";
    case HighsRootWorkAction::kStopSeparation:
      return "stop_separation";
    case HighsRootWorkAction::kYieldToTree:
      return "yield_to_tree";
    case HighsRootWorkAction::kStopGlobal:
      return "stop_global";
  }
  return "unknown";
}

const char* HighsRootWorkController::reasonName(HighsRootWorkReason reason) {
  switch (reason) {
    case HighsRootWorkReason::kNone:
      return "none";
    case HighsRootWorkReason::kDisabled:
      return "disabled";
    case HighsRootWorkReason::kGlobalLimit:
      return "global_limit";
    case HighsRootWorkReason::kTreeReserve:
      return "tree_reserve";
    case HighsRootWorkReason::kSeparationRoundLimit:
      return "separation_round_limit";
    case HighsRootWorkReason::kSeparationTimeLimit:
      return "separation_time_limit";
    case HighsRootWorkReason::kSeparationLowMarginalValue:
      return "separation_low_marginal_value";
    case HighsRootWorkReason::kHeuristicTimeLimit:
      return "heuristic_time_limit";
  }
  return "unknown";
}
