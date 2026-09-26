/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "mip/HighsRootWork.h"

#include <algorithm>
#include <cmath>

void HighsRootWorkController::beginStage(const HighsRootWorkConfig& config,
                                         double now, double absolute_deadline) {
  config_ = config;
  snapshot_ = HighsRootWorkSnapshot{};
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

  if (atTreeReserve(now))
    return {HighsRootWorkAction::kYieldToTree,
            HighsRootWorkReason::kTreeReserve};

  if (phase == HighsRootWorkPhase::kSeparation && separationLimited()) {
    const HighsRootWorkReason reason =
        config_.max_separation_rounds >= 0 &&
                snapshot_.separation_rounds >= config_.max_separation_rounds
            ? HighsRootWorkReason::kSeparationRoundLimit
            : HighsRootWorkReason::kSeparationTimeLimit;
    return {HighsRootWorkAction::kYieldToTree, reason};
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
  if (phase == HighsRootWorkPhase::kSeparation) {
    snapshot_.separation_time += accounted;
    if (produced_separation_round) ++snapshot_.separation_rounds;
  } else if (isHeuristic(phase)) {
    snapshot_.heuristic_time += accounted;
  }
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
    case HighsRootWorkPhase::kRens:
    case HighsRootWorkPhase::kFeasibilityPump:
      return true;
    case HighsRootWorkPhase::kInitialLp:
    case HighsRootWorkPhase::kSeparation:
    case HighsRootWorkPhase::kAnalyticCenter:
    case HighsRootWorkPhase::kRestart:
    case HighsRootWorkPhase::kCleanup:
    case HighsRootWorkPhase::kTree:
      return false;
  }
  return false;
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
    case HighsRootWorkPhase::kRens:
      return "rens";
    case HighsRootWorkPhase::kFeasibilityPump:
      return "feasibility_pump";
    case HighsRootWorkPhase::kRestart:
      return "restart";
    case HighsRootWorkPhase::kCleanup:
      return "cleanup";
    case HighsRootWorkPhase::kTree:
      return "tree";
  }
  return "unknown";
}

const char* HighsRootWorkController::actionName(HighsRootWorkAction action) {
  switch (action) {
    case HighsRootWorkAction::kContinue:
      return "continue";
    case HighsRootWorkAction::kSkipOptional:
      return "skip_optional";
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
    case HighsRootWorkReason::kHeuristicTimeLimit:
      return "heuristic_time_limit";
  }
  return "unknown";
}
