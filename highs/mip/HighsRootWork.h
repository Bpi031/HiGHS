/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef HIGHS_ROOT_WORK_H_
#define HIGHS_ROOT_WORK_H_

#include <cstdint>
#include <limits>

enum class HighsRootWorkPhase : uint8_t {
  kInitialLp,
  kSeparation,
  kAnalyticCenter,
  kRounding,
  kReducedCostHeuristic,
  kRens,
  kFeasibilityPump,
  kRestart,
  kCleanup,
  kTree,
};

enum class HighsRootWorkAction : uint8_t {
  kContinue,
  kSkipOptional,
  kYieldToTree,
  kStopGlobal,
};

enum class HighsRootWorkReason : uint8_t {
  kNone,
  kDisabled,
  kGlobalLimit,
  kTreeReserve,
  kSeparationRoundLimit,
  kSeparationTimeLimit,
  kHeuristicTimeLimit,
};

struct HighsRootWorkConfig {
  bool enabled = false;
  int64_t max_separation_rounds = -1;
  double max_separation_time = -1.0;
  double max_heuristic_time = -1.0;
  double max_single_helper_time = -1.0;
  double tree_reserve_time = -1.0;
};

struct HighsRootWorkDecision {
  HighsRootWorkAction action = HighsRootWorkAction::kContinue;
  HighsRootWorkReason reason = HighsRootWorkReason::kNone;

  HighsRootWorkDecision() {}
  HighsRootWorkDecision(HighsRootWorkAction action, HighsRootWorkReason reason)
      : action(action), reason(reason) {}
};

struct HighsRootWorkSnapshot {
  int64_t epoch = -1;
  int64_t separation_rounds = 0;
  double separation_time = 0.0;
  double heuristic_time = 0.0;
  double stage_start = 0.0;
  double absolute_deadline = std::numeric_limits<double>::infinity();
  bool tree_entered = false;
};

class HighsRootWorkController {
 public:
  void beginStage(const HighsRootWorkConfig& config, double now,
                  double absolute_deadline);
  void beginEpoch();

  HighsRootWorkDecision beforeOptional(HighsRootWorkPhase phase,
                                       double now) const;
  double optionalAllowance(HighsRootWorkPhase phase, double now) const;
  void recordCompleted(HighsRootWorkPhase phase, double elapsed,
                       bool produced_separation_round = false);
  void recordTreeEntry();

  bool enabled() const { return config_.enabled; }
  const HighsRootWorkConfig& config() const { return config_; }
  const HighsRootWorkSnapshot& snapshot() const { return snapshot_; }

  static const char* phaseName(HighsRootWorkPhase phase);
  static const char* actionName(HighsRootWorkAction action);
  static const char* reasonName(HighsRootWorkReason reason);

 private:
  bool atGlobalLimit(double now) const;
  bool atTreeReserve(double now) const;
  bool separationLimited() const;
  bool heuristicLimited() const;
  static bool isHeuristic(HighsRootWorkPhase phase);

  HighsRootWorkConfig config_;
  HighsRootWorkSnapshot snapshot_;
};

#endif
