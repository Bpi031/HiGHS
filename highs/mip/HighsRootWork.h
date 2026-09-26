/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef HIGHS_ROOT_WORK_H_
#define HIGHS_ROOT_WORK_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

enum class HighsRootWorkPhase : uint8_t {
  kInitialLp,
  kSeparation,
  kAnalyticCenter,
  kRounding,
  kReducedCostHeuristic,
  kIncumbentCompletion,
  kRens,
  kFeasibilityPump,
  kLpReoptimization,
  kSymmetry,
  kRestart,
  kCleanup,
  kTree,
  kCount,
};

enum class HighsRootWorkAction : uint8_t {
  kContinue,
  kSkipOptional,
  kStopSeparation,
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
  kSeparationLowMarginalValue,
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
  bool separation_stopped = false;
  int64_t consecutive_low_value_separation_rounds = 0;
  double last_separation_marginal_value = -1.0;
};

struct HighsRootWorkObservation {
  double time = 0.0;
  int64_t lp_iterations = 0;
  int64_t improving_solutions = 0;
  double incumbent = std::numeric_limits<double>::infinity();
  double dual_bound = -std::numeric_limits<double>::infinity();
  int64_t active_lp_rows = 0;
  int64_t active_lp_nonzeros = 0;
  int64_t cut_pool_rows = 0;
  int64_t fractional_integers = 0;
};

struct HighsRootWorkActivityAccount {
  double wall_time = 0.0;
  int64_t lp_iterations = 0;
  int64_t calls = 0;
  int64_t successes = 0;
  int64_t accepted_incumbents = 0;
  double primal_gain = 0.0;
  double dual_gain = 0.0;
  int64_t cuts_generated = 0;
  int64_t cut_pool_rows_added = 0;
  int64_t lp_rows_added = 0;
  int64_t lp_nonzeros_added = 0;
  int64_t last_fractional_integers = -1;
};

constexpr size_t kHighsRootWorkPhaseCount =
    static_cast<size_t>(HighsRootWorkPhase::kCount);

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
  void recordCompleted(HighsRootWorkPhase phase,
                       const HighsRootWorkObservation& before,
                       const HighsRootWorkObservation& after,
                       int64_t cuts_generated = 0,
                       bool produced_separation_round = false);
  void recordSeparationStop() { snapshot_.separation_stopped = true; }
  void recordTreeEntry();

  bool enabled() const { return config_.enabled; }
  const HighsRootWorkConfig& config() const { return config_; }
  const HighsRootWorkSnapshot& snapshot() const { return snapshot_; }
  const HighsRootWorkActivityAccount& activity(HighsRootWorkPhase phase) const;

  static const char* phaseName(HighsRootWorkPhase phase);
  static const char* actionName(HighsRootWorkAction action);
  static const char* reasonName(HighsRootWorkReason reason);

 private:
  bool atGlobalLimit(double now) const;
  bool atTreeReserve(double now) const;
  bool separationLimited() const;
  bool heuristicLimited() const;
  void updateSeparationMarginalValue(const HighsRootWorkObservation& before,
                                     const HighsRootWorkObservation& after);
  static bool isHeuristic(HighsRootWorkPhase phase);
  static size_t phaseIndex(HighsRootWorkPhase phase);

  HighsRootWorkConfig config_;
  HighsRootWorkSnapshot snapshot_;
  std::array<HighsRootWorkActivityAccount, kHighsRootWorkPhaseCount>
      activities_;
};

#endif
