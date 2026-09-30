/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef HIGHS_NODE_LP_EFFORT_H_
#define HIGHS_NODE_LP_EFFORT_H_

#include <array>
#include <cstddef>
#include <cstdint>

#include "lp_data/HConst.h"

struct HighsNodeLpEffortConfig {
  bool enabled = false;
  HighsInt minimum_iterations = 10000;
  HighsInt maximum_iterations = 250000;
  double quantile_multiplier = 2.0;
  HighsInt retry_multiplier = 2;
  HighsInt maximum_yields = 3;
};

enum class HighsNodeLpEffortOutcome {
  kCompleted,
  kNumericalFailure,
};

struct HighsNodeLpEffortObservation {
  HighsInt depth = 0;
  HighsInt iteration_count = 0;
  HighsInt row_count = 0;
  HighsInt column_count = 0;
  HighsInt basis_age = 0;
  HighsNodeLpEffortOutcome outcome = HighsNodeLpEffortOutcome::kCompleted;
  bool improved_bound = false;
};

struct HighsNodeLpEffortSummary {
  static constexpr size_t kHistogramBuckets = 32;

  int64_t attempts = 0;
  int64_t completed = 0;
  int64_t yielded = 0;
  int64_t serial_completions = 0;
  int64_t numerical_failures = 0;
  int64_t total_iterations = 0;
  int64_t completed_iterations = 0;
  int64_t yielded_iterations = 0;
  HighsInt maximum_iterations = 0;
  int64_t task_leases = 0;
  double task_lease_total_seconds = 0.0;
  double task_lease_max_seconds = 0.0;
  // Completed work is the only source used to learn future budgets. Yielded
  // work is right-censored evidence and is retained separately.
  std::array<int64_t, kHistogramBuckets> histogram{};
  std::array<int64_t, kHistogramBuckets> yielded_histogram{};
  std::array<int64_t, kHistogramBuckets> task_lease_microseconds_histogram{};
};

// Deterministic controller for limiting one node LP without changing the
// mathematical search. A limited node must be requeued by the caller; this
// class never declares a node infeasible, bounded, or pruned.
class HighsNodeLpEffortController {
 public:
  void configure(const HighsNodeLpEffortConfig& config);

  bool enabled() const { return config_.enabled; }

  HighsInt iterationLimit(HighsInt first_root_iterations, HighsInt depth,
                          HighsInt yield_count) const;

  void recordAttempt();
  void recordCompleted(const HighsNodeLpEffortObservation& observation);
  void recordYield(HighsInt iteration_count);
  void recordSerialCompletion();
  void recordTaskLease(double seconds);

  const HighsNodeLpEffortConfig& config() const { return config_; }
  const HighsNodeLpEffortSummary& summary() const { return summary_; }

  HighsInt percentileIterations(double quantile, HighsInt depth = -1) const;

 private:
  static constexpr size_t kDepthBuckets = 4;
  static constexpr size_t kRecentWindowSize = 256;
  static constexpr size_t kMinimumLocalSamples = 8;

  static size_t histogramBucket(HighsInt iterations);
  static HighsInt histogramUpperBound(size_t bucket);
  static size_t depthBucket(HighsInt depth);
  void recordIterations(
      HighsInt iterations,
      std::array<int64_t, HighsNodeLpEffortSummary::kHistogramBuckets>&
          histogram);
  void recordCompletedWindow(HighsInt depth, HighsInt iterations);
  HighsInt percentileCompletedWindow(size_t bucket, double quantile) const;

  HighsNodeLpEffortConfig config_;
  HighsNodeLpEffortSummary summary_;
  std::array<std::array<HighsInt, kRecentWindowSize>, kDepthBuckets>
      completed_windows_{};
  std::array<size_t, kDepthBuckets> completed_window_sizes_{};
  std::array<size_t, kDepthBuckets> completed_window_positions_{};
};

#endif
