/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "mip/HighsNodeLpEffort.h"

#include <algorithm>
#include <cmath>
#include <limits>

void HighsNodeLpEffortController::configure(
    const HighsNodeLpEffortConfig& config) {
  config_ = config;
  config_.minimum_iterations = std::max<HighsInt>(1, config_.minimum_iterations);
  config_.maximum_iterations =
      std::max(config_.minimum_iterations, config_.maximum_iterations);
  if (!std::isfinite(config_.quantile_multiplier) ||
      config_.quantile_multiplier < 1.0)
    config_.quantile_multiplier = 1.0;
  config_.retry_multiplier = std::max<HighsInt>(1, config_.retry_multiplier);
  config_.maximum_yields = std::max<HighsInt>(0, config_.maximum_yields);
}

HighsInt HighsNodeLpEffortController::iterationLimit(
    HighsInt first_root_iterations, HighsInt depth,
    HighsInt yield_count) const {
  if (!config_.enabled || yield_count >= config_.maximum_yields)
    return kHighsIInf;

  HighsInt learned = percentileIterations(0.75, depth);
  if (learned <= 0) {
    // The root may seed the controller, but a difficult root must not become a
    // permanent floor for every tree node.
    learned = std::max<HighsInt>(config_.minimum_iterations,
                                 std::max<HighsInt>(0, first_root_iterations) /
                                     4);
  }

  long double budget =
      static_cast<long double>(learned) * config_.quantile_multiplier;
  // Shallow nodes tend to carry more global information and receive one
  // deterministic increment. Deep nodes retain the learned budget.
  if (depth <= 8) budget *= 1.25L;
  for (HighsInt retry = 0; retry < yield_count; ++retry)
    budget *= config_.retry_multiplier;

  budget = std::max<long double>(budget, config_.minimum_iterations);
  budget = std::min<long double>(budget, config_.maximum_iterations);
  if (budget >= kHighsIInf) return kHighsIInf;
  return static_cast<HighsInt>(std::ceil(budget));
}

void HighsNodeLpEffortController::recordAttempt() { ++summary_.attempts; }

void HighsNodeLpEffortController::recordCompleted(
    const HighsNodeLpEffortObservation& observation) {
  const HighsInt iterations = std::max<HighsInt>(0, observation.iteration_count);
  summary_.total_iterations += iterations;
  summary_.maximum_iterations =
      std::max(summary_.maximum_iterations, iterations);
  if (observation.outcome == HighsNodeLpEffortOutcome::kNumericalFailure) {
    ++summary_.numerical_failures;
    return;
  }
  ++summary_.completed;
  summary_.completed_iterations += iterations;
  recordIterations(iterations, summary_.histogram);
  recordCompletedWindow(observation.depth, iterations);
}

void HighsNodeLpEffortController::recordYield(HighsInt iteration_count) {
  ++summary_.yielded;
  const HighsInt iterations = std::max<HighsInt>(0, iteration_count);
  summary_.total_iterations += iterations;
  summary_.yielded_iterations += iterations;
  summary_.maximum_iterations =
      std::max(summary_.maximum_iterations, iterations);
  recordIterations(iterations, summary_.yielded_histogram);
}

void HighsNodeLpEffortController::recordSerialCompletion() {
  ++summary_.serial_completions;
}

void HighsNodeLpEffortController::recordTaskLease(double seconds) {
  if (!std::isfinite(seconds) || seconds < 0.0) return;
  ++summary_.task_leases;
  summary_.task_lease_total_seconds += seconds;
  summary_.task_lease_max_seconds =
      std::max(summary_.task_lease_max_seconds, seconds);
  const long double microseconds =
      std::ceil(static_cast<long double>(seconds) * 1000000.0L);
  const HighsInt bounded_microseconds = static_cast<HighsInt>(std::min<
      long double>(microseconds, std::numeric_limits<HighsInt>::max()));
  ++summary_.task_lease_microseconds_histogram[
      histogramBucket(bounded_microseconds)];
}

HighsInt HighsNodeLpEffortController::percentileIterations(
    double quantile, HighsInt depth) const {
  quantile = std::max(0.0, std::min(1.0, quantile));
  if (depth >= 0) {
    size_t bucket = depthBucket(depth);
    if (completed_window_sizes_[bucket] >= kMinimumLocalSamples)
      return percentileCompletedWindow(bucket, quantile);
    while (bucket > 0) {
      --bucket;
      if (completed_window_sizes_[bucket] != 0)
        return percentileCompletedWindow(bucket, quantile);
    }
  }

  int64_t samples = 0;
  for (int64_t count : summary_.histogram) samples += count;
  if (samples == 0) return 0;
  const int64_t target =
      std::max<int64_t>(1, static_cast<int64_t>(std::ceil(quantile * samples)));
  int64_t cumulative = 0;
  for (size_t bucket = 0; bucket != summary_.histogram.size(); ++bucket) {
    cumulative += summary_.histogram[bucket];
    if (cumulative >= target) return histogramUpperBound(bucket);
  }
  return summary_.maximum_iterations;
}

size_t HighsNodeLpEffortController::histogramBucket(HighsInt iterations) {
  uint64_t value = static_cast<uint64_t>(std::max<HighsInt>(1, iterations));
  size_t bucket = 0;
  while (value > 1 &&
         bucket + 1 < HighsNodeLpEffortSummary::kHistogramBuckets) {
    value >>= 1;
    ++bucket;
  }
  return bucket;
}

HighsInt HighsNodeLpEffortController::histogramUpperBound(size_t bucket) {
  if (bucket + 1 >= HighsNodeLpEffortSummary::kHistogramBuckets)
    return std::numeric_limits<HighsInt>::max();
  const uint64_t upper = uint64_t{1} << (bucket + 1);
  return upper >= static_cast<uint64_t>(std::numeric_limits<HighsInt>::max())
             ? std::numeric_limits<HighsInt>::max()
             : static_cast<HighsInt>(upper);
}

size_t HighsNodeLpEffortController::depthBucket(HighsInt depth) {
  depth = std::max<HighsInt>(0, depth);
  if (depth <= 8) return 0;
  if (depth <= 32) return 1;
  if (depth <= 128) return 2;
  return 3;
}

void HighsNodeLpEffortController::recordIterations(
    HighsInt iterations,
    std::array<int64_t, HighsNodeLpEffortSummary::kHistogramBuckets>&
        histogram) {
  ++histogram[histogramBucket(iterations)];
}

void HighsNodeLpEffortController::recordCompletedWindow(
    HighsInt depth, HighsInt iterations) {
  const size_t bucket = depthBucket(depth);
  const size_t position = completed_window_positions_[bucket];
  completed_windows_[bucket][position] = iterations;
  completed_window_positions_[bucket] = (position + 1) % kRecentWindowSize;
  completed_window_sizes_[bucket] =
      std::min(kRecentWindowSize, completed_window_sizes_[bucket] + 1);
}

HighsInt HighsNodeLpEffortController::percentileCompletedWindow(
    size_t bucket, double quantile) const {
  const size_t size = completed_window_sizes_[bucket];
  if (size == 0) return 0;
  std::array<HighsInt, kRecentWindowSize> sorted = completed_windows_[bucket];
  std::sort(sorted.begin(), sorted.begin() + size);
  const size_t target = std::max<size_t>(
      1, static_cast<size_t>(std::ceil(quantile * static_cast<double>(size))));
  return sorted[std::min(size, target) - 1];
}
