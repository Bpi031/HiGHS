/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "mip/HighsNodeSeparation.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {
template <typename T>
void setDefault(T& value, T profile_value) {
  if (value < 0) value = profile_value;
}

int64_t ratioLimit(double ratio, int64_t base) {
  if (ratio < 0.0) return -1;
  if (!std::isfinite(ratio) || base <= 0) return 0;
  const long double limit =
      std::floor(static_cast<long double>(ratio) * base);
  if (limit >= std::numeric_limits<int64_t>::max())
    return std::numeric_limits<int64_t>::max();
  return static_cast<int64_t>(std::max<long double>(0.0, limit));
}
}  // namespace

void HighsNodeSeparationController::applyProfileDefaults(
    HighsNodeSeparationConfig& config, HighsNodeCutMode mode) {
  switch (mode) {
    case HighsNodeCutMode::kOff:
    case HighsNodeCutMode::kUnlimitedLegacy:
      return;
    case HighsNodeCutMode::kAutomatic:
      // Automatic is resolved to a fixed profile in beginNode.
      return;
    case HighsNodeCutMode::kConservative:
      setDefault(config.maximum_depth, HighsInt{8});
      setDefault(config.node_frequency, HighsInt{4});
      setDefault(config.max_rounds, HighsInt{1});
      setDefault(config.max_seconds, 5.0);
      setDefault(config.max_lp_iterations, int64_t{50000});
      setDefault(config.max_added_rows, HighsInt{250});
      setDefault(config.max_added_nonzeros, int64_t{25000});
      setDefault(config.max_selected_per_origin, HighsInt{100});
      setDefault(config.max_row_growth_ratio, 0.01);
      setDefault(config.max_nonzero_growth_ratio, 0.02);
      setDefault(config.low_value_patience, HighsInt{1});
      return;
    case HighsNodeCutMode::kModerate:
      setDefault(config.maximum_depth, HighsInt{24});
      setDefault(config.node_frequency, HighsInt{2});
      setDefault(config.max_rounds, HighsInt{2});
      setDefault(config.max_seconds, 15.0);
      setDefault(config.max_lp_iterations, int64_t{150000});
      setDefault(config.max_added_rows, HighsInt{1000});
      setDefault(config.max_added_nonzeros, int64_t{100000});
      setDefault(config.max_selected_per_origin, HighsInt{400});
      setDefault(config.max_row_growth_ratio, 0.03);
      setDefault(config.max_nonzero_growth_ratio, 0.05);
      setDefault(config.low_value_patience, HighsInt{2});
      return;
    case HighsNodeCutMode::kAggressive:
      setDefault(config.maximum_depth, HighsInt{64});
      setDefault(config.node_frequency, HighsInt{1});
      setDefault(config.max_rounds, HighsInt{4});
      setDefault(config.max_seconds, 30.0);
      setDefault(config.max_lp_iterations, int64_t{500000});
      setDefault(config.max_added_rows, HighsInt{4000});
      setDefault(config.max_added_nonzeros, int64_t{500000});
      setDefault(config.max_selected_per_origin, HighsInt{1500});
      setDefault(config.max_row_growth_ratio, 0.10);
      setDefault(config.max_nonzero_growth_ratio, 0.15);
      setDefault(config.low_value_patience, HighsInt{2});
      return;
  }
}

void HighsNodeSeparationController::beginNode(
    const HighsNodeSeparationConfig& config,
    const HighsNodeSeparationContext& context) {
  config_ = config;
  initial_ = context;
  summary_ = HighsNodeSeparationSummary{};
  summary_.controller_enabled = config.enabled;
  summary_.requested_mode = config.mode;
  summary_.effective_mode = config.mode;
  summary_.opportunity_sequence = context.opportunity_sequence;
  summary_.tree_nodes_processed = context.tree_nodes_processed;
  summary_.depth = context.depth;

  if (!config_.enabled) {
    summary_.stop_reason = HighsNodeSeparationReason::kControllerDisabled;
    return;
  }

  if (config_.mode == HighsNodeCutMode::kAutomatic) {
    summary_.effective_mode = context.depth <= 4
                                  ? HighsNodeCutMode::kModerate
                                  : HighsNodeCutMode::kConservative;
    config_.mode = summary_.effective_mode;
  }
  applyProfileDefaults(config_, config_.mode);

  if (config_.mode == HighsNodeCutMode::kOff) {
    summary_.eligible = false;
    summary_.stop_reason = HighsNodeSeparationReason::kModeOff;
    return;
  }
  if (config_.maximum_depth >= 0 && context.depth > config_.maximum_depth) {
    summary_.eligible = false;
    summary_.stop_reason = HighsNodeSeparationReason::kDepthLimit;
    return;
  }
  if (config_.node_frequency > 1 &&
      context.opportunity_sequence % config_.node_frequency != 0) {
    summary_.eligible = false;
    summary_.stop_reason = HighsNodeSeparationReason::kFrequencySkip;
  }
}

HighsNodeSeparationDecision HighsNodeSeparationController::limitedByGrowth()
    const {
  if (config_.max_added_rows >= 0 &&
      summary_.cuts_added >= config_.max_added_rows)
    return {false, HighsNodeSeparationReason::kRowLimit};
  if (config_.max_added_nonzeros >= 0 &&
      summary_.selected_nonzeros >= config_.max_added_nonzeros)
    return {false, HighsNodeSeparationReason::kNonzeroLimit};

  const int64_t row_growth_limit =
      ratioLimit(config_.max_row_growth_ratio, initial_.model_rows);
  if (row_growth_limit >= 0 && summary_.cuts_added >= row_growth_limit)
    return {false, HighsNodeSeparationReason::kRowGrowthLimit};

  const int64_t nonzero_growth_limit = ratioLimit(
      config_.max_nonzero_growth_ratio, initial_.model_nonzeros);
  if (nonzero_growth_limit >= 0 &&
      summary_.selected_nonzeros >= nonzero_growth_limit)
    return {false, HighsNodeSeparationReason::kNonzeroGrowthLimit};

  return {true, HighsNodeSeparationReason::kNone};
}

HighsNodeSeparationDecision HighsNodeSeparationController::beforeRound(
    const HighsNodeSeparationContext& current) const {
  if (!config_.enabled)
    return {true, HighsNodeSeparationReason::kControllerDisabled};
  if (!summary_.eligible) return {false, summary_.stop_reason};
  if (summary_.stop_reason != HighsNodeSeparationReason::kNone)
    return {false, summary_.stop_reason};
  if (config_.max_rounds >= 0 && summary_.rounds >= config_.max_rounds)
    return {false, HighsNodeSeparationReason::kRoundLimit};
  if (config_.max_seconds >= 0.0 &&
      current.time - initial_.time >= config_.max_seconds)
    return {false, HighsNodeSeparationReason::kTimeLimit};
  if (config_.max_lp_iterations >= 0 &&
      current.lp_iterations - initial_.lp_iterations >=
          config_.max_lp_iterations)
    return {false, HighsNodeSeparationReason::kLpIterationLimit};
  return limitedByGrowth();
}

void HighsNodeSeparationController::recordRound(
    const HighsNodeSeparationRoundResult& round) {
  ++summary_.rounds;
  summary_.elapsed_seconds += std::max(0.0, round.elapsed_seconds);
  summary_.lp_iterations += std::max<int64_t>(0, round.lp_iterations);
  summary_.candidates_generated +=
      std::max<HighsInt>(0, round.candidates_generated);
  summary_.cuts_selected += std::max<HighsInt>(0, round.cuts_selected);
  summary_.cuts_added += std::max<HighsInt>(0, round.cuts_added);
  summary_.selected_nonzeros +=
      std::max<int64_t>(0, round.selected_nonzeros);
  for (size_t i = 0; i != summary_.generated_by_origin.size(); ++i)
    summary_.generated_by_origin[i] +=
        std::max<int64_t>(0, round.generated_by_origin[i]);
  for (size_t i = 0; i != summary_.selected_by_origin.size(); ++i)
    summary_.selected_by_origin[i] +=
        std::max<HighsInt>(0, round.selected_by_origin[i]);
  summary_.generator_model_row_uses +=
      std::max<int64_t>(0, round.generator_model_row_uses);
  summary_.generator_cut_pool_row_uses +=
      std::max<int64_t>(0, round.generator_cut_pool_row_uses);
  summary_.rejected_cut_pool_row_uses +=
      std::max<int64_t>(0, round.rejected_cut_pool_row_uses);
  const double dual_gain =
      std::max(0.0, round.objective_after - round.objective_before);
  summary_.dual_gain += dual_gain;
  summary_.fractional_reduction += std::max<HighsInt>(
      0, round.fractional_before - round.fractional_after);

  bool low_value = false;
  if (config_.min_dual_gain_per_second >= 0.0) {
    const double rate =
        dual_gain / std::max(1e-9, round.elapsed_seconds);
    low_value |= rate < config_.min_dual_gain_per_second;
  }
  if (config_.min_dual_gain_per_1000_iterations >= 0.0) {
    const double rate =
        1000.0 * dual_gain / std::max<int64_t>(1, round.lp_iterations);
    low_value |= rate < config_.min_dual_gain_per_1000_iterations;
  }

  if (low_value)
    ++summary_.consecutive_low_value_rounds;
  else
    summary_.consecutive_low_value_rounds = 0;

  if (config_.low_value_patience > 0 &&
      summary_.consecutive_low_value_rounds >= config_.low_value_patience)
    summary_.stop_reason = HighsNodeSeparationReason::kLowMarginalValue;
}

void HighsNodeSeparationController::stop(HighsNodeSeparationReason reason) {
  if (summary_.stop_reason == HighsNodeSeparationReason::kNone)
    summary_.stop_reason = reason;
}

HighsInt HighsNodeSeparationController::remainingRows() const {
  int64_t remaining = std::numeric_limits<HighsInt>::max();
  if (config_.max_added_rows >= 0)
    remaining = std::min<int64_t>(
        remaining, config_.max_added_rows - summary_.cuts_added);
  const int64_t growth_limit =
      ratioLimit(config_.max_row_growth_ratio, initial_.model_rows);
  if (growth_limit >= 0)
    remaining =
        std::min<int64_t>(remaining, growth_limit - summary_.cuts_added);
  return static_cast<HighsInt>(std::max<int64_t>(0, remaining));
}

int64_t HighsNodeSeparationController::remainingNonzeros() const {
  int64_t remaining = std::numeric_limits<int64_t>::max();
  if (config_.max_added_nonzeros >= 0)
    remaining = std::min(
        remaining, config_.max_added_nonzeros - summary_.selected_nonzeros);
  const int64_t growth_limit = ratioLimit(
      config_.max_nonzero_growth_ratio, initial_.model_nonzeros);
  if (growth_limit >= 0)
    remaining = std::min(remaining,
                         growth_limit - summary_.selected_nonzeros);
  return std::max<int64_t>(0, remaining);
}

std::array<HighsInt, kHighsCutOriginCount>
HighsNodeSeparationController::remainingCutsPerOrigin() const {
  std::array<HighsInt, kHighsCutOriginCount> remaining;
  for (size_t i = 0; i != remaining.size(); ++i) {
    if (config_.max_selected_per_origin < 0)
      remaining[i] = -1;
    else
      remaining[i] = std::max<HighsInt>(
          0, config_.max_selected_per_origin - summary_.selected_by_origin[i]);
  }
  return remaining;
}

HighsNodeCutMode HighsNodeSeparationController::modeFromInt(HighsInt value) {
  if (value < static_cast<HighsInt>(HighsNodeCutMode::kOff) ||
      value > static_cast<HighsInt>(HighsNodeCutMode::kUnlimitedLegacy))
    return HighsNodeCutMode::kAutomatic;
  return static_cast<HighsNodeCutMode>(value);
}

const char* HighsNodeSeparationController::modeName(HighsNodeCutMode mode) {
  switch (mode) {
    case HighsNodeCutMode::kOff:
      return "off";
    case HighsNodeCutMode::kAutomatic:
      return "automatic";
    case HighsNodeCutMode::kConservative:
      return "conservative";
    case HighsNodeCutMode::kModerate:
      return "moderate";
    case HighsNodeCutMode::kAggressive:
      return "aggressive";
    case HighsNodeCutMode::kUnlimitedLegacy:
      return "unlimited_legacy";
  }
  return "unknown";
}

const char* HighsNodeSeparationController::reasonName(
    HighsNodeSeparationReason reason) {
  switch (reason) {
    case HighsNodeSeparationReason::kNone:
      return "none";
    case HighsNodeSeparationReason::kControllerDisabled:
      return "controller_disabled";
    case HighsNodeSeparationReason::kModeOff:
      return "mode_off";
    case HighsNodeSeparationReason::kDepthLimit:
      return "depth_limit";
    case HighsNodeSeparationReason::kFrequencySkip:
      return "frequency_skip";
    case HighsNodeSeparationReason::kRoundLimit:
      return "round_limit";
    case HighsNodeSeparationReason::kTimeLimit:
      return "time_limit";
    case HighsNodeSeparationReason::kLpIterationLimit:
      return "lp_iteration_limit";
    case HighsNodeSeparationReason::kRowLimit:
      return "row_limit";
    case HighsNodeSeparationReason::kNonzeroLimit:
      return "nonzero_limit";
    case HighsNodeSeparationReason::kRowGrowthLimit:
      return "row_growth_limit";
    case HighsNodeSeparationReason::kNonzeroGrowthLimit:
      return "nonzero_growth_limit";
    case HighsNodeSeparationReason::kLowMarginalValue:
      return "low_marginal_value";
    case HighsNodeSeparationReason::kNoCuts:
      return "no_cuts";
    case HighsNodeSeparationReason::kLpNotOptimal:
      return "lp_not_optimal";
    case HighsNodeSeparationReason::kIntegral:
      return "integral";
    case HighsNodeSeparationReason::kObjectiveLimit:
      return "objective_limit";
  }
  return "unknown";
}
