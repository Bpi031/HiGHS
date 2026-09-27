/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef HIGHS_NODE_SEPARATION_H_
#define HIGHS_NODE_SEPARATION_H_

#include <array>
#include <cstdint>

#include "mip/HighsCutPool.h"
#include "util/HighsInt.h"

enum class HighsNodeCutMode : HighsInt {
  kOff = 0,
  kAutomatic = 1,
  kConservative = 2,
  kModerate = 3,
  kAggressive = 4,
  kUnlimitedLegacy = 5,
};

enum class HighsNodeSeparationReason : uint8_t {
  kNone,
  kControllerDisabled,
  kModeOff,
  kDepthLimit,
  kFrequencySkip,
  kRoundLimit,
  kTimeLimit,
  kLpIterationLimit,
  kRowLimit,
  kNonzeroLimit,
  kRowGrowthLimit,
  kNonzeroGrowthLimit,
  kLowMarginalValue,
  kNoCuts,
  kLpNotOptimal,
  kIntegral,
  kObjectiveLimit,
};

struct HighsNodeSeparationConfig {
  bool enabled = false;
  HighsNodeCutMode mode = HighsNodeCutMode::kAutomatic;
  HighsInt maximum_depth = -1;
  HighsInt node_frequency = -1;
  HighsInt max_rounds = -1;
  double max_seconds = -1.0;
  int64_t max_lp_iterations = -1;
  HighsInt max_added_rows = -1;
  int64_t max_added_nonzeros = -1;
  HighsInt max_selected_per_origin = -1;
  double max_row_growth_ratio = -1.0;
  double max_nonzero_growth_ratio = -1.0;
  HighsInt low_value_patience = -1;
  double min_dual_gain_per_second = -1.0;
  double min_dual_gain_per_1000_iterations = -1.0;
  bool base_model_rows_only = false;
};

struct HighsNodeSeparationContext {
  int64_t opportunity_sequence = 0;
  int64_t tree_nodes_processed = 0;
  HighsInt depth = 0;
  HighsInt model_rows = 0;
  int64_t model_nonzeros = 0;
  HighsInt active_rows = 0;
  int64_t active_nonzeros = 0;
  int64_t lp_iterations = 0;
  double time = 0.0;
  double objective = 0.0;
  HighsInt fractional_integers = 0;
};

struct HighsNodeSeparationRoundResult {
  HighsInt bound_changes = 0;
  HighsInt candidates_generated = 0;
  HighsInt cuts_selected = 0;
  HighsInt cuts_added = 0;
  int64_t selected_nonzeros = 0;
  std::array<int64_t, kHighsCutOriginCount> generated_by_origin{};
  std::array<HighsInt, kHighsCutOriginCount> selected_by_origin{};
  int64_t generator_model_row_uses = 0;
  int64_t generator_cut_pool_row_uses = 0;
  int64_t rejected_cut_pool_row_uses = 0;
  int64_t lp_iterations = 0;
  double elapsed_seconds = 0.0;
  double objective_before = 0.0;
  double objective_after = 0.0;
  HighsInt fractional_before = 0;
  HighsInt fractional_after = 0;
  HighsInt active_rows_before = 0;
  HighsInt active_rows_after = 0;
  int64_t active_nonzeros_before = 0;
  int64_t active_nonzeros_after = 0;
  bool lp_optimal = true;
  bool node_infeasible = false;
};

struct HighsNodeSeparationSummary {
  bool controller_enabled = false;
  bool eligible = true;
  HighsNodeCutMode requested_mode = HighsNodeCutMode::kAutomatic;
  HighsNodeCutMode effective_mode = HighsNodeCutMode::kAutomatic;
  HighsNodeSeparationReason stop_reason = HighsNodeSeparationReason::kNone;
  int64_t opportunity_sequence = 0;
  int64_t tree_nodes_processed = 0;
  HighsInt depth = 0;
  HighsInt rounds = 0;
  double elapsed_seconds = 0.0;
  int64_t lp_iterations = 0;
  HighsInt candidates_generated = 0;
  HighsInt cuts_selected = 0;
  HighsInt cuts_added = 0;
  int64_t selected_nonzeros = 0;
  std::array<int64_t, kHighsCutOriginCount> generated_by_origin{};
  std::array<HighsInt, kHighsCutOriginCount> selected_by_origin{};
  int64_t generator_model_row_uses = 0;
  int64_t generator_cut_pool_row_uses = 0;
  int64_t rejected_cut_pool_row_uses = 0;
  double dual_gain = 0.0;
  HighsInt fractional_reduction = 0;
  HighsInt consecutive_low_value_rounds = 0;
};

struct HighsNodeSeparationDecision {
  bool continue_separation = true;
  HighsNodeSeparationReason reason = HighsNodeSeparationReason::kNone;

  HighsNodeSeparationDecision() {}
  HighsNodeSeparationDecision(bool continue_separation,
                              HighsNodeSeparationReason reason)
      : continue_separation(continue_separation), reason(reason) {}
};

class HighsNodeSeparationController {
 public:
  void beginNode(const HighsNodeSeparationConfig& config,
                 const HighsNodeSeparationContext& context);

  HighsNodeSeparationDecision beforeRound(
      const HighsNodeSeparationContext& current) const;

  void recordRound(const HighsNodeSeparationRoundResult& round);
  void stop(HighsNodeSeparationReason reason);

  HighsInt remainingRows() const;
  int64_t remainingNonzeros() const;
  std::array<HighsInt, kHighsCutOriginCount> remainingCutsPerOrigin() const;

  bool enabled() const { return config_.enabled; }
  bool eligible() const { return summary_.eligible; }
  const HighsNodeSeparationConfig& config() const { return config_; }
  const HighsNodeSeparationSummary& summary() const { return summary_; }

  static HighsNodeCutMode modeFromInt(HighsInt value);
  static const char* modeName(HighsNodeCutMode mode);
  static const char* reasonName(HighsNodeSeparationReason reason);

 private:
  static void applyProfileDefaults(HighsNodeSeparationConfig& config,
                                   HighsNodeCutMode mode);
  HighsNodeSeparationDecision limitedByGrowth() const;

  HighsNodeSeparationConfig config_;
  HighsNodeSeparationContext initial_;
  HighsNodeSeparationSummary summary_;
};

#endif
