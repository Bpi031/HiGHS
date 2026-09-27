#include <limits>
#include <vector>

#include "HCheckConfig.h"
#include "Highs.h"
#include "catch.hpp"
#include "mip/HighsNodeSeparation.h"

namespace {
HighsNodeSeparationContext context(HighsInt depth = 1,
                                   int64_t opportunity_sequence = 0) {
  HighsNodeSeparationContext value;
  value.depth = depth;
  value.opportunity_sequence = opportunity_sequence;
  value.model_rows = 1000;
  value.model_nonzeros = 10000;
  value.active_rows = 1000;
  value.active_nonzeros = 10000;
  value.time = 10.0;
  value.lp_iterations = 100;
  value.objective = 20.0;
  value.fractional_integers = 10;
  return value;
}
}  // namespace

TEST_CASE("node-separation-disabled-is-nonintervening",
          "[highs_node_separation]") {
  HighsNodeSeparationController controller;
  controller.beginNode(HighsNodeSeparationConfig{}, context());
  const HighsNodeSeparationDecision decision =
      controller.beforeRound(context());
  REQUIRE(decision.continue_separation);
  REQUIRE(decision.reason ==
          HighsNodeSeparationReason::kControllerDisabled);
  REQUIRE_FALSE(controller.summary().controller_enabled);
}

TEST_CASE("node-separation-profiles-expand-and-remain-bounded",
          "[highs_node_separation]") {
  for (HighsInt raw_mode = 2; raw_mode <= 4; ++raw_mode) {
    HighsNodeSeparationConfig config;
    config.enabled = true;
    config.mode = HighsNodeSeparationController::modeFromInt(raw_mode);
    HighsNodeSeparationController controller;
    controller.beginNode(config, context());
    REQUIRE(controller.eligible());
    REQUIRE(controller.config().max_rounds > 0);
    REQUIRE(controller.config().max_seconds > 0.0);
    REQUIRE(controller.config().max_lp_iterations > 0);
    REQUIRE(controller.config().max_added_rows > 0);
    REQUIRE(controller.config().max_added_nonzeros > 0);
    REQUIRE(controller.config().max_selected_per_origin > 0);
    REQUIRE(controller.config().max_row_growth_ratio > 0.0);
    REQUIRE(controller.config().max_nonzero_growth_ratio > 0.0);
  }
}

TEST_CASE("node-separation-depth-and-frequency-are-exact",
          "[highs_node_separation]") {
  HighsNodeSeparationConfig config;
  config.enabled = true;
  config.mode = HighsNodeCutMode::kModerate;
  config.maximum_depth = 3;
  config.node_frequency = 4;

  HighsNodeSeparationController depth_controller;
  depth_controller.beginNode(config, context(4, 8));
  REQUIRE_FALSE(depth_controller.eligible());
  REQUIRE(depth_controller.summary().stop_reason ==
          HighsNodeSeparationReason::kDepthLimit);

  HighsNodeSeparationController frequency_controller;
  frequency_controller.beginNode(config, context(3, 9));
  REQUIRE_FALSE(frequency_controller.eligible());
  REQUIRE(frequency_controller.summary().stop_reason ==
          HighsNodeSeparationReason::kFrequencySkip);

  HighsNodeSeparationController eligible_controller;
  eligible_controller.beginNode(config, context(3, 12));
  REQUIRE(eligible_controller.eligible());
}

TEST_CASE("node-separation-limits-are-independent",
          "[highs_node_separation]") {
  HighsNodeSeparationConfig config;
  config.enabled = true;
  config.mode = HighsNodeCutMode::kUnlimitedLegacy;
  config.max_rounds = 1;
  config.max_seconds = 2.0;
  config.max_lp_iterations = 100;
  config.max_added_rows = 20;
  config.max_added_nonzeros = 200;
  config.max_selected_per_origin = 3;
  config.max_row_growth_ratio = 0.05;
  config.max_nonzero_growth_ratio = 0.05;

  HighsNodeSeparationController controller;
  controller.beginNode(config, context());
  REQUIRE(controller.remainingRows() == 20);
  REQUIRE(controller.remainingNonzeros() == 200);

  HighsNodeSeparationRoundResult round;
  round.cuts_added = 20;
  round.cuts_selected = 20;
  round.selected_nonzeros = 200;
  round.generated_by_origin[static_cast<size_t>(HighsCutOrigin::kTableau)] = 5;
  round.selected_by_origin[static_cast<size_t>(HighsCutOrigin::kTableau)] = 3;
  round.generator_model_row_uses = 12;
  round.generator_cut_pool_row_uses = 4;
  round.lp_iterations = 50;
  round.elapsed_seconds = 1.0;
  round.objective_before = 20.0;
  round.objective_after = 21.0;
  controller.recordRound(round);

  REQUIRE(controller.remainingRows() == 0);
  REQUIRE(controller.remainingNonzeros() == 0);
  REQUIRE(controller.remainingCutsPerOrigin()[static_cast<size_t>(
              HighsCutOrigin::kTableau)] == 0);
  REQUIRE(controller.summary().generator_model_row_uses == 12);
  REQUIRE(controller.summary().generator_cut_pool_row_uses == 4);
  const HighsNodeSeparationDecision decision =
      controller.beforeRound(context());
  REQUIRE_FALSE(decision.continue_separation);
  REQUIRE(decision.reason == HighsNodeSeparationReason::kRoundLimit);
}

TEST_CASE("node-separation-cut-origin-names-are-stable",
          "[highs_node_separation]") {
  REQUIRE(std::string(HighsCutPool::originName(HighsCutOrigin::kImplication)) ==
          "implication");
  REQUIRE(std::string(HighsCutPool::originName(HighsCutOrigin::kClique)) ==
          "clique");
  REQUIRE(std::string(HighsCutPool::originName(HighsCutOrigin::kTableau)) ==
          "tableau");
  REQUIRE(std::string(HighsCutPool::originName(HighsCutOrigin::kPath)) ==
          "path");
  REQUIRE(std::string(HighsCutPool::originName(HighsCutOrigin::kModK)) ==
          "modk");
}

TEST_CASE("node-separation-low-value-stop-is-phase-local",
          "[highs_node_separation]") {
  HighsNodeSeparationConfig config;
  config.enabled = true;
  config.mode = HighsNodeCutMode::kUnlimitedLegacy;
  config.low_value_patience = 2;
  config.min_dual_gain_per_1000_iterations = 1.0;
  HighsNodeSeparationController controller;
  controller.beginNode(config, context());

  HighsNodeSeparationRoundResult round;
  round.lp_iterations = 1000;
  round.elapsed_seconds = 1.0;
  round.objective_before = 100.0;
  round.objective_after = 100.5;
  controller.recordRound(round);
  REQUIRE(controller.summary().stop_reason ==
          HighsNodeSeparationReason::kNone);
  controller.recordRound(round);
  REQUIRE(controller.summary().stop_reason ==
          HighsNodeSeparationReason::kLowMarginalValue);
}

TEST_CASE("node-separation-unlimited-controller-preserves-search",
          "[highs_node_separation]") {
  const std::string filename =
      std::string(HIGHS_DIR) + "/check/instances/rgn.mps";

  struct Evidence {
    HighsModelStatus status;
    double objective;
    double bound;
    int64_t nodes;
    HighsInt iterations;
    std::vector<double> solution;
  };

  auto solve = [&](bool controller) {
    Highs highs;
    // The native test executable shares HiGHS's global scheduler across test
    // cases. Reset before configuring this deterministic identity run so a
    // preceding parallel test cannot leak its scheduler state into the arm.
    highs.resetGlobalScheduler(true);
    REQUIRE(highs.setOptionValue("output_flag", false) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("threads", 1) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("random_seed", 42) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_rel_gap", 0.0) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_abs_gap", 0.0) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_max_nodes", 25) == HighsStatus::kOk);
    REQUIRE(highs.readModel(filename) == HighsStatus::kOk);
    if (controller) {
      REQUIRE(highs.setOptionValue("mip_node_separation_controller", true) ==
              HighsStatus::kOk);
      REQUIRE(highs.setOptionValue("mip_node_cut_mode", 5) ==
              HighsStatus::kOk);
    }
    REQUIRE(highs.run() != HighsStatus::kError);
    const HighsInfo& info = highs.getInfo();
    Evidence evidence{highs.getModelStatus(),
                      info.objective_function_value,
                      info.mip_dual_bound,
                      info.mip_node_count,
                      info.simplex_iteration_count,
                      highs.getSolution().col_value};
    highs.resetGlobalScheduler(true);
    return evidence;
  };

  const Evidence baseline = solve(false);
  const Evidence controlled = solve(true);
  REQUIRE(controlled.status == baseline.status);
  REQUIRE(controlled.objective == baseline.objective);
  REQUIRE(controlled.bound == baseline.bound);
  REQUIRE(controlled.nodes == baseline.nodes);
  REQUIRE(controlled.iterations == baseline.iterations);
  REQUIRE(controlled.solution == baseline.solution);
}

TEST_CASE("node-separation-base-row-gate-keeps-a-valid-search",
          "[highs_node_separation]") {
  Highs highs;
  highs.resetGlobalScheduler(true);
  REQUIRE(highs.setOptionValue("output_flag", false) == HighsStatus::kOk);
  REQUIRE(highs.setOptionValue("threads", 1) == HighsStatus::kOk);
  REQUIRE(highs.setOptionValue("random_seed", 42) == HighsStatus::kOk);
  REQUIRE(highs.setOptionValue("mip_max_nodes", 25) == HighsStatus::kOk);
  REQUIRE(highs.setOptionValue("mip_node_separation_controller", true) ==
          HighsStatus::kOk);
  REQUIRE(highs.setOptionValue("mip_node_cut_mode", 3) == HighsStatus::kOk);
  REQUIRE(highs.setOptionValue("mip_node_cut_base_model_rows_only", true) ==
          HighsStatus::kOk);
  REQUIRE(highs.readModel(std::string(HIGHS_DIR) + "/check/instances/rgn.mps") ==
          HighsStatus::kOk);
  REQUIRE(highs.run() != HighsStatus::kError);
  REQUIRE(highs.getInfo().primal_solution_status == kSolutionStatusFeasible);
  highs.resetGlobalScheduler(true);
}
