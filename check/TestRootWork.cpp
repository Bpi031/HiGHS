#include <limits>

#include "HCheckConfig.h"
#include "Highs.h"
#include "catch.hpp"
#include "mip/HighsRootWork.h"

TEST_CASE("root-work-disabled", "[highs_root_work]") {
  HighsRootWorkController controller;
  controller.beginStage(HighsRootWorkConfig{}, 10.0, 20.0);
  controller.beginEpoch();
  controller.recordCompleted(HighsRootWorkPhase::kSeparation, 50.0, true);

  const HighsRootWorkDecision decision =
      controller.beforeOptional(HighsRootWorkPhase::kSeparation, 100.0);
  REQUIRE(decision.action == HighsRootWorkAction::kContinue);
  REQUIRE(decision.reason == HighsRootWorkReason::kDisabled);
  REQUIRE(controller.snapshot().separation_rounds == 0);
}

TEST_CASE("root-work-cumulative-across-restarts", "[highs_root_work]") {
  HighsRootWorkConfig config;
  config.enabled = true;
  config.max_separation_rounds = 2;
  HighsRootWorkController controller;
  controller.beginStage(config, 0.0, 100.0);

  controller.beginEpoch();
  controller.recordCompleted(HighsRootWorkPhase::kSeparation, 1.0, true);
  controller.beginEpoch();
  controller.recordCompleted(HighsRootWorkPhase::kSeparation, 2.0, true);

  const HighsRootWorkDecision decision =
      controller.beforeOptional(HighsRootWorkPhase::kSeparation, 4.0);
  REQUIRE(controller.snapshot().epoch == 1);
  REQUIRE(controller.snapshot().separation_rounds == 2);
  REQUIRE(controller.snapshot().separation_time == 3.0);
  REQUIRE(decision.action == HighsRootWorkAction::kYieldToTree);
  REQUIRE(decision.reason == HighsRootWorkReason::kSeparationRoundLimit);
}

TEST_CASE("root-work-heuristic-budget-is-local-decision", "[highs_root_work]") {
  HighsRootWorkConfig config;
  config.enabled = true;
  config.max_heuristic_time = 5.0;
  HighsRootWorkController controller;
  controller.beginStage(config, 0.0, 100.0);
  controller.beginEpoch();
  controller.recordCompleted(HighsRootWorkPhase::kRens, 6.0);

  const HighsRootWorkDecision helper =
      controller.beforeOptional(HighsRootWorkPhase::kRens, 7.0);
  const HighsRootWorkDecision separation =
      controller.beforeOptional(HighsRootWorkPhase::kSeparation, 7.0);
  REQUIRE(helper.action == HighsRootWorkAction::kSkipOptional);
  REQUIRE(helper.reason == HighsRootWorkReason::kHeuristicTimeLimit);
  REQUIRE(separation.action == HighsRootWorkAction::kContinue);
}

TEST_CASE("root-work-tree-reserve-and-global-limit", "[highs_root_work]") {
  HighsRootWorkConfig config;
  config.enabled = true;
  config.tree_reserve_time = 10.0;
  HighsRootWorkController controller;
  controller.beginStage(config, 0.0, 100.0);
  controller.beginEpoch();

  HighsRootWorkDecision decision =
      controller.beforeOptional(HighsRootWorkPhase::kRens, 90.0);
  REQUIRE(decision.action == HighsRootWorkAction::kYieldToTree);
  REQUIRE(decision.reason == HighsRootWorkReason::kTreeReserve);

  decision = controller.beforeOptional(HighsRootWorkPhase::kRens, 100.0);
  REQUIRE(decision.action == HighsRootWorkAction::kStopGlobal);
  REQUIRE(decision.reason == HighsRootWorkReason::kGlobalLimit);
}

TEST_CASE("root-work-unlimited-and-nonfinite-accounting", "[highs_root_work]") {
  HighsRootWorkConfig config;
  config.enabled = true;
  HighsRootWorkController controller;
  controller.beginStage(config, 0.0, std::numeric_limits<double>::infinity());
  controller.beginEpoch();
  controller.recordCompleted(HighsRootWorkPhase::kSeparation,
                             std::numeric_limits<double>::infinity(), true);
  controller.recordCompleted(HighsRootWorkPhase::kRens, -5.0);

  REQUIRE(controller.snapshot().separation_time == 0.0);
  REQUIRE(controller.snapshot().heuristic_time == 0.0);
  REQUIRE(controller.beforeOptional(HighsRootWorkPhase::kSeparation, 1e100)
              .action == HighsRootWorkAction::kContinue);
}

TEST_CASE("root-work-helper-allowance-composes-remaining-budgets",
          "[highs_root_work]") {
  HighsRootWorkConfig config;
  config.enabled = true;
  config.max_heuristic_time = 12.0;
  config.max_single_helper_time = 7.0;
  config.tree_reserve_time = 10.0;
  HighsRootWorkController controller;
  controller.beginStage(config, 0.0, 100.0);
  controller.beginEpoch();

  REQUIRE(controller.optionalAllowance(HighsRootWorkPhase::kRens, 20.0) == 7.0);
  controller.recordCompleted(HighsRootWorkPhase::kRens, 9.0);
  REQUIRE(controller.optionalAllowance(HighsRootWorkPhase::kRens, 20.0) == 3.0);
  REQUIRE(controller.optionalAllowance(HighsRootWorkPhase::kAnalyticCenter,
                                       85.0) == 5.0);
  REQUIRE(controller.optionalAllowance(HighsRootWorkPhase::kRens, 90.0) == 0.0);
}

TEST_CASE("root-work-zero-budgets-yield-or-skip-without-global-stop",
          "[highs_root_work]") {
  HighsRootWorkConfig config;
  config.enabled = true;
  config.max_separation_rounds = 0;
  config.max_separation_time = 0.0;
  config.max_heuristic_time = 0.0;
  config.max_single_helper_time = 0.0;
  HighsRootWorkController controller;
  controller.beginStage(config, 0.0, 100.0);
  controller.beginEpoch();

  const HighsRootWorkDecision separation =
      controller.beforeOptional(HighsRootWorkPhase::kSeparation, 1.0);
  const HighsRootWorkDecision heuristic =
      controller.beforeOptional(HighsRootWorkPhase::kRens, 1.0);
  REQUIRE(separation.action == HighsRootWorkAction::kYieldToTree);
  REQUIRE(separation.reason == HighsRootWorkReason::kSeparationRoundLimit);
  REQUIRE(heuristic.action == HighsRootWorkAction::kSkipOptional);
  REQUIRE(heuristic.reason == HighsRootWorkReason::kHeuristicTimeLimit);
  REQUIRE(controller.optionalAllowance(HighsRootWorkPhase::kRens, 1.0) == 0.0);
}

TEST_CASE("root-work-new-stage-resets-cumulative-and-tree-state",
          "[highs_root_work]") {
  HighsRootWorkConfig config;
  config.enabled = true;
  config.max_separation_rounds = 1;
  HighsRootWorkController controller;
  controller.beginStage(config, 0.0, 100.0);
  controller.beginEpoch();
  controller.recordCompleted(HighsRootWorkPhase::kSeparation, 3.0, true);
  controller.recordTreeEntry();

  REQUIRE(controller.snapshot().separation_rounds == 1);
  REQUIRE(controller.snapshot().tree_entered);

  controller.beginStage(config, 20.0, 120.0);
  controller.beginEpoch();
  REQUIRE(controller.snapshot().epoch == 0);
  REQUIRE(controller.snapshot().separation_rounds == 0);
  REQUIRE(controller.snapshot().separation_time == 0.0);
  REQUIRE_FALSE(controller.snapshot().tree_entered);
  REQUIRE(
      controller.beforeOptional(HighsRootWorkPhase::kSeparation, 21.0).action ==
      HighsRootWorkAction::kContinue);
}

TEST_CASE("root-work-active-path-preserves-tiny-mip-optimum",
          "[highs_root_work]") {
  const std::string filename =
      std::string(HIGHS_DIR) + "/check/instances/rgn.mps";

  auto solve = [&](bool enabled) {
    Highs highs;
    REQUIRE(highs.setOptionValue("output_flag", false) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("threads", 1) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_rel_gap", 0.0) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_abs_gap", 0.0) == HighsStatus::kOk);
    REQUIRE(highs.readModel(filename) == HighsStatus::kOk);
    if (enabled) {
      REQUIRE(highs.setOptionValue("mip_root_work_budget", true) ==
              HighsStatus::kOk);
      REQUIRE(highs.setOptionValue("mip_root_work_max_separation_rounds", 0) ==
              HighsStatus::kOk);
      REQUIRE(highs.setOptionValue("mip_root_work_max_separation_time", 0.0) ==
              HighsStatus::kOk);
      REQUIRE(highs.setOptionValue("mip_root_work_max_heuristic_time", 0.0) ==
              HighsStatus::kOk);
      REQUIRE(highs.setOptionValue("mip_root_work_max_single_helper_time",
                                   0.0) == HighsStatus::kOk);
      REQUIRE(highs.setOptionValue("mip_root_work_tree_reserve_time", -1.0) ==
              HighsStatus::kOk);
    }
    REQUIRE(highs.run() == HighsStatus::kOk);
    const HighsModelStatus status = highs.getModelStatus();
    const double objective = highs.getInfo().objective_function_value;
    highs.resetGlobalScheduler(true);
    return std::make_pair(status, objective);
  };

  const auto upstream = solve(false);
  const auto budgeted = solve(true);
  REQUIRE(upstream.first == HighsModelStatus::kOptimal);
  REQUIRE(budgeted.first == HighsModelStatus::kOptimal);
  REQUIRE(budgeted.second == Approx(upstream.second).margin(1e-7));
}
