#include <limits>
#include <vector>

#include "HCheckConfig.h"
#include "Highs.h"
#include "catch.hpp"
#include "lp_data/HighsOptions.h"
#include "mip/HighsPrimalHeuristics.h"
#include "mip/HighsRootWork.h"

TEST_CASE("sub-MIP provenance names are exact", "[highs_incumbent]") {
  REQUIRE(std::string(HighsPrimalHeuristics::subMipMethodName(
              HighsSubMipOrigin::kRootReducedCost)) == "root_reduced_cost");
  REQUIRE(std::string(HighsPrimalHeuristics::subMipMethodName(
              HighsSubMipOrigin::kRootRens)) == "rens");
  REQUIRE(std::string(HighsPrimalHeuristics::subMipMethodName(
              HighsSubMipOrigin::kTreeRins)) == "rins");
  REQUIRE(std::string(HighsPrimalHeuristics::subMipParentPhaseName(
              HighsSubMipOrigin::kRootRens)) == "root");
  REQUIRE(std::string(HighsPrimalHeuristics::subMipParentPhaseName(
              HighsSubMipOrigin::kTreeRens)) == "tree");
}

TEST_CASE("bounded root helper sub-MIPs cannot recurse",
          "[highs_incumbent]") {
  HighsOptions options;
  options.mip_heuristic_run_rens = true;
  options.mip_heuristic_run_rins = true;
  options.mip_heuristic_run_root_reduced_cost = true;

  HighsPrimalHeuristics::configureSubMipOptions(
      options, HighsSubMipOrigin::kRootRens);

  REQUIRE_FALSE(options.mip_heuristic_run_rens);
  REQUIRE_FALSE(options.mip_heuristic_run_rins);
  REQUIRE_FALSE(options.mip_heuristic_run_root_reduced_cost);
}

TEST_CASE("tree heuristic sub-MIPs preserve the native portfolio",
          "[highs_incumbent]") {
  HighsOptions options;
  options.mip_heuristic_run_rens = true;
  options.mip_heuristic_run_rins = true;
  options.mip_heuristic_run_root_reduced_cost = true;

  HighsPrimalHeuristics::configureSubMipOptions(
      options, HighsSubMipOrigin::kTreeRins);

  REQUIRE(options.mip_heuristic_run_rens);
  REQUIRE(options.mip_heuristic_run_rins);
  REQUIRE(options.mip_heuristic_run_root_reduced_cost);
}

TEST_CASE("heuristic retries share one deadline", "[highs_incumbent]") {
  REQUIRE(HighsPrimalHeuristics::remainingHeuristicTime(10.0, 12.0, 7.0) ==
          Approx(5.0));
  REQUIRE(HighsPrimalHeuristics::remainingHeuristicTime(10.0, 20.0, 7.0) ==
          Approx(0.0));
  REQUIRE(HighsPrimalHeuristics::remainingHeuristicTime(
              10.0, 20.0, std::numeric_limits<double>::infinity()) ==
          kHighsInf);
}

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
  REQUIRE(decision.action == HighsRootWorkAction::kStopSeparation);
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
  REQUIRE(separation.action == HighsRootWorkAction::kStopSeparation);
  REQUIRE(separation.reason == HighsRootWorkReason::kSeparationRoundLimit);
  REQUIRE(heuristic.action == HighsRootWorkAction::kSkipOptional);
  REQUIRE(heuristic.reason == HighsRootWorkReason::kHeuristicTimeLimit);
  REQUIRE(controller.optionalAllowance(HighsRootWorkPhase::kRens, 1.0) == 0.0);
}

TEST_CASE("root-work-low-marginal-separation-is-phase-local",
          "[highs_root_work]") {
  HighsRootWorkConfig config;
  config.enabled = true;
  HighsRootWorkController controller;
  controller.beginStage(config, 0.0, 100.0);
  controller.beginEpoch();

  auto record = [&](double start, double finish, double before_bound,
                    double after_bound) {
    HighsRootWorkObservation before;
    before.time = start;
    before.lp_iterations = static_cast<int64_t>(start * 1000.0);
    before.incumbent = 1000.0;
    before.dual_bound = before_bound;
    before.active_lp_rows = 100;
    before.active_lp_nonzeros = 1000;
    HighsRootWorkObservation after = before;
    after.time = finish;
    after.lp_iterations += 10000;
    after.dual_bound = after_bound;
    after.active_lp_rows += 100;
    after.active_lp_nonzeros += 1000;
    controller.recordCompleted(HighsRootWorkPhase::kSeparation, before, after,
                               10, true);
  };

  record(0.0, 1.0, 100.0, 110.0);
  record(1.0, 2.0, 110.0, 120.0);
  record(2.0, 3.0, 120.0, 120.0);
  REQUIRE_FALSE(controller.snapshot().separation_stopped);
  record(3.0, 4.0, 120.0, 120.0);
  REQUIRE(controller.snapshot().separation_stopped);

  const HighsRootWorkDecision separation =
      controller.beforeOptional(HighsRootWorkPhase::kSeparation, 5.0);
  const HighsRootWorkDecision reduced_cost =
      controller.beforeOptional(HighsRootWorkPhase::kReducedCostHeuristic, 5.0);
  const HighsRootWorkDecision completion =
      controller.beforeOptional(HighsRootWorkPhase::kIncumbentCompletion, 5.0);
  REQUIRE(separation.action == HighsRootWorkAction::kStopSeparation);
  REQUIRE(separation.reason ==
          HighsRootWorkReason::kSeparationLowMarginalValue);
  REQUIRE(reduced_cost.action == HighsRootWorkAction::kContinue);
  REQUIRE(completion.action == HighsRootWorkAction::kContinue);
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

TEST_CASE("root-work-records-separate-activity-evidence", "[highs_root_work]") {
  HighsRootWorkConfig config;
  config.enabled = true;
  HighsRootWorkController controller;
  controller.beginStage(config, 0.0, 100.0);
  controller.beginEpoch();

  HighsRootWorkObservation before;
  before.time = 1.0;
  before.lp_iterations = 10;
  before.improving_solutions = 1;
  before.incumbent = 100.0;
  before.dual_bound = 20.0;
  before.active_lp_rows = 10;
  before.active_lp_nonzeros = 30;
  before.cut_pool_rows = 2;
  before.fractional_integers = 5;

  HighsRootWorkObservation after;
  after.time = 3.0;
  after.lp_iterations = 25;
  after.improving_solutions = 2;
  after.incumbent = 90.0;
  after.dual_bound = 26.0;
  after.active_lp_rows = 12;
  after.active_lp_nonzeros = 40;
  after.cut_pool_rows = 3;
  after.fractional_integers = 4;

  controller.recordCompleted(HighsRootWorkPhase::kSeparation, before, after, 3,
                             true);

  const HighsRootWorkActivityAccount& account =
      controller.activity(HighsRootWorkPhase::kSeparation);
  REQUIRE(account.calls == 1);
  REQUIRE(account.successes == 1);
  REQUIRE(account.wall_time == 2.0);
  REQUIRE(account.lp_iterations == 15);
  REQUIRE(account.accepted_incumbents == 1);
  REQUIRE(account.primal_gain == 10.0);
  REQUIRE(account.dual_gain == 6.0);
  REQUIRE(account.cuts_generated == 3);
  REQUIRE(account.cut_pool_rows_added == 1);
  REQUIRE(account.lp_rows_added == 2);
  REQUIRE(account.lp_nonzeros_added == 10);
  REQUIRE(account.last_fractional_integers == 4);
  REQUIRE(controller.snapshot().separation_rounds == 1);
  REQUIRE(controller.snapshot().separation_time == 2.0);
}

TEST_CASE("root-work-unlimited-telemetry-preserves-deterministic-search",
          "[highs_root_work]") {
  const std::string filename =
      std::string(HIGHS_DIR) + "/check/instances/rgn.mps";

  struct SolveEvidence {
    HighsModelStatus status;
    double objective;
    int64_t nodes;
    HighsInt simplex_iterations;
    std::vector<double> solution;
  };

  auto solve = [&](bool telemetry) {
    Highs highs;
    REQUIRE(highs.setOptionValue("output_flag", false) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("threads", 1) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("random_seed", 42) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_rel_gap", 0.0) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_abs_gap", 0.0) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_max_nodes", 1) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_root_max_separation_rounds", 2) ==
            HighsStatus::kOk);
    REQUIRE(highs.readModel(filename) == HighsStatus::kOk);
    if (telemetry)
      REQUIRE(highs.setOptionValue("mip_root_work_budget", true) ==
              HighsStatus::kOk);
    // A deterministic node limit is expected to return warning/limited status;
    // only a native error invalidates the identity experiment.
    REQUIRE(highs.run() != HighsStatus::kError);
    const HighsInfo& info = highs.getInfo();
    SolveEvidence evidence{highs.getModelStatus(),
                           info.objective_function_value, info.mip_node_count,
                           info.simplex_iteration_count,
                           highs.getSolution().col_value};
    highs.resetGlobalScheduler(true);
    return evidence;
  };

  const SolveEvidence baseline = solve(false);
  const SolveEvidence telemetry = solve(true);
  REQUIRE(telemetry.status == baseline.status);
  REQUIRE(telemetry.objective == baseline.objective);
  REQUIRE(telemetry.nodes == baseline.nodes);
  REQUIRE(telemetry.simplex_iterations == baseline.simplex_iterations);
  REQUIRE(telemetry.solution == baseline.solution);
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
