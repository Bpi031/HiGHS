#include <cmath>
#include <limits>
#include <memory>

#include "HCheckConfig.h"
#include "Highs.h"
#include "catch.hpp"
#include "mip/HighsNodeLpEffort.h"
#include "mip/HighsNodeQueue.h"
#include "mip/HighsSearch.h"

TEST_CASE("node-lp-effort-applies-only-to-authoritative-tree-search",
          "[highs_node_lp_effort]") {
  REQUIRE(HighsSearch::boundedNodeLpEffortApplies(false, true));
  REQUIRE_FALSE(HighsSearch::boundedNodeLpEffortApplies(true, true));
  REQUIRE_FALSE(HighsSearch::boundedNodeLpEffortApplies(false, false));
}

TEST_CASE("node-lp-effort-resume-basis-requires-complete-dimensions",
          "[highs_node_lp_effort]") {
  HighsBasis basis;
  basis.valid = true;
  basis.row_status.resize(2, HighsBasisStatus::kBasic);
  basis.col_status.resize(3, HighsBasisStatus::kLower);
  REQUIRE(HighsSearch::resumableBasisMatches(basis, 2, 3));
  REQUIRE_FALSE(HighsSearch::resumableBasisMatches(basis, 1, 3));
  REQUIRE_FALSE(HighsSearch::resumableBasisMatches(basis, 2, 2));
  basis.valid = false;
  REQUIRE_FALSE(HighsSearch::resumableBasisMatches(basis, 2, 3));
}

TEST_CASE("node-lp-effort-disabled-is-an-identity",
          "[highs_node_lp_effort]") {
  HighsNodeLpEffortController controller;
  REQUIRE(controller.iterationLimit(1000000, 1, 0) == kHighsIInf);
}

TEST_CASE("node-lp-effort-root-seed-is-bounded",
          "[highs_node_lp_effort]") {
  HighsNodeLpEffortConfig config;
  config.enabled = true;
  config.minimum_iterations = 10000;
  config.maximum_iterations = 250000;
  config.quantile_multiplier = 2.0;
  HighsNodeLpEffortController controller;
  controller.configure(config);

  REQUIRE(controller.iterationLimit(10000000, 20, 0) == 250000);
  REQUIRE(controller.iterationLimit(40000, 20, 0) == 20000);
}

TEST_CASE("node-lp-effort-learns-and-retries-deterministically",
          "[highs_node_lp_effort]") {
  HighsNodeLpEffortConfig config;
  config.enabled = true;
  config.minimum_iterations = 1000;
  config.maximum_iterations = 100000;
  config.quantile_multiplier = 2.0;
  config.retry_multiplier = 2;
  config.maximum_yields = 3;
  HighsNodeLpEffortController controller;
  controller.configure(config);

  for (HighsInt iterations : {1000, 1500, 2000, 3000}) {
    HighsNodeLpEffortObservation observation;
    observation.depth = 4;
    observation.iteration_count = iterations;
    controller.recordCompleted(observation);
  }

  const HighsInt first = controller.iterationLimit(1000000, 20, 0);
  const HighsInt retry = controller.iterationLimit(1000000, 20, 1);
  REQUIRE(first == 4000);
  REQUIRE(retry == 8000);
  REQUIRE(controller.iterationLimit(1000000, 20, 3) == kHighsIInf);
  REQUIRE(controller.percentileIterations(0.75) == 2048);
}

TEST_CASE("node-lp-effort-telemetry-is-bounded",
          "[highs_node_lp_effort]") {
  HighsNodeLpEffortController controller;
  HighsNodeLpEffortConfig config;
  config.enabled = true;
  controller.configure(config);
  controller.recordAttempt();
  controller.recordYield(12000);
  controller.recordAttempt();
  HighsNodeLpEffortObservation observation;
  observation.iteration_count = 8000;
  controller.recordCompleted(observation);
  controller.recordSerialCompletion();
  controller.recordTaskLease(1.5);
  controller.recordTaskLease(0.5);
  controller.recordTaskLease(1100.0);
  controller.recordTaskLease(2200.0);
  controller.recordTaskLease(-1.0);

  const HighsNodeLpEffortSummary& summary = controller.summary();
  REQUIRE(summary.attempts == 2);
  REQUIRE(summary.yielded == 1);
  REQUIRE(summary.completed == 1);
  REQUIRE(summary.serial_completions == 1);
  REQUIRE(summary.total_iterations == 20000);
  REQUIRE(summary.completed_iterations == 8000);
  REQUIRE(summary.yielded_iterations == 12000);
  REQUIRE(summary.maximum_iterations == 12000);
  REQUIRE(summary.histogram[12] == 1);
  REQUIRE(summary.yielded_histogram[13] == 1);
  REQUIRE(summary.task_leases == 4);
  REQUIRE(summary.task_lease_total_seconds == 3302.0);
  REQUIRE(summary.task_lease_max_seconds == 2200.0);
  REQUIRE(summary.task_lease_microseconds_histogram[18] == 1);
  REQUIRE(summary.task_lease_microseconds_histogram[20] == 1);
  REQUIRE(summary.task_lease_microseconds_histogram[30] == 1);
  REQUIRE(summary.task_lease_microseconds_histogram[31] == 1);
  const double median_upper =
      HighsNodeLpEffortController::taskLeasePercentileUpperSeconds(
          summary.task_lease_microseconds_histogram, 0.50);
  const double p95_upper =
      HighsNodeLpEffortController::taskLeasePercentileUpperSeconds(
          summary.task_lease_microseconds_histogram, 0.95);
  REQUIRE(median_upper >= 1.5);
  REQUIRE(median_upper < 2.1);
  REQUIRE(p95_upper >= 2200.0);
  REQUIRE(std::isfinite(p95_upper));
}

TEST_CASE("node-lp-effort-learns-only-from-recent-completions",
          "[highs_node_lp_effort]") {
  HighsNodeLpEffortConfig config;
  config.enabled = true;
  config.minimum_iterations = 1;
  config.maximum_iterations = 1000000;
  config.quantile_multiplier = 1.0;
  HighsNodeLpEffortController controller;
  controller.configure(config);

  for (HighsInt iterations = 100; iterations <= 800; iterations += 100) {
    HighsNodeLpEffortObservation observation;
    observation.depth = 40;
    observation.iteration_count = iterations;
    controller.recordCompleted(observation);
  }
  controller.recordYield(100000);
  HighsNodeLpEffortObservation failure;
  failure.depth = 40;
  failure.iteration_count = 200000;
  failure.outcome = HighsNodeLpEffortOutcome::kNumericalFailure;
  controller.recordCompleted(failure);
  HighsNodeLpEffortObservation limit;
  limit.depth = 40;
  limit.iteration_count = 50000;
  limit.outcome = HighsNodeLpEffortOutcome::kLimitReached;
  controller.recordCompleted(limit);

  REQUIRE(controller.iterationLimit(1000000, 40, 0) == 600);
  REQUIRE(controller.summary().completed == 8);
  REQUIRE(controller.summary().yielded == 1);
  REQUIRE(controller.summary().numerical_failures == 1);
  REQUIRE(controller.summary().limit_stops == 1);
  REQUIRE(controller.summary().total_iterations == 353600);
  REQUIRE(controller.summary().completed_iterations == 3600);
  REQUIRE(controller.summary().yielded_iterations == 100000);
}

TEST_CASE("node-lp-effort-small-depth-window-seeds-from-parent",
          "[highs_node_lp_effort]") {
  HighsNodeLpEffortConfig config;
  config.enabled = true;
  config.minimum_iterations = 1;
  config.maximum_iterations = 1000000;
  config.quantile_multiplier = 1.0;
  HighsNodeLpEffortController controller;
  controller.configure(config);

  for (HighsInt iterations : {1000, 2000, 3000, 4000}) {
    HighsNodeLpEffortObservation observation;
    observation.depth = 20;
    observation.iteration_count = iterations;
    controller.recordCompleted(observation);
  }
  for (HighsInt iterations : {10000, 20000}) {
    HighsNodeLpEffortObservation observation;
    observation.depth = 60;
    observation.iteration_count = iterations;
    controller.recordCompleted(observation);
  }

  REQUIRE(controller.iterationLimit(1000000, 60, 0) == 3000);
}

TEST_CASE("yielded-node-resume-envelope-preserves-proof-state",
          "[highs_node_lp_effort]") {
  HighsNodeQueue queue;
  queue.setNumCol(0);
  auto basis = std::make_shared<HighsBasis>();

  queue.emplaceNode({}, {}, 17.0, 19.0, 4, basis, 2);
  HighsNodeQueue::OpenNode node = std::move(queue.popBestNode());

  REQUIRE(node.lower_bound == 17.0);
  REQUIRE(node.estimate == 19.0);
  REQUIRE(node.depth == 4);
  REQUIRE(node.basis == basis);
  REQUIRE(node.yield_count == 2);
  REQUIRE(queue.empty());
}

TEST_CASE("forced-node-lp-yields-preserve-terminal-proof",
          "[highs_node_lp_effort]") {
  struct Outcome {
    HighsModelStatus status;
    double objective;
    double dual_bound;
  };
  const std::string filename =
      std::string(HIGHS_DIR) + "/check/instances/bell5.mps";

  auto solve = [&](bool bounded) {
    Highs highs;
    highs.resetGlobalScheduler(true);
    REQUIRE(highs.setOptionValue("output_flag", false) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("threads", 1) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("parallel", kHighsOffString) ==
            HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_rel_gap", 0.0) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_abs_gap", 0.0) == HighsStatus::kOk);
    if (bounded) {
      REQUIRE(highs.setOptionValue("mip_node_lp_effort", true) ==
              HighsStatus::kOk);
      REQUIRE(highs.setOptionValue("mip_node_lp_effort_min_iterations", 1) ==
              HighsStatus::kOk);
      REQUIRE(highs.setOptionValue("mip_node_lp_effort_max_iterations", 1) ==
              HighsStatus::kOk);
      REQUIRE(highs.setOptionValue("mip_node_lp_effort_max_yields", 3) ==
              HighsStatus::kOk);
    }
    REQUIRE(highs.readModel(filename) == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    const HighsInfo& info = highs.getInfo();
    return Outcome{highs.getModelStatus(), info.objective_function_value,
                   info.mip_dual_bound};
  };

  const Outcome baseline = solve(false);
  const Outcome yielded = solve(true);
  REQUIRE(baseline.status == HighsModelStatus::kOptimal);
  REQUIRE(yielded.status == baseline.status);
  REQUIRE(std::abs(yielded.objective - baseline.objective) <= 1e-6);
  REQUIRE(std::abs(yielded.dual_bound - baseline.dual_bound) <= 1e-6);
}
