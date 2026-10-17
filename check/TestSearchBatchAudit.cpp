#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "HCheckConfig.h"
#include "Highs.h"
#include "catch.hpp"
#include "mip/HighsSearchBatchAudit.h"

namespace {
HighsSearchBatchRecord sample(int64_t issued = 4) {
  HighsSearchBatchRecord result;
  result.capacity = result.allocated = 4;
  result.threads = 4;
  result.issued = issued;
  result.started_s = 100;
  result.joined_s = 110;
  result.reconciled_s = 113;
  return result;
}
}  // namespace

TEST_CASE("joined-batch-tail-is-not-worker-CPU-time", "[highs_batch_audit]") {
  auto batch = sample();
  batch.observeTask(101, 103);
  batch.observeTask(101, 104);
  batch.observeTask(102, 105);
  batch.observeTask(100, 109);
  REQUIRE(batch.complete());
  REQUIRE(batch.task_interval_s == 17);
  REQUIRE(batch.spanSeconds() == 10);
  REQUIRE(batch.tailSeconds() == 4);
  REQUIRE(batch.reconciliationSeconds() == 3);
  HighsSearchBatchAudit audit;
  audit.record(batch);
  REQUIRE(audit.batches() == 1);
  REQUIRE(audit.valid() == 1);
  REQUIRE(audit.taskIntervalSeconds() == 17);
  REQUIRE(audit.spanSeconds() == 10);
  REQUIRE(audit.tailSeconds() == 4);
  REQUIRE(audit.reconciliationSeconds() == 3);
  auto serial = sample(1);
  serial.observeTask(100, 109);
  REQUIRE(serial.complete());
  REQUIRE(serial.tailSeconds() == 0);
}

TEST_CASE("joined-batch-missing-and-invalid-observations-stay-unavailable",
          "[highs_batch_audit]") {
  HighsSearchBatchAudit audit;
  auto absent = sample();
  REQUIRE_FALSE(absent.complete());
  audit.record(absent);
  const std::array<double, 5> bad{{-1, 99, 111,
      (std::numeric_limits<double>::infinity)(),
      std::numeric_limits<double>::quiet_NaN()}};
  for (double time : bad) {
    auto batch = sample(1);
    batch.observeTask(time, 109);
    REQUIRE_FALSE(batch.complete());
    audit.record(batch);
    auto returned = sample(1);
    returned.observeTask(100, time);
    REQUIRE_FALSE(returned.complete());
    audit.record(returned);
  }
  auto overflow = sample(1);
  overflow.tasks_observed = (std::numeric_limits<int64_t>::max)();
  overflow.observeTask(100, 109);
  REQUIRE_FALSE(overflow.valid);
  REQUIRE(overflow.tasks_observed == (std::numeric_limits<int64_t>::max)());
  REQUIRE(audit.invalid() == 11);
  REQUIRE(audit.valid() == 0);
  REQUIRE(audit.retained() == 0);
  REQUIRE(audit.taskIntervalSeconds() == 0);
  REQUIRE_THROWS_AS(audit.at(0), std::out_of_range);
  auto huge = sample(1);
  huge.started_s = 0;
  huge.joined_s = huge.reconciled_s = (std::numeric_limits<double>::max)();
  huge.observeTask(0, huge.joined_s);
  REQUIRE(huge.complete());
  audit.record(huge);
  audit.record(huge);
  REQUIRE(std::isfinite(audit.spanSeconds()));
  REQUIRE(audit.spanSeconds() == huge.joined_s);
  REQUIRE(audit.taskIntervalSeconds() == huge.joined_s);
}

TEST_CASE("joined-batch-independent-interval-oracle-and-clock-metamorphisms",
          "[highs_batch_audit]") {
  uint64_t state = 42;
  for (int trial = 0; trial < 4096; ++trial) {
    std::array<double, 4> starts{}, ends{};
    double area = 0;
    for (size_t i = 0; i < starts.size(); ++i) {
      state = state * 6364136223846793005ULL + 1;
      starts[i] = 100 + double((state >> 24) & 15) / 8;
      ends[i] = starts[i] + double((state >> 32) & 31) / 8;
      area += ends[i] - starts[i];
    }
    auto sorted = ends;
    std::sort(sorted.begin(), sorted.end());
    auto batch = sample(), transformed = sample();
    transformed.started_s = 2 * batch.started_s + 128;
    transformed.joined_s = 2 * batch.joined_s + 128;
    transformed.reconciled_s = 2 * batch.reconciled_s + 128;
    for (size_t i = 0; i < starts.size(); ++i) {
      const size_t index = (i + trial) % starts.size();
      batch.observeTask(starts[index], ends[index]);
      transformed.observeTask(2 * starts[index] + 128, 2 * ends[index] + 128);
    }
    REQUIRE(batch.complete());
    REQUIRE(transformed.complete());
    REQUIRE(batch.task_interval_s == area);
    REQUIRE(batch.tailSeconds() == sorted[3] - sorted[2]);
    REQUIRE(transformed.task_interval_s == 2 * area);
    REQUIRE(transformed.tailSeconds() == 2 * batch.tailSeconds());
    REQUIRE(transformed.spanSeconds() == 2 * batch.spanSeconds());
    REQUIRE(transformed.reconciliationSeconds() == 2 * batch.reconciliationSeconds());
  }
}

TEST_CASE("joined-batch-retention-keeps-prefix-and-large-later-tails",
          "[highs_batch_audit]") {
  HighsSearchBatchAudit audit;
  double expected_tail = 0;
  for (int epoch = 0; epoch < 256; ++epoch) {
    auto batch = sample(2);
    batch.epoch = epoch;
    const double tail = double(epoch) / 256;
    batch.observeTask(100, 108);
    batch.observeTask(100, 108 + tail);
    expected_tail += tail;
    audit.record(batch);
  }
  REQUIRE(audit.retained() == 128);
  REQUIRE(audit.batches() == 256);
  REQUIRE(audit.valid() == 256);
  REQUIRE(audit.dropped() == 128);
  REQUIRE(audit.tailSeconds() == expected_tail);
  REQUIRE(audit.maxTailSeconds() == 255.0 / 256);
  for (size_t i = 0; i < 64; ++i) REQUIRE(audit.at(i).epoch == int64_t(i));
  for (size_t i = 64; i < 128; ++i) REQUIRE(audit.at(i).epoch >= 192);
}

TEST_CASE("native-thread-capacity-is-not-limited-to-four",
          "[highs_batch_audit][highs_batch_integration]") {
  struct Result { double objective, bound; int64_t nodes, iterations; };
  auto solve = [&](HighsInt threads, HighsInt report_level, const char* exchange) {
    Highs highs;
    highs.resetGlobalScheduler(true);
    REQUIRE(highs.setOptionValue("log_to_console", false) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("threads", threads) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("parallel", "on") == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_incumbent_exchange", exchange) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_report_level", report_level) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_allow_restart", false) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_rel_gap", 0.0) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_abs_gap", 0.0) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("time_limit", 15.0) == HighsStatus::kOk);
    bool unexpected_batch_observation = false;
    highs.setCallback([&](int kind, const std::string& message,
                          const HighsCallbackOutput*, HighsCallbackInput*, void*) {
      unexpected_batch_observation |= kind == kCallbackLogging &&
          message.find("MIP-SearchBatch:") != std::string::npos;
    });
    REQUIRE(highs.startCallback(kCallbackLogging) == HighsStatus::kOk);
    REQUIRE(highs.readModel(std::string(HIGHS_DIR) + "/check/instances/bell5.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
    // The measured observer was rejected; retain only the independent ledger.
    REQUIRE_FALSE(unexpected_batch_observation);
    const auto& lp = highs.getLp();
    const auto& solution = highs.getSolution();
    REQUIRE(solution.value_valid);
    REQUIRE(solution.col_value.size() == size_t(lp.num_col_));
    auto matrix = lp.a_matrix_;
    matrix.ensureColwise();
    std::vector<long double> activity(lp.num_row_, 0);
    long double direct_objective = lp.offset_;
    for (HighsInt col = 0; col < lp.num_col_; ++col) {
      const double value = solution.col_value[col];
      REQUIRE(std::isfinite(value));
      REQUIRE(value >= lp.col_lower_[col] - 1e-6);
      REQUIRE(value <= lp.col_upper_[col] + 1e-6);
      if (!lp.integrality_.empty() && lp.integrality_[col] != HighsVarType::kContinuous)
        REQUIRE(std::abs(value - std::round(value)) <= 1e-6);
      direct_objective += static_cast<long double>(lp.col_cost_[col]) * value;
      for (HighsInt k = matrix.start_[col]; k < matrix.start_[col + 1]; ++k)
        activity[matrix.index_[k]] += static_cast<long double>(matrix.value_[k]) * value;
    }
    for (HighsInt row = 0; row < lp.num_row_; ++row) {
      REQUIRE(activity[row] >= lp.row_lower_[row] - 1e-6);
      REQUIRE(activity[row] <= lp.row_upper_[row] + 1e-6);
    }
    const auto& info = highs.getInfo();
    REQUIRE(std::abs(direct_objective - info.objective_function_value) <= 1e-6);
    const Result result{info.objective_function_value, info.mip_dual_bound,
                        info.mip_node_count, info.simplex_iteration_count};
    highs.resetGlobalScheduler(true);
    return result;
  };
  for (const char* exchange : {"off", "deterministic"}) {
    for (HighsInt threads : {1, 2, 4, 8, 10}) {
      INFO("threads=" << threads << " exchange=" << exchange);
      const auto before = solve(threads, 1, exchange),
                 after = solve(threads, 2, exchange);
      REQUIRE(after.objective == before.objective);
      REQUIRE(after.bound == before.bound);
      REQUIRE(after.nodes == before.nodes);
      REQUIRE(after.iterations == before.iterations);
    }
  }
}
