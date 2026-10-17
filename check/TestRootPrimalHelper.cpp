#include <limits>
#include <future>
#include <thread>

#include "catch.hpp"
#include "mip/HighsRootPrimalHelper.h"

namespace {
HighsLp helperModel() {
  HighsLp lp;
  lp.num_col_ = 5;
  lp.num_row_ = 5;
  lp.col_cost_.assign(5, 1);
  lp.col_lower_.assign(5, 0);
  lp.col_upper_.assign(5, 1);
  lp.integrality_.assign(5, HighsVarType::kInteger);
  lp.row_lower_.assign(5, 1);
  lp.row_upper_.assign(5, kHighsInf);
  lp.a_matrix_.start_ = {0};
  for (int j = 0; j != 5; ++j) {
    lp.a_matrix_.index_.push_back(j);
    lp.a_matrix_.index_.push_back((j + 4) % 5);
    lp.a_matrix_.value_.insert(lp.a_matrix_.value_.end(), {1, 1});
    lp.a_matrix_.start_.push_back(lp.a_matrix_.index_.size());
  }
  return lp;
}
}  // namespace

TEST_CASE("root primal helper is candidate-only and joins before release",
          "[root_primal_helper]") {
  for (int threads : {2, 4}) {
    Highs::resetGlobalScheduler(true);
    highs::parallel::initialize_scheduler(threads);
    Highs owner;
    owner.setOptionValue("output_flag", false);
    owner.setOptionValue("threads", threads);
    owner.setOptionValue("presolve", "off");
    const auto lp = helperModel();
    REQUIRE(owner.passModel(lp) == HighsStatus::kOk);
    const auto options = owner.getOptions();
    for (bool seeded : {false, true}) {
      HighsSolveCapacity capacity(threads, 1024 * 1024);
      HighsRootPrimalHelper helper;
      REQUIRE(helper.start(
          lp, seeded ? std::vector<double>(5, 1) : std::vector<double>{},
          options, capacity, 1.0));
      REQUIRE(helper.seeded() == seeded);
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (!helper.finished() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
      helper.join();
      helper.join();
      REQUIRE_FALSE(helper.candidate().empty());
      const auto check = highsVerifyCandidate(
          lp, helper.candidate(), nullptr, options.mip_feasibility_tolerance);
      REQUIRE(check.feasible);
      double cost = 0;
      for (double value : helper.candidate()) cost += value;
      REQUIRE(cost >= 3 - 1e-6);  // exhaustive odd-cycle minimum
      REQUIRE(cost <= 5 + 1e-6);
      REQUIRE(helper.weightedAttempted());
      REQUIRE(helper.weightedSeconds() >= 0);
      REQUIRE(helper.weightedWork() <= helper.work());
      REQUIRE(helper.feasibilitySeconds() + helper.completionSeconds() +
              helper.weightedSeconds() <= helper.elapsed());
      if (seeded) REQUIRE(helper.feasibilitySeconds() == 0);
      REQUIRE(owner.getModelStatus() == HighsModelStatus::kNotset);
      REQUIRE(owner.getOptions().mip_feasibility_tolerance ==
              options.mip_feasibility_tolerance);
      HighsSolveCapacity::Lease all;
      REQUIRE(capacity.reserve(threads, 1024 * 1024, all));
    }
  }
}

TEST_CASE("root helper declines exhausted budget and cancels safely",
          "[root_primal_helper]") {
  Highs::resetGlobalScheduler(true);
  highs::parallel::initialize_scheduler(2);
  Highs owner;
  owner.setOptionValue("output_flag", false);
  owner.setOptionValue("threads", 2);
  const auto lp = helperModel();
  HighsSolveCapacity empty(2, 0), capacity(2, 1024 * 1024);
  HighsRootPrimalHelper helper;
  REQUIRE_FALSE(helper.start(lp, {}, owner.getOptions(), empty, 1));
  REQUIRE_FALSE(helper.start(lp, {}, owner.getOptions(), capacity, 0));
  REQUIRE_FALSE(helper.start(lp, {}, owner.getOptions(), capacity,
                             std::numeric_limits<double>::infinity()));
  REQUIRE(helper.start(lp, {1}, owner.getOptions(), capacity, 1));
  REQUIRE_FALSE(helper.seeded());
  helper.join();
  REQUIRE(owner.getModelStatus() == HighsModelStatus::kNotset);
  REQUIRE_FALSE(helper.start(lp, {}, owner.getOptions(), capacity, 1));
}

TEST_CASE("cold root helper seeks feasibility and leaves owner objective intact",
          "[root_primal_helper][root_cold_start]") {
  for (const ObjSense sense : {ObjSense::kMinimize, ObjSense::kMaximize}) {
    Highs::resetGlobalScheduler(true);
    highs::parallel::initialize_scheduler(2);
    Highs owner;
    owner.setOptionValue("output_flag", false);
    owner.setOptionValue("threads", 2);
    owner.setOptionValue("presolve", "off");
    auto lp = helperModel();
    lp.sense_ = sense;
    lp.offset_ = 123;
    REQUIRE(owner.passModel(lp) == HighsStatus::kOk);
    HighsSolveCapacity capacity(2, 1024 * 1024);
    HighsRootPrimalHelper helper;
    REQUIRE(helper.start(lp, {}, owner.getOptions(), capacity, 2));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!helper.finished() && std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    helper.join();
    REQUIRE(helper.feasibilityMode());
    REQUIRE(highsVerifyCandidate(lp, helper.candidate(), nullptr, 1e-6).feasible);
    // The bounded weighted follow-up may change integer choices, but cannot
    // mutate the owner or reuse private proof bounds.
    REQUIRE(helper.weightedAttempted());
    const auto checked = highsVerifyCandidate(lp, helper.candidate(), nullptr, 1e-6);
    REQUIRE(checked.objective == Approx(123 + (sense == ObjSense::kMinimize ? 3 : 5)));
    REQUIRE(owner.getLp().col_cost_ == lp.col_cost_);
    REQUIRE(owner.getLp().sense_ == sense);
    REQUIRE(owner.getLp().offset_ == 123);
    REQUIRE(owner.getModelStatus() == HighsModelStatus::kNotset);
  }
}

TEST_CASE("cancelled queued root helper retains a checked starting seed",
          "[root_primal_helper][root_cold_start]") {
  Highs::resetGlobalScheduler(true);
  highs::parallel::initialize_scheduler(2);
  Highs owner;
  owner.setOptionValue("output_flag", false);
  owner.setOptionValue("threads", 2);
  auto lp = helperModel();
  std::promise<void> started, release;
  auto ready = started.get_future();
  auto released = release.get_future();
  highs::parallel::TaskGroup blocker;
  blocker.spawn([&]() { started.set_value(); released.wait(); });
  ready.wait();
  HighsSolveCapacity capacity(2, 1024 * 1024);
  HighsRootPrimalHelper helper;
  const std::vector<double> seed(5, 1);
  const bool launched = helper.start(lp, seed, owner.getOptions(), capacity, 2);
  helper.join();
  release.set_value();
  blocker.taskWait();
  REQUIRE(launched);
  REQUIRE(helper.seeded());
  REQUIRE_FALSE(helper.feasibilityMode());
  REQUIRE(helper.candidate() == seed);
  REQUIRE(helper.work() == 0);
  REQUIRE_FALSE(helper.weightedAttempted());
  REQUIRE(helper.weightedAllowance() == 0);
}

TEST_CASE("queued root helper cannot renew an expired allowance",
          "[root_primal_helper][root_cold_start]") {
  for (bool seeded : {false, true}) {
    Highs::resetGlobalScheduler(true);
    highs::parallel::initialize_scheduler(2);
    Highs owner;
    owner.setOptionValue("output_flag", false);
    owner.setOptionValue("threads", 2);
    const auto lp = helperModel();
    std::promise<void> started, release;
    auto ready = started.get_future();
    auto released = release.get_future();
    highs::parallel::TaskGroup blocker;
    blocker.spawn([&]() { started.set_value(); released.wait(); });
    ready.wait();
    HighsSolveCapacity capacity(2, 1024 * 1024);
    HighsRootPrimalHelper helper;
    const std::vector<double> seed = seeded ? std::vector<double>(5, 1)
                                           : std::vector<double>{};
    const auto before = std::chrono::steady_clock::now();
    const bool launched = helper.start(lp, seed, owner.getOptions(), capacity, 0.5);
    std::this_thread::sleep_until(before + std::chrono::seconds(1));
    release.set_value();
    blocker.taskWait();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (launched && !helper.finished() && std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    const bool finished = helper.finished();
    helper.join();
    REQUIRE(launched);
    REQUIRE(finished);
    REQUIRE(helper.candidate() == seed);
    REQUIRE(helper.elapsed() >= 0.5);
    REQUIRE(helper.work() == 0);
    REQUIRE(helper.feasibilitySeconds() == 0);
    REQUIRE(helper.completionSeconds() == 0);
    REQUIRE(helper.weightedSeconds() == 0);
    REQUIRE(helper.weightedAllowance() == 0);
    REQUIRE_FALSE(helper.weightedAttempted());
    REQUIRE(owner.getModelStatus() == HighsModelStatus::kNotset);
    HighsSolveCapacity::Lease all;
    REQUIRE(capacity.reserve(2, 1024 * 1024, all));
  }
}

TEST_CASE("cold root helper completes loose auxiliary values at fixed integer choices",
          "[root_primal_helper][root_cold_completion]") {
  for (const ObjSense sense : {ObjSense::kMinimize, ObjSense::kMaximize}) {
    for (const HighsVarType auxiliary : {HighsVarType::kContinuous,
                                       HighsVarType::kImplicitInteger}) {
      Highs::resetGlobalScheduler(true);
      highs::parallel::initialize_scheduler(2);
      Highs owner;
      owner.setOptionValue("output_flag", false);
      owner.setOptionValue("threads", 2);
      owner.setOptionValue("presolve", "off");
      HighsLp lp;
      lp.num_col_ = 2;
      lp.num_row_ = 2;
      lp.sense_ = sense;
      lp.offset_ = 123;
      lp.col_cost_ = {0, sense == ObjSense::kMinimize ? 1.0 : -1.0};
      lp.col_lower_ = {0, 0};
      lp.col_upper_ = {1, 100};
      lp.integrality_ = {HighsVarType::kInteger, auxiliary};
      lp.row_lower_ = {1, 0};
      lp.row_upper_ = {1, kHighsInf};
      lp.a_matrix_.start_ = {0, 2, 3};
      lp.a_matrix_.index_ = {0, 1, 1};
      lp.a_matrix_.value_ = {1, -1, 1};
      REQUIRE(owner.passModel(lp) == HighsStatus::kOk);
      HighsSolveCapacity capacity(2, 1024 * 1024);
      HighsRootPrimalHelper helper;
      REQUIRE(helper.start(lp, {}, owner.getOptions(), capacity, 2));
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
      while (!helper.finished() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
      helper.join();
      REQUIRE(highsVerifyCandidate(lp, helper.candidate(), nullptr, 1e-6).feasible);
      REQUIRE(helper.candidate()[0] == 1);
      REQUIRE(helper.candidate()[1] == Approx(1));
      REQUIRE(helper.weightedAttempted());
      REQUIRE(owner.getLp().col_cost_ == lp.col_cost_);
      REQUIRE(owner.getModelStatus() == HighsModelStatus::kNotset);
    }
  }
}
