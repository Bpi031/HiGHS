#include <atomic>
#include <limits>
#include <random>
#include <thread>

#include "Highs.h"
#include "catch.hpp"
#include "mip/HighsRootLpPortfolio.h"
#include "parallel/HighsParallel.h"

namespace {
HighsLp portfolioLp(bool maximize = false, double scale = 1) {
  HighsLp lp;
  lp.num_col_ = 2;
  lp.num_row_ = 1;
  lp.col_cost_ = {scale, 2 * scale};
  lp.col_lower_ = {0, 0};
  lp.col_upper_ = {2, 2};
  lp.row_lower_ = {1};
  lp.row_upper_ = {1};
  lp.offset_ = 7 * scale;
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_ = {0, 1, 2};
  lp.a_matrix_.index_ = {0, 0};
  lp.a_matrix_.value_ = {1, 1};
  lp.sense_ = maximize ? ObjSense::kMaximize : ObjSense::kMinimize;
  return lp;
}
}  // namespace

TEST_CASE("root capacity reservations never overcommit", "[root_portfolio]") {
  HighsSolveCapacity capacity(2, 100);
  HighsSolveCapacity::Lease a, b, c;
  REQUIRE(capacity.reserve(1, 60, a));
  REQUIRE_FALSE(capacity.reserve(1, 60, b));
  REQUIRE_FALSE(capacity.reserve(2, 20, b));
  REQUIRE_FALSE(capacity.reserve(1, 0, a));
  REQUIRE(capacity.reserve(1, 40, b));
  REQUIRE_FALSE(capacity.reserve(1, 0, c));
  a.release();
  a.release();
  REQUIRE(capacity.reserve(1, 60, c));
  b.release();
  c.release();
  REQUIRE_FALSE(capacity.reserve(0, 0, a));
  REQUIRE_FALSE(capacity.reserve(1, std::numeric_limits<size_t>::max(), a));
  std::atomic<int> acquired{0};
  std::atomic<int> attempted{0};
  auto reserve = [&]() {
    HighsSolveCapacity::Lease lease;
    if (capacity.reserve(2, 100, lease)) acquired.fetch_add(1);
    attempted.fetch_add(1);
    while (attempted.load() < 2) std::this_thread::yield();
  };
  std::thread one(reserve), two(reserve);
  one.join();
  two.join();
  REQUIRE(acquired.load() == 1);
  REQUIRE(capacity.reserve(2, 100, a));
}

TEST_CASE("root LP race installs only a checked basis", "[root_portfolio]") {
  for (int threads : {1, 2, 4}) {
    Highs::resetGlobalScheduler(true);
    highs::parallel::initialize_scheduler(threads);
    for (bool maximize : {false, true})
      for (double scale : {0.125, 1.0, 128.0}) {
        CAPTURE(threads, maximize, scale);
        Highs owner;
        owner.setOptionValue("output_flag", false);
        owner.setOptionValue("threads", threads);
        owner.setOptionValue("presolve", "off");
        const auto lp = portfolioLp(maximize, scale);
        REQUIRE(owner.passModel(lp) == HighsStatus::kOk);
        const auto before = owner.getOptions();
        HighsSolveCapacity capacity(threads, 1024 * 1024);
        const auto result =
            HighsRootLpPortfolio::offerBasis(owner, capacity, 10);
        REQUIRE(result.attempted == (threads > 1));
        REQUIRE(result.basis_installed == (threads > 1));
        REQUIRE(owner.getModelStatus() == HighsModelStatus::kNotset);
        REQUIRE(owner.getOptions().primal_feasibility_tolerance ==
                before.primal_feasibility_tolerance);
        REQUIRE(owner.getOptions().dual_feasibility_tolerance ==
                before.dual_feasibility_tolerance);
        REQUIRE(owner.run() == HighsStatus::kOk);
        REQUIRE(owner.getModelStatus() == HighsModelStatus::kOptimal);
        REQUIRE(owner.getObjectiveValue() ==
                Approx((maximize ? 9 : 8) * scale));
        REQUIRE(HighsRootLpPortfolio::checkedOptimalBasis(
            lp, owner.getSolution(), owner.getBasis(), before));
        auto bad = owner.getSolution();
        bad.col_dual[0] += 1;
        REQUIRE_FALSE(HighsRootLpPortfolio::checkedOptimalBasis(
            lp, bad, owner.getBasis(), before));
        bad = owner.getSolution();
        bad.row_value[0] = std::numeric_limits<double>::quiet_NaN();
        REQUIRE_FALSE(HighsRootLpPortfolio::checkedOptimalBasis(
            lp, bad, owner.getBasis(), before));
        auto malformed = lp;
        malformed.a_matrix_.index_[0] = lp.num_row_;
        REQUIRE_FALSE(HighsRootLpPortfolio::checkedOptimalBasis(
            malformed, owner.getSolution(), owner.getBasis(), before));
        malformed = lp;
        malformed.a_matrix_.start_.clear();
        REQUIRE_FALSE(HighsRootLpPortfolio::checkedOptimalBasis(
            malformed, owner.getSolution(), owner.getBasis(), before));
        auto no_basis = owner.getBasis();
        no_basis.col_status.assign(lp.num_col_, HighsBasisStatus::kBasic);
        no_basis.row_status.assign(lp.num_row_, HighsBasisStatus::kBasic);
        REQUIRE_FALSE(HighsRootLpPortfolio::checkedOptimalBasis(
            lp, owner.getSolution(), no_basis, before));
        REQUIRE_FALSE(
            HighsRootLpPortfolio::offerBasis(owner, capacity, 10).attempted);
      }
  }
  Highs::resetGlobalScheduler(true);
}

TEST_CASE("root LP race respects capacity deadline and no proof authority",
          "[root_portfolio]") {
  Highs::resetGlobalScheduler(true);
  highs::parallel::initialize_scheduler(2);
  for (int kind : {0, 1, 2, 3}) {
    CAPTURE(kind);
    Highs owner;
    owner.setOptionValue("output_flag", false);
    owner.setOptionValue("threads", 2);
    auto lp = portfolioLp();
    if (kind == 1) lp.row_lower_[0] = lp.row_upper_[0] = 5;  // infeasible
    if (kind == 2) {
      lp.col_cost_[0] = -1;
      lp.col_upper_[0] = kHighsInf;
      lp.row_lower_[0] = -kHighsInf;
      lp.row_upper_[0] = kHighsInf;  // unbounded
    }
    if (kind == 3)
      lp.integrality_ = {HighsVarType::kInteger, HighsVarType::kInteger};
    REQUIRE(owner.passModel(lp) == HighsStatus::kOk);
    HighsSolveCapacity empty(2, 0), capacity(2, 1024 * 1024);
    REQUIRE_FALSE(HighsRootLpPortfolio::offerBasis(owner, empty, 10).attempted);
    REQUIRE_FALSE(
        HighsRootLpPortfolio::offerBasis(owner, capacity, 0).attempted);
    REQUIRE_FALSE(
        HighsRootLpPortfolio::offerBasis(owner, capacity, -1).attempted);
    REQUIRE_FALSE(HighsRootLpPortfolio::offerBasis(
                      owner, capacity, std::numeric_limits<double>::infinity())
                      .attempted);
    const auto result = HighsRootLpPortfolio::offerBasis(owner, capacity, 10);
    REQUIRE(result.attempted == (kind != 3));
    REQUIRE(result.basis_installed == (kind == 0));
    REQUIRE(owner.getModelStatus() == HighsModelStatus::kNotset);
  }
  Highs::resetGlobalScheduler(true);
}

TEST_CASE("root portfolio MIP integration preserves exact optimum",
          "[root_portfolio]") {
  Highs::resetGlobalScheduler(true);
  for (bool enabled : {false, true}) {
    Highs mip;
    mip.setOptionValue("output_flag", false);
    mip.setOptionValue("threads", 2);
    mip.setOptionValue("presolve", "off");
    mip.setOptionValue("time_limit", 10.0);
    REQUIRE(mip.setOptionValue("mip_root_lp_portfolio", enabled) ==
            HighsStatus::kOk);
    auto lp = portfolioLp();
    lp.row_lower_ = {1.5};
    lp.row_upper_ = {kHighsInf};
    lp.integrality_ = {HighsVarType::kInteger, HighsVarType::kInteger};
    REQUIRE(mip.passModel(lp) == HighsStatus::kOk);
    REQUIRE(mip.run() == HighsStatus::kOk);
    REQUIRE(mip.getModelStatus() == HighsModelStatus::kOptimal);
    REQUIRE(mip.getObjectiveValue() == Approx(9));
  }
  Highs::resetGlobalScheduler(true);
}

TEST_CASE("root portfolio signed rows match an analytic optimum",
          "[root_portfolio]") {
  Highs::resetGlobalScheduler(true);
  highs::parallel::initialize_scheduler(2);
  std::mt19937 random(581);
  for (int sample = 0; sample < 100; ++sample) {
    HighsLp lp;
    lp.num_col_ = lp.num_row_ = 8;
    lp.a_matrix_.format_ = MatrixFormat::kColwise;
    lp.a_matrix_.start_ = {0};
    lp.offset_ = sample - 50;
    lp.sense_ = sample % 2 ? ObjSense::kMinimize : ObjSense::kMaximize;
    double expected = lp.offset_;
    for (int j = 0; j < 8; ++j) {
      const double cost = int(random() % 19) - 9;
      const double a = (random() % 2 ? -1 : 1) * (1 << (random() % 3));
      lp.col_cost_.push_back(cost);
      lp.col_lower_.push_back(-10);
      lp.col_upper_.push_back(10);
      lp.row_lower_.push_back(std::min(-2 * a, 3 * a));
      lp.row_upper_.push_back(std::max(-2 * a, 3 * a));
      lp.a_matrix_.index_.push_back(j);
      lp.a_matrix_.value_.push_back(a);
      lp.a_matrix_.start_.push_back(j + 1);
      expected += cost * (static_cast<int>(lp.sense_) * cost >= 0 ? -2 : 3);
    }
    Highs owner;
    owner.setOptionValue("output_flag", false);
    owner.setOptionValue("threads", 2);
    owner.setOptionValue("presolve", "off");
    REQUIRE(owner.passModel(lp) == HighsStatus::kOk);
    HighsSolveCapacity capacity(2, 1024 * 1024);
    const auto result = HighsRootLpPortfolio::offerBasis(owner, capacity, 10);
    CAPTURE(sample, expected, result.attempted, result.winner,
            static_cast<int>(result.status[0]),
            static_cast<int>(result.status[1]));
    REQUIRE(owner.run() == HighsStatus::kOk);
    REQUIRE(owner.getModelStatus() == HighsModelStatus::kOptimal);
    CAPTURE(owner.getSolution().col_value, owner.getSolution().col_dual,
            owner.getSolution().row_value, owner.getSolution().row_dual,
            lp.col_cost_, lp.row_lower_, lp.row_upper_, lp.a_matrix_.value_);
    REQUIRE(owner.getObjectiveValue() == Approx(expected));
    REQUIRE(result.basis_installed);
  }
  Highs::resetGlobalScheduler(true);
}

TEST_CASE("root portfolio polls external cancellation only on its owner",
          "[root_portfolio]") {
  Highs::resetGlobalScheduler(true);
  highs::parallel::initialize_scheduler(4);
  const auto owner_thread = std::this_thread::get_id();
  for (bool stop : {false, true}) {
    Highs owner;
    owner.setOptionValue("output_flag", false);
    owner.setOptionValue("threads", 4);
    REQUIRE(owner.passModel(portfolioLp()) == HighsStatus::kOk);
    HighsSolveCapacity capacity(4, 1024 * 1024);
    int polls = 0;
    bool foreign_thread = false;
    auto result = HighsRootLpPortfolio::offerBasis(owner, capacity, 10, [&]() {
      foreign_thread |= std::this_thread::get_id() != owner_thread;
      ++polls;
      return stop && polls >= 2;
    });
    REQUIRE(polls >= 2);
    REQUIRE_FALSE(foreign_thread);
    REQUIRE(result.cancelled == stop);
    REQUIRE(result.basis_installed == !stop);
    REQUIRE(owner.getModelStatus() == HighsModelStatus::kNotset);
    HighsSolveCapacity::Lease remaining;
    REQUIRE(capacity.reserve(4, 1024 * 1024, remaining));
  }
  Highs::resetGlobalScheduler(true);
}

TEST_CASE(
    "callback bearing MIP activates the root race without concurrent dispatch",
    "[root_portfolio]") {
  Highs::resetGlobalScheduler(true);
  Highs mip;
  mip.setOptionValue("output_flag", true);
  mip.setOptionValue("log_to_console", false);
  mip.setOptionValue("threads", 4);
  mip.setOptionValue("presolve", "off");
  mip.setOptionValue("time_limit", 10.0);
  mip.setOptionValue("mip_root_lp_portfolio", true);
  mip.setOptionValue("mip_heuristic_run_feasibility_jump", false);
  const auto owner_thread = std::this_thread::get_id();
  bool observed = false;
  bool foreign_thread = false;
  mip.setCallback(
      [&](int kind, const std::string& message, const HighsCallbackOutput*,
          HighsCallbackInput*, void*) {
        if (kind == kCallbackMipInterrupt)
          foreign_thread |= std::this_thread::get_id() != owner_thread;
        if (kind == kCallbackLogging &&
            message.find(
                "MIP-RootPortfolio: version=1 attempted=1 installed=1") !=
                std::string::npos)
          observed = true;
      },
      nullptr);
  mip.startCallback(kCallbackMipInterrupt);
  mip.startCallback(kCallbackLogging);
  auto lp = portfolioLp();
  lp.row_lower_ = {1.5};
  lp.row_upper_ = {kHighsInf};
  lp.integrality_ = {HighsVarType::kInteger, HighsVarType::kInteger};
  REQUIRE(mip.passModel(lp) == HighsStatus::kOk);
  REQUIRE(mip.run() == HighsStatus::kOk);
  REQUIRE(mip.getModelStatus() == HighsModelStatus::kOptimal);
  REQUIRE(mip.getObjectiveValue() == Approx(9));
  REQUIRE(observed);
  REQUIRE_FALSE(foreign_thread);
  Highs::resetGlobalScheduler(true);
}

TEST_CASE("root portfolio retains serial behavior for deterministic scheduling",
          "[root_portfolio]") {
  Highs::resetGlobalScheduler(true);
  Highs mip;
  mip.setOptionValue("output_flag", true);
  mip.setOptionValue("log_to_console", false);
  mip.setOptionValue("threads", 4);
  mip.setOptionValue("presolve", "off");
  mip.setOptionValue("time_limit", 10.0);
  mip.setOptionValue("mip_root_lp_portfolio", true);
  mip.setOptionValue("mip_root_primal_helper", true);
  mip.setOptionValue("mip_root_cut_exchange", true);
  mip.setOptionValue("mip_flexible_node_dispatch", true);
  mip.setOptionValue("mip_incumbent_exchange", "deterministic");
  bool observed = false;
  mip.setCallback(
      [&](int kind, const std::string& message, const HighsCallbackOutput*,
          HighsCallbackInput*, void*) {
        if (kind == kCallbackLogging &&
            (message.find("MIP-RootPortfolio:") != std::string::npos ||
             message.find("MIP-RootPrimal:") != std::string::npos ||
             message.find("MIP-RootCutExchange:") != std::string::npos ||
             message.find("MIP-Dispatch:") != std::string::npos))
          observed = true;
      },
      nullptr);
  mip.startCallback(kCallbackLogging);
  auto lp = portfolioLp();
  lp.row_lower_ = {1.5};
  lp.row_upper_ = {kHighsInf};
  lp.integrality_ = {HighsVarType::kInteger, HighsVarType::kInteger};
  REQUIRE(mip.passModel(lp) == HighsStatus::kOk);
  REQUIRE(mip.run() == HighsStatus::kOk);
  REQUIRE(mip.getModelStatus() == HighsModelStatus::kOptimal);
  REQUIRE(mip.getObjectiveValue() == Approx(9));
  REQUIRE_FALSE(observed);
  Highs::resetGlobalScheduler(true);
}
