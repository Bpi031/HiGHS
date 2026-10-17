// #include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <type_traits>

#include "HCheckConfig.h"
#include "Highs.h"
#include "mip/HighsCandidateVerifier.h"
#include "catch.hpp"

TEST_CASE("pure-candidate-verifier-fails-closed-without-mutating-model",
          "[rost_callback_refinement]") {
  HighsLp lp;
  lp.num_col_ = 2; lp.num_row_ = 1; lp.offset_ = 7;
  lp.col_cost_ = {-3, 2}; lp.col_lower_ = {0, 0}; lp.col_upper_ = {1, 4};
  lp.integrality_ = {HighsVarType::kInteger, HighsVarType::kContinuous};
  lp.row_lower_ = {1}; lp.row_upper_ = {3};
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_ = {0, 1, 2}; lp.a_matrix_.index_ = {0, 0};
  lp.a_matrix_.value_ = {1, 1};
  const std::vector<double> valid{1, 0.5};
  const auto checked = highsVerifyCandidate(lp, valid, nullptr, 1e-6);
  REQUIRE(checked.feasible);
  REQUIRE(static_cast<double>(checked.objective) == 5);
  REQUIRE(lp.offset_ == 7);
  REQUIRE((lp.col_cost_ == std::vector<double>{-3, 2}));
  for (const auto& columns : std::vector<std::vector<double>>{
      {}, {1}, {1, 0, 0}, {0.5, 1}, {2, 0}, {0, 0},
      {1, std::numeric_limits<double>::quiet_NaN()}, {1, kHighsInf}})
    REQUIRE_FALSE(highsVerifyCandidate(lp, columns, nullptr, 1e-6).feasible);
  for (const auto& rows : std::vector<std::vector<double>>{
      {}, {1, 2}, {std::numeric_limits<double>::quiet_NaN()}, {kHighsInf}})
    REQUIRE_FALSE(highsVerifyCandidate(lp, valid, &rows, 1e-6).feasible);
  REQUIRE_FALSE(highsVerifyCandidate(lp, valid, nullptr, -1).feasible);
  lp.offset_ = std::numeric_limits<double>::quiet_NaN();
  REQUIRE_FALSE(highsVerifyCandidate(lp, valid, nullptr, 1e-6).feasible);
}
#include "HighsExternalApi.h"
#include "lp_data/HConst.h"
#include "lp_data/HighsCallback.h"
#include "interfaces/highs_c_api.h"
#include "lp_data/HighsRefinementBudget.h"

const bool dev_run = false;

TEST_CASE("legacy callback repair preserves the original model and admission flag",
          "[rost_callback_refinement]") {
  Highs parent;
  parent.resetGlobalScheduler(true);
  parent.setOptionValue("output_flag", false);
  parent.setOptionValue("threads", 1);
  REQUIRE(parent.addVar(0, 1) == HighsStatus::kOk);
  REQUIRE(parent.addVar(0, 1) == HighsStatus::kOk);
  const HighsInt columns[] = {0, 1};
  const double coefficients[] = {1, 1};
  REQUIRE(parent.addRow(-kHighsInf, 1, 2, columns, coefficients) == HighsStatus::kOk);
  HighsCallbackInput input;
  input.highs = &parent;
  input.user_solution = {0, kHighsUndefined};
  input.user_has_solution = true;
  bool expect_success = false;
  SECTION("valid partial completion") { expect_success = true; }
  SECTION("out of original bounds") { input.user_solution = {2, -1}; }
  SECTION("infeasible partial assignment") { input.user_solution = {1, 1}; }
  SECTION("nonfinite assignment") {
    input.user_solution = {std::numeric_limits<double>::quiet_NaN(), 0};
  }
  SECTION("already cancelled") {
    static std::atomic<bool> cancelled{true};
    input.refinement_cancel = &cancelled;
  }
  SECTION("parent time allowance exhausted") { parent.setOptionValue("time_limit", 0.0); }
  SECTION("missing owner") { input.highs = nullptr; }
  SECTION("wrong vector size") { input.user_solution = {0}; }
  SECTION("fractional integer") {
    REQUIRE(parent.changeColIntegrality(0, HighsVarType::kInteger) == HighsStatus::kOk);
    input.user_solution = {0.5, 0};
  }
  HighsCallbackDataIn wire = input;
  const double* storage = wire.user_solution;
  const auto status = Highs_repairCallbackSolution(&wire);
  if (expect_success) {
    REQUIRE(status == static_cast<HighsInt>(HighsStatus::kOk));
    REQUIRE(input.user_has_solution);
    auto original = parent.getLp();
    original.ensureColwise();
    REQUIRE(highsVerifyCandidate(original, input.user_solution, nullptr, 1e-6).feasible);
  } else {
    CHECK(status != static_cast<HighsInt>(HighsStatus::kOk));
    CHECK_FALSE(input.user_has_solution);
  }
  CHECK(wire.user_has_solution == (input.user_has_solution ? 1 : 0));
  CHECK(wire.user_solution == storage);
  CHECK(input.user_solution.data() == storage);
  // The adapter must not resurrect the rejected partial assignment.
  input = wire;
  CHECK(input.user_has_solution == expect_success);
  CHECK(parent.getLp().col_lower_ == std::vector<double>{0, 0});
  CHECK(parent.getLp().col_upper_ == std::vector<double>{1, 1});
  CHECK(parent.getModelStatus() == HighsModelStatus::kNotset);
  parent.resetGlobalScheduler(true);
}

TEST_CASE("refinement work budget spans LP invocations without double counting",
          "[rost_callback_refinement]") {
  std::atomic<bool> cancel{false};
  const auto deadline = HighsRefinementBudget::Clock::now() + std::chrono::seconds(60);
  HighsRefinementBudget budget(10, deadline, &cancel);
  HighsRefinementLpWork first(&budget), nested(&budget);
  first.begin();
  first.observe(3, -1);
  first.observe(3, 0);
  REQUIRE(budget.remaining() == 7);
  nested.begin();
  nested.observe(2, 1);
  REQUIRE(budget.remaining() == 4);
  first.observe(4, -1);
  REQUIRE(budget.remaining() == 3);
  REQUIRE(budget.simplex_iterations == 6);
  REQUIRE(budget.ipm_iterations == 1);
  REQUIRE(budget.lp_runs == 2);
  first.begin();
  first.observe(-1, 3);
  REQUIRE(budget.stopStatus() == HighsModelStatus::kIterationLimit);
  REQUIRE(budget.lp_runs == 3);
  cancel.store(true);
  REQUIRE(budget.stopStatus() == HighsModelStatus::kInterrupt);
  cancel.store(false);
  budget.deadline = HighsRefinementBudget::Clock::now() - std::chrono::seconds(1);
  REQUIRE(budget.stopStatus() == HighsModelStatus::kTimeLimit);
  // Saturate telemetry safely rather than wrapping the allowance.
  first.observe(std::numeric_limits<int64_t>::max(), -1);
  REQUIRE(budget.simplex_iterations == std::numeric_limits<int64_t>::max());
  REQUIRE(budget.remaining() == 0);
}

TEST_CASE("legacy callback repair handles semi domains and C errors",
          "[rost_callback_refinement]") {
  HighsCallbackDataIn invalid{};
  invalid.user_has_solution = 1;
  CHECK(Highs_repairCallbackSolution(nullptr) == kHighsStatusError);
  CHECK(Highs_repairCallbackSolution(&invalid) == kHighsStatusError);
  CHECK(invalid.user_has_solution == 0);
  Highs parent;
  parent.resetGlobalScheduler(true);
  parent.setOptionValue("output_flag", false);
  parent.setOptionValue("threads", 1);
  REQUIRE(parent.addVar(2, 5) == HighsStatus::kOk);
  for (auto type : {HighsVarType::kSemiContinuous, HighsVarType::kSemiInteger}) {
    REQUIRE(parent.changeColIntegrality(0, type) == HighsStatus::kOk);
    for (double value : {0.0, 1.0, 2.0, 2.5, 5.0, 6.0}) {
      HighsCallbackInput input;
      input.highs = &parent;
      input.user_solution = {value};
      input.user_has_solution = true;
      HighsCallbackDataIn wire = input;
      wire.user_interrupt = 1;
      const bool expected = value == 0 || (value >= 2 && value <= 5 &&
          (type == HighsVarType::kSemiContinuous || value == std::round(value)));
      const auto status = Highs_repairCallbackSolution(&wire);
      CHECK((status == kHighsStatusOk) == expected);
      CHECK(input.user_has_solution == expected);
      CHECK(wire.user_has_solution == expected);
      CHECK(wire.user_interrupt == 1);
      CHECK(input.user_solution == std::vector<double>{value});
    }
  }
  parent.resetGlobalScheduler(true);
}

TEST_CASE("callback reset clears unavailable work and cut counters",
          "[rost_callback_refinement]") {
  HighsCallback callback(nullptr);
  callback.data_out.mip_total_lp_iterations = 12345;
  callback.data_out.cutpool_num_col = 27;
  callback.data_out.cutpool_num_cut = 19;
  callback.clearHighsCallbackOutput();
  CHECK(callback.data_out.mip_total_lp_iterations == -1);
  CHECK(callback.data_out.cutpool_num_col == 0);
  CHECK(callback.data_out.cutpool_num_cut == 0);
  const HighsCallbackDataOut output = callback.data_out;
  CHECK(output.mip_total_lp_iterations == -1);
  CHECK(output.cutpool_num_col == 0);
  CHECK(output.cutpool_num_cut == 0);
}

TEST_CASE("callback repair agrees with exhaustive partial assignment oracle",
          "[rost_callback_refinement]") {
  Highs parent;
  parent.resetGlobalScheduler(true);
  parent.setOptionValue("output_flag", false);
  parent.setOptionValue("threads", 1);
  parent.setOptionValue("time_limit", 5.0);
  for (int col = 0; col < 4; ++col) {
    REQUIRE(parent.addVar(0, 1) == HighsStatus::kOk);
    REQUIRE(parent.changeColIntegrality(col, HighsVarType::kInteger) == HighsStatus::kOk);
  }
  const HighsInt first[] = {0, 1}, second[] = {1, 2, 3};
  const double values[] = {1, 1, 1};
  REQUIRE(parent.addRow(-kHighsInf, 1, 2, first, values) == HighsStatus::kOk);
  REQUIRE(parent.addRow(1, kHighsInf, 3, second, values) == HighsStatus::kOk);
  const double symbols[] = {kHighsUndefined, 0, 1, -1, 2};
  for (const char* presolve : {"off", "on"}) {
    parent.setOptionValue("presolve", presolve);
    for (int pattern = 0; pattern < 625; ++pattern) {
      INFO("presolve=" << presolve << " pattern=" << pattern);
      std::vector<double> partial(4);
      int code = pattern;
      for (double& value : partial) { value = symbols[code % 5]; code /= 5; }
      bool exists = false;
      for (int bits = 0; bits < 16; ++bits) {
        int x[4];
        bool matches = true;
        for (int col = 0; col < 4; ++col) {
          x[col] = (bits >> col) & 1;
          if (partial[col] != kHighsUndefined && partial[col] != x[col]) matches = false;
        }
        exists |= matches && x[0] + x[1] <= 1 && x[1] + x[2] + x[3] >= 1;
      }
      HighsCallbackInput input;
      input.highs = &parent;
      input.user_solution = partial;
      input.user_has_solution = true;
      HighsCallbackDataIn wire = input;
      const auto status = Highs_repairCallbackSolution(&wire);
      REQUIRE((status == kHighsStatusOk) == exists);
      REQUIRE(input.user_has_solution == exists);
      REQUIRE(wire.user_has_solution == exists);
      if (exists) {
        const auto& x = input.user_solution;
        for (int col = 0; col < 4; ++col) {
          REQUIRE((x[col] == 0 || x[col] == 1));
          if (partial[col] != kHighsUndefined) REQUIRE(x[col] == partial[col]);
        }
        REQUIRE(x[0] + x[1] <= 1);
        REQUIRE(x[1] + x[2] + x[3] >= 1);
      }
      REQUIRE(parent.getModelStatus() == HighsModelStatus::kNotset);
    }
  }
  parent.resetGlobalScheduler(true);
}

TEST_CASE("callback refinement iteration budget reaches the MIP root LP",
          "[rost_callback_refinement]") {
  Highs parent;
  parent.resetGlobalScheduler(true);
  parent.setOptionValue("output_flag", false);
  parent.setOptionValue("threads", 1);
  parent.setOptionValue("presolve", "off");
  parent.setOptionValue("mip_root_presolve_only", true);
  HighsLp lp;
  // Disjoint odd cycles: the LP needs several pivots, and its optimum is
  // fractional. Neither the domain nor a single pivot proves the MIP optimum.
  lp.num_col_ = lp.num_row_ = 60;
  lp.col_cost_.assign(60, -1);
  lp.col_lower_.assign(60, 0);
  lp.col_upper_.assign(60, 1);
  lp.integrality_.assign(60, HighsVarType::kInteger);
  lp.row_lower_.assign(60, -kHighsInf);
  lp.row_upper_.assign(60, 1);
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_.clear();
  for (HighsInt col = 0; col < 60; ++col) {
    lp.a_matrix_.start_.push_back(2 * col);
    lp.a_matrix_.index_.push_back(col);
    lp.a_matrix_.index_.push_back(col - col % 3 + (col + 1) % 3);
    lp.a_matrix_.value_.insert(lp.a_matrix_.value_.end(), {1, 1});
  }
  lp.a_matrix_.start_.push_back(120);
  REQUIRE(parent.passModel(lp) == HighsStatus::kOk);
  HighsCallbackInput input;
  input.highs = &parent;
  input.user_solution.assign(60, kHighsUndefined);
  input.user_has_solution = true;
  CHECK(input.refineSolution(3, 100, 1, true) == HighsStatus::kWarning);
  CHECK_FALSE(input.user_has_solution);
  CHECK(input.refinement_status ==
        static_cast<HighsInt>(HighsModelStatus::kIterationLimit));
  CHECK(input.refinement_simplex_iterations <= 1);
  CHECK(input.refinement_ipm_iterations == 0);
  CHECK(input.refinement_lp_runs == 1);
  REQUIRE(parent.getModelStatus() == HighsModelStatus::kNotset);
  input.user_solution.assign(60, kHighsUndefined);
  input.user_has_solution = true;
  REQUIRE(input.refineSolution(3, 100, 10000, true) == HighsStatus::kOk);
  const auto checked = highsVerifyCandidate(lp, input.user_solution, nullptr, 1e-6);
  REQUIRE(checked.feasible);
  REQUIRE(static_cast<double>(checked.objective) == Approx(-20));
  REQUIRE(input.refinement_simplex_iterations > 1);
  // The public simplex setting alone did not restrict the nested MIP root.
  // This control also protects ordinary parent solves from helper-only changes.
  parent.setOptionValue("simplex_iteration_limit", 1);
  REQUIRE(parent.run() == HighsStatus::kOk);
  REQUIRE(parent.getModelStatus() == HighsModelStatus::kOptimal);
  REQUIRE(parent.getInfo().simplex_iteration_count > 1);
  REQUIRE(parent.getObjectiveValue() == Approx(-20));

  lp.integrality_.clear();
  REQUIRE(parent.passModel(lp) == HighsStatus::kOk);
  input.user_solution.assign(60, kHighsUndefined);
  input.user_has_solution = true;
  REQUIRE(input.refineSolution(3, 100, 1, true) == HighsStatus::kWarning);
  REQUIRE_FALSE(input.user_has_solution);
  REQUIRE(input.refinement_status ==
          static_cast<HighsInt>(HighsModelStatus::kIterationLimit));
  REQUIRE(input.refinement_simplex_iterations == 1);
  input.user_solution.assign(60, kHighsUndefined);
  input.user_has_solution = true;
  const auto continuous_status = input.refineSolution(3, 100, 10000, true);
  INFO("continuous status=" << input.refinement_status << " simplex="
       << input.refinement_simplex_iterations << " ipm="
       << input.refinement_ipm_iterations);
  REQUIRE(continuous_status == HighsStatus::kOk);
  const auto continuous = highsVerifyCandidate(lp, input.user_solution, nullptr, 1e-6);
  REQUIRE(continuous.feasible);
  REQUIRE(static_cast<double>(continuous.objective) == Approx(-30));
  parent.resetGlobalScheduler(true);
}

TEST_CASE("dense callback rejects malformed submissions atomically",
          "[rost_callback_refinement]") {
  Highs parent;
  parent.setOptionValue("output_flag", false);
  REQUIRE(parent.addVar(0, 1) == HighsStatus::kOk);
  REQUIRE(parent.addVar(0, 1) == HighsStatus::kOk);
  HighsCallbackInput input;
  input.highs = &parent;
  input.user_solution = {0, 0};
  const double values[] = {1, 1};
  REQUIRE(input.setSolution(-1, values) == HighsStatus::kError);
  REQUIRE(input.setSolution(3, values) == HighsStatus::kError);
  REQUIRE(input.setSolution(1, nullptr) == HighsStatus::kError);
  REQUIRE(input.setSolution(0, nullptr) == HighsStatus::kOk);
  REQUIRE_FALSE(input.user_has_solution);
  for (double invalid : {std::numeric_limits<double>::quiet_NaN(),
                         -std::numeric_limits<double>::infinity(), -1.0, 2.0}) {
    const double bad[] = {1, invalid};
    REQUIRE(input.setSolution(2, bad) == HighsStatus::kError);
    REQUIRE(input.user_solution == std::vector<double>{0, 0});
    REQUIRE_FALSE(input.user_has_solution);
  }
  const double partial[] = {kHighsUndefined, 1};
  REQUIRE(input.setSolution(2, partial) == HighsStatus::kOk);
  REQUIRE(input.user_solution[0] == kHighsUndefined);
  REQUIRE(input.user_has_solution);
  REQUIRE(input.setSolution(1, values) == HighsStatus::kOk);
  REQUIRE(input.user_solution == std::vector<double>{1, 1});
  const double bad[] = {0, 2};
  REQUIRE(input.setSolution(2, bad) == HighsStatus::kError);
  REQUIRE(input.user_solution == std::vector<double>{1, 1});
  REQUIRE(input.user_has_solution);
  REQUIRE(input.setSolution(0, nullptr) == HighsStatus::kOk);
  REQUIRE(input.user_has_solution);
  input.user_solution.clear();
  REQUIRE(input.setSolution(1, values) == HighsStatus::kError);
  input.highs = nullptr;
  REQUIRE(input.setSolution(1, values) == HighsStatus::kError);
}

TEST_CASE("sparse callback rejects malformed submissions atomically",
          "[rost_callback_refinement]") {
  Highs parent;
  parent.setOptionValue("output_flag", false);
  REQUIRE(parent.addVar(0, 1) == HighsStatus::kOk);
  REQUIRE(parent.addVar(0, 1) == HighsStatus::kOk);
  HighsCallbackInput input;
  input.highs = &parent;
  input.user_solution = {0, 0};
  const std::vector<double> original = input.user_solution;
  const double values[] = {1, 1};
  for (HighsInt invalid : {-1, 2, 3}) {
    const HighsInt indices[] = {0, invalid};
    REQUIRE(input.setSolution(2, indices, values) == HighsStatus::kError);
    REQUIRE(input.user_solution == original);
    REQUIRE_FALSE(input.user_has_solution);
  }
  const HighsInt indices[] = {0, 1};
  REQUIRE(input.setSolution(-1, indices, values) == HighsStatus::kError);
  REQUIRE(input.setSolution(1, nullptr, values) == HighsStatus::kError);
  REQUIRE(input.setSolution(1, indices, nullptr) == HighsStatus::kError);
  REQUIRE(input.setSolution(0, nullptr, nullptr) == HighsStatus::kOk);
  for (double invalid : {std::numeric_limits<double>::quiet_NaN(),
                         -std::numeric_limits<double>::infinity()}) {
    const double bad[] = {1, invalid};
    REQUIRE(input.setSolution(2, indices, bad) == HighsStatus::kError);
    REQUIRE(input.user_solution == original);
    REQUIRE_FALSE(input.user_has_solution);
  }
  REQUIRE(input.setSolution(2, indices, values) == HighsStatus::kOk);
  REQUIRE(input.user_solution == std::vector<double>{1, 1});
  REQUIRE(input.user_has_solution);
  const double unspecified[] = {kHighsUndefined, 1};
  REQUIRE(input.setSolution(2, indices, unspecified) == HighsStatus::kOk);
  REQUIRE(input.user_solution[0] == kHighsUndefined);
  REQUIRE(input.setSolution(2, indices, values) == HighsStatus::kOk);
  const HighsInt duplicate[] = {1, 1};
  const double last_wins[] = {1, 0};
  REQUIRE(input.setSolution(2, duplicate, last_wins) == HighsStatus::kWarning);
  REQUIRE(input.user_solution == std::vector<double>{1, 0});
  const HighsInt invalid = parent.getNumCol();
  REQUIRE(input.setSolution(1, &invalid, values) == HighsStatus::kError);
  REQUIRE(input.user_solution == std::vector<double>{1, 0});
  REQUIRE(input.user_has_solution);
  input.user_solution.clear();
  REQUIRE(input.setSolution(1, indices, values) == HighsStatus::kError);

  Highs empty;
  empty.setOptionValue("output_flag", false);
  input.highs = &empty;
  REQUIRE(input.setSolution(1, indices, values) == HighsStatus::kError);
  input.highs = nullptr;
  REQUIRE(input.setSolution(1, indices, values) == HighsStatus::kError);
}

TEST_CASE("C callback setters survive input roundtrip without losing interrupt",
          "[rost_callback_refinement]") {
  Highs parent;
  parent.setOptionValue("output_flag", false);
  REQUIRE(parent.addVar(0, 1) == HighsStatus::kOk);
  REQUIRE(parent.addVar(0, 1) == HighsStatus::kOk);
  for (bool sparse : {false, true}) {
    HighsCallbackInput input;
    input.highs = &parent;
    input.user_solution = {0, 0};
    auto data = static_cast<HighsCallbackDataIn>(input);
    data.user_interrupt = 1;
    const double values[] = {1, 1};
    const HighsInt indices[] = {0, 1};
    const HighsInt status = sparse
        ? Highs_setCallbackSparseSolution(&data, 2, indices, values)
        : Highs_setCallbackSolution(&data, 2, values);
    REQUIRE(status == kHighsStatusOk);
    REQUIRE(data.user_has_solution == 1);
    REQUIRE(data.user_interrupt == 1);
    input = data;
    // The roundtrip imports flags; it must not copy the entire solution vector.
    REQUIRE((std::is_same<decltype(input = data), HighsCallbackInput&>::value));
    REQUIRE(input.user_has_solution);
    REQUIRE(input.user_interrupt);
    REQUIRE(input.user_solution == std::vector<double>{1, 1});
    const double bad[] = {0, 2};
    const HighsInt invalid_status = sparse
        ? Highs_setCallbackSparseSolution(&data, 2, indices, bad)
        : Highs_setCallbackSolution(&data, 2, bad);
    REQUIRE(invalid_status == kHighsStatusError);
    input = data;
    REQUIRE(input.user_has_solution);
    REQUIRE(input.user_interrupt);
    REQUIRE(input.user_solution == std::vector<double>{1, 1});
  }
}

TEST_CASE("bounded callback completion keeps policy locks and buffer identity",
          "[rost_callback_refinement]") {
  Highs parent;
  parent.resetGlobalScheduler(true);
  parent.setOptionValue("output_flag", false);
  HighsLp lp;
  lp.num_col_ = 2;
  lp.num_row_ = 1;
  lp.col_cost_ = {0, 1};
  lp.col_lower_ = {0, 0};
  lp.col_upper_ = {1, 100};
  lp.integrality_ = {HighsVarType::kInteger, HighsVarType::kContinuous};
  lp.row_lower_ = {1};
  lp.row_upper_ = {kHighsInf};
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_ = {0, 1, 2};
  lp.a_matrix_.index_ = {0, 0};
  lp.a_matrix_.value_ = {1, 1};
  REQUIRE(parent.passModel(lp) == HighsStatus::kOk);
  HighsCallbackInput input;
  input.highs = &parent;
  input.user_solution = {0, kHighsUndefined};
  input.user_has_solution = true;
  HighsCallbackDataIn data = input;
  double* buffer = data.user_solution;
  REQUIRE(Highs_refineCallbackSolution(&data, 1, 100, 10000, 1) == 0);
  REQUIRE(data.user_solution == buffer);
  REQUIRE(data.user_has_solution == 1);
  REQUIRE(data.user_solution[0] == 0);
  REQUIRE(data.user_solution[1] == Approx(1));
  CHECK(Highs_getCallbackRefinementStatus(&data) ==
        static_cast<HighsInt>(HighsModelStatus::kOptimal));
  // The restricted optimum is not the parent optimum or its proof status.
  REQUIRE(parent.getModelStatus() == HighsModelStatus::kNotset);
  REQUIRE(parent.getLp().col_lower_[0] == 0);
  REQUIRE(parent.getLp().col_upper_[0] == 1);
  REQUIRE(parent.run() == HighsStatus::kOk);
  REQUIRE(parent.getObjectiveValue() == Approx(0));

  input.user_has_solution = true;
  REQUIRE(input.refineSolution(0, 100, 10000, true) == HighsStatus::kError);
  REQUIRE_FALSE(input.user_has_solution);
  input.user_solution = {0, kHighsUndefined};
  input.user_has_solution = true;
  data = input;
  REQUIRE(Highs_refineCallbackSolution(&data, 1e-12, 100, 10000, 1) == 1);
  REQUIRE(Highs_getCallbackRefinementStatus(&data) ==
          static_cast<HighsInt>(HighsModelStatus::kTimeLimit));
  REQUIRE_FALSE(input.user_has_solution);
  input.user_has_solution = true;
  data = input;
  REQUIRE(Highs_refineCallbackSolution(&data, 1, 100, 10000, 1) == 0);
  REQUIRE(data.user_has_solution == 1);
  REQUIRE(data.user_solution[1] == Approx(1));
  input.user_solution = {0.5, kHighsUndefined};
  input.user_has_solution = true;
  REQUIRE(input.refineSolution(1, 100, 10000, true) == HighsStatus::kError);
  REQUIRE_FALSE(input.user_has_solution);
  REQUIRE(Highs_refineCallbackSolution(nullptr, 1, 1, 1, 1) == -1);
  input.user_has_solution = true;
  data = input;
  REQUIRE(Highs_refineCallbackSolution(&data, 1, 1, 1, 2) == -1);
  REQUIRE(data.user_has_solution == 0);
  REQUIRE_FALSE(input.user_has_solution);
  parent.resetGlobalScheduler(true);
}

TEST_CASE("callback refinement retains objective locks and rejects infeasible prefixes",
          "[rost_callback_refinement]") {
  Highs parent;
  parent.resetGlobalScheduler(true);
  parent.setOptionValue("output_flag", false);
  HighsLp lp;
  lp.num_col_ = 2;
  lp.num_row_ = 2;
  lp.col_cost_ = {0, 1};
  lp.col_lower_ = {0, 0};
  lp.col_upper_ = {1, 100};
  lp.integrality_ = {HighsVarType::kInteger, HighsVarType::kContinuous};
  lp.row_lower_ = {1, 0};
  lp.row_upper_ = {kHighsInf, 0};
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_ = {0, 2, 3};
  lp.a_matrix_.index_ = {0, 1, 0};
  lp.a_matrix_.value_ = {1, 1, 1};
  REQUIRE(parent.passModel(lp) == HighsStatus::kOk);
  HighsCallbackInput input;
  input.highs = &parent;
  input.user_solution = {0, kHighsUndefined};
  input.user_has_solution = true;
  REQUIRE(input.refineSolution(1, 100, 10000, true) == HighsStatus::kOk);
  REQUIRE(input.user_solution[1] == Approx(1));
  input.user_solution = {1, kHighsUndefined};
  input.user_has_solution = true;
  REQUIRE(input.refineSolution(1, 100, 10000, false) == HighsStatus::kWarning);
  REQUIRE_FALSE(input.user_has_solution);
  REQUIRE(parent.getNumRow() == 2);
  REQUIRE(parent.getModelStatus() == HighsModelStatus::kNotset);
  parent.resetGlobalScheduler(true);
}

TEST_CASE("bulk callback fixings preserve free columns parent bounds and buffer",
          "[rost_callback_refinement]") {
  Highs parent;
  parent.resetGlobalScheduler(true);
  parent.setOptionValue("output_flag", false);
  parent.setOptionValue("threads", 1);
  HighsLp lp;
  lp.num_col_ = 2048;
  lp.col_cost_.assign(lp.num_col_, 1);
  lp.col_lower_.assign(lp.num_col_, 0);
  lp.col_upper_.assign(lp.num_col_, 2);
  lp.integrality_.assign(lp.num_col_, HighsVarType::kInteger);
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_.assign(lp.num_col_ + 1, 0);
  REQUIRE(parent.passModel(lp) == HighsStatus::kOk);
  HighsCallbackInput input;
  input.highs = &parent;
  input.user_has_solution = true;
  input.user_solution.assign(lp.num_col_, kHighsUndefined);
  for (HighsInt col = 0; col < lp.num_col_; col += 2) input.user_solution[col] = 1;
  double* buffer = input.user_solution.data();
  REQUIRE(input.refineSolution(2, 100, 10000, true) == HighsStatus::kOk);
  REQUIRE(input.user_solution.data() == buffer);
  REQUIRE(input.user_has_solution);
  for (HighsInt col = 0; col < lp.num_col_; ++col)
    REQUIRE(input.user_solution[col] == Approx(col % 2 == 0 ? 1.0 : 0.0));
  REQUIRE(parent.getLp().col_lower_ == lp.col_lower_);
  REQUIRE(parent.getLp().col_upper_ == lp.col_upper_);
  REQUIRE(parent.getModelStatus() == HighsModelStatus::kNotset);
  parent.resetGlobalScheduler(true);
}

TEST_CASE("live master query admits bounded original-space completion",
          "[rost_callback_refinement]") {
  Highs parent;
  parent.resetGlobalScheduler(true);
  parent.setOptionValue("output_flag", false);
  parent.setOptionValue("presolve", "off");
  HighsLp lp;
  lp.num_col_ = 2;
  lp.num_row_ = 1;
  lp.col_cost_ = {0, 1};
  lp.col_lower_ = {0, 0};
  lp.col_upper_ = {1, 100};
  lp.integrality_ = {HighsVarType::kInteger, HighsVarType::kContinuous};
  lp.row_lower_ = {1};
  lp.row_upper_ = {kHighsInf};
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_ = {0, 1, 2};
  lp.a_matrix_.index_ = {0, 0};
  lp.a_matrix_.value_ = {1, 1};
  REQUIRE(parent.passModel(lp) == HighsStatus::kOk);
  HighsSolution start;
  start.col_value = {0, 50};
  REQUIRE(parent.setSolution(start) == HighsStatus::kOk);
  int completions = 0;
  parent.setCallback([&](int kind, const std::string&, const HighsCallbackOutput* out,
                         HighsCallbackInput* in, void*) {
    if (kind != kCallbackMipUserSolution || out->mip_solution.size() != 2 ||
        out->mip_solution[1] <= 1) return;
    const double partial[] = {out->mip_solution[0], kHighsUndefined};
    if (in->setSolution(2, partial) == HighsStatus::kOk &&
        in->refineSolution(1, 100, 10000, true) == HighsStatus::kOk)
      ++completions;
  });
  parent.startCallback(kCallbackMipUserSolution);
  REQUIRE(parent.run() == HighsStatus::kOk);
  REQUIRE(completions >= 1);
  REQUIRE(parent.getModelStatus() == HighsModelStatus::kOptimal);
  REQUIRE(parent.getObjectiveValue() == Approx(0));
  parent.resetGlobalScheduler(true);
}

TEST_CASE("live refinement does not apply user objective scaling twice",
          "[rost_callback_refinement]") {
  Highs parent;
  parent.resetGlobalScheduler(true);
  parent.setOptionValue("output_flag", false);
  parent.setOptionValue("threads", 1);
  parent.setOptionValue("presolve", "off");
  // The parent scale is valid, but applying it twice exceeds large-cost limits.
  REQUIRE(parent.setOptionValue("user_objective_scale", 40) == HighsStatus::kOk);
  REQUIRE(parent.addVar(0, 1) == HighsStatus::kOk);
  REQUIRE(parent.addVar(0, 100) == HighsStatus::kOk);
  REQUIRE(parent.changeColIntegrality(0, HighsVarType::kInteger) == HighsStatus::kOk);
  REQUIRE(parent.changeColCost(1, 1) == HighsStatus::kOk);
  const HighsInt indices[] = {0, 1};
  const double values[] = {1, 1};
  REQUIRE(parent.addRow(1, kHighsInf, 2, indices, values) == HighsStatus::kOk);
  HighsSolution start;
  start.col_value = {0, 50};
  REQUIRE(parent.setSolution(start) == HighsStatus::kOk);
  int attempts = 0, completions = 0;
  HighsStatus set_status = HighsStatus::kError, refine_status = HighsStatus::kError;
  HighsInt refinement_status = -1;
  double returned = -1, observed_cost = -1;
  parent.setCallback([&](int kind, const std::string&, const HighsCallbackOutput*,
                         HighsCallbackInput* in, void*) {
    if (kind != kCallbackMipUserSolution || attempts++) return;
    const double partial[] = {0, kHighsUndefined};
    observed_cost = in->highs->getLp().col_cost_[1];
    set_status = in->setSolution(2, partial);
    refine_status = in->refineSolution(1, 100, 10000, true);
    refinement_status = in->refinement_status;
    returned = in->user_solution[1];
    if (set_status == HighsStatus::kOk && refine_status == HighsStatus::kOk)
      ++completions;
  });
  parent.startCallback(kCallbackMipUserSolution);
  REQUIRE(parent.run() != HighsStatus::kError);
  REQUIRE(attempts > 0);
  INFO("set=" << int(set_status) << " refine=" << int(refine_status)
       << " model=" << refinement_status << " cost=" << observed_cost
       << " value=" << returned);
  REQUIRE(completions == 1);
  REQUIRE(returned == Approx(1));
  REQUIRE(parent.getModelStatus() == HighsModelStatus::kOptimal);
  REQUIRE(parent.getObjectiveValue() == Approx(0));
  REQUIRE(parent.getLp().col_cost_[1] == 1);
  parent.resetGlobalScheduler(true);
}

enum class CallbackProposal {
  kBetter, kBadRow, kFractional, kBadBound, kWorse, kNonfinite, kDimensions,
  kStale, kDuplicate
};

TEST_CASE("tagged-repairs-acknowledge-and-continue-before-another-batch",
          "[rost_callback_refinement]") {
  Highs highs;
  highs.resetGlobalScheduler(true);
  highs.setOptionValue("output_flag", false);
  highs.setOptionValue("threads", 1);
  highs.setOptionValue("presolve", "off");
  highs.setOptionValue("mip_candidate_checkpoint", true);
  HighsLp lp;
  lp.num_col_ = 2; lp.num_row_ = 1;
  lp.col_cost_ = {0, 1}; lp.col_lower_ = {0, 0}; lp.col_upper_ = {1, 100};
  lp.integrality_ = {HighsVarType::kInteger, HighsVarType::kContinuous};
  lp.row_lower_ = {1}; lp.row_upper_ = {kHighsInf};
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_ = {0, 1, 2}; lp.a_matrix_.index_ = {0, 0};
  lp.a_matrix_.value_ = {1, 1};
  REQUIRE(highs.passModel(lp) == HighsStatus::kOk);
  HighsSolution start; start.col_value = {0, 50};
  REQUIRE(highs.setSolution(start) == HighsStatus::kOk);
  int offered = 0, receipts = 0;
  highs.setCallback([&](int kind, const std::string&, const HighsCallbackOutput* out,
                        HighsCallbackInput* in, void*) {
    if (kind != kCallbackMipUserSolution || receipts == 3) return;
    if (offered > 0) {
      REQUIRE(in->admission.candidate_id == offered);
      REQUIRE(in->admission.status == kHighsAdmissionImproved);
      REQUIRE(out->mip_total_lp_iterations == 0);
      REQUIRE(out->mip_node_count == 0);
      ++receipts;
    }
    if (offered == 3) return;
    in->user_solution = {0, 40.0 - offered * 10.0};
    in->user_has_solution = true;
    HighsCallbackDataIn input = *in;
    REQUIRE(Highs_tagCallbackSolution(&input, ++offered, in->admission_generation) == 0);
  });
  REQUIRE(highs.startCallback(kCallbackMipUserSolution) == HighsStatus::kOk);
  REQUIRE(highs.run() == HighsStatus::kOk);
  REQUIRE(offered == 3);
  REQUIRE(receipts == 3);
  REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
  REQUIRE(highs.getObjectiveValue() == Approx(0));
}

static void checkCallbackAdmissionObjectiveSpace(const ObjSense sense,
                                                const double offset,
                                                const double fixed_cost,
                                                const double scale,
                                                const bool presolve,
                                                const CallbackProposal kind =
                                                    CallbackProposal::kBetter) {
  INFO("sense=" << static_cast<int>(sense) << " offset=" << offset
                << " fixed_cost=" << fixed_cost << " scale=" << scale
                << " presolve=" << presolve << " proposal=" << int(kind));
  Highs highs;
  highs.resetGlobalScheduler(true);
  REQUIRE(highs.setOptionValue("log_to_console", false) == HighsStatus::kOk);
  REQUIRE(highs.setOptionValue("threads", 1) == HighsStatus::kOk);
  REQUIRE(highs.setOptionValue("presolve", presolve ? "on" : "off") ==
          HighsStatus::kOk);
  REQUIRE(highs.setOptionValue("mip_heuristic_run_feasibility_jump", false) ==
          HighsStatus::kOk);
  // An odd-cycle cover stays fractional at the LP optimum. Its trivial
  // all-zero/all-one candidates cannot mimic the submitted five-column cover.
  const HighsInt n = 9;
  const double sign = static_cast<int>(sense);
  HighsLp lp;
  lp.num_col_ = n + 1;
  lp.num_row_ = n;
  lp.sense_ = sense;
  lp.offset_ = offset * scale;
  lp.col_cost_.assign(n + 1, sign * scale);
  lp.col_cost_[n] = sign * fixed_cost * scale;
  lp.col_lower_.assign(n + 1, 0);
  lp.col_lower_[n] = 1;
  lp.col_upper_.assign(n + 1, 1);
  lp.integrality_.assign(n + 1, HighsVarType::kInteger);
  lp.row_lower_.assign(n, 1);
  lp.row_upper_.assign(n, kHighsInf);
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_.clear();
  for (HighsInt col = 0; col < n; ++col) {
    lp.a_matrix_.start_.push_back(lp.a_matrix_.value_.size());
    lp.a_matrix_.index_.push_back(col);
    lp.a_matrix_.index_.push_back((col + n - 1) % n);
    lp.a_matrix_.value_.insert(lp.a_matrix_.value_.end(), {1, 1});
  }
  lp.a_matrix_.start_.push_back(lp.a_matrix_.value_.size());
  lp.a_matrix_.start_.push_back(lp.a_matrix_.value_.size());
  REQUIRE(lp.a_matrix_.start_.size() == static_cast<size_t>(n + 2));
  REQUIRE(lp.a_matrix_.value_.size() == static_cast<size_t>(2 * n));
  REQUIRE(highs.passModel(lp) == HighsStatus::kOk);
  if (presolve) {
    REQUIRE(highs.presolve() == HighsStatus::kOk);
    REQUIRE(highs.getPresolvedLp().num_col_ > 0);
    REQUIRE(highs.getPresolvedLp().num_col_ < lp.num_col_);
  }
  HighsSolution start;
  start.col_value.assign(n + 1, 1);
  std::vector<double> proposed(n + 1, 1);
  for (HighsInt col = 1; col < n; col += 2) proposed[col] = 0;
  HighsInt start_count = n;
  if (kind == CallbackProposal::kWorse) {
    start.col_value = proposed;
    start.col_value[1] = 1;
    start_count = 6;
    proposed.assign(n + 1, 1);
  } else if (kind == CallbackProposal::kBadRow) {
    for (HighsInt col = 0; col < n; ++col) proposed[col] = 0;
  } else if (kind == CallbackProposal::kFractional) {
    for (HighsInt col = 0; col < n; ++col) proposed[col] = 0.5;
  } else if (kind == CallbackProposal::kBadBound) {
    proposed[n] = 0;
  } else if (kind == CallbackProposal::kNonfinite) {
    proposed[n] = std::numeric_limits<double>::quiet_NaN();
  }
  REQUIRE(highs.setSolution(start) == HighsStatus::kOk);
  const double expected = (offset + sign * (fixed_cost + 5)) * scale;
  const double start_objective =
      (offset + sign * (fixed_cost + start_count)) * scale;
  int submissions = 0;
  bool admitted_before_root_lp = false;
  bool submitted_vector_transformed = false;
  bool worsened_parent = false;
  highs.setCallback([&](int callback_kind, const std::string&,
                        const HighsCallbackOutput* out,
                        HighsCallbackInput* in, void*) {
    if (callback_kind == kCallbackMipUserSolution && submissions == 0) {
      REQUIRE(out->mip_primal_bound == Approx(start_objective));
      in->user_solution = proposed;
      in->user_has_solution = true;
      if (kind == CallbackProposal::kDimensions) in->user_solution.pop_back();
      HighsCallbackDataIn input = *in;
      REQUIRE(Highs_tagCallbackSolution(&input, 1, in->admission_generation +
              (kind == CallbackProposal::kStale ? 1 : 0)) == 0);
      if (kind == CallbackProposal::kDuplicate) in->last_candidate_id = 2;
      ++submissions;
    }
    if (callback_kind == kCallbackMipSolution && submissions == 1 &&
        out->mip_solution == proposed)
      submitted_vector_transformed = true;
    if (callback_kind == kCallbackMipLogging && submissions == 1 &&
        sign * out->mip_primal_bound > sign * start_objective + 1e-6)
      worsened_parent = true;
    if (callback_kind == kCallbackMipLogging && submissions == 1 &&
        out->mip_total_lp_iterations == 0 &&
        out->mip_primal_bound == Approx(expected))
      admitted_before_root_lp = true;
  });
  highs.startCallback(kCallbackMipUserSolution);
  highs.startCallback(kCallbackMipLogging);
  highs.startCallback(kCallbackMipSolution);
  REQUIRE(highs.run() == HighsStatus::kOk);
  CHECK(submissions == 1);
  HighsCallbackSolutionOutcome outcome = {};
  REQUIRE(Highs_getLastCallbackSolutionOutcome(&highs, 1, &outcome) == 0);
  CHECK(outcome.candidate_id == 1);
  CHECK(outcome.generation > 0);
  CHECK(outcome.objective_before == Approx(start_objective));
  CHECK(outcome.admission_time >= 0);
  if (kind == CallbackProposal::kBetter) {
    CHECK(outcome.status == kHighsAdmissionImproved);
    CHECK(outcome.sequence_after == outcome.sequence_before + 1);
    CHECK(outcome.objective_after == Approx(expected));
    CHECK(outcome.submitted_objective == Approx(expected));
  } else {
    const int status = kind == CallbackProposal::kWorse ? kHighsAdmissionNoImprovement
        : kind == CallbackProposal::kNonfinite ? kHighsAdmissionNonfinite
        : kind == CallbackProposal::kDimensions ? kHighsAdmissionInvalidDimensions
        : kind == CallbackProposal::kStale ? kHighsAdmissionStale
        : kind == CallbackProposal::kDuplicate ? kHighsAdmissionDuplicate
        : kHighsAdmissionInfeasible;
    CHECK(outcome.status == status);
    CHECK(outcome.sequence_after == outcome.sequence_before);
    CHECK(outcome.objective_after == Approx(start_objective));
  }
  const auto saved = outcome;
  CHECK(Highs_getLastCallbackSolutionOutcome(&highs, 2, &outcome) == -1);
  CHECK(outcome.candidate_id == saved.candidate_id);
  CHECK(Highs_getLastCallbackSolutionOutcome(nullptr, 1, &outcome) == -1);
  if (kind == CallbackProposal::kBetter) {
    CHECK(admitted_before_root_lp);
    CHECK(submitted_vector_transformed);
  } else if (kind == CallbackProposal::kBadRow ||
             kind == CallbackProposal::kFractional ||
             kind == CallbackProposal::kBadBound ||
             kind == CallbackProposal::kNonfinite) {
    CHECK_FALSE(submitted_vector_transformed);
  }
  CHECK_FALSE(worsened_parent);
  CHECK(highs.getModelStatus() == HighsModelStatus::kOptimal);
  CHECK(highs.getObjectiveValue() == Approx(expected));
  for (HighsInt row = 0; row < n; ++row)
    CHECK(highs.getSolution().col_value[row] +
              highs.getSolution().col_value[(row + 1) % n] >=
          1 - 1e-6);
  highs.resetGlobalScheduler(true);
}

TEST_CASE("callback admission respects sense offset and exact objective scaling",
          "[rost_callback_refinement][rost_callback_admission]") {
  for (const auto sense : {ObjSense::kMinimize, ObjSense::kMaximize})
    for (const double offset : {-1000.0, 0.0, 1000.0})
      for (const double scale : {1.0, 0.0625})
        checkCallbackAdmissionObjectiveSpace(sense, offset, 0, scale, false);
}

TEST_CASE("callback admission accounts for objective contributions removed by presolve",
          "[rost_callback_refinement][rost_callback_admission]") {
  for (const auto sense : {ObjSense::kMinimize, ObjSense::kMaximize})
    for (const double fixed_cost : {-1000.0, 1000.0})
      for (const double scale : {1.0, 0.0625})
        checkCallbackAdmissionObjectiveSpace(sense, 0, fixed_cost, scale, true);
}

TEST_CASE("callback admission rejects invalid and worse original-space proposals",
          "[rost_callback_refinement][rost_callback_admission]") {
  for (const auto sense : {ObjSense::kMinimize, ObjSense::kMaximize})
    for (const bool presolve : {false, true})
      for (const auto proposal :
           {CallbackProposal::kBadRow, CallbackProposal::kFractional,
            CallbackProposal::kBadBound, CallbackProposal::kWorse,
            CallbackProposal::kNonfinite, CallbackProposal::kDimensions,
            CallbackProposal::kStale, CallbackProposal::kDuplicate})
        checkCallbackAdmissionObjectiveSpace(sense, -1000, 1000, 0.0625,
                                            presolve, proposal);
}

const double egout_optimal_objective = 568.1007;
const double egout_objective_target = 610;
const HighsInt adlittle_ipm_iteration_limit = 5;
const HighsInt adlittle_simplex_iteration_limit = 30;

const HighsInt kLogBufferSize = kIoBufferSize;
const HighsInt kUserCallbackNoData = -1;
const HighsInt kUserCallbackData = 99;

char printed_log[kLogBufferSize];

using std::memset;
using std::strcmp;
using std::strcpy;
using std::strlen;
using std::strncmp;
using std::strstr;

struct MipData {
  HighsInt num_col;
  HighsVarType* integrality;
};

struct UserMipSolution {
  double optimal_objective_value;
  std::vector<double> optimal_solution;
  HighsInt require_external_solution_query_origin;
};

// Callback that saves message for comparison
HighsCallbackFunctionType myLogCallback =
    [](int callback_type, const std::string& message,
       const HighsCallbackOutput* data_out, HighsCallbackInput* data_in,
       void* user_callback_data) { strcpy(printed_log, message.c_str()); };

HighsCallbackFunctionType userMipSolutionCallback =
    [](int callback_type, const std::string& message,
       const HighsCallbackOutput* data_out, HighsCallbackInput* data_in,
       void* user_callback_data) {
      if (dev_run) {
        printf(
            "MipSolutionCallback with objective = %15.8g and bounds [%15.8g, "
            "%15.8g]",
            data_out->objective_function_value, data_out->mip_dual_bound,
            data_out->mip_primal_bound);
        MipData callback_data = *(static_cast<MipData*>(user_callback_data));
        HighsInt num_col = callback_data.num_col;
        HighsVarType* integrality = callback_data.integrality;
        HighsInt num_integer = 0;
        for (HighsInt iCol = 0; iCol < num_col; iCol++)
          if (integrality[iCol] == HighsVarType::kInteger) num_integer++;
        if (num_integer < 50) {
          printf(" and solution [");
          for (HighsInt iCol = 0; iCol < num_col; iCol++) {
            if (integrality[iCol] != HighsVarType::kInteger) continue;
            double value = data_out->mip_solution[iCol];
            if (std::abs(value) < 1e-5) {
              printf("0");
            } else if (std::abs(value - 1) < 1e-5) {
              printf("1");
            } else {
              bool printed = false;
              for (HighsInt k = 2; k < 10; k++) {
                if (std::abs(value - k) < 1e-5) {
                  printf("%1d", int(k));
                  printed = true;
                }
              }
              if (printed) continue;
              for (HighsInt k = 10; k < 999; k++) {
                if (std::abs(value - k) < 1e-5) {
                  printf(" %d ", int(k));
                  printed = true;
                }
              }
              if (printed) continue;
              printf("*");
            }
          }
          printf("]\n");
        } else {
          printf("\n");
        }
        fflush(stdout);
      }
    };

HighsCallbackFunctionType userInterruptCallback =
    [](int callback_type, const std::string& message,
       const HighsCallbackOutput* data_out, HighsCallbackInput* data_in,
       void* user_callback_data) {
      // Extract local_callback_data from user_callback_data unless it
      // is nullptr
      if (callback_type == kCallbackMipImprovingSolution) {
        // Use local_callback_data to maintain the objective value from
        // the previous callback
        assert(user_callback_data);
        // Extract the double value pointed to from void* user_callback_data
        const double local_callback_data = *(double*)user_callback_data;
        if (dev_run)
          printf(
              "userCallback(type %2d; data %11.4g): %s with objective %g and "
              "solution[0] = %g\n",
              callback_type, local_callback_data, message.c_str(),
              data_out->objective_function_value, data_out->mip_solution[0]);
        REQUIRE(local_callback_data >= data_out->objective_function_value);
        REQUIRE(data_out->mip_solution_source >= 0);
        REQUIRE(data_out->mip_solution_sequence >= 0);
        // Update the double value pointed to from void* user_callback_data
        *(double*)user_callback_data = data_out->objective_function_value;
      } else {
        const int local_callback_data =
            user_callback_data ? static_cast<int>(reinterpret_cast<intptr_t>(
                                     user_callback_data))
                               : kUserCallbackNoData;
        if (user_callback_data) {
          REQUIRE(local_callback_data == kUserCallbackData);
        } else {
          REQUIRE(local_callback_data == kUserCallbackNoData);
        }
        if (callback_type == kCallbackLogging) {
          if (dev_run) printf("Callback: %s", message.c_str());
          //            printf("userInterruptCallback(type %2d; data %2d): %s",
          //                   callback_type, local_callback_data,
          //                   message.c_str());
        } else if (callback_type == kCallbackSimplexInterrupt) {
          if (dev_run)
            printf(
                "userInterruptCallback(type %2d; data %2d): %s with iteration "
                "count = "
                "%d\n",
                callback_type, local_callback_data, message.c_str(),
                int(data_out->simplex_iteration_count));
          data_in->user_interrupt = data_out->simplex_iteration_count >
                                    adlittle_simplex_iteration_limit;
        } else if (callback_type == kCallbackIpmInterrupt) {
          if (dev_run)
            printf(
                "userInterruptCallback(type %2d; data %2d): %s with iteration "
                "count = "
                "%d\n",
                callback_type, local_callback_data, message.c_str(),
                int(data_out->ipm_iteration_count));
          data_in->user_interrupt =
              data_out->ipm_iteration_count > adlittle_ipm_iteration_limit;
        } else if (callback_type == kCallbackMipInterrupt) {
          if (dev_run)
            printf(
                "userInterruptCallback(type %2d; data %2d): %s with Bounds "
                "(%11.4g, %11.4g); Gap = %11.4g; Objective = "
                "%g\n",
                callback_type, local_callback_data, message.c_str(),
                data_out->mip_dual_bound, data_out->mip_primal_bound,
                data_out->mip_gap, data_out->objective_function_value);
          data_in->user_interrupt =
              data_out->objective_function_value < egout_objective_target;
        }
      }
    };

HighsCallbackFunctionType userMipCutPoolCallback =
    [](int callback_type, const std::string& message,
       const HighsCallbackOutput* data_out, HighsCallbackInput* data_in,
       void* user_callback_data) {
      if (dev_run) {
        printf("userMipCutPoolCallback: dim(%2d, %2d, %2d)\n",
               int(data_out->cutpool_num_col), int(data_out->cutpool_num_cut),
               int(data_out->cutpool_value.size()));
        for (HighsInt iCut = 0; iCut < data_out->cutpool_num_cut; iCut++) {
          printf("Cut %d\n", int(iCut));
          for (HighsInt iEl = data_out->cutpool_start[iCut];
               iEl < data_out->cutpool_start[iCut + 1]; iEl++) {
            printf("   %2d %11.5g\n", int(data_out->cutpool_index[iEl]),
                   data_out->cutpool_value[iEl]);
          }
        }
      }
    };

HighsCallbackFunctionType userkMipUserSolution =
    [](int callback_type, const std::string& message,
       const HighsCallbackOutput* data_out, HighsCallbackInput* data_in,
       void* user_callback_data) {
      UserMipSolution callback_data =
          *(static_cast<UserMipSolution*>(user_callback_data));
      if (data_out->external_solution_query_origin ==
          callback_data.require_external_solution_query_origin) {
        if (data_out->mip_primal_bound >
            callback_data.optimal_objective_value) {
          // If current objective value is not optimal, pass the
          // optimal solution as a user solution
          if (dev_run)
            printf(
                "userkMipUserSolution: origin = %d; %g = mip_primal_bound > "
                "optimal_objective_value = %g\n",
                int(data_out->external_solution_query_origin),
                data_out->mip_primal_bound,
                callback_data.optimal_objective_value);
          data_in->user_has_solution = true;
          data_in->user_solution = callback_data.optimal_solution;
        }
      }
    };

HighsCallbackFunctionType userkMipUserSetSolution =
    [](int callback_type, const std::string& message,
       const HighsCallbackOutput* data_out, HighsCallbackInput* data_in,
       void* user_callback_data) {
      const auto& callback_data =
          *(static_cast<UserMipSolution*>(user_callback_data));
      if (data_out->external_solution_query_origin ==
          callback_data.require_external_solution_query_origin) {
        if (dev_run)
          printf(
              "userkMipUserSetSolution: origin = %d; %g = mip_primal_bound > "
              "optimal_objective_value = %g\n",
              int(data_out->external_solution_query_origin),
              data_out->mip_primal_bound,
              callback_data.optimal_objective_value);

        data_in->setSolution(callback_data.optimal_solution.size(),
                             callback_data.optimal_solution.data());
      }
    };

HighsCallbackFunctionType userkMipUserSetPartialSolution =
    [](int callback_type, const std::string& message,
       const HighsCallbackOutput* data_out, HighsCallbackInput* data_in,
       void* user_callback_data) {
      const auto& callback_data =
          *(static_cast<UserMipSolution*>(user_callback_data));
      if (data_out->external_solution_query_origin ==
          callback_data.require_external_solution_query_origin) {
        if (dev_run)
          printf(
              "userkMipUserSetPartialSolution: origin = %d; %g = "
              "mip_primal_bound > "
              "optimal_objective_value = %g\n",
              int(data_out->external_solution_query_origin),
              data_out->mip_primal_bound,
              callback_data.optimal_objective_value);

        // get every other index
        std::vector<HighsInt> index;
        std::vector<double> value;

        for (HighsInt i = 0;
             i < static_cast<HighsInt>(callback_data.optimal_solution.size());
             i++) {
          if (i % 2 == 0) {
            index.push_back(i);
            value.push_back(callback_data.optimal_solution[i]);
          }
        }

        data_in->setSolution(index.size(), index.data(), value.data());
        data_in->repairSolution();
      }
    };

std::function<void(int, const std::string&, const HighsCallbackOutput*,
                   HighsCallbackInput*, void*)>
    userDataCallback = [](int callback_type, const std::string& message,
                          const HighsCallbackOutput* data_out,
                          HighsCallbackInput* data_in,
                          void* user_callback_data) {
      assert(callback_type == kCallbackMipInterrupt ||
             callback_type == kCallbackMipLogging ||
             callback_type == kCallbackMipImprovingSolution);
      if (dev_run)
        printf(
            "userDataCallback: Node count = %" PRId64
            "; LP total iterations = %" PRId64
            "; Time = %6.2f; "
            "Bounds (%11.4g, %11.4g); Gap = %11.4g; Objective = %11.4g: %s\n",
            data_out->mip_node_count, data_out->mip_total_lp_iterations,
            data_out->running_time, data_out->mip_dual_bound,
            data_out->mip_primal_bound, data_out->mip_gap,
            data_out->objective_function_value, message.c_str());
    };

TEST_CASE("my-callback-logging", "[highs_callback]") {
  bool output_flag = true;  // Still runs quietly
  bool log_to_console = false;
  HighsInt log_dev_level = kHighsLogDevLevelInfo;
  HighsLogOptions log_options;
  log_options.clear();
  log_options.log_stream = stdout;
  log_options.output_flag = &output_flag;
  log_options.log_to_console = &log_to_console;
  log_options.log_dev_level = &log_dev_level;
  log_options.user_callback = myLogCallback;
  log_options.user_callback_active = true;

  highsLogDev(log_options, HighsLogType::kInfo, "Hi %s!", "HiGHS");
  if (dev_run) printf("Log callback yields \"%s\"\n", printed_log);
  REQUIRE(strcmp(printed_log, "Hi HiGHS!") == 0);

  // Check that nothing is printed if the type is VERBOSE when
  // log_dev_level is kHighsLogDevLevelInfo;
  *printed_log = '\0';
  highsLogDev(log_options, HighsLogType::kVerbose, "Hi %s!", "HiGHS");
  REQUIRE(*printed_log == '\0');

  {
    char long_message[sizeof(printed_log)];
    memset(long_message, 'H', sizeof(long_message));
    long_message[sizeof(long_message) - 2] = '\0';
    long_message[sizeof(long_message) - 1] = '\n';
    highsLogDev(log_options, HighsLogType::kInfo, long_message);
    if (dev_run) printf("Log callback yields \"%s\"\n", printed_log);
    REQUIRE(strncmp(printed_log, "HHHH", 4) == 0);
    REQUIRE(strlen(printed_log) <= sizeof(printed_log));
  }

  highsLogUser(log_options, HighsLogType::kInfo, "Hello %s!\n", "HiGHS");
  REQUIRE(strlen(printed_log) > 9);
  REQUIRE(strcmp(printed_log, "Hello HiGHS!\n") == 0);

  {
    char long_message[sizeof(printed_log)];
    memset(long_message, 'H', sizeof(long_message));
    long_message[sizeof(long_message) - 2] = '\0';
    long_message[sizeof(long_message) - 1] = '\n';
    highsLogUser(log_options, HighsLogType::kWarning, long_message);
    if (dev_run) printf("Log callback yields \"%s\"\n", printed_log);
    REQUIRE(strstr(printed_log, "HHHH") != nullptr);
    REQUIRE(strlen(printed_log) <= sizeof(printed_log));
  }
}

TEST_CASE("highs-callback-logging", "[highs_callback]") {
  // Uses userInterruptCallback to start logging lines with
  // "userInterruptCallback(kUserCallbackData): " since
  // Highs::setCallback has second argument p_user_callback_data
  std::string filename = std::string(HIGHS_DIR) + "/check/instances/avgas.mps";
  int user_callback_data = kUserCallbackData;
  void* p_user_callback_data =
      reinterpret_cast<void*>(static_cast<intptr_t>(user_callback_data));
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  highs.setCallback(userInterruptCallback, p_user_callback_data);
  highs.startCallback(kCallbackLogging);
  highs.readModel(filename);
  highs.run();

  highs.resetGlobalScheduler(true);
}

TEST_CASE("highs-callback-solution-basis-logging", "[highs_callback]") {
  std::string filename = std::string(HIGHS_DIR) + "/check/instances/avgas.mps";
  int user_callback_data = kUserCallbackData;
  void* p_user_callback_data =
      reinterpret_cast<void*>(static_cast<intptr_t>(user_callback_data));
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  highs.readModel(filename);
  highs.run();
  highs.setCallback(userInterruptCallback, p_user_callback_data);
  highs.startCallback(kCallbackLogging);
  if (dev_run) highs.writeSolution("", kSolutionStylePretty);
  if (dev_run) highs.writeBasis("");

  highs.resetGlobalScheduler(true);
}

TEST_CASE("highs-callback-simplex-interrupt", "[highs_callback]") {
  std::string filename =
      std::string(HIGHS_DIR) + "/check/instances/adlittle.mps";
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  highs.setCallback(userInterruptCallback);
  highs.startCallback(kCallbackSimplexInterrupt);
  highs.readModel(filename);
  HighsStatus status = highs.run();
  REQUIRE(status == HighsStatus::kWarning);
  REQUIRE(highs.getModelStatus() == HighsModelStatus::kInterrupt);
  REQUIRE(highs.getInfo().simplex_iteration_count >
          adlittle_simplex_iteration_limit);

  highs.resetGlobalScheduler(true);
}

TEST_CASE("highs-callback-ipm-interrupt", "[highs_callback]") {
  std::string filename =
      std::string(HIGHS_DIR) + "/check/instances/adlittle.mps";
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  highs.setCallback(userInterruptCallback);
  highs.startCallback(kCallbackIpmInterrupt);
  highs.readModel(filename);
  highs.setOptionValue("solver", kIpxString);
  REQUIRE(highs.run() == HighsStatus::kWarning);
  REQUIRE(highs.getModelStatus() == HighsModelStatus::kInterrupt);
  REQUIRE(highs.getInfo().ipm_iteration_count ==
          adlittle_ipm_iteration_limit + 1);

  highs.readModel(filename);

  if (HighsExternalApi::isAvailable<HighsExtras::hipo>()) {
    REQUIRE(highs.setOptionValue("solver", kHipoString) == HighsStatus::kOk);
    ;
    REQUIRE(highs.run() == HighsStatus::kWarning);
    REQUIRE(highs.getModelStatus() == HighsModelStatus::kInterrupt);
    REQUIRE(highs.getInfo().ipm_iteration_count ==
            adlittle_ipm_iteration_limit + 1);
  }
  highs.resetGlobalScheduler(true);
}

TEST_CASE("highs-callback-mip-interrupt", "[highs_callback]") {
  std::string filename = std::string(HIGHS_DIR) + "/check/instances/egout.mps";
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  highs.setOptionValue("presolve", kHighsOffString);
  highs.setCallback(userInterruptCallback);
  highs.startCallback(kCallbackMipInterrupt);
  highs.readModel(filename);
  HighsStatus status = highs.run();
  REQUIRE(status == HighsStatus::kWarning);
  REQUIRE(highs.getModelStatus() == HighsModelStatus::kInterrupt);
  REQUIRE(highs.getInfo().objective_function_value > egout_optimal_objective);

  highs.resetGlobalScheduler(true);
}

TEST_CASE("highs-callback-mip-improving", "[highs_callback]") {
  std::string filename = std::string(HIGHS_DIR) + "/check/instances/egout.mps";
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  highs.setOptionValue("presolve", kHighsOffString);
  double user_callback_data = kHighsInf;
  void* p_user_callback_data = (void*)(&user_callback_data);
  highs.setCallback(userInterruptCallback, p_user_callback_data);
  highs.startCallback(kCallbackMipImprovingSolution);
  highs.readModel(filename);
  highs.run();

  highs.resetGlobalScheduler(true);
}

TEST_CASE("highs-callback-mip-data", "[highs_callback]") {
  std::string filename = std::string(HIGHS_DIR) + "/check/instances/egout.mps";
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  highs.setOptionValue("presolve", kHighsOffString);
  highs.setCallback(userDataCallback);
  highs.startCallback(kCallbackMipImprovingSolution);
  highs.startCallback(kCallbackMipLogging);
  highs.readModel(filename);
  highs.run();

  highs.resetGlobalScheduler(true);
}

TEST_CASE("highs-callback-mip-solution", "[highs_callback]") {
  std::string filename = std::string(HIGHS_DIR) + "/check/instances/egout.mps";
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  highs.setOptionValue("presolve", kHighsOffString);
  highs.readModel(filename);
  // To print the values of the integer variables in the callback,
  // need the number of columns and the integrality. Set this up in a
  // struct to be passed via user_callback_data
  HighsLp lp = highs.getLp();
  MipData user_callback_data;
  user_callback_data.num_col = int(lp.num_col_);
  user_callback_data.integrality = lp.integrality_.data();
  void* p_user_callback_data = &user_callback_data;

  highs.setCallback(userMipSolutionCallback, p_user_callback_data);
  highs.startCallback(kCallbackMipSolution);
  highs.run();

  highs.resetGlobalScheduler(true);
}

TEST_CASE("highs-callback-mip-cut-pool", "[highs_callback]") {
  std::string filename = std::string(HIGHS_DIR) + "/check/instances/flugpl.mps";
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  highs.readModel(filename);
  highs.setCallback(userMipCutPoolCallback);
  highs.startCallback(kCallbackMipGetCutPool);
  highs.run();

  highs.resetGlobalScheduler(true);
}

static void runMipUserSolutionTest(
    HighsCallbackFunctionType callback_function) {
  const std::vector<std::string> model = {"p0548", "flugpl", "gt2", "egout",
                                          "sp150x300d"};
  const std::vector<HighsInt> require_origin = {0, 1, 2, 3, 4};  //, 4, 5, 6};
  assert(model.size() == require_origin.size());
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  highs.setOptionValue("mip_rel_gap", 0);
  HighsInt from_model = 0;
  HighsInt to_model = HighsInt(model.size());
  for (HighsInt iModel = from_model; iModel < to_model; iModel++) {
    const std::string filename =
        std::string(HIGHS_DIR) + "/check/instances/" + model[iModel] + ".mps";
    highs.readModel(filename);
    highs.run();
    std::vector<double> optimal_solution = highs.getSolution().col_value;
    double objective_function_value0 = highs.getInfo().objective_function_value;
    highs.clearSolver();

    UserMipSolution user_callback_data;
    user_callback_data.optimal_objective_value = objective_function_value0;
    user_callback_data.optimal_solution = optimal_solution;
    user_callback_data.require_external_solution_query_origin =
        require_origin[iModel];
    void* p_user_callback_data = (void*)(&user_callback_data);

    highs.setCallback(callback_function, p_user_callback_data);
    highs.startCallback(kCallbackMipUserSolution);
    highs.run();
    highs.stopCallback(kCallbackMipUserSolution);
    double objective_function_value1 = highs.getInfo().objective_function_value;
    double objective_diff =
        std::fabs(objective_function_value1 - objective_function_value0) /
        std::max(1.0, std::fabs(objective_function_value0));
    REQUIRE(objective_diff < 1e-12);
  }
  highs.resetGlobalScheduler(true);
}

TEST_CASE("highs-callback-mip-user-solution", "[highs-callback]") {
  runMipUserSolutionTest(userkMipUserSolution);
}

TEST_CASE("highs-callback-mip-user-set-solution", "[highs-callback]") {
  runMipUserSolutionTest(userkMipUserSetSolution);
}

TEST_CASE("highs-callback-mip-user-set-partial-solution", "[highs-callback]") {
  runMipUserSolutionTest(userkMipUserSetPartialSolution);
}

// HighsCCallbackType
static void cstyle_userCallback(const int callback_type, const char* message,
                                const HighsCallbackDataOut* data_out,
                                HighsCallbackDataIn* data_in,
                                void* user_callback_data) {
  HighsCallbackType cbType = static_cast<HighsCallbackType>(callback_type);
  HighsCallbackOutput* cbOut =
      static_cast<HighsCallbackOutput*>(data_out->cbdata);

  assert(user_callback_data != nullptr);
  double local_callback_data = *static_cast<double*>(user_callback_data);

  if (dev_run) {
    switch (cbType) {
      case HighsCallbackType::kCallbackLogging:
        printf("userCallback(%11.4g): %s\n", local_callback_data, message);
        break;

      case HighsCallbackType::kCallbackMipImprovingSolution:
        printf("userCallback(%11.4g): improving solution with objective = %g\n",
               local_callback_data, cbOut->objective_function_value);
        printf("userCallback(%11.4g): improving solution with value[0] = %g\n",
               local_callback_data, cbOut->mip_solution[0]);
        break;

      case HighsCallbackType::kCallbackMipLogging:
        printf("userCallback(%11.4g): MIP logging\n", local_callback_data);
        break;

      case HighsCallbackType::kCallbackMipInterrupt:
        printf("userCallback(%11.4g): MIP interrupt\n", local_callback_data);
        break;
      default:
        break;
    }
  }

  if (cbType == HighsCallbackType::kCallbackMipLogging ||
      cbType == HighsCallbackType::kCallbackMipInterrupt) {
    data_in->user_interrupt = true;
  }
}

// test that the c callback also works in c++
TEST_CASE("highs-callback-mip-user-solution-c", "[highs-callback]") {
  std::string filename = std::string(HIGHS_DIR) + "/check/instances/egout.mps";
  Highs highs;
  highs.setOptionValue("output_flag", dev_run);
  highs.setOptionValue("presolve", kHighsOffString);
  highs.readModel(filename);

  double my_callback_data = 5.5;
  highs.setCallback(cstyle_userCallback, static_cast<void*>(&my_callback_data));
  highs.startCallback(kCallbackMipSolution);

  highs.run();
  highs.resetGlobalScheduler(true);
}
