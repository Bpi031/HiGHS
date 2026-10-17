#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>

#include "Highs.h"
#include "catch.hpp"
#include "io/HMPSIO.h"
#include "mip/HighsMipSolverData.h"

namespace {
// Deliberately independent of the solver's sparse matrix and objective helpers.
struct TinyIntegerModel {
  std::vector<int> cost;
  std::vector<std::vector<int>> rows;
  std::vector<int> lower, upper;
  int offset = 0;
  bool maximize = false;
  bool binary = false;
};

bool feasible(const TinyIntegerModel& m, const std::vector<int>& x) {
  for (int value : x)
    if (value < (m.binary ? 0 : -1) || value > (m.binary ? 1 : 2)) return false;
  for (size_t i = 0; i < m.rows.size(); ++i) {
    int activity = std::inner_product(m.rows[i].begin(), m.rows[i].end(),
                                      x.begin(), 0);
    if (activity < m.lower[i] || activity > m.upper[i]) return false;
  }
  return true;
}

int score(const TinyIntegerModel& m, const std::vector<int>& x) {
  return m.offset +
         std::inner_product(m.cost.begin(), m.cost.end(), x.begin(), 0);
}

struct ExactAnswer {
  bool exists = false;
  int objective = 0;
  std::vector<int> witness;
};

ExactAnswer enumerate(const TinyIntegerModel& m) {
  ExactAnswer answer;
  const size_t n = m.cost.size();
  for (unsigned mask = 0; mask < (1u << (2 * n)); ++mask) {
    std::vector<int> x(n);
    for (size_t j = 0; j < n; ++j) x[j] = int((mask >> (2 * j)) & 3u) - 1;
    if (!feasible(m, x)) continue;
    const int value = score(m, x);
    if (!answer.exists || (m.maximize ? value > answer.objective
                                      : value < answer.objective)) {
      answer.exists = true;
      answer.objective = value;
      answer.witness = x;
    }
  }
  return answer;
}

TinyIntegerModel generate(uint32_t seed) {
  std::mt19937 rng(seed);
  TinyIntegerModel m;
  m.binary = seed % 3 == 0;
  const size_t n = 3 + rng() % 4;
  std::vector<int> witness(n);
  for (size_t j = 0; j < n; ++j) {
    m.cost.push_back(int(rng() % 15) - 7);
    witness[j] = m.binary ? int(rng() % 2) : int(rng() % 4) - 1;
  }
  m.offset = int(rng() % 21) - 10;
  m.maximize = seed % 2;
  for (int i = 0; i < 8; ++i) {
    std::vector<int> row(n);
    for (int& v : row) v = int(rng() % 7) - 3;
    const int activity = std::inner_product(row.begin(), row.end(),
                                           witness.begin(), 0);
    m.rows.push_back(row);
    m.lower.push_back(activity - int(rng() % 3));
    m.upper.push_back(activity + int(rng() % 3));
  }
  // Include exact infeasibility, not just known-feasible random instances.
  if (seed % 5 == 0) {
    std::vector<int> row(n, 0);
    row[0] = 1;
    m.rows.push_back(row);
    m.lower.push_back(3);
    m.upper.push_back(4);
  }
  return m;
}

HighsLp encode(const TinyIntegerModel& m, int scale = 1) {
  HighsLp lp;
  lp.num_col_ = m.cost.size();
  lp.num_row_ = m.rows.size();
  lp.sense_ = m.maximize ? ObjSense::kMaximize : ObjSense::kMinimize;
  lp.offset_ = m.offset;
  lp.col_cost_.assign(m.cost.begin(), m.cost.end());
  lp.col_lower_.assign(lp.num_col_, m.binary ? 0 : -1);
  lp.col_upper_.assign(lp.num_col_, m.binary ? 1 : 2);
  lp.integrality_.assign(lp.num_col_, HighsVarType::kInteger);
  lp.a_matrix_.format_ = MatrixFormat::kRowwise;
  lp.a_matrix_.start_ = {0};
  for (size_t i = 0; i < m.rows.size(); ++i) {
    lp.row_lower_.push_back(scale * m.lower[i]);
    lp.row_upper_.push_back(scale * m.upper[i]);
    for (size_t j = 0; j < m.cost.size(); ++j) {
      if (!m.rows[i][j]) continue;
      lp.a_matrix_.index_.push_back(j);
      lp.a_matrix_.value_.push_back(scale * m.rows[i][j]);
    }
    lp.a_matrix_.start_.push_back(lp.a_matrix_.value_.size());
  }
  return lp;
}

void configure(Highs& h, int threads = 1) {
  h.resetGlobalScheduler(true);
  REQUIRE(h.setOptionValue("output_flag", false) == HighsStatus::kOk);
  REQUIRE(h.setOptionValue("threads", threads) == HighsStatus::kOk);
  REQUIRE(h.setOptionValue("time_limit", 10.0) == HighsStatus::kOk);
  REQUIRE(h.setOptionValue("mip_rel_gap", 0.0) == HighsStatus::kOk);
  REQUIRE(h.setOptionValue("mip_abs_gap", 0.0) == HighsStatus::kOk);
}

void checkAnswer(Highs& h, const TinyIntegerModel& m, const ExactAnswer& exact) {
  REQUIRE(h.run() == HighsStatus::kOk);
  if (!exact.exists) {
    REQUIRE(h.getModelStatus() == HighsModelStatus::kInfeasible);
    return;
  }
  if (h.getModelStatus() != HighsModelStatus::kOptimal) {
    h.setOptionValue("output_flag", true);
    h.writeModel("robustness-failure.lp");
  }
  REQUIRE(h.getModelStatus() == HighsModelStatus::kOptimal);
  const auto& values = h.getSolution().col_value;
  REQUIRE(h.getSolution().value_valid);
  REQUIRE(values.size() == m.cost.size());
  std::vector<int> x;
  for (double v : values) {
    REQUIRE(std::isfinite(v));
    REQUIRE(v >= -1.000001);
    REQUIRE(v <= 2.000001);
    REQUIRE(std::abs(v - std::round(v)) < 1e-6);
    x.push_back(int(std::round(v)));
  }
  REQUIRE(feasible(m, x));
  REQUIRE(score(m, x) == exact.objective);
  REQUIRE(h.getInfo().objective_function_value ==
          Approx(exact.objective).margin(1e-6));
  HighsInt exponent = 0;
  REQUIRE(h.getOptionValue("user_objective_scale", exponent) == HighsStatus::kOk);
  const double bound = std::ldexp(h.getInfo().mip_dual_bound - m.offset,
                                 -exponent) + m.offset;
  REQUIRE(bound == Approx(exact.objective).margin(1e-6));
}

unsigned boundedEnv(const char* name, unsigned fallback, unsigned maximum) {
  const char* text = std::getenv(name);
  if (!text) return fallback;
  char* end = nullptr;
  unsigned long value = std::strtoul(text, &end, 10);
  REQUIRE(end != text);
  REQUIRE(*end == '\0');
  REQUIRE(value <= maximum);
  return unsigned(value);
}
}  // namespace

TEST_CASE("robustness assertion build contract", "[highs_robustness]") {
#ifdef NDEBUG
  REQUIRE(std::getenv("ROST_REQUIRE_NATIVE_ASSERTIONS") == nullptr);
#else
  REQUIRE(true);
#endif
}

TEST_CASE("tiny integer oracle survives presolve transformations and workers",
          "[highs_robustness][highs_property]") {
  const unsigned first = boundedEnv("ROST_HIGHS_TEST_SEED", 1, 1000000000);
  const unsigned count = boundedEnv("ROST_HIGHS_TEST_CASES", 24, 10000);
  REQUIRE(count > 0);
  for (unsigned k = 0; k < count; ++k) {
    const unsigned seed = first + k;
    CAPTURE(seed);
    const auto original = generate(seed);
    const auto exact = enumerate(original);
    for (int variant = 0; variant < 6; ++variant) {
      CAPTURE(variant);
      auto m = original;
      if (variant == 2) {
        std::reverse(m.cost.begin(), m.cost.end());
        for (auto& row : m.rows) std::reverse(row.begin(), row.end());
        std::reverse(m.rows.begin(), m.rows.end());
        std::reverse(m.lower.begin(), m.lower.end());
        std::reverse(m.upper.begin(), m.upper.end());
      }
      if (variant == 3) {
        m.rows.push_back(m.rows.front());
        m.lower.push_back(m.lower.front());
        m.upper.push_back(m.upper.front());
      }
      Highs h;
      configure(h, variant == 3 || variant == 4 ? 4 : 1);
      REQUIRE(h.setOptionValue("presolve", variant == 1 || variant >= 4 ? "off" : "on") ==
              HighsStatus::kOk);
      if (variant >= 3) {
        REQUIRE(h.setOptionValue("parallel", "on") == HighsStatus::kOk);
        REQUIRE(h.setOptionValue("mip_root_work_budget", true) == HighsStatus::kOk);
        REQUIRE(h.setOptionValue("mip_root_work_max_separation_rounds", 1) ==
                HighsStatus::kOk);
        REQUIRE(h.setOptionValue("user_objective_scale", -2) == HighsStatus::kOk);
        if (variant >= 4)
          REQUIRE(h.setOptionValue("mip_root_work_max_single_helper_time",
                                   variant == 4 ? 0.0 : 0.001) == HighsStatus::kOk);
      }
      REQUIRE(h.passModel(encode(m, variant == 2 ? 8 : 1)) == HighsStatus::kOk);
      checkAnswer(h, m, exact);
      if (exact.exists) {
        const auto warm = h.getSolution();
        REQUIRE(h.clearSolver() == HighsStatus::kOk);
        REQUIRE(h.setSolution(warm) == HighsStatus::kOk);
        checkAnswer(h, m, exact);
      }
      h.resetGlobalScheduler(true);
    }
  }
}

TEST_CASE("independent integer checker rejects corrupted witnesses",
          "[highs_robustness]") {
  TinyIntegerModel m;
  m.cost = {2, 3};
  m.rows = {{1, 1}};
  m.lower = {1};
  m.upper = {1};
  m.offset = 7;
  auto exact = enumerate(m);
  REQUIRE(exact.exists);
  REQUIRE(exact.objective == 8);  // (2,-1), including signed integer domains.
  REQUIRE_FALSE(feasible(m, {0, 0}));
  REQUIRE_FALSE(feasible(m, {3, -2}));
  REQUIRE(score(m, {1, 0}) != exact.objective);
  m.lower[0] = 5;
  m.upper[0] = 6;
  REQUIRE_FALSE(enumerate(m).exists);
}

TEST_CASE("deterministic worker capacity preserves enumerated optima",
          "[highs_robustness][highs_worker_capacity]") {
  for (unsigned seed = 1; seed <= 24; ++seed) {
    const auto model = generate(seed);
    const auto exact = enumerate(model);
    for (int threads : {1, 2, 4, 8, 10}) {
      CAPTURE(seed, threads, model.maximize, exact.exists);
      Highs highs;
      configure(highs, threads);
      REQUIRE(highs.setOptionValue("parallel", "on") == HighsStatus::kOk);
      REQUIRE(highs.setOptionValue("presolve", "off") == HighsStatus::kOk);
      REQUIRE(highs.setOptionValue("mip_incumbent_exchange", "deterministic") ==
              HighsStatus::kOk);
      REQUIRE(highs.passModel(encode(model)) == HighsStatus::kOk);
      checkAnswer(highs, model, exact);
    }
  }
}

TEST_CASE("cold and checked nonoptimal seeds preserve enumerated optima",
          "[highs_robustness][highs_seeded_ramp]") {
  unsigned nonoptimal_seeds = 0;
  unsigned infeasible_models = 0;
  for (unsigned seed = 1; seed <= 36; ++seed) {
    auto model = generate(seed);
    // Leave alternatives for seed improvement; keep contradictory controls.
    if (seed % 5 != 0) {
      model.rows.resize(2);
      model.lower.resize(2);
      model.upper.resize(2);
    }
    const auto exact = enumerate(model);
    auto reverse = model;
    reverse.maximize = !model.maximize;
    const auto worst = enumerate(reverse);
    REQUIRE(worst.exists == exact.exists);
    if (exact.exists) {
      REQUIRE(feasible(model, worst.witness));
      REQUIRE(score(model, worst.witness) == worst.objective);
      nonoptimal_seeds += worst.objective != exact.objective;
    } else {
      ++infeasible_models;
    }
    for (int threads : {1, 2, 4, 8}) {
      for (bool seeded : {false, true}) {
        CAPTURE(seed, threads, seeded, exact.exists, model.maximize);
        Highs highs;
        configure(highs, threads);
        REQUIRE(highs.setOptionValue("parallel", "on") == HighsStatus::kOk);
        REQUIRE(highs.setOptionValue("presolve", "off") == HighsStatus::kOk);
        REQUIRE(highs.setOptionValue("mip_incumbent_exchange", "deterministic") ==
                HighsStatus::kOk);
        REQUIRE(highs.passModel(encode(model)) == HighsStatus::kOk);
        if (seeded && exact.exists) {
          HighsSolution candidate;
          candidate.value_valid = true;
          candidate.col_value.assign(worst.witness.begin(), worst.witness.end());
          REQUIRE(highs.setSolution(candidate) == HighsStatus::kOk);
        }
        checkAnswer(highs, model, exact);
      }
    }
  }
  REQUIRE(nonoptimal_seeds >= 8);
  REQUIRE(infeasible_models >= 7);
}

TEST_CASE("LP engines and convex QP have analytic original-model answers",
          "[highs_robustness]") {
  for (const std::string method : {"simplex", "ipm"}) {
    CAPTURE(method);
    Highs h;
    configure(h);
    HighsLp lp;
    lp.num_col_ = 2;
    lp.num_row_ = 1;
    lp.col_cost_ = {2, 3};
    lp.col_lower_ = {0, 0};
    lp.col_upper_ = {1, 1};
    lp.row_lower_ = {1};
    lp.row_upper_ = {kHighsInf};
    lp.a_matrix_.format_ = MatrixFormat::kRowwise;
    lp.a_matrix_.start_ = {0, 2};
    lp.a_matrix_.index_ = {0, 1};
    lp.a_matrix_.value_ = {1, 1};
    REQUIRE(h.setOptionValue("presolve", "off") == HighsStatus::kOk);
    REQUIRE(h.setOptionValue("solver", method) == HighsStatus::kOk);
    REQUIRE(h.passModel(lp) == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);
    REQUIRE(h.getModelStatus() == HighsModelStatus::kOptimal);
    const auto& s = h.getSolution();
    REQUIRE(s.value_valid);
    REQUIRE(s.dual_valid);
    REQUIRE(s.col_value[0] == Approx(1).margin(1e-6));
    REQUIRE(s.col_value[1] == Approx(0).margin(1e-6));
    REQUIRE(h.getInfo().objective_function_value == Approx(2).margin(1e-6));
    // Stationarity and complementarity, without using getDualObjectiveValue.
    REQUIRE(s.row_dual[0] >= -1e-6);
    for (int j = 0; j < 2; ++j)
      REQUIRE(lp.col_cost_[j] - s.row_dual[0] - s.col_dual[j] ==
              Approx(0).margin(1e-6));
    REQUIRE(s.row_dual[0] * (s.col_value[0] + s.col_value[1] - 1) ==
            Approx(0).margin(1e-6));
    h.resetGlobalScheduler(true);
  }
  Highs h;
  configure(h);
  HighsModel qp;
  qp.lp_.num_col_ = 2;
  qp.lp_.num_row_ = 0;
  qp.lp_.col_cost_ = {-2, -8};
  qp.lp_.col_lower_ = {-10, -10};
  qp.lp_.col_upper_ = {10, 10};
  qp.lp_.a_matrix_.start_ = {0, 0, 0};
  qp.hessian_.dim_ = 2;
  qp.hessian_.format_ = HessianFormat::kTriangular;
  qp.hessian_.start_ = {0, 1, 2};
  qp.hessian_.index_ = {0, 1};
  qp.hessian_.value_ = {2, 4};
  REQUIRE(h.passModel(qp) == HighsStatus::kOk);
  REQUIRE(h.run() == HighsStatus::kOk);
  REQUIRE(h.getModelStatus() == HighsModelStatus::kOptimal);
  const auto& x = h.getSolution().col_value;
  REQUIRE(x[0] == Approx(1).margin(1e-6));
  REQUIRE(x[1] == Approx(2).margin(1e-6));
  REQUIRE(x[0] * x[0] + 2 * x[1] * x[1] - 2 * x[0] - 8 * x[1] ==
          Approx(-9).margin(1e-6));
  h.resetGlobalScheduler(true);
}

TEST_CASE("tolerance envelopes and model lifecycle do not invent proof",
          "[highs_robustness]") {
  for (const double tolerance : {1e-6, 1e-8}) {
    Highs h;
    configure(h);
    REQUIRE(h.setOptionValue("primal_feasibility_tolerance", tolerance) ==
            HighsStatus::kOk);
    REQUIRE(h.addCol(1, 0, 1, 0, nullptr, nullptr) == HighsStatus::kOk);
    for (const double multiplier : {0.25, 4.0}) {
      HighsSolution candidate;
      candidate.value_valid = true;
      candidate.col_value = {1 + multiplier * tolerance};
      REQUIRE(h.setSolution(candidate) == HighsStatus::kOk);
      bool valid, integral, feasible_result;
      const auto status = h.assessPrimalSolution(valid, integral, feasible_result);
      REQUIRE(status != HighsStatus::kError);
      REQUIRE(valid);
      REQUIRE(feasible_result == (multiplier < 1));
    }
    REQUIRE(h.run() == HighsStatus::kOk);
    REQUIRE(h.getModelStatus() == HighsModelStatus::kOptimal);
    REQUIRE(h.changeColBounds(0, 0, kHighsInf) == HighsStatus::kOk);
    REQUIRE(h.changeColCost(0, -1) == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);
    REQUIRE(h.getModelStatus() == HighsModelStatus::kUnbounded);
    REQUIRE(h.clear() == HighsStatus::kOk);
    configure(h);
    REQUIRE(h.addCol(1, 0, 1, 0, nullptr, nullptr) == HighsStatus::kOk);
    HighsInt col = 0;
    double value = 1;
    REQUIRE(h.addRow(2, kHighsInf, 1, &col, &value) == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);
    REQUIRE(h.getModelStatus() == HighsModelStatus::kInfeasible);
    h.resetGlobalScheduler(true);
  }
}

TEST_CASE("mixed integer completion cannot inherit the relaxation objective",
          "[highs_robustness]") {
  Highs h;
  configure(h);
  HighsLp lp;
  lp.num_col_ = 2;
  lp.num_row_ = 1;
  lp.col_cost_ = {1, 10};
  lp.col_lower_ = {0, 0};
  lp.col_upper_ = {1, 1};
  lp.integrality_ = {HighsVarType::kInteger, HighsVarType::kContinuous};
  lp.row_lower_ = {0.5};
  lp.row_upper_ = {kHighsInf};
  lp.a_matrix_.format_ = MatrixFormat::kRowwise;
  lp.a_matrix_.start_ = {0, 2};
  lp.a_matrix_.index_ = {0, 1};
  lp.a_matrix_.value_ = {1, 1};
  REQUIRE(h.passModel(lp) == HighsStatus::kOk);
  REQUIRE(h.run() == HighsStatus::kOk);
  REQUIRE(h.getModelStatus() == HighsModelStatus::kOptimal);
  REQUIRE(h.getInfo().objective_function_value == Approx(1));
  // Fixed x=0 requires positive continuous slack, rather than reusing zero.
  REQUIRE(h.changeColBounds(0, 0, 0) == HighsStatus::kOk);
  REQUIRE(h.run() == HighsStatus::kOk);
  REQUIRE(h.getModelStatus() == HighsModelStatus::kOptimal);
  REQUIRE(h.getSolution().col_value[1] == Approx(0.5));
  REQUIRE(h.getInfo().objective_function_value == Approx(5));
  REQUIRE(h.clearIntegrality() == HighsStatus::kOk);
  REQUIRE(h.changeColBounds(0, 0, 1) == HighsStatus::kOk);
  REQUIRE(h.run() == HighsStatus::kOk);
  REQUIRE(h.getModelStatus() == HighsModelStatus::kOptimal);
  REQUIRE(h.getInfo().objective_function_value == Approx(0.5));
  h.resetGlobalScheduler(true);
}

TEST_CASE("domain assignment preserves infeasible rollback position",
          "[highs_robustness]") {
  Highs owner;
  configure(owner);
  REQUIRE(owner.addCol(2, 0, 4, 0, nullptr, nullptr) == HighsStatus::kOk);
  REQUIRE(owner.changeColIntegrality(0, HighsVarType::kInteger) == HighsStatus::kOk);
  HighsCallback callback(&owner);
  HighsSolution solution;
  HighsMipSolver mip(callback, owner.getOptions(), owner.getLp(), solution);
  mip.mipdata_.reset(new HighsMipSolverData(mip));
  auto& data = *mip.mipdata_;
  data.feastol = 1e-6;
  data.getDomain().setupObjectivePropagation();
  HighsDomain source(data.getDomain());
  source.changeBound(HighsBoundType::kLower, 0, 1);
  source.markInfeasible();
  HighsDomain copied(source);
  HighsDomain assigned(data.getDomain());
  assigned.markInfeasible();  // Deliberately a different rollback position.
  assigned = source;
  copied.backtrackToGlobal();
  assigned.backtrackToGlobal();
  REQUIRE_FALSE(copied.infeasible());
  REQUIRE_FALSE(assigned.infeasible());
  REQUIRE(assigned.col_lower_ == copied.col_lower_);
  REQUIRE(copied.getObjectiveLowerBound() == 0);
  REQUIRE(assigned.getObjectiveLowerBound() == copied.getObjectiveLowerBound());
}

TEST_CASE("truncated MPS rows return a parser error without aborting",
          "[highs_robustness]") {
  const char* filename = "robustness-truncated-rows.mps";
  for (const char* text : {"ROWS", "NAME truncated\nROWS\n",
                           "NAME truncated\nROWS\n N obj\n L capacity\n"}) {
    {
      std::ofstream file(filename);
      REQUIRE(file.good());
      file << text;
      REQUIRE(file.good());
    }
    Highs h;
    configure(h);
    const auto status = h.readModel(filename);
    std::remove(filename);
    REQUIRE(status == HighsStatus::kError);
    REQUIRE(h.getNumCol() == 0);
  }
}

TEST_CASE("domain stack reset cannot discard an infeasibility proof",
          "[highs_robustness][highs_domain_rollback]") {
  for (bool branching : {false, true}) {
    CAPTURE(branching);
    Highs owner;
    configure(owner);
    REQUIRE(owner.addCol(2, 0, 4, 0, nullptr, nullptr) == HighsStatus::kOk);
    REQUIRE(owner.changeColIntegrality(0, HighsVarType::kInteger) == HighsStatus::kOk);
    HighsCallback callback(&owner);
    HighsSolution solution;
    HighsMipSolver mip(callback, owner.getOptions(), owner.getLp(), solution);
    mip.mipdata_.reset(new HighsMipSolverData(mip));
    auto& data = *mip.mipdata_;
    data.feastol = 1e-6;
    data.upper_limit = 1;
    data.getDomain().setupObjectivePropagation();
    HighsDomain worker(data.getDomain());
    // The new global bound cannot beat the incumbent. Activity updates roll
    // back at the infeasibility position, but the bound remains on the stack.
    worker.changeBound(HighsBoundType::kLower, 0, 1);
    REQUIRE(worker.infeasible());
    const auto changes = worker.getDomainChangeStack().size();
    if (branching)
      worker.setDomainChangeStack({}, {});
    else
      worker.setDomainChangeStack({});
    REQUIRE(worker.infeasible());
    REQUIRE(worker.getDomainChangeStack().size() == changes);
    worker.backtrackToGlobal();
    REQUIRE_FALSE(worker.infeasible());
    REQUIRE(worker.col_lower_[0] == 0);
    REQUIRE(worker.getObjectiveLowerBound() == 0);
  }
}

TEST_CASE("empty matrices still check constant row feasibility",
          "[highs_robustness]") {
  for (bool integer : {false, true}) {
    for (const char* solver : {"choose", "ipm"}) {
      for (double lower : {0.0, 1.0}) {
        CAPTURE(integer, solver, lower);
        Highs h;
        configure(h);
        REQUIRE(h.setOptionValue("solver", solver) == HighsStatus::kOk);
        REQUIRE(h.addCol(2, 0, 4, 0, nullptr, nullptr) == HighsStatus::kOk);
        if (integer)
          REQUIRE(h.changeColIntegrality(0, HighsVarType::kInteger) == HighsStatus::kOk);
        REQUIRE(h.addRow(lower, kHighsInf, 0, nullptr, nullptr) == HighsStatus::kOk);
        REQUIRE(h.run() == HighsStatus::kOk);
        REQUIRE(h.getModelStatus() == (lower == 0 ? HighsModelStatus::kOptimal
                                                 : HighsModelStatus::kInfeasible));
        if (lower == 0) {
          REQUIRE(h.getInfo().objective_function_value == 0);
          REQUIRE(h.getSolution().col_value[0] == 0);
        }
        h.resetGlobalScheduler(true);
      }
    }
  }
}

TEST_CASE("malformed MPS pairs and numeric tokens fail without aborting",
          "[highs_robustness]") {
  const char* filename = "robustness-malformed-pairs.mps";
  const std::string prefix = "NAME test\nROWS\n N obj\n G row\nCOLUMNS\n";
  for (const std::string& body : {
           std::string(" x obj 1 row 1 junk\nRHS\n rhs row 1\n"),
           std::string(" x obj 1 row\nRHS\n rhs row 1\n"),
           std::string(" x obj 1 row 1\nRHS\n rhs row 1 obj 0 junk\n"),
           std::string(" x obj 1 row 1\nRHS\n rhs row 1 obj\n"),
           std::string(" x obj rubbish row 1\nRHS\n rhs row 1\n"),
           std::string(" x obj 1suffix row 1\nRHS\n rhs row 1\n"),
           std::string(" x obj nan row 1\nRHS\n rhs row 1\n")}) {
    CAPTURE(body);
    {
      std::ofstream file(filename);
      file << prefix << body << "ENDATA\n";
      REQUIRE(file.good());
    }
    Highs h;
    configure(h);
    const auto status = h.readModel(filename);
    std::remove(filename);
    REQUIRE(status == HighsStatus::kError);
    REQUIRE(h.getNumCol() == 0);
  }
  // Valid Fortran exponents and integer bounds beyond HighsInt must still read.
  {
    std::ofstream file(filename);
    file << prefix << " x obj 1D+2 row 1\nRHS\n rhs row 1\n"
                      "BOUNDS\n UI bound x 1e30\nENDATA\n";
    REQUIRE(file.good());
  }
  Highs h;
  configure(h);
  const auto status = h.readModel(filename);
  std::remove(filename);
  REQUIRE(status != HighsStatus::kError);
  REQUIRE(h.getLp().col_cost_[0] == 100);
  REQUIRE(h.run() != HighsStatus::kError);
  REQUIRE(h.getModelStatus() == HighsModelStatus::kOptimal);
  REQUIRE(h.getInfo().objective_function_value == 100);
}

TEST_CASE("integer bounds need not fit the reduced cost cache index type",
          "[highs_robustness]") {
  for (double limit : {kHighsInf, 1e12}) {
    for (double sign : {-1.0, 1.0}) {
      CAPTURE(limit, sign);
      Highs h;
      configure(h);
      REQUIRE(h.setOptionValue("presolve", "off") == HighsStatus::kOk);
      REQUIRE(h.addCol(sign, sign > 0 ? 0 : -limit,
                       sign > 0 ? limit : 0, 0, nullptr, nullptr) == HighsStatus::kOk);
      REQUIRE(h.changeColIntegrality(0, HighsVarType::kInteger) == HighsStatus::kOk);
      HighsInt column = 0;
      REQUIRE(h.addRow(1.5, kHighsInf, 1, &column, &sign) == HighsStatus::kOk);
      REQUIRE(h.run() == HighsStatus::kOk);
      REQUIRE(h.getModelStatus() == HighsModelStatus::kOptimal);
      REQUIRE(h.getInfo().objective_function_value == Approx(2));
      REQUIRE(h.getSolution().col_value[0] == Approx(2 * sign));
    }
  }
}

TEST_CASE("info move binds records to the destination object",
          "[highs_robustness][highs_info_lifetime]") {
  Highs h;
  configure(h);
  HighsInfo source;
  REQUIRE_FALSE(source.valid);
  REQUIRE_FALSE(h.getInfo().valid);
  source.valid = true;
  source.objective_function_value = 11;
  HighsInfo moved(std::move(source));
  moved.objective_function_value = 22;
  double value = 0;
  REQUIRE(getLocalInfoValue(h.getOptions().log_options, "objective_function_value",
                           moved.valid, moved.records, value) == InfoStatus::kOk);
  REQUIRE(value == 22);
  REQUIRE(source.objective_function_value == 11);
  HighsInfo copied(moved);
  copied.objective_function_value = 33;
  REQUIRE(getLocalInfoValue(h.getOptions().log_options, "objective_function_value",
                           copied.valid, copied.records, value) == InfoStatus::kOk);
  REQUIRE(value == 33);
}

TEST_CASE("fixed MPS lines are bounded across blanks and malformed markers",
          "[highs_robustness]") {
  std::string missing_second_value = "    x         obj       1";
  missing_second_value.resize(39, ' ');
  missing_second_value += "other\n";
  for (const std::string& prefix : {std::string("\n"), std::string("   \n"),
                                    std::string("\r\n\t\n")}) {
    std::istringstream file(prefix + "    x         obj       1\n");
    char line[128], flag[2] = {0, 0};
    double data[3] = {};
    HighsVarType type = HighsVarType::kContinuous;
    REQUIRE(load_mpsLine(file, type, 128, line, flag, data));
    REQUIRE(data[0] == 1);
    REQUIRE_FALSE(file.bad());
  }
  for (const std::string& text : {
           std::string("  ") + char(0) + "  x         obj       1\n",
           std::string(200, 'x'),
           missing_second_value,
           std::string("    MARK0000  'MARKER'\n"),
           std::string("    MARK0000  'MARKER'                 'INTBAD'\n")}) {
    CAPTURE(text);
    std::istringstream file(text);
    char line[128], flag[2] = {0, 0};
    double data[3] = {};
    HighsVarType type = HighsVarType::kContinuous;
    REQUIRE_FALSE(load_mpsLine(file, type, 128, line, flag, data));
    REQUIRE(file.bad());
  }
  for (const auto& marker : {std::make_pair("INTORG", HighsVarType::kInteger),
                              std::make_pair("INTEND", HighsVarType::kContinuous)}) {
    std::istringstream file(std::string("    MARK0000  'MARKER'                 '") +
                            marker.first + "'\n    x         obj       1\n");
    char line[128], flag[2] = {0, 0};
    double data[3] = {};
    HighsVarType type = HighsVarType::kContinuous;
    REQUIRE(load_mpsLine(file, type, 128, line, flag, data));
    REQUIRE(type == marker.second);
  }
}

TEST_CASE("fixed MPS parse failure bypasses success-only metadata checks",
          "[highs_robustness]") {
  const char* filename = "robustness-fixed-rejection.mps";
  for (const std::string& text : {std::string("NAME bad\nROWS\n L row\nENDATA\n"),
                                  std::string("NAME a") + char(0) + "b\nROWS\n"}) {
    {
      std::ofstream file(filename, std::ios::binary);
      file << text;
      REQUIRE(file.good());
    }
    Highs h;
    configure(h);
    REQUIRE(h.setOptionValue("mps_parser_type_free", false) == HighsStatus::kOk);
    REQUIRE(h.readModel(filename) == HighsStatus::kError);
    REQUIRE(h.getNumCol() == 0);
    std::remove(filename);
  }
}

namespace {
struct RunIoFiles {
  const std::string basis = "robustness-run-io.basis";
  const std::string solution = "robustness-run-io.sol";
  const std::string model = "robustness-run-io.lp";
  ~RunIoFiles() {
    std::remove(basis.c_str());
    std::remove(solution.c_str());
    std::remove(model.c_str());
  }
};

void configureRunIoLp(Highs& h) {
  configure(h);
  REQUIRE(h.setOptionValue("presolve", "off") == HighsStatus::kOk);
  REQUIRE(h.setOptionValue("solver", "simplex") == HighsStatus::kOk);
  REQUIRE(h.setOptionValue("simplex_strategy", 1) == HighsStatus::kOk);
  REQUIRE(h.addCol(1, 0, 1, 0, nullptr, nullptr) == HighsStatus::kOk);
  REQUIRE(h.addCol(2, 0, 1, 0, nullptr, nullptr) == HighsStatus::kOk);
  const HighsInt indices[] = {0, 1};
  const double values[] = {1, 1};
  REQUIRE(h.addRow(1, kHighsInf, 2, indices, values) == HighsStatus::kOk);
  REQUIRE(h.passColName(0, "x") == HighsStatus::kOk);
  REQUIRE(h.passColName(1, "y") == HighsStatus::kOk);
  REQUIRE(h.passRowName(0, "cover") == HighsStatus::kOk);
}
}  // namespace

TEST_CASE("Run output files cannot erase a solver resource warning",
          "[highs_robustness][run_io_status]") {
  RunIoFiles files;
  for (int exports = 0; exports < 4; ++exports) {
    CAPTURE(exports);
    Highs h;
    configureRunIoLp(h);
    REQUIRE(h.setOptionValue("simplex_iteration_limit", 0) == HighsStatus::kOk);
    if (exports & 1)
      REQUIRE(h.setOptionValue("solution_file", files.solution) == HighsStatus::kOk);
    if (exports & 2)
      REQUIRE(h.setOptionValue("write_basis_file", files.basis) == HighsStatus::kOk);
    CHECK(h.run() == HighsStatus::kWarning);
    REQUIRE(h.getModelStatus() == HighsModelStatus::kIterationLimit);
    REQUIRE(h.getInfo().simplex_iteration_count == 0);
  }
}

TEST_CASE("Run file operations cannot erase an earlier I/O error",
          "[highs_robustness][run_io_status]") {
  RunIoFiles files;
  Highs solved;
  configureRunIoLp(solved);
  REQUIRE(solved.run() == HighsStatus::kOk);
  REQUIRE(solved.getModelStatus() == HighsModelStatus::kOptimal);
  REQUIRE(solved.getInfo().objective_function_value == Approx(1.0));
  REQUIRE(solved.getSolution().col_value[0] == Approx(1.0));
  REQUIRE(solved.getSolution().col_value[1] == Approx(0.0));
  REQUIRE(solved.writeBasis(files.basis) == HighsStatus::kOk);
  // The parent is a regular file, guaranteeing an invalid child path.
  const std::string unavailable = files.basis + "/unavailable";
  for (int failure = 0; failure < 4; ++failure) {
    CAPTURE(failure);
    Highs h;
    configureRunIoLp(h);
    if (failure == 0) {
      REQUIRE(h.setOptionValue("read_solution_file", unavailable) == HighsStatus::kOk);
      REQUIRE(h.setOptionValue("read_basis_file", files.basis) == HighsStatus::kOk);
    } else if (failure == 1) {
      REQUIRE(h.setOptionValue("read_basis_file", unavailable) == HighsStatus::kOk);
      REQUIRE(h.setOptionValue("write_model_file", files.model) == HighsStatus::kOk);
    } else {
      if (failure == 2) {
        REQUIRE(h.setOptionValue("solution_file", unavailable) == HighsStatus::kOk);
      } else {
        REQUIRE(h.setOptionValue("write_iis_model_file", unavailable + ".mps") ==
                HighsStatus::kOk);
        REQUIRE(h.setOptionValue("solution_file", files.solution) == HighsStatus::kOk);
      }
      REQUIRE(h.setOptionValue("write_basis_file", files.basis) == HighsStatus::kOk);
    }
    CHECK(h.run() == HighsStatus::kError);
  }
}
