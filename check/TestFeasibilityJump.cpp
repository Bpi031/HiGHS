#include "Highs.h"
#include "catch.hpp"
#include "mip/feasibilityjump.hh"

namespace external_feasibilityjump {

// Test-only access keeps score recomputation independent of the search walk.
struct FeasibilityJumpTestAccess {
  static void init(FeasibilityJumpSolver& solver, double* start) {
    solver.init(start);
  }

  static Move move(FeasibilityJumpSolver& solver, uint32_t column) {
    return solver.bestMove(column);
  }

  static Move rescore(FeasibilityJumpSolver& solver, uint32_t column,
                      double weight) {
    solver.objectiveWeight = weight;
    solver.resetMoves(column);
    return move(solver, column);
  }

  static void bumpWeights(FeasibilityJumpSolver& solver) {
    solver.updateWeights();
  }

  static double objectiveWeight(const FeasibilityJumpSolver& solver) {
    return solver.objectiveWeight;
  }

  static bool good(const FeasibilityJumpSolver& solver, uint32_t column) {
    return solver.goodVarsSetIdx[column] != -1;
  }

  static void largeIncrement(FeasibilityJumpSolver& solver) {
    solver.weightUpdateIncrement = 2.0e20;
  }
};

}  // namespace external_feasibilityjump

namespace {

using namespace external_feasibilityjump;

constexpr double kFjTolerance = 1e-5;

void addRow(Problem& problem, RowType sense, double rhs, double coefficient,
            double weight = 1.0) {
  int column = 0;
  const int row =
      problem.addConstraint(sense, rhs, 1, &column, &coefficient, 0);
  REQUIRE(row >= 0);
  REQUIRE(row < static_cast<int>(problem.constraints.size()));
  problem.constraints[row].weight = weight;
}

double jumpValue(Problem& problem, double start) {
  problem.resetIncumbent(&start);
  JumpMove jump(kFjTolerance);
  jump.init(problem);
  jump.updateValue(problem, 0);
  double value = NAN;
  jump.forEachVarMove(0, [&](Move& move) { value = move.value; });
  return value;
}

double weightedViolation(double value, double lowerWeight,
                          double upperWeight) {
  return lowerWeight * std::max(0.0, 2.0 - value) +
         upperWeight * std::max(0.0, value - 1.0);
}

RowType reverseSense(RowType sense) {
  if (sense == Lte) return Gte;
  if (sense == Gte) return Lte;
  return Equal;
}

}  // namespace

TEST_CASE("FJ objective scores reward cost reduction, not cost increase",
          "[highs_feasibility_jump]") {
  Highs highs;
  REQUIRE(highs.setOptionValue("output_flag", false) == HighsStatus::kOk);
  for (const double cost : {-3.0, 0.0, 3.0}) {
    for (double start : {0.0, 1.0}) {
      for (const double weight : {0.5, 1.0, 3.0}) {
        CAPTURE(cost, start, weight);
        FeasibilityJumpSolver solver(highs.getOptions().log_options);
        solver.addVar(Integer, 0, 1, cost);
        FeasibilityJumpTestAccess::init(solver, &start);
        const Move move = FeasibilityJumpTestAccess::rescore(solver, 0, weight);
        REQUIRE(move.value == 1.0 - start);
        const double costChange = cost * (move.value - start);
        REQUIRE(move.score == Approx(-weight * costChange));
        REQUIRE(FeasibilityJumpTestAccess::good(solver, 0) ==
                (costChange < 0.0));
      }
    }
  }
}

TEST_CASE("FJ objective weight updates match full rescoring and good moves",
          "[highs_feasibility_jump]") {
  Highs highs;
  REQUIRE(highs.setOptionValue("output_flag", false) == HighsStatus::kOk);
  for (const double cost : {-3.0, 0.0, 3.0}) {
    for (double start : {0.0, 1.0}) {
      FeasibilityJumpSolver solver(highs.getOptions().log_options);
      solver.addVar(Integer, 0, 1, cost);
      FeasibilityJumpTestAccess::init(solver, &start);
      for (int bump = 1; bump <= 4; ++bump) {
        CAPTURE(cost, start, bump);
        FeasibilityJumpTestAccess::bumpWeights(solver);
        const Move incremental = FeasibilityJumpTestAccess::move(solver, 0);
        const double weight =
            FeasibilityJumpTestAccess::objectiveWeight(solver);
        REQUIRE(weight == bump);
        REQUIRE(incremental.score ==
                Approx(-weight * cost * (incremental.value - start)));
        REQUIRE(FeasibilityJumpTestAccess::good(solver, 0) ==
                (incremental.score > 0.0));
        const Move recomputed =
            FeasibilityJumpTestAccess::rescore(solver, 0, weight);
        REQUIRE(incremental.value == recomputed.value);
        REQUIRE(incremental.score == Approx(recomputed.score));
      }
    }
  }
}

TEST_CASE("FJ objective weight rescaling preserves cost direction",
          "[highs_feasibility_jump]") {
  Highs highs;
  REQUIRE(highs.setOptionValue("output_flag", false) == HighsStatus::kOk);
  for (const double cost : {-1.0, 1.0}) {
    double start = 0;
    FeasibilityJumpSolver solver(highs.getOptions().log_options);
    solver.addVar(Integer, 0, 1, cost);
    FeasibilityJumpTestAccess::init(solver, &start);
    FeasibilityJumpTestAccess::largeIncrement(solver);
    FeasibilityJumpTestAccess::bumpWeights(solver);
    REQUIRE(FeasibilityJumpTestAccess::objectiveWeight(solver) == Approx(2.0));
    REQUIRE(FeasibilityJumpTestAccess::move(solver, 0).score ==
            Approx(-2.0 * cost));
    REQUIRE(FeasibilityJumpTestAccess::good(solver, 0) == (cost < 0.0));
  }
}

TEST_CASE("FJ signed rows choose the least weighted violation",
          "[highs_feasibility_jump]") {
  // These rows deliberately conflict: x >= 2 and x <= 1. Enumerate x in
  // [0,4] to check weighted repair quality without claiming feasibility.
  for (const bool reverseLower : {false, true}) {
    for (const bool reverseUpper : {false, true}) {
      for (const double lowerWeight : {1.0, 2.0, 3.0}) {
        for (const double upperWeight : {1.0, 2.0, 3.0}) {
          for (const double start : {0.0, 4.0}) {
            CAPTURE(reverseLower, reverseUpper, lowerWeight, upperWeight,
                    start);
            Problem problem(kFjTolerance, kFjTolerance);
            problem.addVar(Integer, 0, 4, 0);
            addRow(problem, reverseLower ? Lte : Gte,
                   reverseLower ? -2 : 2, reverseLower ? -1 : 1, lowerWeight);
            addRow(problem, reverseUpper ? Gte : Lte,
                   reverseUpper ? -1 : 1, reverseUpper ? -1 : 1, upperWeight);
            const double value = jumpValue(problem, start);
            REQUIRE(std::isfinite(value));
            REQUIRE(value >= 0);
            REQUIRE(value <= 4);
            REQUIRE(value == std::round(value));
            double minimum = std::numeric_limits<double>::infinity();
            for (int candidate = 0; candidate <= 4; ++candidate)
              minimum = std::min(
                  minimum, weightedViolation(candidate, lowerWeight,
                                             upperWeight));
            REQUIRE(weightedViolation(value, lowerWeight, upperWeight) ==
                    Approx(minimum));
          }
        }
      }
    }
  }
}

TEST_CASE("FJ row polarity preserves jumps before integer rounding",
          "[highs_feasibility_jump]") {
  for (const VarType type : {Integer, Continuous}) {
    for (const RowType sense : {Lte, Gte, Equal}) {
      for (const double coefficient : {0.5, 1.0, 2.0}) {
        for (const double rhs : {1.0, 2.5, 5.0}) {
          for (const double start : {0.0, 4.0}) {
            CAPTURE(type, sense, coefficient, rhs, start);
            Problem positive(kFjTolerance, kFjTolerance);
            Problem negative(kFjTolerance, kFjTolerance);
            positive.addVar(type, 0, 4, 0);
            negative.addVar(type, 0, 4, 0);
            addRow(positive, sense, rhs, coefficient);
            addRow(negative, reverseSense(sense), -rhs, -coefficient);
            REQUIRE(jumpValue(positive, start) ==
                    Approx(jumpValue(negative, start)));
          }
        }
      }
    }
  }
  Problem integer(kFjTolerance, kFjTolerance);
  integer.addVar(Integer, 0, 4, 0);
  addRow(integer, Lte, -5, -2);
  REQUIRE(jumpValue(integer, 0) == 3);

  Problem upper(kFjTolerance, kFjTolerance);
  upper.addVar(Integer, -std::numeric_limits<double>::infinity(), 4, 0);
  addRow(upper, Gte, -5, -2);
  REQUIRE(jumpValue(upper, 4) == 2);
}

TEST_CASE("FJ negative rows respect unbounded and singleton domains",
          "[highs_feasibility_jump]") {
  const double infinity = std::numeric_limits<double>::infinity();
  Problem lower(kFjTolerance, kFjTolerance);
  lower.addVar(Integer, 0, infinity, 0);
  addRow(lower, Lte, -2, -1);
  REQUIRE(jumpValue(lower, 0) == 2);

  Problem upper(kFjTolerance, kFjTolerance);
  upper.addVar(Integer, -infinity, 4, 0);
  addRow(upper, Gte, -1, -1);
  REQUIRE(jumpValue(upper, 4) == 1);

  Problem fixed(kFjTolerance, kFjTolerance);
  fixed.addVar(Integer, 2, 2, 0);
  addRow(fixed, Lte, -2, -1);
  REQUIRE(jumpValue(fixed, 2) == 2);
}

TEST_CASE("FJ public search finds and reports a signed-row feasible point",
          "[highs_feasibility_jump]") {
  Highs highs;
  REQUIRE(highs.setOptionValue("output_flag", false) == HighsStatus::kOk);
  for (const int seed : {0, 42, 43}) {
    FeasibilityJumpSolver solver(highs.getOptions().log_options, seed);
    solver.addVar(Integer, 0, 4, 1);
    int column = 0;
    double coefficient = -1;
    solver.addConstraint(Lte, -2, 1, &column, &coefficient, 0);
    double start = 0;
    int solutions = 0;
    solver.solve(&start, [&](FJStatus status) {
      if (!status.solution)
        return status.totalEffort > 100 ? Terminate : Continue;
      ++solutions;
      REQUIRE(status.numVars == 1);
      REQUIRE(status.solution[0] >= 2);
      REQUIRE(status.solution[0] <= 4);
      REQUIRE(status.solution[0] == std::round(status.solution[0]));
      REQUIRE(status.solutionObjectiveValue == status.solution[0]);
      REQUIRE(status.solution[0] == 2);
      return Terminate;
    });
    REQUIRE(solutions == 1);
  }
}

TEST_CASE("FJ tiny MIP min and max agree with independent enumeration",
          "[highs_feasibility_jump]") {
  Highs::resetGlobalScheduler(true);
  for (const ObjSense sense : {ObjSense::kMinimize, ObjSense::kMaximize}) {
    double optimum = sense == ObjSense::kMinimize ? kHighsInf : -kHighsInf;
    for (int x = 0; x <= 1; ++x) {
      for (int y = 0; y <= 1; ++y) {
        for (int z = 0; z <= 1; ++z) {
          if (x + y < 1 || y + z < 1 || x > z) continue;
          const double objective = 3 * x - 2 * y + z;
          optimum = sense == ObjSense::kMinimize
                        ? std::min(optimum, objective)
                        : std::max(optimum, objective);
        }
      }
    }
    for (const bool enabled : {false, true}) {
      for (const bool presolve : {false, true}) {
        for (const int seed : {42, 43}) {
          CAPTURE(sense, enabled, presolve, seed);
          Highs highs;
          REQUIRE(highs.setOptionValue("output_flag", false) ==
                  HighsStatus::kOk);
          REQUIRE(highs.setOptionValue("threads", 1) == HighsStatus::kOk);
          REQUIRE(highs.setOptionValue("random_seed", seed) ==
                  HighsStatus::kOk);
          REQUIRE(highs.setOptionValue("time_limit", 5.0) == HighsStatus::kOk);
          REQUIRE(highs.setOptionValue("presolve", presolve ? "on" : "off") ==
                  HighsStatus::kOk);
          REQUIRE(highs.setOptionValue("mip_heuristic_run_feasibility_jump",
                                       enabled) == HighsStatus::kOk);
          HighsLp lp;
          lp.num_col_ = 3;
          lp.num_row_ = 3;
          lp.sense_ = sense;
          lp.col_cost_ = {3, -2, 1};
          lp.col_lower_ = {0, 0, 0};
          lp.col_upper_ = {1, 1, 1};
          lp.integrality_.assign(3, HighsVarType::kInteger);
          lp.row_lower_ = {1, -kHighsInf, -kHighsInf};
          lp.row_upper_ = {kHighsInf, -1, 0};
          lp.a_matrix_.format_ = MatrixFormat::kRowwise;
          lp.a_matrix_.start_ = {0, 2, 4, 6};
          lp.a_matrix_.index_ = {0, 1, 1, 2, 0, 2};
          lp.a_matrix_.value_ = {1, 1, -1, -1, 1, -1};
          REQUIRE(highs.passModel(lp) == HighsStatus::kOk);
          REQUIRE(highs.run() == HighsStatus::kOk);
          REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
          REQUIRE(highs.getInfo().objective_function_value == Approx(optimum));
          const auto& solution = highs.getSolution();
          REQUIRE(solution.value_valid);
          REQUIRE(solution.col_value.size() == 3);
          highs.resetGlobalScheduler(true);
          for (const double value : solution.col_value) {
            REQUIRE(value >= -kFjTolerance);
            REQUIRE(value <= 1 + kFjTolerance);
            REQUIRE(value == Approx(std::round(value)).margin(kFjTolerance));
          }
          const double x = solution.col_value[0];
          const double y = solution.col_value[1];
          const double z = solution.col_value[2];
          REQUIRE(x + y >= 1 - kFjTolerance);
          REQUIRE(y + z >= 1 - kFjTolerance);
          REQUIRE(x <= z + kFjTolerance);
          highs.resetGlobalScheduler(true);
        }
      }
    }
  }
}
