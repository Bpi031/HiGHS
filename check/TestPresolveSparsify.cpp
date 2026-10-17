#include <algorithm>
#include <numeric>
#include <random>

#include "Highs.h"
#include "catch.hpp"

namespace {

HighsLp missingColumnModel() {
  HighsLp lp;
  lp.num_col_ = 5;
  lp.num_row_ = 2;
  lp.a_matrix_.num_col_ = lp.num_col_;
  lp.a_matrix_.num_row_ = lp.num_row_;
  lp.col_cost_ = {-3, 2, -1, 4, -2};
  lp.col_lower_.assign(5, 0);
  lp.col_upper_.assign(5, 1);
  lp.row_lower_ = {2, -kHighsInf};
  lp.row_upper_ = {2, 2};
  lp.a_matrix_.format_ = MatrixFormat::kRowwise;
  lp.a_matrix_.start_ = {0, 4, 8};
  lp.a_matrix_.index_ = {0, 1, 2, 3, 1, 2, 3, 4};
  lp.a_matrix_.value_.assign(8, 1);
  return lp;
}

void configureSparsify(Highs& highs) {
  REQUIRE(highs.setOptionValue("output_flag", false) == HighsStatus::kOk);
  REQUIRE(highs.setOptionValue("threads", 1) == HighsStatus::kOk);
  REQUIRE(highs.setOptionValue("lp_presolve_requires_basis_postsolve", false) ==
          HighsStatus::kOk);
  REQUIRE(highs.setOptionValue("presolve_rule_test", kPresolveRuleSparsify) ==
          HighsStatus::kOk);
}

double checkedObjective(const HighsLp& original,
                        const HighsSolution& solution) {
  REQUIRE(solution.value_valid);
  REQUIRE(solution.col_value.size() == size_t(original.num_col_));
  if (solution.dual_valid) {
    REQUIRE(solution.row_dual.size() == size_t(original.num_row_));
    REQUIRE(solution.col_dual.size() == size_t(original.num_col_));
  }
  double objective = original.offset_;
  for (HighsInt col = 0; col < original.num_col_; ++col) {
    const double value = solution.col_value[col];
    REQUIRE(std::isfinite(value));
    REQUIRE(value >= original.col_lower_[col] - 1e-7);
    REQUIRE(value <= original.col_upper_[col] + 1e-7);
    if (!original.integrality_.empty() &&
        original.integrality_[col] == HighsVarType::kInteger)
      REQUIRE(std::abs(value - std::round(value)) <= 1e-7);
    objective += original.col_cost_[col] * value;
  }
  HighsLp rowwise = original;
  rowwise.a_matrix_.num_row_ = original.num_row_;
  rowwise.a_matrix_.num_col_ = original.num_col_;
  rowwise.a_matrix_.ensureRowwise();
  std::vector<long double> aty(original.num_col_, 0.0);
  for (HighsInt row = 0; row < original.num_row_; ++row) {
    long double activity = 0;
    for (HighsInt pos = rowwise.a_matrix_.start_[row];
         pos < rowwise.a_matrix_.start_[row + 1]; ++pos) {
      HighsInt col = rowwise.a_matrix_.index_[pos];
      activity += static_cast<long double>(rowwise.a_matrix_.value_[pos]) *
                  solution.col_value[col];
      if (solution.dual_valid)
        aty[col] += static_cast<long double>(rowwise.a_matrix_.value_[pos]) *
                    solution.row_dual[row];
    }
    REQUIRE(activity >= original.row_lower_[row] - 1e-7);
    REQUIRE(activity <= original.row_upper_[row] + 1e-7);
  }
  if (solution.dual_valid) {
    long double dual_objective = original.offset_;
    for (HighsInt row = 0; row < original.num_row_; ++row) {
      const double dual = solution.row_dual[row];
      REQUIRE(std::isfinite(dual));
      if (dual == 0.0) continue;
      const double bound =
          dual > 0 ? original.row_lower_[row] : original.row_upper_[row];
      REQUIRE(std::isfinite(bound));
      dual_objective += static_cast<long double>(dual) * bound;
    }
    for (HighsInt col = 0; col < original.num_col_; ++col) {
      REQUIRE(original.col_cost_[col] - static_cast<double>(aty[col]) ==
              Approx(solution.col_dual[col]).margin(1e-7));
      const double dual = solution.col_dual[col];
      REQUIRE(std::isfinite(dual));
      dual_objective +=
          static_cast<long double>(dual) *
          (dual > 0 ? original.col_lower_[col] : original.col_upper_[col]);
    }
    REQUIRE(static_cast<double>(dual_objective) ==
            Approx(objective).margin(1e-7));
  }
  return objective;
}

HighsLp permutedModel(const HighsLp& input, unsigned seed) {
  HighsLp lp = input;
  std::mt19937 random(seed);
  std::vector<HighsInt> columns(lp.num_col_);
  std::iota(columns.begin(), columns.end(), 0);
  std::shuffle(columns.begin(), columns.end(), random);
  for (HighsInt col = 0; col < lp.num_col_; ++col) {
    lp.col_cost_[columns[col]] = input.col_cost_[col];
    lp.col_lower_[columns[col]] = input.col_lower_[col];
    lp.col_upper_[columns[col]] = input.col_upper_[col];
    if (!input.integrality_.empty())
      lp.integrality_[columns[col]] = input.integrality_[col];
  }
  lp.a_matrix_.start_.assign(1, 0);
  lp.a_matrix_.index_.clear();
  lp.a_matrix_.value_.clear();
  for (HighsInt row = 0; row < lp.num_row_; ++row) {
    const HighsInt from = seed % 2 ? lp.num_row_ - 1 - row : row;
    lp.row_lower_[row] = input.row_lower_[from];
    lp.row_upper_[row] = input.row_upper_[from];
    for (HighsInt pos = input.a_matrix_.start_[from];
         pos < input.a_matrix_.start_[from + 1]; ++pos) {
      lp.a_matrix_.index_.push_back(columns[input.a_matrix_.index_[pos]]);
      lp.a_matrix_.value_.push_back(input.a_matrix_.value_[pos]);
    }
    lp.a_matrix_.start_.push_back(lp.a_matrix_.value_.size());
  }
  return lp;
}

void scaleRow(HighsLp& lp, HighsInt row, double scale) {
  for (HighsInt pos = lp.a_matrix_.start_[row];
       pos < lp.a_matrix_.start_[row + 1]; ++pos)
    lp.a_matrix_.value_[pos] *= scale;
  lp.row_lower_[row] *= scale;
  lp.row_upper_[row] *= scale;
  if (scale < 0) std::swap(lp.row_lower_[row], lp.row_upper_[row]);
}

double binaryOptimum(const HighsLp& lp) {
  REQUIRE(lp.num_col_ <= 10);
  double best = kHighsInf;
  for (unsigned bits = 0; bits < (1u << lp.num_col_); ++bits) {
    bool feasible = true;
    for (HighsInt row = 0; row < lp.num_row_; ++row) {
      double activity = 0;
      for (HighsInt pos = lp.a_matrix_.start_[row];
           pos < lp.a_matrix_.start_[row + 1]; ++pos)
        activity += lp.a_matrix_.value_[pos] *
                    ((bits >> lp.a_matrix_.index_[pos]) & 1u);
      feasible = feasible && activity >= lp.row_lower_[row] &&
                 activity <= lp.row_upper_[row];
    }
    if (!feasible) continue;
    double value = lp.offset_;
    for (HighsInt col = 0; col < lp.num_col_; ++col)
      value += lp.col_cost_[col] * ((bits >> col) & 1u);
    best = std::min(best, value);
  }
  return best;
}

}  // namespace

TEST_CASE("sparsify-known-missing-column", "[highs_test_presolve_sparsify]") {
  Highs highs;
  configureSparsify(highs);
  REQUIRE(highs.passModel(missingColumnModel()) == HighsStatus::kOk);
  REQUIRE(highs.presolve() == HighsStatus::kOk);
  HighsLp reduced = highs.getPresolvedLp();
  REQUIRE(reduced.num_col_ == 5);
  REQUIRE(reduced.num_row_ == 2);
  REQUIRE(reduced.a_matrix_.numNz() == 6);
  reduced.a_matrix_.ensureRowwise();
  REQUIRE(reduced.a_matrix_.start_ == std::vector<HighsInt>{0, 4, 6});
  REQUIRE(reduced.a_matrix_.index_ == std::vector<HighsInt>{0, 1, 2, 3, 0, 4});
  REQUIRE(reduced.a_matrix_.value_ == std::vector<double>{1, 1, 1, 1, -1, 1});
  REQUIRE(reduced.row_upper_[1] == 0);
  REQUIRE(reduced.row_lower_[1] == -kHighsInf);
  REQUIRE(highs.getPresolveLog().phase[kPresolvePhaseSparsify].call == 1);
  REQUIRE(highs.getPresolveLog().reconciled);
  highs.resetGlobalScheduler(true);
}

TEST_CASE("sparsify-permutation-scale-postsolve",
          "[highs_test_presolve_sparsify]") {
  for (unsigned seed = 0; seed < 12; ++seed) {
    for (double scale : {0.125, 1.0, 8.0}) {
      CAPTURE(seed, scale);
      HighsLp original = missingColumnModel();
      if (seed % 4 == 1) original.row_lower_[1] = 1;
      scaleRow(original, 0, scale);
      scaleRow(original, 1, seed % 3 ? 1.0 : -1.0);
      original = permutedModel(original, seed);
      Highs baseline;
      REQUIRE(baseline.setOptionValue("output_flag", false) ==
              HighsStatus::kOk);
      REQUIRE(baseline.setOptionValue("threads", 1) == HighsStatus::kOk);
      REQUIRE(baseline.setOptionValue("presolve", "off") == HighsStatus::kOk);
      REQUIRE(baseline.passModel(original) == HighsStatus::kOk);
      REQUIRE(baseline.run() == HighsStatus::kOk);
      REQUIRE(baseline.getModelStatus() == HighsModelStatus::kOptimal);
      const double objective =
          checkedObjective(original, baseline.getSolution());
      Highs h;
      configureSparsify(h);
      REQUIRE(h.passModel(original) == HighsStatus::kOk);
      for (int repeat = 0; repeat < 2; ++repeat) {
        REQUIRE(h.presolve() == HighsStatus::kOk);
        REQUIRE(h.getPresolvedLp().a_matrix_.numNz() == 6);
        Highs reduced;
        REQUIRE(reduced.setOptionValue("output_flag", false) ==
                HighsStatus::kOk);
        REQUIRE(reduced.setOptionValue("threads", 1) == HighsStatus::kOk);
        REQUIRE(reduced.setOptionValue("presolve", "off") == HighsStatus::kOk);
        REQUIRE(reduced.passModel(h.getPresolvedLp()) == HighsStatus::kOk);
        REQUIRE(reduced.run() == HighsStatus::kOk);
        REQUIRE(reduced.getModelStatus() == HighsModelStatus::kOptimal);
        REQUIRE(h.postsolve(reduced.getSolution()) == HighsStatus::kOk);
        REQUIRE(h.getSolution().dual_valid);
        REQUIRE(h.getInfo().num_dual_infeasibilities == 0);
        REQUIRE(checkedObjective(original, h.getSolution()) ==
                Approx(objective).margin(1e-7));
      }
    }
  }
  Highs::resetGlobalScheduler(true);
}

TEST_CASE("sparsify-time-limit-is-not-an-infeasibility-proof",
          "[highs_test_presolve_sparsify]") {
  Highs h;
  configureSparsify(h);
  // Standalone presolve's entry check excludes zero; use an expired positive
  // budget so this tests its timeout status rather than periodic polling.
  REQUIRE(h.setOptionValue("time_limit", 1e-12) == HighsStatus::kOk);
  REQUIRE(h.passModel(missingColumnModel()) == HighsStatus::kOk);
  REQUIRE(h.presolve() == HighsStatus::kWarning);
  REQUIRE(h.getModelPresolveStatus() == HighsPresolveStatus::kTimeout);
  REQUIRE(h.getModelStatus() != HighsModelStatus::kInfeasible);
  REQUIRE(h.clearSolver() == HighsStatus::kOk);
  REQUIRE(h.setOptionValue("time_limit", 5.0) == HighsStatus::kOk);
  REQUIRE(h.presolve() == HighsStatus::kOk);
  REQUIRE(h.getPresolvedLp().a_matrix_.numNz() == 6);
  Highs::resetGlobalScheduler(true);
}

TEST_CASE("sparsify-reduction-limit-postsolve",
          "[highs_test_presolve_sparsify]") {
  HighsLp original = missingColumnModel();
  original.num_col_ = 10;
  original.num_row_ = 4;
  original.a_matrix_.num_col_ = 10;
  original.a_matrix_.num_row_ = 4;
  original.a_matrix_.start_ = {0, 4, 8, 12, 16};
  for (HighsInt pos = 0; pos < 8; ++pos) {
    original.a_matrix_.index_.push_back(original.a_matrix_.index_[pos] + 5);
    original.a_matrix_.value_.push_back(1);
  }
  for (HighsInt col = 0; col < 5; ++col) {
    original.col_cost_.push_back(original.col_cost_[col]);
    original.col_lower_.push_back(0);
    original.col_upper_.push_back(1);
  }
  original.row_lower_ = {2, -kHighsInf, 2, -kHighsInf};
  original.row_upper_ = {2, 2, 2, 2};
  Highs baseline;
  REQUIRE(baseline.setOptionValue("output_flag", false) == HighsStatus::kOk);
  REQUIRE(baseline.setOptionValue("threads", 1) == HighsStatus::kOk);
  REQUIRE(baseline.setOptionValue("presolve", "off") == HighsStatus::kOk);
  REQUIRE(baseline.passModel(original) == HighsStatus::kOk);
  REQUIRE(baseline.run() == HighsStatus::kOk);
  REQUIRE(baseline.getModelStatus() == HighsModelStatus::kOptimal);
  const double optimum = checkedObjective(original, baseline.getSolution());
  for (HighsInt limit : {1, -1}) {
    Highs h;
    configureSparsify(h);
    REQUIRE(h.setOptionValue("presolve_reduction_limit", limit) ==
            HighsStatus::kOk);
    REQUIRE(h.passModel(original) == HighsStatus::kOk);
    REQUIRE(h.presolve() == HighsStatus::kOk);
    REQUIRE(h.getPresolvedLp().a_matrix_.numNz() == (limit == 1 ? 14 : 12));
    Highs reduced;
    REQUIRE(reduced.setOptionValue("output_flag", false) == HighsStatus::kOk);
    REQUIRE(reduced.setOptionValue("threads", 1) == HighsStatus::kOk);
    REQUIRE(reduced.setOptionValue("presolve", "off") == HighsStatus::kOk);
    REQUIRE(reduced.passModel(h.getPresolvedLp()) == HighsStatus::kOk);
    REQUIRE(reduced.run() == HighsStatus::kOk);
    REQUIRE(reduced.getModelStatus() == HighsModelStatus::kOptimal);
    REQUIRE(h.postsolve(reduced.getSolution()) == HighsStatus::kOk);
    REQUIRE(checkedObjective(original, h.getSolution()) ==
            Approx(optimum).margin(1e-7));
  }
  Highs::resetGlobalScheduler(true);
}

TEST_CASE("sparsify-rejects-unsafe-or-unprofitable-patterns",
          "[highs_test_presolve_sparsify]") {
  for (int variant = 0; variant < 4; ++variant) {
    CAPTURE(variant);
    HighsLp lp = missingColumnModel();
    if (variant == 0) {
      // Two missing source entries, not the single permitted fill.
      lp.a_matrix_.index_.erase(lp.a_matrix_.index_.begin() + 4);
      lp.a_matrix_.value_.erase(lp.a_matrix_.value_.begin() + 4);
      --lp.a_matrix_.start_.back();
    } else if (variant == 1) {
      scaleRow(lp, 1, 2048);
    } else if (variant == 2) {
      // Near ratios would leave a small but non-removable coefficient.
      lp.a_matrix_.value_[4] += 1e-7;
    } else {
      // Every scale cancels only one entry: no gain after the one fill.
      lp.a_matrix_.value_[5] = 2;
      lp.a_matrix_.value_[6] = 3;
    }
    Highs h;
    configureSparsify(h);
    REQUIRE(h.passModel(lp) == HighsStatus::kOk);
    REQUIRE(h.presolve() == HighsStatus::kOk);
    REQUIRE(h.getPresolvedLp().a_matrix_.numNz() == lp.a_matrix_.numNz());
  }
}

TEST_CASE("sparsify-respects-basis-and-rule-disable",
          "[highs_test_presolve_sparsify]") {
  for (bool basis_required : {true, false}) {
    Highs h;
    configureSparsify(h);
    if (basis_required)
      REQUIRE(h.setOptionValue("lp_presolve_requires_basis_postsolve", true) ==
              HighsStatus::kOk);
    else
      REQUIRE(h.setOptionValue("presolve_rule_off",
                               HighsInt{1} << kPresolveRuleSparsify) ==
              HighsStatus::kOk);
    REQUIRE(h.passModel(missingColumnModel()) == HighsStatus::kOk);
    REQUIRE(h.presolve() == HighsStatus::kOk);
    REQUIRE(h.getPresolvedLp().a_matrix_.numNz() == 8);
    REQUIRE(h.getPresolveLog().phase[kPresolvePhaseSparsify].call == 0);
  }
}

TEST_CASE("sparsify-small-binary-diversity-oracle",
          "[highs_test_presolve_sparsify]") {
  std::mt19937 random(2601005);
  for (unsigned trial = 0; trial < 64; ++trial) {
    CAPTURE(trial);
    HighsLp lp;
    lp.num_col_ = 5 + trial % 4;
    lp.col_lower_.assign(lp.num_col_, 0);
    lp.col_upper_.assign(lp.num_col_, 1);
    lp.integrality_.assign(lp.num_col_, HighsVarType::kInteger);
    for (HighsInt col = 0; col < lp.num_col_; ++col)
      lp.col_cost_.push_back(int(random() % 19) - 9);
    lp.a_matrix_.format_ = MatrixFormat::kRowwise;
    lp.a_matrix_.start_.assign(1, 0);
    for (HighsInt row = 0; row < 5; ++row) {
      double planted_activity = 0;
      for (HighsInt col = 0; col < lp.num_col_; ++col) {
        if ((row == 0 && col == lp.num_col_ - 1) || (row == 1 && col == 0))
          continue;
        double value = row < 2 ? 1.0 : int(random() % 7) - 3;
        if (!value) continue;
        lp.a_matrix_.index_.push_back(col);
        lp.a_matrix_.value_.push_back(value);
        planted_activity += value * (col % 2);
      }
      lp.a_matrix_.start_.push_back(lp.a_matrix_.value_.size());
      lp.row_lower_.push_back(row == 0 ? planted_activity : -kHighsInf);
      lp.row_upper_.push_back(planted_activity + (row < 2 ? 0 : random() % 3));
    }
    lp.num_row_ = 5;
    lp.a_matrix_.num_col_ = lp.num_col_;
    lp.a_matrix_.num_row_ = lp.num_row_;
    REQUIRE(lp.a_matrix_.start_.size() == size_t(lp.num_row_ + 1));
    REQUIRE(lp.a_matrix_.numNz() == HighsInt(lp.a_matrix_.value_.size()));
    // Include independently provable infeasible cases, not just planted ones.
    if (trial % 16 == 0) lp.row_lower_[0] = lp.row_upper_[0] = lp.num_col_;
    lp = permutedModel(lp, trial);
    const double optimum = binaryOptimum(lp);
    REQUIRE((optimum == kHighsInf) == (trial % 16 == 0));
    for (int mode = 0; mode < 3; ++mode) {
      CAPTURE(mode);
      Highs h;
      REQUIRE(h.setOptionValue("output_flag", false) == HighsStatus::kOk);
      REQUIRE(h.setOptionValue("threads", 1) == HighsStatus::kOk);
      REQUIRE(h.setOptionValue("time_limit", 5.0) == HighsStatus::kOk);
      REQUIRE(h.setOptionValue("mip_rel_gap", 0.0) == HighsStatus::kOk);
      REQUIRE(h.setOptionValue("mip_abs_gap", 0.0) == HighsStatus::kOk);
      if (mode == 0)
        REQUIRE(h.setOptionValue("presolve", "off") == HighsStatus::kOk);
      if (mode == 2)
        REQUIRE(h.setOptionValue("presolve_rule_test", kPresolveRuleSparsify) ==
                HighsStatus::kOk);
      REQUIRE(h.passModel(lp) == HighsStatus::kOk);
      REQUIRE(h.run() == HighsStatus::kOk);
      if (optimum == kHighsInf) {
        REQUIRE(h.getModelStatus() == HighsModelStatus::kInfeasible);
      } else {
        REQUIRE(h.getModelStatus() == HighsModelStatus::kOptimal);
        REQUIRE(checkedObjective(lp, h.getSolution()) ==
                Approx(optimum).margin(1e-7));
        REQUIRE(h.getInfo().objective_function_value ==
                Approx(optimum).margin(1e-7));
      }
    }
  }
  Highs::resetGlobalScheduler(true);
}

TEST_CASE("sparsify-integrality-fill-guards",
          "[highs_test_presolve_sparsify]") {
  for (int variant = 0; variant < 4; ++variant) {
    CAPTURE(variant);
    HighsLp lp = missingColumnModel();
    lp.integrality_.assign(5, HighsVarType::kInteger);
    if (variant == 1) lp.col_upper_[0] = 3;
    if (variant == 2) {
      // Continuous source equality must not fill an integer-bearing target.
      for (HighsInt col = 0; col < 4; ++col)
        lp.integrality_[col] = HighsVarType::kContinuous;
    }
    if (variant == 3) lp.integrality_[0] = HighsVarType::kContinuous;
    Highs h;
    configureSparsify(h);
    REQUIRE(h.passModel(lp) == HighsStatus::kOk);
    REQUIRE(h.presolve() == HighsStatus::kOk);
    REQUIRE(h.getPresolvedLp().a_matrix_.numNz() ==
            ((variant == 1 || variant == 2) ? 8 : 6));
    REQUIRE(h.getPresolveLog().reconciled);
  }
  Highs::resetGlobalScheduler(true);
}
