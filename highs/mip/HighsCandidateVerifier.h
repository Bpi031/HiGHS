#ifndef HIGHS_CANDIDATE_VERIFIER_H_
#define HIGHS_CANDIDATE_VERIFIER_H_

#include <algorithm>
#include <cmath>
#include <vector>
#include "lp_data/HighsLpUtils.h"
#include "util/HighsCDouble.h"

struct HighsCandidateCheck {
  bool feasible = false;
  double bound_violation = kHighsInf;
  double row_violation = kHighsInf;
  double integrality_violation = kHighsInf;
  HighsCDouble objective = kHighsInf;
};

// Pure verification against a validated, stable model snapshot. No repair LP,
// callback, postsolve-stack traversal, incumbent publication or proof mutation.
// Supplied row activities must come from this vector and the same model.
inline HighsCandidateCheck highsVerifyCandidate(
    const HighsLp& lp, const std::vector<double>& columns,
    const std::vector<double>* supplied_rows, double tolerance) {
  HighsCandidateCheck result;
  if (!std::isfinite(tolerance) || tolerance < 0 || lp.num_col_ < 0 ||
      lp.num_row_ < 0 || columns.size() != static_cast<size_t>(lp.num_col_) ||
      (supplied_rows && supplied_rows->size() != static_cast<size_t>(lp.num_row_)) ||
      !std::all_of(columns.begin(), columns.end(),
                   [](double v) { return std::isfinite(v); }))
    return result;
  result.bound_violation = 0;
  result.row_violation = 0;
  result.integrality_violation = 0;
  result.objective = lp.offset_;
  for (HighsInt i = 0; i < lp.num_col_; ++i) {
    const double value = columns[i];
    result.objective += lp.col_cost_[i] * value;
    if (!lp.integrality_.empty() && lp.integrality_[i] == HighsVarType::kInteger)
      result.integrality_violation = std::max(
          result.integrality_violation, std::abs(value - std::round(value)));
    if (value < lp.col_lower_[i] - tolerance)
      result.bound_violation = std::max(result.bound_violation, lp.col_lower_[i] - value);
    else if (value > lp.col_upper_[i] + tolerance)
      result.bound_violation = std::max(result.bound_violation, value - lp.col_upper_[i]);
  }
  if (!std::isfinite(static_cast<double>(result.objective))) return result;
  std::vector<double> computed_rows;
  if (!supplied_rows && lp.num_row_ > 0) {
    // The quad accumulator requires column-major input. Never ignore its
    // error and then index an empty row buffer in a release build.
    if (!lp.a_matrix_.isColwise() ||
        calculateRowValuesQuad(lp, columns, computed_rows) != HighsStatus::kOk ||
        computed_rows.size() != static_cast<size_t>(lp.num_row_))
      return result;
  }
  const auto& rows = supplied_rows ? *supplied_rows : computed_rows;
  for (HighsInt i = 0; i < lp.num_row_; ++i) {
    const double value = rows[i];
    if (!std::isfinite(value)) {
      result.row_violation = kHighsInf;
      return result;
    }
    if (value < lp.row_lower_[i] - tolerance)
      result.row_violation = std::max(result.row_violation, lp.row_lower_[i] - value);
    else if (value > lp.row_upper_[i] + tolerance)
      result.row_violation = std::max(result.row_violation, value - lp.row_upper_[i]);
  }
  result.feasible = result.bound_violation <= tolerance &&
      result.row_violation <= tolerance && result.integrality_violation <= tolerance;
  return result;
}

#endif
