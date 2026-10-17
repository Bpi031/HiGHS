/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file lp_data/HighsCallback.cpp
 * @brief
 */
#include "lp_data/HighsCallback.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <utility>

#include "Highs.h"
#include "lp_data/HighsStatus.h"
#include "lp_data/HighsRefinementBudget.h"
#include "util/HighsIntegers.h"

void HighsCallback::clearHighsCallbackOutput() {
  this->data_out.log_type = HighsLogType::kInfo;
  this->data_out.running_time = -1;
  this->data_out.simplex_iteration_count = -1;
  this->data_out.ipm_iteration_count = -1;
  this->data_out.pdlp_iteration_count = -1;
  this->data_out.objective_function_value = -kHighsInf;
  this->data_out.mip_node_count = -1;
  this->data_out.mip_total_lp_iterations = -1;
  this->data_out.mip_primal_bound = kHighsInf;
  this->data_out.mip_dual_bound = -kHighsInf;
  this->data_out.mip_gap = -1;
  this->data_out.mip_solution.clear();
  this->data_out.mip_solution_source = -1;
  this->data_out.mip_solution_sequence = -1;
  this->data_out.mip_solution_worker_id = -1;
  this->data_out.mip_solution_discovery_node = -1;
  this->data_out.mip_solution_discovery_lp_iterations = -1;
  this->data_out.mip_solution_work_epoch = -1;
  this->data_out.mip_solution_worker_sequence = -1;
  this->data_out.mip_solution_discovery_time = -1.0;
  this->data_out.mip_solution_publication_time = -1.0;
  this->data_out.mip_solution_admission_time = -1.0;
  this->data_out.cutpool_start.clear();
  this->data_out.cutpool_num_col = 0;
  this->data_out.cutpool_num_cut = 0;
  this->data_out.cutpool_index.clear();
  this->data_out.cutpool_value.clear();
  this->data_out.cutpool_lower.clear();
  this->data_out.cutpool_upper.clear();
  this->data_out.external_solution_query_origin =
      ExternalMipSolutionQueryOrigin::kExternalMipSolutionQueryOriginAfterSetup;
}

void HighsCallback::clearHighsCallbackInput() {
  size_t num_col = highs != nullptr ? highs->getNumCol() : 0;

  // make sure buffer size is correct and reset the contents if previously used
  if (this->data_in.user_has_solution ||
      num_col != this->data_in.user_solution.size()) {
    this->data_in.user_solution.assign(num_col, kHighsUndefined);
  }

  this->data_in.user_interrupt = false;
  this->data_in.user_has_solution = false;
  this->data_in.candidate_id = 0;
  this->data_in.candidate_generation = -1;
}

void HighsCallback::clear() {
  this->refinement_budget = nullptr;
  this->user_callback = nullptr;
  this->primal_service_callback = nullptr;
  this->primal_service_data = nullptr;
  this->user_callback_data = nullptr;
  this->active.assign(kNumCallbackType, false);
  this->clearHighsCallbackOutput();
  this->clearHighsCallbackInput();
}

bool HighsCallback::callbackActive(const int callback_type) {
  // Check that callback function has been defined
  if (!this->user_callback) return false;
  // Check that callback_type is within range
  const bool callback_type_ok =
      callback_type >= kCallbackMin && callback_type <= kCallbackMax;
  assert(callback_type_ok);
  if (!callback_type_ok) return false;
  // Don't call callback if it is not active
  assert(this->active.size() > 0);
  if (!this->active[callback_type]) return false;
  return true;
}

bool HighsCallback::callbackAction(const int callback_type,
                                   std::string message) {
  if (!callbackActive(callback_type)) return false;
  this->user_callback(callback_type, message.c_str(), &this->data_out,
                      &this->data_in, this->user_callback_data);
  // Assess any action
  bool action = this->data_in.user_interrupt;

  // Check for no action if case not handled internally
  if (callback_type == kCallbackMipImprovingSolution ||
      callback_type == kCallbackMipSolution ||
      callback_type == kCallbackMipLogging ||
      callback_type == kCallbackMipGetCutPool ||
      callback_type == kCallbackMipDefineLazyConstraints ||
      callback_type == kCallbackMipUserSolution)
    assert(!action);
  return action;
}

// Conversions for C API

// Convert HighsCallbackDataOut to HighsCCallbackDataOut
HighsCallbackOutput::operator HighsCallbackDataOut() const {
  HighsCallbackDataOut data;
  data.cbdata = static_cast<void*>(const_cast<HighsCallbackOutput*>(this));
  data.log_type = static_cast<int>(log_type);
  data.running_time = running_time;
  data.simplex_iteration_count = simplex_iteration_count;
  data.ipm_iteration_count = ipm_iteration_count;
  data.pdlp_iteration_count = pdlp_iteration_count;
  data.objective_function_value = objective_function_value;

  data.mip_node_count = mip_node_count;
  data.mip_total_lp_iterations = mip_total_lp_iterations;
  data.mip_primal_bound = mip_primal_bound;
  data.mip_dual_bound = mip_dual_bound;
  data.mip_gap = mip_gap;
  data.mip_solution_size = mip_solution.size();
  data.mip_solution =
      mip_solution.empty() ? nullptr : const_cast<double*>(mip_solution.data());
  data.mip_solution_source = mip_solution_source;
  data.mip_solution_sequence = mip_solution_sequence;
  data.mip_solution_worker_id = mip_solution_worker_id;
  data.mip_solution_discovery_node = mip_solution_discovery_node;
  data.mip_solution_discovery_lp_iterations =
      mip_solution_discovery_lp_iterations;
  data.mip_solution_work_epoch = mip_solution_work_epoch;
  data.mip_solution_worker_sequence = mip_solution_worker_sequence;
  data.mip_solution_discovery_time = mip_solution_discovery_time;
  data.mip_solution_publication_time = mip_solution_publication_time;
  data.mip_solution_admission_time = mip_solution_admission_time;

  data.cutpool_num_col = cutpool_num_col;
  data.cutpool_num_cut = cutpool_lower.size();
  data.cutpool_num_nz = cutpool_value.size();
  data.cutpool_start = cutpool_start.empty()
                           ? nullptr
                           : const_cast<HighsInt*>(cutpool_start.data());
  data.cutpool_index = cutpool_index.empty()
                           ? nullptr
                           : const_cast<HighsInt*>(cutpool_index.data());
  data.cutpool_value = cutpool_value.empty()
                           ? nullptr
                           : const_cast<double*>(cutpool_value.data());
  data.cutpool_lower = cutpool_lower.empty()
                           ? nullptr
                           : const_cast<double*>(cutpool_lower.data());
  data.cutpool_upper = cutpool_upper.empty()
                           ? nullptr
                           : const_cast<double*>(cutpool_upper.data());

  data.external_solution_query_origin =
      static_cast<HighsInt>(external_solution_query_origin);
  return data;
}

HighsCallbackInput::operator HighsCallbackDataIn() const {
  HighsCallbackDataIn data;
  data.cbdata = static_cast<void*>(const_cast<HighsCallbackInput*>(this));
  data.user_interrupt = user_interrupt ? 1 : 0;
  data.user_has_solution = user_has_solution ? 1 : 0;
  data.user_solution_size = user_solution.size();
  data.user_solution = user_solution.empty()
                           ? nullptr
                           : const_cast<double*>(user_solution.data());
  return data;
}

// we assume that user_solution.data() == data_in.user_solution
// and that user_solution.size() == data_in.user_solution_size
HighsCallbackInput& HighsCallbackInput::operator=(
    const HighsCallbackDataIn& data_in) {
  assert(user_solution.data() == data_in.user_solution);
  user_interrupt = data_in.user_interrupt != 0;
  user_has_solution = data_in.user_has_solution != 0;
  return *this;
}

HighsStatus HighsCallbackInput::setSolution(HighsInt num_entries,
                                            const double* value) {
  if (!highs || num_entries < 0 || (num_entries > 0 && !value))
    return HighsStatus::kError;
  if (num_entries == 0) return HighsStatus::kOk;
  const auto& lp = highs->getLp();
  const auto& options = highs->getOptions();
  if (num_entries > lp.num_col_ ||
      user_solution.size() != static_cast<size_t>(lp.num_col_))
    return HighsStatus::kError;

  // Validate the entire submission before replacing any previous candidate.
  for (HighsInt i = 0; i < num_entries; ++i) {
    if (value[i] != kHighsUndefined &&
        (!std::isfinite(value[i]) ||
         value[i] < lp.col_lower_[i] - options.primal_feasibility_tolerance ||
         value[i] > lp.col_upper_[i] + options.primal_feasibility_tolerance))
      return HighsStatus::kError;
  }
  for (HighsInt i = 0; i < num_entries; ++i) user_solution[i] = value[i];
  user_has_solution = true;
  return HighsStatus::kOk;
}

// User provides a partial solution
HighsStatus HighsCallbackInput::setSolution(HighsInt num_entries,
                                            const HighsInt* index,
                                            const double* value) {
  if (!highs || num_entries < 0 ||
      (num_entries > 0 && (!index || !value)))
    return HighsStatus::kError;
  if (num_entries == 0) {
    // No solution provided, so nothing to do
    return HighsStatus::kOk;
  }

  const auto& options = highs->getOptions();
  const auto& lp = highs->getLp();

  // Warn about duplicates in index
  if (user_solution.size() != static_cast<size_t>(lp.num_col_))
    return HighsStatus::kError;

  HighsStatus return_status = HighsStatus::kOk;
  HighsInt num_duplicates = 0;
  std::vector<bool> is_set(lp.num_col_, false);

  for (HighsInt iX = 0; iX < num_entries; iX++) {
    HighsInt iCol = index[iX];
    if (iCol < 0 || iCol >= lp.num_col_) {
      highsLogUser(options.log_options, HighsLogType::kError,
                   "setSolution: User solution index %d has value %d out of "
                   "range [0, %d)",
                   int(iX), int(iCol), int(lp.num_col_));
      return HighsStatus::kError;
    } else if (value[iX] != kHighsUndefined &&
               (!std::isfinite(value[iX]) ||
                value[iX] < lp.col_lower_[iCol] -
                                options.primal_feasibility_tolerance ||
                lp.col_upper_[iCol] + options.primal_feasibility_tolerance <
                    value[iX])) {
      highsLogUser(options.log_options, HighsLogType::kError,
                   "setSolution: User solution value %d of %g is infeasible "
                   "for bounds [%g, %g]",
                   int(iX), value[iX], lp.col_lower_[iCol],
                   lp.col_upper_[iCol]);
      return HighsStatus::kError;
    }
    if (is_set[iCol]) num_duplicates++;
    is_set[iCol] = true;
  }
  if (num_duplicates > 0) {
    highsLogUser(options.log_options, HighsLogType::kWarning,
                 "setSolution: User set of indices has %d duplicate%s: last "
                 "value used\n",
                 int(num_duplicates), num_duplicates > 1 ? "s" : "");
    return_status = HighsStatus::kWarning;
  }
  // assign solution values to user_solution
  for (HighsInt i = 0; i < num_entries; i++) {
    user_solution[index[i]] = value[i];
  }

  user_has_solution = true;
  return return_status;
}

HighsStatus HighsCallbackInput::refineSolution(
    double seconds, HighsInt nodes, HighsInt lp_iterations,
    bool require_optimal) {
  const bool supplied = user_has_solution;
  user_has_solution = false;
  refinement_status = static_cast<HighsInt>(HighsModelStatus::kNotset);
  refinement_simplex_iterations = 0;
  refinement_ipm_iterations = 0;
  refinement_lp_runs = 0;
  if (!highs || !supplied || !std::isfinite(seconds) || seconds <= 0 ||
      nodes <= 0 || lp_iterations <= 0 ||
      user_solution.size() != static_cast<size_t>(highs->getNumCol()))
    return HighsStatus::kError;

  if (refinement_cancel && refinement_cancel->load()) {
    refinement_status = static_cast<HighsInt>(HighsModelStatus::kInterrupt);
    return HighsStatus::kWarning;
  }

  const auto& parent_options = highs->getOptions();
  seconds = std::min(seconds, parent_options.time_limit - highs->getRunTime());
  if (seconds <= 0) {
    refinement_status = static_cast<HighsInt>(HighsModelStatus::kTimeLimit);
    return HighsStatus::kWarning;
  }
  const auto deadline = std::chrono::steady_clock::now() +
      std::chrono::duration<double>(seconds);
  HighsRefinementBudget budget(lp_iterations, deadline, refinement_cancel);
  const auto& lp = highs->getLp();
  std::vector<HighsInt> fixed_columns;
  std::vector<double> fixed_values;
  fixed_columns.reserve(lp.num_col_);
  fixed_values.reserve(lp.num_col_);
  for (HighsInt col = 0; col < lp.num_col_; ++col) {
    double value = user_solution[col];
    if (value == kHighsUndefined) continue;
    if (!std::isfinite(value)) return HighsStatus::kError;
    const HighsVarType type = lp.integrality_.empty()
                                 ? HighsVarType::kContinuous : lp.integrality_[col];
    if (type != HighsVarType::kContinuous && type != HighsVarType::kInteger)
      return HighsStatus::kError;
    if (type == HighsVarType::kInteger) {
      if (!HighsIntegers::isIntegral(value, parent_options.mip_feasibility_tolerance))
        return HighsStatus::kError;
      value = std::round(value);
    }
    if (value < lp.col_lower_[col] || value > lp.col_upper_[col])
      return HighsStatus::kError;
    fixed_columns.push_back(col);
    fixed_values.push_back(value);
  }
  Highs clone;
  HighsOptions child_options = parent_options;
  // A live callback sees the already user-scaled parent model. Keep those
  // coordinates: applying the parent's scale again can overflow/reject it.
  child_options.user_objective_scale = 0;
  child_options.user_bound_scale = 0;
  child_options.output_flag = false;
  if (clone.passOptions(child_options) == HighsStatus::kError ||
      clone.passModel(highs->getModel()) == HighsStatus::kError)
    return HighsStatus::kError;
  clone.setOptionValue("output_flag", false);
  clone.setOptionValue("parallel", "off");
  clone.setOptionValue("time_limit", seconds);
  clone.setOptionValue("mip_max_nodes", nodes);
  clone.setOptionValue("simplex_iteration_limit", lp_iterations);
  clone.setOptionValue("ipm_iteration_limit", lp_iterations);
  clone.setOptionValue("mip_lp_age_limit", parent_options.mip_lp_age_limit);
  clone.setOptionValue("mip_rel_gap", 0.0);
  clone.setOptionValue("mip_abs_gap", 0.0);
  // Prevent recursive helper work and keep all available work in this model.
  clone.setOptionValue("mip_heuristic_run_rins", false);
  clone.setOptionValue("mip_heuristic_run_rens", false);
  clone.setOptionValue("mip_heuristic_run_root_reduced_cost", false);
  // One invalidation/assessment pass instead of one per fixed assignment.
  if (!fixed_columns.empty() &&
      clone.changeColsBounds(static_cast<HighsInt>(fixed_columns.size()),
          fixed_columns.data(), fixed_values.data(), fixed_values.data()) ==
              HighsStatus::kError)
    return HighsStatus::kError;
  HighsRefinementLpWork direct_lp_work(&budget);
  if (!clone.getLp().isMip()) direct_lp_work.begin();
  clone.setCallback([&](int kind, const std::string&, const HighsCallbackOutput* output,
                        HighsCallbackInput* input, void*) {
    if (kind == kCallbackSimplexInterrupt)
      direct_lp_work.observe(output->simplex_iteration_count, -1);
    else if (kind == kCallbackIpmInterrupt)
      direct_lp_work.observe(-1, output->ipm_iteration_count);
    input->user_interrupt = budget.stopStatus() != HighsModelStatus::kNotset;
  });
  clone.callback_.refinement_budget = &budget;
  for (int kind : {kCallbackMipInterrupt, kCallbackSimplexInterrupt,
                   kCallbackIpmInterrupt})
    clone.startCallback(kind);
  if (refinement_cancel && refinement_cancel->load()) {
    refinement_status = static_cast<HighsInt>(HighsModelStatus::kInterrupt);
    return HighsStatus::kWarning;
  }
  const double remaining = std::chrono::duration<double>(
      deadline - std::chrono::steady_clock::now()).count();
  if (remaining <= 0) {
    refinement_status = static_cast<HighsInt>(HighsModelStatus::kTimeLimit);
    return HighsStatus::kWarning;
  }
  // Include model copying and fixing in the allowance, including presolve
  // intervals where the child does not invoke an interrupt callback.
  clone.setOptionValue("time_limit", remaining);
  const HighsStatus run_status = clone.run();
  if (!clone.getLp().isMip())
    direct_lp_work.observe(clone.getInfo().simplex_iteration_count,
                           clone.getInfo().ipm_iteration_count);
  refinement_status = static_cast<HighsInt>(clone.getModelStatus());
  if ((clone.getModelStatus() == HighsModelStatus::kInterrupt ||
       clone.getModelStatus() == HighsModelStatus::kIterationLimit) &&
      budget.stopStatus() != HighsModelStatus::kNotset)
    refinement_status = static_cast<HighsInt>(budget.stopStatus());
  refinement_simplex_iterations = budget.simplex_iterations;
  refinement_ipm_iterations = budget.ipm_iterations;
  refinement_lp_runs = budget.lp_runs;
  if (run_status == HighsStatus::kError) return HighsStatus::kError;
  const auto& info = clone.getInfo();
  const auto& solution = clone.getSolution();
  if ((require_optimal && clone.getModelStatus() != HighsModelStatus::kOptimal) ||
      info.primal_solution_status != kSolutionStatusFeasible ||
      !solution.value_valid || solution.col_value.size() != user_solution.size() ||
      !std::isfinite(info.max_primal_infeasibility) ||
      info.max_primal_infeasibility > parent_options.mip_feasibility_tolerance ||
      (clone.getLp().isMip() &&
       (!std::isfinite(info.max_integrality_violation) ||
        info.max_integrality_violation > parent_options.mip_feasibility_tolerance)))
    return HighsStatus::kWarning;
  // Preserve the C callback buffer: its pointer must survive callback return.
  std::copy(solution.col_value.begin(), solution.col_value.end(), user_solution.begin());
  user_has_solution = true;
  return HighsStatus::kOk;
}

HighsStatus HighsCallbackInput::repairSolution() {
  const bool supplied = user_has_solution;
  user_has_solution = false;
  if (!highs || !supplied ||
      user_solution.size() != static_cast<size_t>(highs->getNumCol()))
    return HighsStatus::kError;
  const auto started = std::chrono::steady_clock::now();
  const auto& options = highs->getOptions();
  const auto& lp = highs->getLp();
  const double allowance = options.time_limit - highs->getRunTime();
  auto stopped = [&]() {
    return (refinement_cancel && refinement_cancel->load()) ||
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
            >= allowance;
  };
  if (stopped()) return HighsStatus::kError;

  std::vector<HighsInt> columns;
  std::vector<double> values;
  std::vector<HighsInt> semi_columns;
  std::vector<HighsVarType> fixed_types;
  for (HighsInt col = 0; col < lp.num_col_; ++col) {
    double value = user_solution[col];
    if (value == kHighsUndefined) continue;
    if (!std::isfinite(value)) return HighsStatus::kError;
    const auto type = lp.integrality_.empty() ? HighsVarType::kContinuous
                                              : lp.integrality_[col];
    if (type == HighsVarType::kInteger || type == HighsVarType::kSemiInteger) {
      if (!HighsIntegers::isIntegral(value, options.mip_feasibility_tolerance))
        return HighsStatus::kError;
      value = std::round(value);
    }
    // Fixing must restrict the original domain, not silently replace it.
    const bool semi_zero = value == 0 &&
        (type == HighsVarType::kSemiContinuous || type == HighsVarType::kSemiInteger);
    if (!semi_zero && (value < lp.col_lower_[col] || value > lp.col_upper_[col]))
      return HighsStatus::kError;
    columns.push_back(col);
    values.push_back(value);
    if (type == HighsVarType::kSemiContinuous || type == HighsVarType::kSemiInteger) {
      semi_columns.push_back(col);
      fixed_types.push_back(HighsVarType::kContinuous);
    }
  }

  Highs clone;
  HighsOptions child_options = options;
  child_options.user_objective_scale = child_options.user_bound_scale = 0;
  child_options.output_flag = false;
  child_options.parallel = kHighsOffString;
  if (clone.passOptions(child_options) == HighsStatus::kError ||
      clone.passModel(highs->getModel()) == HighsStatus::kError ||
      (!columns.empty() && clone.changeColsBounds(
          static_cast<HighsInt>(columns.size()), columns.data(), values.data(),
          values.data()) == HighsStatus::kError))
    return HighsStatus::kError;
  // A semi-variable still permits zero even when its nonzero interval is
  // fixed. Remove that disjunction only for the explicitly fixed columns.
  if (!semi_columns.empty() && clone.changeColsIntegrality(
          static_cast<HighsInt>(semi_columns.size()), semi_columns.data(),
          fixed_types.data()) == HighsStatus::kError)
    return HighsStatus::kError;
  bool found = false;
  clone.setCallback([&](int kind, const std::string&, const HighsCallbackOutput*,
                        HighsCallbackInput* input, void*) {
    if (kind == kCallbackMipSolution) found = true;
    else input->user_interrupt = found || stopped();
  });
  for (int kind : {kCallbackMipSolution, kCallbackMipInterrupt,
                   kCallbackSimplexInterrupt, kCallbackIpmInterrupt})
    clone.startCallback(kind);
  if (stopped()) return HighsStatus::kError;
  clone.setOptionValue("time_limit", std::max(0.0, allowance -
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()));
  if (clone.run() == HighsStatus::kError) return HighsStatus::kError;
  HighsSolution solution = clone.getSolution();
  HighsLp colwise;
  const HighsLp* verification = &lp;
  if (!lp.a_matrix_.isColwise()) {
    colwise = lp;
    colwise.ensureColwise();
    verification = &colwise;
  }
  if (!solution.value_valid || solution.col_value.size() != user_solution.size() ||
      clone.getInfo().primal_solution_status != kSolutionStatusFeasible ||
      !std::all_of(solution.col_value.begin(), solution.col_value.end(),
                   [](double value) { return std::isfinite(value); }) ||
      calculateRowValuesQuad(*verification, solution) != HighsStatus::kOk ||
      !std::all_of(solution.row_value.begin(), solution.row_value.end(),
                   [](double value) { return std::isfinite(value); }))
    return HighsStatus::kError;
  bool valid = false, integral = false, feasible = false;
  if (assessLpPrimalSolution("callback repair", clone.getOptions(), *verification, solution,
                            valid, integral, feasible) == HighsStatus::kError ||
      !valid || !integral || !feasible)
    return HighsStatus::kError;
  // Keep the C callback's borrowed storage alive across the repair call.
  std::copy(solution.col_value.begin(), solution.col_value.end(), user_solution.begin());
  user_has_solution = true;
  return HighsStatus::kOk;
}
