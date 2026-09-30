/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "mip/HighsMipWorker.h"

#include <algorithm>

#include "mip/HighsMipSolverData.h"
#include "mip/MipTimer.h"

HighsMipWorker::HighsMipWorker(const HighsMipSolver& mipsolver,
                               HighsLpRelaxation* lp, HighsDomain* domain,
                               HighsCutPool* cutpool,
                               HighsConflictPool* conflictpool,
                               HighsPseudocost* pseudocost,
                               HighsInt worker_id)
    : mipsolver_(mipsolver),
      mipdata_(*mipsolver_.mipdata_),
      lp_(lp),
      globaldom_(domain),
      cutpool_(cutpool),
      conflictpool_(conflictpool),
      pseudocost_(pseudocost),
      worker_id_(worker_id),
      randgen(mipsolver.options_mip_->random_seed) {
  upper_bound = mipdata_.upper_bound;
  upper_limit = mipdata_.upper_limit;
  optimality_limit = mipdata_.optimality_limit;
  heuristics_allowed = true;
  HighsNodeLpEffortConfig node_lp_effort_config;
  // Node effort and incumbent exchange are independent mechanisms. Keeping a
  // separate switch is required for identity and interaction experiments.
  node_lp_effort_config.enabled =
      !mipsolver.submip && mipsolver.options_mip_->mip_node_lp_effort;
  node_lp_effort_config.minimum_iterations =
      mipsolver.options_mip_->mip_node_lp_effort_min_iterations;
  node_lp_effort_config.maximum_iterations =
      mipsolver.options_mip_->mip_node_lp_effort_max_iterations;
  node_lp_effort_config.maximum_yields =
      mipsolver.options_mip_->mip_node_lp_effort_max_yields;
  node_lp_effort_.configure(node_lp_effort_config);
  search_ptr_ =
      std::unique_ptr<HighsSearch>(new HighsSearch(*this, getPseudocost()));
  sepa_ptr_ = std::unique_ptr<HighsSeparation>(new HighsSeparation(*this));
  search_ptr_->setLpRelaxation(lp_);
  sepa_ptr_->setLpRelaxation(lp_);
}

HighsInt HighsMipWorker::nodeLpIterationLimit(HighsInt depth,
                                               HighsInt yield_count) const {
  return node_lp_effort_.iterationLimit(
      static_cast<HighsInt>(std::min<int64_t>(
          kHighsIInf, std::max<int64_t>(0, mipdata_.firstrootlpiters))),
      depth, yield_count);
}

const HighsMipSolver& HighsMipWorker::getMipSolver() const {
  return mipsolver_;
}

void HighsMipWorker::resetSearch() {
  search_ptr_.reset();
  search_ptr_ =
      std::unique_ptr<HighsSearch>(new HighsSearch(*this, getPseudocost()));
  search_ptr_->setLpRelaxation(lp_);
}

void HighsMipWorker::resetSepa() {
  sepa_ptr_.reset();
  sepa_ptr_ = std::unique_ptr<HighsSeparation>(new HighsSeparation(*this));
  sepa_ptr_->setLpRelaxation(lp_);
}

bool HighsMipWorker::addIncumbent(const std::vector<double>& sol, double solobj,
                                  int solution_source) {
  if (solobj < upper_bound) {
    // Get the transformed objective and solution if required
    const std::pair<bool, double> transformed_solobj =
        transformNewIntegerFeasibleSolution(sol);
    if (!transformed_solobj.first) return false;
    if (transformed_solobj.second < upper_bound) {
      upper_bound = transformed_solobj.second;
      double new_upper_limit =
          mipdata_.computeNewUpperLimit(upper_bound, 0.0, 0.0);
      if (new_upper_limit < upper_limit) {
        upper_limit = new_upper_limit;
        optimality_limit = mipdata_.computeNewUpperLimit(
            upper_bound, mipsolver_.options_mip_->mip_abs_gap,
            mipsolver_.options_mip_->mip_rel_gap);
      }
    }
    // The master remains the only authority that can admit an incumbent.
    // Preserve every locally interesting candidate in a bounded, worker-owned
    // mailbox until the deterministic synchronization point.
    HighsWorkerCandidate candidate;
    candidate.solution = sol;
    candidate.native_objective = solobj;
    candidate.solution_source = solution_source;
    candidate.metadata.worker_id = worker_id_;
    candidate.metadata.discovery_node =
        mipdata_.num_nodes + (search_ptr_ ? search_ptr_->nnodes : 0);
    candidate.metadata.discovery_lp_iterations =
        mipdata_.total_lp_iterations +
        (search_ptr_ ? search_ptr_->lpiterations : 0);
    candidate.metadata.work_epoch = work_epoch_;
    candidate.metadata.publication_sequence = publication_sequence_++;
    candidate.metadata.discovery_time = mipsolver_.timer_.read();
    candidate.metadata.publication_time = candidate.metadata.discovery_time;
    if (!incumbent_mailbox_.publish(std::move(candidate))) return false;
  }
  return true;
}

std::pair<bool, double> HighsMipWorker::transformNewIntegerFeasibleSolution(
    const std::vector<double>& sol) const {
  HighsSolution solution;
  solution.col_value = sol;
  solution.value_valid = true;

  // Perform primal postsolve to get the original column values
  mipsolver_.mipdata_->postSolveStack.undoPrimal(*mipsolver_.options_mip_,
                                                 solution, -1, true);

  // Determine the row values, as they aren't computed in primal
  // postsolve
  HighsStatus return_status =
      calculateRowValuesQuad(*mipsolver_.orig_model_, solution);
  if (kAllowDeveloperAssert) assert(return_status == HighsStatus::kOk);

  // compute the objective value in the original space
  double bound_violation_ = 0;
  double row_violation_ = 0;
  double integrality_violation_ = 0;

  HighsCDouble mipsolver_quad_objective_value = 0;

  bool feasible = mipsolver_.solutionFeasible(
      mipsolver_.orig_model_, solution.col_value, &solution.row_value,
      bound_violation_, row_violation_, integrality_violation_,
      mipsolver_quad_objective_value);

  const double transformed_solobj = static_cast<double>(
      static_cast<HighsInt>(mipsolver_.orig_model_->sense_) *
          mipsolver_quad_objective_value -
      mipsolver_.model_->offset_);

  return std::make_pair(feasible, transformed_solobj);
}

bool HighsMipWorker::trySolution(const std::vector<double>& solution,
                                 const int solution_source) {
  if (static_cast<int>(solution.size()) != mipsolver_.model_->num_col_)
    return false;

  HighsCDouble obj = 0;

  for (HighsInt i = 0; i != mipsolver_.model_->num_col_; ++i) {
    if (solution[i] < mipsolver_.model_->col_lower_[i] - mipdata_.feastol)
      return false;
    if (solution[i] > mipsolver_.model_->col_upper_[i] + mipdata_.feastol)
      return false;
    if (mipsolver_.variableType(i) == HighsVarType::kInteger &&
        fractionality(solution[i]) > mipdata_.feastol)
      return false;

    obj += mipsolver_.colCost(i) * solution[i];
  }

  for (HighsInt i = 0; i != mipsolver_.model_->num_row_; ++i) {
    double rowactivity = 0.0;

    HighsInt start = mipdata_.ARstart_[i];
    HighsInt end = mipdata_.ARstart_[i + 1];

    for (HighsInt j = start; j != end; ++j)
      rowactivity += solution[mipdata_.ARindex_[j]] * mipdata_.ARvalue_[j];

    if (rowactivity > mipsolver_.rowUpper(i) + mipdata_.feastol) return false;
    if (rowactivity < mipsolver_.rowLower(i) - mipdata_.feastol) return false;
  }

  return addIncumbent(solution, static_cast<double>(obj), solution_source);
}

void HighsMipWorker::resetSepaStats() {
  sepa_stats.numNeighbourhoodQueries = 0;
  sepa_stats.sepa_lp_iterations = 0;
}

void HighsMipWorker::updateHeurStatsLpIters(int64_t lp_iters,
                                            int64_t total_repair_lp,
                                            int64_t total_repair_lp_feasible,
                                            int64_t total_repair_lp_iters) {
  heur_stats.lp_iterations += lp_iters;
  heur_stats.total_repair_lp += total_repair_lp;
  heur_stats.total_repair_lp_feasible += total_repair_lp_feasible;
  heur_stats.total_repair_lp_iterations += total_repair_lp_iters;
}

void HighsMipWorker::updateHeurStatsInfeasObservations(double fixingRate) {
  heur_stats.infeasObservations += fixingRate;
  ++heur_stats.numInfeasObservations;
}

void HighsMipWorker::updateHeurStatsSuccessObservations(double fixingRate) {
  heur_stats.successObservations += fixingRate;
  ++heur_stats.numSuccessObservations;
}

void HighsMipWorker::getHeurStatsValues(
    int64_t& total_repair_lp, int64_t& total_repair_lp_feasible,
    int64_t& total_repair_lp_iterations, int64_t& lp_iterations,
    double& successObservations, HighsInt& numSuccessObservations,
    double& infeasObservations, HighsInt& numInfeasObservations,
    HighsInt& max_submip_level, HighsModelStatus& termination_status) const {
  total_repair_lp = heur_stats.total_repair_lp;
  total_repair_lp_feasible = heur_stats.total_repair_lp_feasible;
  total_repair_lp_iterations = heur_stats.total_repair_lp_iterations;
  lp_iterations = heur_stats.lp_iterations;
  successObservations = heur_stats.successObservations;
  numSuccessObservations = heur_stats.numSuccessObservations;
  infeasObservations = heur_stats.infeasObservations;
  numInfeasObservations = heur_stats.numInfeasObservations;
  max_submip_level = heur_stats.max_submip_level;
  termination_status = heur_stats.termination_status_;
}

void HighsMipWorker::resetHeurStats() {
  heur_stats.total_repair_lp = 0;
  heur_stats.total_repair_lp_feasible = 0;
  heur_stats.total_repair_lp_iterations = 0;
  heur_stats.lp_iterations = 0;
  heur_stats.max_submip_level = 0;
  heur_stats.termination_status_ = HighsModelStatus::kNotset;
  heur_stats.successObservations = 0;
  heur_stats.numSuccessObservations = 0;
  heur_stats.infeasObservations = 0;
  heur_stats.numInfeasObservations = 0;
}
