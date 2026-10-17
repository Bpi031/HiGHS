/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#ifndef MIP_HIGHS_MIP_SOLVER_H_
#define MIP_HIGHS_MIP_SOLVER_H_

#include <cstdint>
#include <limits>

#include "Highs.h"
#include "lp_data/HighsCallback.h"
#include "lp_data/HighsOptions.h"
#include "mip/HighsCandidateCheckpoint.h"
#include "mip/HighsAssignmentSupport.h"
#include "mip/HighsSolveCapacity.h"
#include "parallel/HighsHardwareTopology.h"
#include "parallel/HighsParallel.h"

struct HighsMipSolverData;
class HighsCutPool;
struct HighsPseudocostInitialization;
class HighsCliqueTable;
class HighsImplications;
class HighsChildProposalSink;

struct HighsTerminator {
  HighsInt num_instance;
  HighsInt my_instance;
  HighsModelStatus* record;
  void clear();
  void initialise(HighsInt num_instance_, HighsInt my_instance_,
                  HighsModelStatus* record_);
  HighsInt concurrency() const;
  void terminate();
  bool terminated() const;
  HighsModelStatus terminationStatus() const;
  void report(const HighsLogOptions log_options) const;
};

class HighsMipSolver {
 public:
  HighsCallback* callback_;
  const HighsOptions* options_mip_;
  const HighsLp* model_;
  const HighsLp* orig_model_;
  std::vector<HighsInt> primal_partition_groups;
  HighsInt primal_partition_free_groups = 0;
  HighsAssignmentSlots primal_assignment_slots;
  // Diagnostic-only writes by the terminal child's serial first-LP owner.
  mutable double child_first_lp_seconds = -1;
  mutable int64_t child_first_lp_iterations = 0;
  HighsModelStatus modelstatus_;
  std::vector<double> solution_;
  double solution_objective_;
  double bound_violation_;
  double integrality_violation_;
  double row_violation_;
  // The following are only to return data to HiGHS, and are set in
  // HighsMipSolver::cleanupSolve
  double dual_bound_;
  double primal_bound_;
  double gap_;
  int64_t node_count_;
  int64_t total_lp_iterations_;
  double primal_dual_integral_;

  FILE* improving_solution_file_;
  std::vector<HighsObjectiveSolution> saved_objective_and_solution_;

  bool submip;
  // Non-owning, solveSubMip-scoped, serial producer; not a public callback.
  HighsChildProposalSink* child_proposal_sink = nullptr;
  // Restricted helper work only; exhaustion never proves infeasibility.
  int64_t helper_lp_limit = std::numeric_limits<int64_t>::max();
  mutable int64_t helper_lp_work = 0;
  HighsCandidateCheckpoint candidate_checkpoint;
  const HighsCandidateCheckpoint* parent_candidate_checkpoint = nullptr;
  uint64_t parent_candidate_sequence = 0;
  bool parentCheckpointRequested() const {
    return parent_candidate_checkpoint &&
           parent_candidate_checkpoint->requestedSince(parent_candidate_sequence);
  }
  HighsInt submip_level;
  HighsInt max_submip_level;
  const HighsBasis* rootbasis;
  const HighsPseudocostInitialization* pscostinit;
  const HighsCliqueTable* clqtableinit;
  const HighsImplications* implicinit;

  std::unique_ptr<HighsMipSolverData> mipdata_;
  std::unique_ptr<HighsSolveCapacity> solve_capacity;

  HighsModelStatus termination_status_;
  HighsTerminator terminator_;

  void run();

  HighsInt numCol() const { return model_->num_col_; }

  HighsInt numRow() const { return model_->num_row_; }

  HighsInt numNonzero() const { return model_->a_matrix_.numNz(); }

  const double* colCost() const { return model_->col_cost_.data(); }

  double colCost(HighsInt col) const { return model_->col_cost_[col]; }

  const double* rowLower() const { return model_->row_lower_.data(); }

  double rowLower(HighsInt col) const { return model_->row_lower_[col]; }

  const double* rowUpper() const { return model_->row_upper_.data(); }

  double rowUpper(HighsInt col) const { return model_->row_upper_[col]; }

  const HighsVarType* variableType() const {
    return model_->integrality_.data();
  }

  HighsVarType variableType(HighsInt col) const {
    return model_->integrality_[col];
  }

  bool isColIntegral(const HighsInt col) const {
    return variableType(col) != HighsVarType::kContinuous;
  }

  bool isColInteger(const HighsInt col) const {
    return variableType(col) == HighsVarType::kInteger;
  }

  bool isColContinuous(const HighsInt col) const {
    return variableType(col) == HighsVarType::kContinuous;
  }

  HighsMipSolver(HighsCallback& callback, const HighsOptions& options,
                 const HighsLp& lp, const HighsSolution& solution,
                 bool submip = false, HighsInt submip_level = 0);

  ~HighsMipSolver();

  template <class F>
  void runTask(F&& f, highs::parallel::TaskGroup& tg, bool parallel_lock,
               bool force_serial,
               const std::vector<HighsInt>& indices = std::vector<HighsInt>(1,
                                                                            0));

  void setModel(const HighsLp& model) {
    model_ = &model;
    solution_objective_ = kHighsInf;
  }

  mutable HighsTimer timer_;
  HighsProfiling* profiling_ = nullptr;

  void cleanupSolve();
  void solvingReport(const std::string& solutionstatus) const;

  void runMipPresolve(const HighsInt presolve_reduction_limit);
  const HighsLp& getPresolvedModel() const;
  HighsPresolveStatus getPresolveStatus() const;
  presolve::HighsPostsolveStack getPostsolveStack() const;
  const HighsPresolveLog& getPresolveLog() const;

  void callbackGetCutPool() const;
  bool solutionFeasible(const HighsLp* lp, const std::vector<double>& col_value,
                        const std::vector<double>* pass_row_value,
                        double& bound_violation, double& row_violation,
                        double& integrality_violation, HighsCDouble& obj) const;

  std::vector<HighsModelStatus> initialiseTerminatorRecord(
      HighsInt num_instance) const;
  void initialiseTerminator(HighsInt num_instance_ = 0,
                            HighsInt my_instance_ = kNoThreadInstance,
                            HighsModelStatus* record_ = nullptr);
  void initialiseTerminator(const HighsMipSolver& mip_solver);
  bool terminate() const {
    return this->termination_status_ != HighsModelStatus::kNotset;
  }
  HighsModelStatus terminationStatus() const {
    return this->termination_status_;
  }
  void setParallelLock(bool lock) const;
  void setProfiling(HighsProfiling* profiling);
  bool deterministicIncumbentExchange() const {
    return options_mip_->mip_incumbent_exchange == "deterministic";
  }
  bool opportunisticIncumbentExchange() const {
    return !submip && options_mip_->mip_incumbent_exchange == "opportunistic";
  }
  bool boundedIncumbentEpochs() const {
    return deterministicIncumbentExchange() || opportunisticIncumbentExchange();
  }
  HighsInt getMaxNumWorkers() const {
    if (highs::parallel::num_threads() == 1 ||
        options_mip_->parallel != kHighsOnString || submip) {
      return 1;
    }
    if (opportunisticIncumbentExchange())
      return std::max(1, highs::parallel::num_threads() -
          (callback_->primal_service_callback ? 1 : 0));
    if (deterministicIncumbentExchange())
      return highsTopologyAwareWorkerCount(highs::parallel::num_threads());
    if (options_mip_->mip_flexible_node_dispatch)
      return highs::parallel::num_threads();
    return static_cast<HighsInt>(
        std::ceil(1.7 * highs::parallel::num_threads()));
  }
};

std::array<char, 128> getGapString(const double gap_,
                                   const double primal_bound_,
                                   const HighsOptions* options_mip_);
#endif
