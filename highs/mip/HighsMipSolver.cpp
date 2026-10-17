/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "mip/HighsMipSolver.h"
#include <fstream>

#include "lp_data/HighsLpUtils.h"
#include "lp_data/HighsModelUtils.h"
#include "mip/HighsCliqueTable.h"
#include "mip/HighsCutPool.h"
#include "mip/HighsDomain.h"
#include "mip/HighsImplications.h"
#include "mip/HighsLpRelaxation.h"
#include "mip/HighsMipSolverData.h"
#include "mip/HighsMipWorker.h"
#include "mip/HighsPseudocost.h"
#include "mip/HighsSearch.h"
#include "mip/HighsSeparation.h"
#include "mip/MipTimer.h"
#include "presolve/HPresolve.h"
#include "presolve/HighsPostsolveStack.h"
#include "presolve/PresolveComponent.h"
#include "util/HighsCDouble.h"
#include "mip/HighsCandidateVerifier.h"
#include "mip/HighsPrimalService.h"
#include "util/HighsIntegers.h"

using std::fabs;

namespace {
uint64_t rinsNeighbourhoodSignature(
    const HighsMipSolverData& mipdata,
    const std::vector<double>& relaxation_solution,
    const std::vector<double>& incumbent) {
  if (incumbent.size() != relaxation_solution.size()) return 0;
  uint64_t signature = 1469598103934665603ULL;
  size_t agreeing_columns = 0;
  for (HighsInt col : mipdata.integral_cols) {
    if (std::abs(relaxation_solution[col] - incumbent[col]) >
        mipdata.feastol)
      continue;
    ++agreeing_columns;
    const int64_t value = static_cast<int64_t>(
        HighsIntegers::nearestInteger(incumbent[col]));
    signature ^= static_cast<uint64_t>(col);
    signature *= 1099511628211ULL;
    signature ^= static_cast<uint64_t>(value);
    signature *= 1099511628211ULL;
  }
  return agreeing_columns == 0 ? 0 : signature;
}
}  // namespace

HighsMipSolver::HighsMipSolver(HighsCallback& callback,
                               const HighsOptions& options, const HighsLp& lp,
                               const HighsSolution& solution, bool submip,
                               HighsInt submip_level)
    : callback_(&callback),
      options_mip_(&options),
      model_(&lp),
      orig_model_(&lp),
      solution_objective_(kHighsInf),
      submip(submip),
      submip_level(submip_level),
      rootbasis(nullptr),
      pscostinit(nullptr),
      clqtableinit(nullptr),
      implicinit(nullptr) {
  timer_.setPrintfFlag(options_mip_->output_flag, options_mip_->log_to_console);
  assert(!submip || submip_level > 0);
  max_submip_level = 0;
  // Initialise empty terminator
  initialiseTerminator();
  assert(termination_status_ == HighsModelStatus::kNotset);
  if (solution.value_valid) {
#ifndef NDEBUG
    // MIP solver doesn't check row residuals, but they should be OK
    // so validate using assert
    bool valid, integral, feasible;
    assessLpPrimalSolution("For debugging: ", options, lp, solution, valid,
                           integral, feasible);
    assert(valid);
#endif
    // Initial solution can be infeasible, but need to set values for violation
    // and objective
    HighsCDouble quad_solution_objective_;
    solutionFeasible(orig_model_, solution.col_value, &solution.row_value,
                     bound_violation_, row_violation_, integrality_violation_,
                     quad_solution_objective_);
    solution_objective_ = double(quad_solution_objective_);
    solution_ = solution.col_value;
  }
}

HighsMipSolver::~HighsMipSolver() = default;

template <class F>
void HighsMipSolver::runTask(F&& f, highs::parallel::TaskGroup& tg,
                             bool parallel_lock, bool force_serial,
                             const std::vector<HighsInt>& indices) {
  if (indices.empty()) return;
  setParallelLock(parallel_lock);
  const bool spawn_tasks = !force_serial && indices.size() > 1 &&
                           !options_mip_->mip_search_simulate_concurrency;
  for (HighsInt i : indices) {
    if (spawn_tasks) {
      tg.spawn([&f, i] { f(i); });
    } else {
      f(i);
    }
  }
  if (spawn_tasks) {
    tg.taskWait();
  }
  setParallelLock(false);
}

void HighsMipSolver::run() {
  solve_capacity.reset(new HighsSolveCapacity(
      std::max(1, highs::parallel::num_threads()),
      size_t{256} * 1024 * 1024));
  modelstatus_ = HighsModelStatus::kNotset;
  if (!submip && !options_mip_->mip_primal_partition_file.empty()) {
    std::ifstream contract(options_mip_->mip_primal_partition_file);
    std::string contract_kind;
    contract >> contract_kind;
    contract.clear();
    contract.seekg(0);
    const bool slot_contract = contract_kind == "ROST_ASSIGNMENT_SLOTS_V1";
    if ((options_mip_->mip_primal_heuristic_schedule != "portfolio-cost-aware" &&
         options_mip_->mip_primal_heuristic_schedule != "portfolio-assignment-adaptive" &&
         options_mip_->mip_primal_heuristic_schedule != "portfolio-assignment-composed" &&
         options_mip_->mip_primal_heuristic_schedule != "portfolio-assignment-progressive" &&
         options_mip_->mip_primal_heuristic_schedule != "portfolio-assignment-opportunities") ||
        (slot_contract ? (options_mip_->mip_primal_heuristic_schedule != "portfolio-cost-aware" ||
                         !HighsAssignmentSupport::read(contract, *orig_model_, primal_assignment_slots)) :
        !HighsPrimalHeuristics::readBinaryPartitionContract(
            contract, *orig_model_, primal_partition_groups,
            primal_partition_free_groups))) {
      highsLogUser(options_mip_->log_options, HighsLogType::kError,
                   "Invalid original-column primal partition contract\n");
      modelstatus_ = HighsModelStatus::kModelError;
      return;
    }
  }
  if (!submip && (options_mip_->mip_primal_heuristic_schedule == "portfolio-assignment-adaptive" ||
                 options_mip_->mip_primal_heuristic_schedule == "portfolio-assignment-composed" ||
                 options_mip_->mip_primal_heuristic_schedule == "portfolio-assignment-progressive" ||
                 options_mip_->mip_primal_heuristic_schedule == "portfolio-assignment-opportunities") &&
      primal_partition_groups.empty()) {
    highsLogUser(options_mip_->log_options, HighsLogType::kError,
                 "Assignment-adaptive portfolio requires an original-column partition contract\n");
    modelstatus_ = HighsModelStatus::kModelError;
    return;
  }
  if (!submip) {
    auto& input = callback_->data_in;
    ++input.admission_generation;
    input.incumbent_sequence = 0;
    input.last_candidate_id = 0;
    input.admission = {};
    input.admission.generation = input.admission_generation;
  }
  if (options_mip_->mip_incumbent_exchange != "off" &&
      options_mip_->mip_incumbent_exchange != "observe" &&
      options_mip_->mip_incumbent_exchange != "deterministic" &&
      options_mip_->mip_incumbent_exchange != "opportunistic") {
    highsLogUser(options_mip_->log_options, HighsLogType::kError,
                 "Unknown mip_incumbent_exchange '%s'\n",
                 options_mip_->mip_incumbent_exchange.c_str());
    modelstatus_ = HighsModelStatus::kModelError;
    return;
  }
  max_submip_level = std::max(submip_level, max_submip_level);
  // Start the timer local to HighsMipSolver - independent of the
  // timer passed from Highs as a pointer that's used in
  // HighsProfiling
  this->timer_.start();
  improving_solution_file_ = nullptr;
  if (!submip && options_mip_->mip_improving_solution_file != "")
    improving_solution_file_ =
        fopen(options_mip_->mip_improving_solution_file.c_str(), "w");

  mipdata_ = decltype(mipdata_)(new HighsMipSolverData(*this));
  for (HighsInt iLp = 0; iLp < static_cast<HighsInt>(mipdata_->lps.size());
       iLp++)
    mipdata_->lps[iLp].setProfiling(this->profiling_);
  assert(profiling_);
  // The solve time clock shouldn't be running on entry
  assert(!profiling_->running(kSolveTime));
  profiling_->start(kPresolveTime);
  profiling_->start(kMipClockInit);
  mipdata_->init();
#ifdef HIGHS_DEBUGSOL
  mipdata_->debugSolution.activate();
  bool debugSolActive = false;
  std::swap(mipdata_->debugSolution.debugSolActive, debugSolActive);
#endif
  profiling_->stop(kMipClockInit);
  profiling_->start(kMipClockRunPresolve);
  mipdata_->runMipPresolve(options_mip_->presolve_reduction_limit);
  profiling_->stop(kMipClockRunPresolve);
  profiling_->stop(kPresolveTime);
  if (profiling_->mip_ && !submip)
    highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
                 "MIP-Timing: %11.2g - completed presolve\n", timer_.read());
  // Identify whether time limit has been reached (in presolve)
  if (modelstatus_ == HighsModelStatus::kNotset &&
      timer_.read() >= options_mip_->time_limit)
    modelstatus_ = HighsModelStatus::kTimeLimit;

  if (modelstatus_ != HighsModelStatus::kNotset) {
    highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
                 "Presolve: %s\n",
                 utilModelStatusToString(modelstatus_).c_str());
    if (modelstatus_ == HighsModelStatus::kOptimal) {
      mipdata_->lower_bound = 0;
      mipdata_->upper_bound = 0;
      mipdata_->transformNewIntegerFeasibleSolution(std::vector<double>());
      mipdata_->saveReportMipSolution(-kHighsInf, kSolutionSourceEmptyMip);
    }
    cleanupSolve();
    return;
  }

  profiling_->start(kSolveTime);
  if (profiling_->mip_ && !submip)
    highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
                 "MIP-Timing: %11.2g - starting  setup\n", timer_.read());
  profiling_->start(kMipClockRunSetup);
#ifdef HIGHS_DEBUGSOL
  mipdata_->debugSolution.debugSolActive = debugSolActive;
#endif
  mipdata_->runSetup();
  profiling_->stop(kMipClockRunSetup);
  if (profiling_->mip_ && !submip)
    highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
                 "MIP-Timing: %11.2g - completed setup\n", timer_.read());

  if (mipdata_->getDomain().infeasible()) {
    cleanupSolve();
    return;
  }
  // Initialise master worker.
  mipdata_->workers.emplace_back(
      *this, &mipdata_->getLp(), &mipdata_->getDomain(),
      &mipdata_->getCutPool(), &mipdata_->getConflictPool(),
      &mipdata_->getPseudoCost(), 0);

  HighsMipWorker& master_worker = mipdata_->workers[0];

restart:
  if (modelstatus_ == HighsModelStatus::kNotset) {
    // Check limits have not been reached before evaluating root node
    if (mipdata_->checkLimits()) {
      cleanupSolve();
      return;
    }
    // Possibly query existence of an external solution
    if (!submip)
      mipdata_->queryExternalSolution(
          solution_objective_, kExternalMipSolutionQueryOriginAfterSetup);

    // Apply the trivial heuristics
    profiling_->start(kMipClockTrivialHeuristics);
    HighsModelStatus returned_model_status = mipdata_->trivialHeuristics();
    profiling_->stop(kMipClockTrivialHeuristics);
    if (modelstatus_ == HighsModelStatus::kNotset &&
        returned_model_status == HighsModelStatus::kInfeasible) {
      // trivialHeuristics can spot trivial infeasibility, so act on it
      modelstatus_ = returned_model_status;
      cleanupSolve();
      return;
    }
    // Apply the feasibility jump heuristic (if enabled)
    if (options_mip_->mip_heuristic_run_feasibility_jump) {
      profiling_->start(kMipClockFeasibilityJump);
      HighsModelStatus returned_model_status = mipdata_->feasibilityJump();
      profiling_->stop(kMipClockFeasibilityJump);
      if (modelstatus_ == HighsModelStatus::kNotset &&
          returned_model_status == HighsModelStatus::kInfeasible) {
        // feasibilityJump can spot trivial infeasibility, so act on it
        modelstatus_ = returned_model_status;
        cleanupSolve();
        return;
      }
    }
    // End of pre-root-node heuristics
    if (callback_->refinement_budget && mipdata_->checkLimits()) {
      cleanupSolve();
      return;
    }
    if (profiling_->mip_ && !submip)
      highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
                   "MIP-Timing: %11.2g - starting evaluate root node\n",
                   timer_.read());
    profiling_->start(kMipClockEvaluateRootNode);
    mipdata_->evaluateRootNode(master_worker);
    profiling_->stop(kMipClockEvaluateRootNode);
    if (this->terminate()) {
      modelstatus_ = this->terminationStatus();
      cleanupSolve();
      return;
    }
    if (profiling_->mip_ && !submip)
      highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
                   "MIP-Timing: %11.2g - completed evaluate root node\n",
                   timer_.read());
    // age 5 times to remove stored but never violated cuts after root
    // separation
    profiling_->start(kMipClockPerformAging0);
    mipdata_->getCutPool().performAging();
    mipdata_->getCutPool().performAging();
    mipdata_->getCutPool().performAging();
    mipdata_->getCutPool().performAging();
    mipdata_->getCutPool().performAging();
    profiling_->stop(kMipClockPerformAging0);
  }
  if (mipdata_->nodequeue.empty() || mipdata_->checkLimits()) {
    cleanupSolve();
    return;
  }

  mipdata_->updateLowerBound(mipdata_->nodequeue.getBestLowerBound());
  mipdata_->printDisplayLine();

  const HighsInt max_num_workers = getMaxNumWorkers();
  const bool primal_portfolio = !submip &&
      mipdata_->heuristic_manager.portfolioEnabled();
  const bool managed_rins_budget = !submip &&
      (primal_portfolio || (mipdata_->heuristic_manager.config().schedule ==
          HighsPrimalHeuristicSchedule::kManagedRins &&
      mipdata_->heuristic_manager.config().rins_budget.max_lp_iterations > 0));
  if (options_mip_->mip_incumbent_exchange != "off" && !submip) {
    const HighsHardwareTopology topology = highsDetectHardwareTopology();
    highsLogUser(
        options_mip_->log_options, HighsLogType::kInfo,
        "MIP-Incumbent-Exchange: schema=rost/highs-incumbent-exchange/v1 "
        "mode=%s configured_threads=%" HIGHSINT_FORMAT
        " workers=%" HIGHSINT_FORMAT " epoch_nodes=%" HIGHSINT_FORMAT
        " logical_cpus=%" HIGHSINT_FORMAT " physical_cores=%" HIGHSINT_FORMAT
        " performance_cores=%" HIGHSINT_FORMAT " numa_nodes=%" HIGHSINT_FORMAT
        " llc_domains=%" HIGHSINT_FORMAT " topology_complete=%d\n",
        options_mip_->mip_incumbent_exchange.c_str(),
        highs::parallel::num_threads(), max_num_workers,
        options_mip_->mip_incumbent_epoch_nodes, topology.logical_cpus,
        topology.physical_cores, topology.performance_cores,
        topology.numa_nodes, topology.last_level_cache_domains,
        topology.complete ? 1 : 0);
  }
  HighsInt num_workers = 1;
  highs::parallel::TaskGroup tg;

  auto destroyOldWorkers = [&]() {
    if (mipdata_->workers.size() <= 1) return;
    while (mipdata_->domains.size() > 1) {
      mipdata_->domains.pop_back();
    }
    while (mipdata_->lps.size() > 1) {
      mipdata_->lps.pop_back();
    }
    while (mipdata_->pseudocosts.size() > 1) {
      mipdata_->pseudocosts.pop_back();
    }
    while (mipdata_->workers.size() > 1) {
      mipdata_->workers.pop_back();
    }
    while (mipdata_->cutpools.size() > 1) {
      mipdata_->cutpools.pop_back();
    }
    while (mipdata_->conflictpools.size() > 1) {
      mipdata_->conflictpools.pop_back();
    }
  };

  auto createNewWorkers = [&](HighsInt num_new_workers) {
    if (num_new_workers <= 0) return;
    // Remove all cuts from non-global pool for copied LP
    mipdata_->lps.emplace_back(mipdata_->getLp());
    // Have to set the global sub-solver call time pointer here for
    // new worker 0, since
    // The new worker solves after installing its node/domain. It needs only
    // the remapped model and basis here, not an extra cleanup solve.
    mipdata_->lps.back().setProfiling(this->profiling_);
    mipdata_->lps.back().removeWorkerSpecificRows(
        opportunisticIncumbentExchange()
            ? HighsLpRelaxation::CutRemovalMode::kModelAndBasisOnly
            : HighsLpRelaxation::CutRemovalMode::kReoptimize);
    for (HighsInt i = 0; i != num_new_workers; ++i) {
      if (i != 0) {
        mipdata_->lps.emplace_back(mipdata_->lps.back());
        mipdata_->lps.back().setProfiling(this->profiling_);
      }
      mipdata_->domains.emplace_back(mipdata_->getDomain());
      mipdata_->cutpools.emplace_back(
          numCol(), options_mip_->mip_pool_age_limit,
          options_mip_->mip_pool_soft_limit, mipdata_->cutpools.size());
      mipdata_->conflictpools.emplace_back(5 * options_mip_->mip_pool_age_limit,
                                           options_mip_->mip_pool_soft_limit);
      mipdata_->domains.back().addCutpool(mipdata_->cutpools.back());
      assert(mipdata_->domains.back().getDomainChangeStack().empty());
      mipdata_->domains.back().addConflictPool(mipdata_->conflictpools.back());
      mipdata_->pseudocosts.emplace_back(*this);
      mipdata_->workers.emplace_back(
          *this, &mipdata_->lps.back(), &mipdata_->domains.back(),
          &mipdata_->cutpools.back(), &mipdata_->conflictpools.back(),
          &mipdata_->pseudocosts.back(), mipdata_->workers.size());
      mipdata_->lps.back().setMipWorker(mipdata_->workers.back());
      mipdata_->lps.back().notifyCutPoolsLpCopied(1);
      mipdata_->workers.back().randgen.initialise(options_mip_->random_seed +
                                                  mipdata_->workers.size() - 1);
      mipdata_->workers.back().nodequeue.setNumCol(numCol());
      mipdata_->debugSolution.registerDomain(
          mipdata_->workers.back().search_ptr_->getLocalDomain());
    }
  };

  // Use case: Change pointers in master worker to local copies of global info
  auto constructAdditionalWorkerData = [&](HighsMipWorker& worker) {
    assert(mipdata_->cutpools.size() == 1 &&
           mipdata_->conflictpools.size() == 1);
    assert(&worker == &mipdata_->workers[0]);
    mipdata_->cutpools.emplace_back(numCol(), options_mip_->mip_pool_age_limit,
                                    options_mip_->mip_pool_soft_limit, 1);
    worker.setCutPool(&mipdata_->cutpools.back());
    mipdata_->conflictpools.emplace_back(5 * options_mip_->mip_pool_age_limit,
                                         options_mip_->mip_pool_soft_limit);
    worker.setConflictPool(&mipdata_->conflictpools.back());
    mipdata_->domains.emplace_back(mipdata_->getDomain());
    worker.setGlobalDomain(&mipdata_->domains.back());
    worker.getGlobalDomain().addCutpool(worker.getCutPool());
    assert(worker.getGlobalDomain().getDomainChangeStack().empty());
    worker.getGlobalDomain().addConflictPool(worker.getConflictPool());
    mipdata_->pseudocosts.emplace_back(*this);
    worker.setPseudocost(&mipdata_->pseudocosts.back());
    worker.getLpRelaxation().setMipWorker(worker);
    worker.resetSearch();
    worker.resetSepa();
    worker.nodequeue.clear();
    worker.nodequeue.setNumCol(numCol());
  };

  int64_t candidate_checked = 0;
  int64_t candidate_admitted = 0;
  int64_t post_admission_queries = 0;
  double candidate_wait_max = 0;
  double candidate_check_seconds = 0;
  int checkpoint_receipts = 0;
  std::shared_ptr<const HighsPrimalSnapshot> joined_primal;
  HighsCallbackInput joined_service_input;
  bool service_failed = false;
  bool service_had_owner = false;
  uint64_t service_queries = 0, service_notifications = 0;
  auto syncSolutions = [&]() -> bool {
    // Candidate-only admission already notified its distinct callback owner.
    // Reconciliation must not emit older joined candidates as new incumbents.
    struct RestoreSolutionCallbacks {
      HighsCallback& cb;
      bool solution, improving, logging;
      ~RestoreSolutionCallbacks() {
        cb.active[kCallbackMipSolution] = solution;
        cb.active[kCallbackMipImprovingSolution] = improving;
        cb.active[kCallbackMipLogging] = logging;
      }
    } restore{*callback_, callback_->active[kCallbackMipSolution],
              callback_->active[kCallbackMipImprovingSolution],
              callback_->active[kCallbackMipLogging]};
    if (service_had_owner) {
      callback_->active[kCallbackMipSolution] = false;
      callback_->active[kCallbackMipImprovingSolution] = false;
      callback_->active[kCallbackMipLogging] = false;
    }
    // Note: Upper bound / limit of workers updated via addIncumbent
    std::vector<HighsWorkerCandidate> candidates;
    bool mailbox_overflow = candidate_checkpoint.overflowed();
    for (HighsMipWorker& worker : mipdata_->workers) {
      if (worker.incumbent_mailbox_.overflowed()) {
        highsLogUser(options_mip_->log_options, HighsLogType::kError,
                     "Incumbent mailbox overflow on worker %" HIGHSINT_FORMAT
                     "; terminating without a trusted result\n",
                     worker.workerId());
        mailbox_overflow = true;
      }
    }
    if (mailbox_overflow) {
      modelstatus_ = HighsModelStatus::kSolveError;
      mipdata_->nodequeue.clear();
      for (HighsMipWorker& worker : mipdata_->workers) {
        worker.incumbent_mailbox_.drain();
        worker.incumbent_mailbox_.clearOverflow();
        worker.attachPrimalStore(nullptr);
      }
      return false;
    }
    for (HighsMipWorker& worker : mipdata_->workers) {
      std::vector<HighsWorkerCandidate> worker_candidates =
          worker.incumbent_mailbox_.drain();
      candidates.insert(candidates.end(),
                        std::make_move_iterator(worker_candidates.begin()),
                        std::make_move_iterator(worker_candidates.end()));
      worker.incumbent_mailbox_.clearOverflow();
    }
    // This also retains a private-repair proposal that has no worker mailbox.
    if (joined_primal) candidates.push_back(joined_primal->candidate);
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const HighsWorkerCandidate& left,
                        const HighsWorkerCandidate& right) {
                       return std::tie(left.metadata.work_epoch,
                                       left.metadata.worker_id,
                                       left.metadata.publication_sequence) <
                              std::tie(right.metadata.work_epoch,
                                       right.metadata.worker_id,
                                       right.metadata.publication_sequence);
                     });
    for (HighsWorkerCandidate& candidate : candidates) {
      const double dequeued = timer_.read();
      const bool same_generation = candidate.metadata.admission_generation ==
                                  callback_->data_in.admission_generation;
      ++candidate_checked;
      const bool accepted = same_generation && mipdata_->addIncumbent(candidate.solution, candidate.native_objective,
                             candidate.solution_source, true, false,
                             &candidate.metadata);
      const double finished = timer_.read();
      candidate_admitted += accepted;
      candidate_wait_max = std::max(candidate_wait_max,
          std::max(0.0, dequeued - candidate.metadata.publication_time));
      candidate_check_seconds += std::max(0.0, finished - dequeued);
      if (options_mip_->mip_candidate_checkpoint && candidate_checked <= 128)
        highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
          "MIP-CandidateAdmission: schema=rost/highs-candidate-admission/v1 worker=%d epoch=%lld sequence=%lld "
          "discovery_s=%.17g publication_s=%.17g dequeue_s=%.17g "
          "finished_s=%.17g accepted=%d same_generation=%d authority=%s\n",
          int(candidate.metadata.worker_id),
          static_cast<long long>(candidate.metadata.work_epoch),
          static_cast<long long>(candidate.metadata.publication_sequence),
          candidate.metadata.discovery_time, candidate.metadata.publication_time,
          dequeued, finished, int(accepted), int(same_generation),
          opportunisticIncumbentExchange() ? "joined_reconciliation" : "joined_master");
    }
    // Any cutoff used by a peer must have a globally retained witness before
    // status/proof construction. A reconciliation failure is not infeasibility.
    bool reconciled = !service_failed;
    if (joined_primal && (mipdata_->incumbent.empty() ||
        mipdata_->upper_bound > joined_primal->upper_bound + mipdata_->feastol))
      reconciled = false;
    for (HighsMipWorker& worker : mipdata_->workers) {
      const auto snapshot = worker.pendingPrimalSnapshot();
      if (snapshot && (mipdata_->incumbent.empty() ||
          mipdata_->upper_bound > snapshot->upper_bound + mipdata_->feastol))
        reconciled = false;
      worker.attachPrimalStore(nullptr);
    }
    if (service_had_owner) {
      auto& input = callback_->data_in;
      input.incumbent_sequence = std::max(input.incumbent_sequence,
                                          joined_service_input.incumbent_sequence);
      input.last_candidate_id = joined_service_input.last_candidate_id;
      input.admission = joined_service_input.admission;
    }
    joined_primal.reset();
    service_had_owner = false;
    if (!reconciled) {
      highsLogUser(options_mip_->log_options, HighsLogType::kError,
                   "Checked peer witness did not reconcile at batch join\n");
      modelstatus_ = HighsModelStatus::kSolveError;
    }
    return reconciled;
  };

  auto syncPools = [&](std::vector<HighsInt>& indices) -> void {
    if (!mipdata_->hasMultipleWorkers() || mipdata_->parallelLockActive())
      return;
    for (const HighsInt i : indices) {
      mipdata_->workers[i].getConflictPool().syncConflictPool(
          mipdata_->getConflictPool());
      mipdata_->workers[i].getCutPool().syncCutPool(*this,
                                                    mipdata_->getCutPool());
    }
    mipdata_->getCutPool().performAging();
    mipdata_->getConflictPool().performAging();
  };

  auto syncGlobalDomain = [&](std::vector<HighsInt>& indices) -> void {
    if (!mipdata_->hasMultipleWorkers()) return;
    for (const HighsInt i : indices) {
      HighsMipWorker& worker = mipdata_->workers[i];
      const auto& domchgstack = worker.getGlobalDomain().getDomainChangeStack();
      for (const HighsDomainChange& domchg : domchgstack) {
        if ((domchg.boundtype == HighsBoundType::kLower &&
             domchg.boundval >
                 mipdata_->getDomain().col_lower_[domchg.column]) ||
            (domchg.boundtype == HighsBoundType::kUpper &&
             domchg.boundval <
                 mipdata_->getDomain().col_upper_[domchg.column])) {
          mipdata_->getDomain().changeBound(domchg,
                                            HighsDomain::Reason::unspecified());
        }
      }
    }
  };

  auto doResetWorkerDomain = [&](HighsInt i) {
    HighsMipWorker& worker = mipdata_->workers[i];
    if (worker.getGlobalDomain().infeasible()) return;
    for (const HighsDomainChange& domchg :
         mipdata_->getDomain().getDomainChangeStack()) {
      worker.getGlobalDomain().changeBound(domchg,
                                           HighsDomain::Reason::unspecified());
      if (worker.getGlobalDomain().infeasible()) return;
    }
#ifndef NDEBUG
    for (HighsInt col = 0; col < numCol(); ++col) {
      assert(mipdata_->getDomain().col_lower_[col] ==
             worker.getGlobalDomain().col_lower_[col]);
      assert(mipdata_->getDomain().col_upper_[col] ==
             worker.getGlobalDomain().col_upper_[col]);
    }
#endif
    worker.getGlobalDomain().setDomainChangeStack(
        std::vector<HighsDomainChange>());
    // Warning: Resetting local domain cannot be done in parallel (changes
    // propagationDomains of main pool)
    worker.search_ptr_->resetLocalDomain();
    worker.getGlobalDomain().clearChangedCols();
  };

  auto resetGlobalDomain = [&](bool force, bool resetWorkers) -> bool {
    // if global propagation found bound changes, we update the domain
    if (!mipdata_->getDomain().getChangedCols().empty() || force) {
      profiling_->start(kMipClockUpdateLocalDomain);
      highsLogDev(options_mip_->log_options, HighsLogType::kInfo,
                  "added %" HIGHSINT_FORMAT " global bound changes\n",
                  (HighsInt)mipdata_->getDomain().getChangedCols().size());
      mipdata_->cliquetable.cleanupFixed(mipdata_->getDomain());
      if (mipdata_->getDomain().infeasible()) {
        profiling_->stop(kMipClockUpdateLocalDomain);
        return false;
      }
      if (mipdata_->hasMultipleWorkers() && resetWorkers) {
        // Sync worker domains here. cleanupFixed might have found extra changes
        std::vector<HighsInt> indices(num_workers);
        std::iota(indices.begin(), indices.end(), 0);
        runTask(doResetWorkerDomain, tg, false, true, indices);
        // All workers are joined. Global bound reconciliation can itself prove
        // that no improving solution remains; do not erase that proof on reset.
        for (const HighsInt i : indices) {
          if (mipdata_->workers[i].getGlobalDomain().infeasible()) {
            mipdata_->getDomain().markInfeasible();
            profiling_->stop(kMipClockUpdateLocalDomain);
            return false;
          }
        }
      }
      for (const HighsInt col : mipdata_->getDomain().getChangedCols())
        mipdata_->implications.cleanupVarbounds(col);

      mipdata_->getDomain().setDomainChangeStack(
          std::vector<HighsDomainChange>());
      if (!mipdata_->hasMultipleWorkers())
        master_worker.search_ptr_->resetLocalDomain();
      mipdata_->getDomain().clearChangedCols();
      mipdata_->removeFixedIndices();
      profiling_->stop(kMipClockUpdateLocalDomain);
    }
    return true;
  };

  auto syncGlobalPseudoCost = [&]() -> void {
    if (!mipdata_->hasMultipleWorkers()) return;
    for (HighsMipWorker& worker : mipdata_->workers) {
      mipdata_->getPseudoCost().flushPseudoCost(worker.getPseudocost());
    }
  };

  auto resetWorkerPseudoCosts = [&](std::vector<HighsInt>& indices) {
    if (!mipdata_->hasMultipleWorkers()) return;
    auto doResetWorkerPseudoCost = [&](HighsInt i) -> void {
      mipdata_->getPseudoCost().syncPseudoCost(
          mipdata_->workers[i].getPseudocost());
    };
    runTask(doResetWorkerPseudoCost, tg, false, false, indices);
  };

  master_worker.resetSearch();
  master_worker.resetSepa();
  master_worker.nodequeue.clear();
  master_worker.nodequeue.setNumCol(numCol());
  master_worker.upper_bound = mipdata_->upper_bound;
  master_worker.upper_limit = mipdata_->upper_limit;
  master_worker.optimality_limit = mipdata_->optimality_limit;
  destroyOldWorkers();
  mipdata_->debugSolution.registerDomain(
      master_worker.search_ptr_->getLocalDomain());

  profiling_->start(kMipClockSearch);
  int64_t numStallNodes = 0;
  int64_t lastLbLeave = 0;
  int64_t numQueueLeaves = 0;
  HighsInt numHugeTreeEstim = 0;
  int64_t numNodesLastCheck = mipdata_->num_nodes;
  int64_t nextCheck = mipdata_->num_nodes;
  double treeweightLastCheck = 0.0;
  double upperLimLastCheck = mipdata_->upper_limit;
  double lowerBoundLastCheck = mipdata_->lower_bound;

  enum class RestartVote { kNoCheck, kNoHugeTree, kHugeTree, kWouldRestart };

  auto nodesInstalled = [&]() -> bool {
    for (const HighsMipWorker& worker : mipdata_->workers) {
      if (worker.search_ptr_->hasNode()) return true;
    }
    return false;
  };

  auto checkRestart = [&](const HighsMipWorker& worker,
                          HighsInt numWorkerVotes) -> RestartVote {
    int64_t nNodes = worker.search_ptr_->nnodes + mipdata_->num_nodes;
    if (!submip && nNodes >= nextCheck && options_mip_->mip_allow_restart) {
      const HighsInt nTreeRestarts =
          mipdata_->numRestarts - mipdata_->numRestartsRoot;
      const HighsCDouble treeWeight =
          worker.search_ptr_->treeweight + mipdata_->pruned_treeweight;
      double currNodeEstim =
          numNodesLastCheck - mipdata_->num_nodes_before_run +
          (nNodes - numNodesLastCheck) * double(1.0 - treeWeight) /
              std::max(double(treeWeight - treeweightLastCheck),
                       mipdata_->epsilon);
      // printf(
      //     "nTreeRestarts: %d, numNodesThisRun: %ld, numNodesLastCheck: %ld,
      //     " "currNodeEstim: %g, " "prunedTreeWeightDelta: %g,
      //     numHugeTreeEstim: %d, numLeavesThisRun:
      //     "
      //     "%ld\n",
      //     nTreeRestarts, mipdata_->num_nodes -
      //     mipdata_->num_nodes_before_run, numNodesLastCheck -
      //     mipdata_->num_nodes_before_run, currNodeEstim, 100.0 *
      //     double(mipdata_->pruned_treeweight - treeweightLastCheck),
      //     numHugeTreeEstim,
      //     mipdata_->num_leaves - mipdata_->num_leaves_before_run);

      bool doRestart = false;

      double activeIntegerRatio =
          1.0 - mipdata_->percentageInactiveIntegers() / 100.0;
      activeIntegerRatio *= activeIntegerRatio;
      bool hugeTree = false;

      if (!doRestart) {
        double gapReduction = 1.0;
        if (worker.upper_limit != kHighsInf) {
          double oldGap = upperLimLastCheck - lowerBoundLastCheck;
          double newGap = worker.upper_limit - mipdata_->lower_bound;
          gapReduction = oldGap / newGap;
        }

        if (gapReduction < 1.0 + (0.05 / activeIntegerRatio) &&
            currNodeEstim >= activeIntegerRatio * 20 *
                                 (nNodes - mipdata_->num_nodes_before_run)) {
          hugeTree = true;
        } else {
          numWorkerVotes = -kHighsIInf;
        }

        int64_t minHugeTreeOffset =
            (mipdata_->num_leaves + worker.search_ptr_->nleaves -
             mipdata_->num_leaves_before_run) /
            1000;
        int64_t minHugeTreeEstim = HighsIntegers::nearestInteger(
            activeIntegerRatio * (10 + minHugeTreeOffset) *
            std::pow(1.5, nTreeRestarts));

        doRestart = numHugeTreeEstim + numWorkerVotes >= minHugeTreeEstim;
      }

      if (doRestart) return RestartVote::kWouldRestart;
      if (hugeTree) return RestartVote::kHugeTree;
      return RestartVote::kNoHugeTree;
    }
    return RestartVote::kNoCheck;
  };

  auto performRestart = [&] {
    highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
                 "\nRestarting search from the root node\n");
    mipdata_->performRestart();
    profiling_->stop(kMipClockSearch);
  };

  auto checkWorkerRestartVotes =
      [&](const std::vector<RestartVote>& votes) -> bool {
    HighsInt numVoters = 0;
    HighsInt numHugeTreeVotes = 0;
    HighsInt numRestartVotes = 0;
    for (const RestartVote vote : votes) {
      if (vote != RestartVote::kNoCheck) {
        ++numVoters;
        if (vote == RestartVote::kWouldRestart) {
          numRestartVotes++;
        } else if (vote == RestartVote::kHugeTree) {
          numHugeTreeVotes++;
        }
      }
    }
    bool forcedRestart = false;
    if (options_mip_->mip_allow_restart && numVoters >= 2 &&
        numRestartVotes / static_cast<double>(numVoters) >= 0.25) {
      forcedRestart = true;
    }
    // Force a restart if enough individual workers vote for it
    if (forcedRestart) {
      performRestart();
      return true;
    }
    // Using joint information after workers are synced, query a restart
    const RestartVote vote = checkRestart(
        master_worker, (numRestartVotes + numHugeTreeVotes + 2) / 2);
    if (vote == RestartVote::kWouldRestart) {
      performRestart();
      return true;
    }

    // Update the appropriate values in case of no restart
    if (vote == RestartVote::kHugeTree) {
      nextCheck = mipdata_->num_nodes + 100;
      numHugeTreeEstim += (numRestartVotes + numHugeTreeVotes + 2) / 2;
    } else if (vote == RestartVote::kNoHugeTree) {
      numHugeTreeEstim = 0;
      treeweightLastCheck = static_cast<double>(mipdata_->pruned_treeweight);
      numNodesLastCheck = mipdata_->num_nodes;
      upperLimLastCheck = mipdata_->upper_limit;
      lowerBoundLastCheck = mipdata_->lower_bound;
    }
    return false;
  };

  auto getSearchIndicesWithNoNodes = [&](std::vector<HighsInt>& indices,
                                         std::vector<RestartVote>& restarts) {
    indices.clear();
    std::fill(restarts.begin(), restarts.end(), RestartVote::kNoCheck);
    const HighsInt num_search_workers =
        std::min(num_workers,
                 static_cast<HighsInt>(mipdata_->nodequeue.numActiveNodes()));
    const HighsInt num_heuristic_workers =
        mipdata_->upper_bound < kHighsInf
            ? std::max(HighsInt{1}, (num_search_workers + 3) / 4)
            : std::max(HighsInt{1}, (num_search_workers + 1) / 2);

    for (HighsInt i = 0; i < num_search_workers; i++) {
      assert(!mipdata_->workers[i].search_ptr_->hasNode());
      indices.emplace_back(i);
      mipdata_->workers[i].setAllowHeuristics(i < num_heuristic_workers);
    }
  };

  auto installNodes = [&](std::vector<HighsInt>& indices,
                          bool& limit_reached) -> void {
    for (const HighsInt i : indices) {
      if ((indices.size() == 1 && numQueueLeaves - lastLbLeave >= 10) ||
          (indices.size() > 1 && i == 0)) {
        mipdata_->workers[i].search_ptr_->installNode(
            mipdata_->nodequeue.popBestBoundNode());
        lastLbLeave = numQueueLeaves;
      } else {
        HighsInt bestBoundNodeStackSize =
            mipdata_->nodequeue.getBestBoundDomchgStackSize();
        double bestBoundNodeLb = mipdata_->nodequeue.getBestLowerBound();
        HighsNodeQueue::OpenNode nextNode(mipdata_->nodequeue.popBestNode());
        if (nextNode.lower_bound == bestBoundNodeLb &&
            (HighsInt)nextNode.domchgstack.size() == bestBoundNodeStackSize)
          lastLbLeave = numQueueLeaves;
        mipdata_->workers[i].search_ptr_->installNode(std::move(nextNode));
      }

      ++numQueueLeaves;

      if (mipdata_->workers[i].search_ptr_->getCurrentEstimate() >=
          mipdata_->upper_limit) {
        ++numStallNodes;
        if (options_mip_->mip_max_stall_nodes != kHighsIInf &&
            numStallNodes >= options_mip_->mip_max_stall_nodes) {
          limit_reached = true;
          modelstatus_ = HighsModelStatus::kSolutionLimit;
          break;
        }
      } else
        numStallNodes = 0;
    }
  };

  auto evaluateNode = [&](HighsInt i) -> HighsSearch::NodeResult {
    if (!mipdata_->parallelLockActive())
      profiling_->start(kMipClockEvaluateNode1);
    const HighsSearch::NodeResult result =
        mipdata_->workers[i].search_ptr_->evaluateNode();
    if (result == HighsSearch::NodeResult::kSubOptimal ||
        result == HighsSearch::NodeResult::kLpIterationLimit ||
        result == HighsSearch::NodeResult::kCandidateCheckpoint) {
      HighsNodeQueue& globalqueue = mipdata_->parallelLockActive()
                                        ? mipdata_->workers[i].nodequeue
                                        : mipdata_->nodequeue;
      mipdata_->workers[i].search_ptr_->currentNodeToQueue(globalqueue);
      if (!mipdata_->parallelLockActive())
        profiling_->stop(kMipClockEvaluateNode1);
      return result;
    }
    if (!mipdata_->parallelLockActive())
      profiling_->stop(kMipClockEvaluateNode1);
    return result;
  };

  auto pruneNode = [&](HighsInt i) -> bool {
    if (!mipdata_->parallelLockActive())
      profiling_->start(kMipClockNodePrunedLoop);
    bool pruned = false;
    if (mipdata_->workers[i].search_ptr_->currentNodePruned()) {
      mipdata_->workers[i].search_ptr_->backtrack();
      mipdata_->workers[i].getGlobalDomain().propagate();
      pruned = true;
      ++mipdata_->workers[i].search_ptr_->getLocalNodes();
      ++mipdata_->workers[i].search_ptr_->getLocalLeaves();
    }
    if (!mipdata_->parallelLockActive())
      profiling_->stop(kMipClockNodePrunedLoop);
    return mipdata_->workers[i].getGlobalDomain().infeasible() || pruned;
  };

  auto separateAndStoreBasis = [&](HighsInt i) -> bool {
    HighsMipWorker& worker = mipdata_->workers[i];
    HighsSearchWorkScope work_scope(worker.search_work,
                                    HighsSearchWorkRole::kSeparation);
    if (options_mip_->mip_allow_cut_separation_at_nodes &&
        !worker.search_work.skipOptional()) {
      if (!mipdata_->parallelLockActive())
        profiling_->start(kMipClockNodeSearchSeparation);
      HighsNodeSeparationConfig node_separation_config;
      node_separation_config.enabled =
          options_mip_->mip_node_separation_controller;
      node_separation_config.mode = HighsNodeSeparationController::modeFromInt(
          options_mip_->mip_node_cut_mode);
      node_separation_config.maximum_depth =
          options_mip_->mip_node_cut_maximum_depth;
      node_separation_config.node_frequency =
          options_mip_->mip_node_cut_frequency;
      node_separation_config.max_rounds =
          options_mip_->mip_node_max_separation_rounds;
      node_separation_config.max_seconds =
          options_mip_->mip_node_max_separation_time;
      node_separation_config.max_lp_iterations =
          options_mip_->mip_node_max_separation_lp_iterations;
      node_separation_config.max_added_rows =
          options_mip_->mip_node_max_cut_rows;
      node_separation_config.max_added_nonzeros =
          options_mip_->mip_node_max_cut_nonzeros;
      node_separation_config.max_selected_per_origin =
          options_mip_->mip_node_max_cuts_per_family;
      node_separation_config.max_row_growth_ratio =
          options_mip_->mip_node_max_row_growth;
      node_separation_config.max_nonzero_growth_ratio =
          options_mip_->mip_node_max_nonzero_growth;
      node_separation_config.low_value_patience =
          options_mip_->mip_node_cut_low_value_patience;
      node_separation_config.min_dual_gain_per_second =
          options_mip_->mip_node_cut_min_gain_per_second;
      node_separation_config.min_dual_gain_per_1000_iterations =
          options_mip_->mip_node_cut_min_gain_per_1000_iterations;
      node_separation_config.base_model_rows_only =
          options_mip_->mip_node_cut_base_model_rows_only;

      HighsNodeSeparationContext node_separation_context;
      node_separation_context.opportunity_sequence =
          worker.sepa_ptr_->nextNodeSeparationOpportunity();
      node_separation_context.tree_nodes_processed =
          mipdata_->num_nodes + worker.search_ptr_->nnodes;
      node_separation_context.depth = worker.search_ptr_->getCurrentDepth();
      node_separation_context.model_rows = numRow();
      node_separation_context.model_nonzeros = numNonzero();
      node_separation_context.active_rows = worker.getLpRelaxation().numRows();
      node_separation_context.active_nonzeros =
          worker.getLpRelaxation().numNonzeros();
      node_separation_context.lp_iterations =
          worker.getLpRelaxation().getNumLpIterations();
      node_separation_context.time = timer_.read();
      node_separation_context.objective =
          worker.getLpRelaxation().getObjective();
      node_separation_context.fractional_integers =
          worker.getLpRelaxation().getFractionalIntegers().size();
      worker.sepa_ptr_->separate(worker.search_ptr_->getLocalDomain(),
                                 node_separation_config,
                                 node_separation_context);
      if (!mipdata_->parallelLockActive())
        profiling_->stop(kMipClockNodeSearchSeparation);
    } else {
      worker.getCutPool().performAging();
    }

    if (worker.getGlobalDomain().infeasible()) {
      worker.search_ptr_->cutoffNode();
      HighsNodeQueue& globalqueue = mipdata_->parallelLockActive()
                                        ? worker.nodequeue
                                        : mipdata_->nodequeue;
      worker.search_ptr_->openNodesToQueue(globalqueue);
      return true;
    }

    if (worker.getLpRelaxation().getStatus() !=
            HighsLpRelaxation::Status::kError &&
        worker.getLpRelaxation().getStatus() !=
            HighsLpRelaxation::Status::kNotSet)
      worker.getLpRelaxation().storeBasis();

    std::shared_ptr<const HighsBasis> basis =
        worker.getLpRelaxation().getStoredBasis();
    if (!basis ||
        !isBasisConsistent(worker.getLpRelaxation().getLp(), *basis)) {
      HighsBasis b = mipdata_->firstrootbasis;
      b.row_status.resize(worker.getLpRelaxation().numRows(),
                          HighsBasisStatus::kBasic);
      basis = std::make_shared<const HighsBasis>(std::move(b));
      worker.getLpRelaxation().setStoredBasis(basis);
    }

    return false;
  };

  auto backtrackPlunge = [&](HighsInt i) {
    if (!mipdata_->parallelLockActive())
      profiling_->start(kMipClockBacktrackPlunge);
    const bool backtrack_plunge =
        mipdata_->workers[i].search_ptr_->backtrackPlunge(
            mipdata_->parallelLockActive() ? mipdata_->workers[i].nodequeue
                                           : mipdata_->nodequeue);
    if (!mipdata_->parallelLockActive())
      profiling_->stop(kMipClockBacktrackPlunge);

    if (!backtrack_plunge) return true;

    assert(mipdata_->workers[i].search_ptr_->hasNode());

    if (mipdata_->workers[i].getConflictPool().getNumConflicts() >
        options_mip_->mip_pool_soft_limit) {
      mipdata_->workers[i].getConflictPool().performAging();
    }
    return false;
  };

  auto runHeuristics = [&](HighsInt i, bool first_tree_lease) -> bool {
    HighsMipWorker& worker = mipdata_->workers[i];
    worker.importPrimalSnapshot();
    if (!mipdata_->parallelLockActive())
      profiling_->start(kMipClockDiveEvaluateNode);
    const HighsSearch::NodeResult evaluate_node_result =
        worker.search_ptr_->evaluateNode();
    if (!mipdata_->parallelLockActive())
      profiling_->stop(kMipClockDiveEvaluateNode);

    if (evaluate_node_result == HighsSearch::NodeResult::kSubOptimal ||
        evaluate_node_result ==
            HighsSearch::NodeResult::kLpIterationLimit ||
        evaluate_node_result == HighsSearch::NodeResult::kCandidateCheckpoint) {
      if (evaluate_node_result == HighsSearch::NodeResult::kLpIterationLimit ||
          evaluate_node_result == HighsSearch::NodeResult::kCandidateCheckpoint) {
        HighsNodeQueue& queue = mipdata_->parallelLockActive()
                                   ? worker.nodequeue
                                   : mipdata_->nodequeue;
        worker.search_ptr_->currentNodeToQueue(queue);
      }
      return true;
    }

    if (worker.search_ptr_->currentNodePruned()) {
      ++worker.search_ptr_->getLocalLeaves();
      return false;
    }

    // With a checked seed, first expose child nodes before optional repair.
    // Cold-start RENS and subsequent tree leases retain their allowance.
    const bool managed_repair = primal_portfolio ||
        (managed_rins_budget && !worker.repairIncumbent().empty());
    if (first_tree_lease && !managed_repair && !submip && max_num_workers > 1 &&
        !worker.repairIncumbent().empty())
      return false;

    if (worker.search_work.checkpointRequested() ||
        (!managed_repair && worker.search_work.skipOptional())) return false;
    HighsSearchWorkScope work_scope(worker.search_work,
                                    HighsSearchWorkRole::kHeuristic);

    if (!mipdata_->parallelLockActive())
      profiling_->start(kMipClockDivePrimalHeuristics);
    if (worker.repairIncumbent().empty()) {
      if (!mipdata_->parallelLockActive())
        profiling_->start(kMipClockDiveRandomizedRounding);
      mipdata_->heuristics.randomizedRounding(
          worker,
          worker.getLpRelaxation().getLpSolver().getSolution().col_value);
      if (!mipdata_->parallelLockActive())
        profiling_->stop(kMipClockDiveRandomizedRounding);
    }
    if (primal_portfolio) {
      mipdata_->heuristics.portfolio(worker,
          worker.getLpRelaxation().getLpSolver().getSolution().col_value);
    } else if (worker.repairIncumbent().empty()) {
      if (options_mip_->mip_heuristic_run_rens &&
          !worker.search_work.skipOptional()) {
        if (!mipdata_->parallelLockActive())
          profiling_->start(kMipClockDiveRens);
        const double helper_started = timer_.read();
        // A blanket root-helper cap can suppress the only cold-start seed.
        // Keep the existing tree RENS allowance; measure it separately.
        const double allowance = kHighsInf;
        mipdata_->heuristics.RENS(
            worker,
            worker.getLpRelaxation().getLpSolver().getSolution().col_value,
            allowance, HighsSubMipOrigin::kTreeRens);
        if (first_tree_lease && !submip && options_mip_->mip_root_work_budget)
          highsLogDev(options_mip_->log_options, HighsLogType::kInfo,
                      "MIP-TreeBootstrap: method=rens allowance=%.6g "
                      "elapsed=%.6g candidate=%s\n",
                      allowance, std::max(0.0, timer_.read() - helper_started),
                      worker.repairIncumbent().empty() ? "false" : "true");
        if (!mipdata_->parallelLockActive())
          profiling_->stop(kMipClockDiveRens);
      }
    } else {
      if (options_mip_->mip_heuristic_run_rins &&
          (managed_repair || !worker.search_work.skipOptional())) {
        if (!mipdata_->parallelLockActive())
          profiling_->start(kMipClockDiveRins);
        const std::vector<double>& relaxation_solution =
            worker.getLpRelaxation().getLpSolver().getSolution().col_value;
        if (mipdata_->heuristic_manager.enabled()) {
          HighsPrimalHeuristicContext context;
          context.method = HighsPrimalHeuristicMethod::kRins;
          context.time = timer_.read();
          context.restart = mipdata_->numRestarts;
          context.worker_id = worker.workerId();
          context.node = mipdata_->num_nodes + worker.search_ptr_->nnodes;
          context.main_lp_iterations = std::max<int64_t>(
              0, mipdata_->total_lp_iterations -
                     mipdata_->heuristic_lp_iterations);
          context.heuristic_lp_iterations =
              mipdata_->heuristic_lp_iterations;
          context.incumbent_sequence = worker.repairIncumbentSequence();
          context.recursion_depth = submip_level;
          context.neighbourhood_signature =
              rinsNeighbourhoodSignature(*mipdata_, relaxation_solution,
                                         worker.repairIncumbent());
          context.has_incumbent = !worker.repairIncumbent().empty();
          context.has_lp_solution =
              relaxation_solution.size() ==
              static_cast<size_t>(numCol());
          const HighsPrimalHeuristicDecision decision =
              mipdata_->heuristic_manager.before(context);
          if (decision.run) {
            HighsPrimalHeuristicOutcome outcome;
            const double helper_started = timer_.read();
            const int64_t helper_lp_started = worker.getHeurLpIterations();
            const HighsSubMipCallBudget* budget =
                mipdata_->heuristic_manager.config().schedule ==
                        HighsPrimalHeuristicSchedule::kManagedRins
                    ? &decision.budget
                    : nullptr;
            if (managed_repair) {
              HighsRepairReservationScope repair(worker.search_work,
                                                 decision.budget.max_lp_iterations);
              if (repair.granted())
                mipdata_->heuristics.RINS(worker, relaxation_solution, budget, &outcome);
              else
                outcome.reservation_denied = true;
            } else {
              mipdata_->heuristics.RINS(worker, relaxation_solution, budget, &outcome);
            }
            outcome.elapsed = std::max(outcome.elapsed, timer_.read() - helper_started);
            outcome.lp_iterations = std::max(
                outcome.lp_iterations, worker.getHeurLpIterations() - helper_lp_started);
            outcome.finished_time = timer_.read();
            mipdata_->heuristic_manager.record(context, outcome,
                                               decision.observation_id);
          }
        } else {
          mipdata_->heuristics.RINS(worker, relaxation_solution);
        }
        if (!mipdata_->parallelLockActive())
          profiling_->stop(kMipClockDiveRins);
      }
    }

    if (!mipdata_->parallelLockActive())
      profiling_->stop(kMipClockDivePrimalHeuristics);

    return worker.getGlobalDomain().infeasible();
  };

  auto dive = [&](HighsInt i, HighsInt nodeLim) -> bool {
    HighsMipWorker& worker = mipdata_->workers[i];
    const int64_t diveNodeLim = nodeLim == kHighsIInf
                                    ? std::numeric_limits<int64_t>::max()
                                    : worker.search_ptr_->nnodes + nodeLim;
    if (!worker.search_ptr_->currentNodePruned()) {
      if (!mipdata_->parallelLockActive()) profiling_->start(kMipClockTheDive);
      const HighsSearch::NodeResult search_dive_result =
          worker.search_ptr_->dive(diveNodeLim);
      if (!mipdata_->parallelLockActive()) profiling_->stop(kMipClockTheDive);
      if (search_dive_result == HighsSearch::NodeResult::kSubOptimal) {
        return true;
      }
      if (search_dive_result == HighsSearch::NodeResult::kTaskWorkLimit) {
        // openNodesToQueue at the epoch boundary owns the suspension.
        return true;
      }
      if (search_dive_result == HighsSearch::NodeResult::kLpSolveError) {
        // Leave the unresolved node open for the joined owner's error exit.
        return true;
      }
      if (search_dive_result ==
          HighsSearch::NodeResult::kLpIterationLimit ||
          search_dive_result == HighsSearch::NodeResult::kCandidateCheckpoint) {
        HighsNodeQueue& queue = mipdata_->parallelLockActive()
                                   ? worker.nodequeue
                                   : mipdata_->nodequeue;
        worker.search_ptr_->currentNodeToQueue(queue);
        return true;
      }
      worker.search_ptr_->getLocalLeaves()++;
    }
    return nodeLim != kHighsIInf && worker.search_ptr_->nnodes >= diveNodeLim;
  };

  std::unique_ptr<HighsPrimalService> primal_service;
  auto processNodes = [&](const std::vector<HighsInt>& indices,
                          std::vector<RestartVote>& restarts,
                          const bool skip_separation, const HighsInt nodeLim,
                          const HighsInt plungeLimit, double avgiter) {
    const uint64_t batch_sequence = candidate_checkpoint.sequence();
    const HighsSearchCoordinator::Generation dispatch_generation{
        "native-epoch-model", "native-epoch-mapping",
        static_cast<uint64_t>(callback_->data_in.admission_generation), 0,
        static_cast<uint64_t>(mipdata_->numRestarts)};
    std::unique_ptr<HighsSearchCoordinator> dispatch;
    std::vector<int64_t> dispatch_nodes(indices.size(), 0);
    if (!submip && options_mip_->mip_flexible_node_dispatch &&
        options_mip_->mip_incumbent_exchange != "deterministic" &&
        indices.size() > 1 &&
        indices.size() <= size_t(highs::parallel::num_threads()) &&
        !options_mip_->mip_search_simulate_concurrency)
      dispatch.reset(new HighsSearchCoordinator(
          dispatch_generation, numCol(), indices.size(),
          size_t{64} * 1024 * 1024, mipdata_->lower_bound));
    std::shared_ptr<HighsPrimalSnapshotStore> primal_store;
    if (opportunisticIncumbentExchange())
      primal_store = std::make_shared<HighsPrimalSnapshotStore>(
          *orig_model_, *model_, mipdata_->feastol,
          callback_->data_in.admission_generation,
          mipdata_->workers[indices.front()].workEpoch(), timer_.read());
    if (primal_store && !mipdata_->incumbent.empty()) {
      HighsWorkerCandidate seed;
      seed.solution = mipdata_->incumbent;
      seed.native_objective = mipdata_->upper_bound;
      seed.solution_source = kSolutionSourceHighsSolution;
      seed.metadata.admission_generation = callback_->data_in.admission_generation;
      seed.metadata.work_epoch = mipdata_->workers[indices.front()].workEpoch();
      seed.metadata.discovery_node = mipdata_->num_nodes;
      seed.metadata.discovery_lp_iterations = mipdata_->total_lp_iterations;
      seed.metadata.discovery_time = seed.metadata.publication_time = timer_.read();
      primal_store->publish(seed, solution_);
    }
    try {
      if (primal_store && callback_->primal_service_callback) {
        if (primal_service) {
          primal_service->nextBatch(
              primal_store, callback_->data_in, mipdata_->lower_bound,
              mipdata_->workers[indices.front()].workEpoch());
        } else {
          primal_service.reset(new HighsPrimalService(
              primal_store, *orig_model_, *model_, *options_mip_,
              mipdata_->postSolveStack, callback_->data_in,
              callback_->primal_service_callback, callback_->primal_service_data,
              mipdata_->lower_bound, mipdata_->workers[indices.front()].workEpoch(),
              options_mip_->parallel == kHighsOnString &&
                  highs::parallel::num_threads() > 1));
        }
      }
      if (primal_service) primal_service->start();
    } catch (const std::exception& error) {
      // No worker owns the service yet. Keep installed nodes unresolved and
      // let joined reconciliation turn the result into an explicit error.
      service_failed = true;
      highsLogUser(options_mip_->log_options, HighsLogType::kError,
                   "Cannot start private primal service: %s\n", error.what());
      return;
    }
    for (HighsInt i : indices)
      mipdata_->workers[i].attachPrimalStore(primal_store);
    for (HighsInt i : indices)
      mipdata_->workers[i].attachPrimalService(primal_service.get());
    auto processNode = [&](HighsInt i) {
      HighsMipWorker& worker = mipdata_->workers[i];
      const size_t dispatch_slot =
          std::find(indices.begin(), indices.end(), i) - indices.begin();
      if (dispatch)
        worker.search_ptr_->setDonationSink([&](const HighsDetachedNode& node) {
          return dispatch->offer(dispatch_generation, node);
        });
      const double lease_started = timer_.read();
      const int64_t lease_started_nodes = worker.search_ptr_->nnodes;
      // Donation changes node ownership, not the receiving lease's allowance.
      auto remainingLeaseNodes = [&]() -> HighsInt {
        if (!dispatch || nodeLim == kHighsIInf) return nodeLim;
        const int64_t used = worker.search_ptr_->nnodes - lease_started_nodes;
        assert(used >= 0);
        return used >= nodeLim ? 0 : nodeLim - HighsInt(used);
      };
      worker.search_work.begin(submip ? 0 : options_mip_->mip_search_work_lp_limit,
                               options_mip_->mip_search_work_enforce, lease_started);
      worker.search_work.attachCheckpoint(
          !submip && options_mip_->mip_candidate_checkpoint
              ? &candidate_checkpoint : nullptr, batch_sequence);
      auto run_lease = [&]() {
        int64_t nodes_explored = 0;
        if (remainingLeaseNodes() == 0) return;
        if (worker.search_work.checkpointRequested()) {
          worker.search_work.recordCandidateYield();
          return;
        }
        if (!skip_separation) {
          const HighsSearch::NodeResult evaluate_result = evaluateNode(i);
          if (evaluate_result ==
              HighsSearch::NodeResult::kLpIterationLimit ||
              evaluate_result == HighsSearch::NodeResult::kCandidateCheckpoint) {
            // The node was requeued with its last certified lower bound and
            // compatible basis. Remove the local search frame without
            // recording a processed node or pruned leaf.
            worker.search_ptr_->backtrack();
            worker.getGlobalDomain().propagate();
            return;
          }
          if (pruneNode(i)) return;
          if (!mipdata_->parallelLockActive()) {
            if (mipdata_->checkLimits()) return;
            mipdata_->printDisplayLine();
          } else {
            if (worker.search_ptr_->checkLocalLimits()) return;
          }
          if (separateAndStoreBasis(i)) return;
        }
        worker.getConflictPool().performAging();
        if (!worker.nodeLpEffortEnabled()) {
          HighsInt iterlimit =
              10 * std::max(avgiter, mipdata_->avgrootlpiters);
          iterlimit = std::max(
              {HighsInt{10000}, iterlimit,
               HighsInt((3 * mipdata_->firstrootlpiters) / 2)});
          worker.getLpRelaxation().setIterationLimit(iterlimit);
        }
        bool considerHeuristics = true;
        while (true) {
          if (remainingLeaseNodes() == 0) break;
          if (worker.search_work.checkpointRequested()) {
            worker.search_work.recordCandidateYield();
            break;
          }
          if (considerHeuristics &&
              (skip_separation || nodeLim == kHighsIInf ||
               boundedIncumbentEpochs()) &&
              worker.getAllowHeuristics() &&
              (mipdata_->moreHeuristicsAllowed() ||
               primal_portfolio ||
               (managed_rins_budget && !worker.repairIncumbent().empty()))) {
            if (runHeuristics(i, skip_separation)) break;
          }
          considerHeuristics = false;
          if (worker.getGlobalDomain().infeasible()) break;
          if (dive(i, remainingLeaseNodes())) break;
          if (worker.search_ptr_->checkLimits(
                  worker.search_ptr_->getLocalNodes())) {
            break;
          }

          if (worker.search_ptr_->getLocalNodes() + nodes_explored >=
              plungeLimit)
            break;
          if (!mipdata_->parallelLockActive()) {
            nodes_explored += worker.search_ptr_->getLocalNodes();
          }
          if (backtrackPlunge(i)) break;
          if (!mipdata_->parallelLockActive()) {
            worker.search_ptr_->flushStatistics(*this);
            mipdata_->printDisplayLine();
          }
        }
        if (nodeLim == kHighsIInf) {
          restarts[i] = checkRestart(worker, 1);
        }
      };
      try {
        run_lease();
        if (dispatch) {
          for (int repeat = 0; repeat != 8; ++repeat) {
            worker.search_ptr_->openNodesToQueue(worker.nodequeue);
            if (remainingLeaseNodes() == 0 ||
                worker.search_ptr_->lpRecoveryFailed() ||
                worker.search_work.checkpointRequested() ||
                worker.search_work.skipOptional() ||
                worker.search_ptr_->checkLocalLimits())
              break;
            HighsSearchCoordinator::Receipt receipt;
            if (!dispatch->take(dispatch_slot, receipt, true)) break;
            worker.search_ptr_->installNode(receipt.node->node());
            if (!dispatch->installed(receipt)) {
              dispatch->fail();
              break;
            }
            run_lease();
          }
          dispatch->retire(dispatch_slot);
        }
      } catch (...) {
        if (!dispatch) throw;
        // Preserve pending receipts on install failure. No early proof exit.
        dispatch->fail();
      }
      worker.search_ptr_->setDonationSink({});
      worker.search_work.finish();
      dispatch_nodes[dispatch_slot] =
          worker.search_ptr_->nnodes - lease_started_nodes;
      worker.recordTaskLease(std::max(0.0, timer_.read() - lease_started));
    };
    runTask(processNode, tg, true, false, indices);
    if (dispatch) {
      service_failed |= dispatch->failed();
      for (auto& node : dispatch->drain())
        mipdata_->nodequeue.emplaceNode(
            std::move(node.changes), std::move(node.branchings), node.bound,
            node.estimate, node.depth, std::move(node.basis), node.yields);
      highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
                   "MIP-Dispatch: version=1 offered=%zu installed=%zu "
                   "coverage_bound=%.17g failed=%d proof=joined_owner\n",
                   dispatch->offered(), dispatch->installed(),
                   dispatch->conservativeBound(), int(dispatch->failed()));
      highsLogDev(options_mip_->log_options, HighsLogType::kInfo,
                  "MIP-DispatchBudget: node_limit=%d max_nodes=%lld\n",
                  int(nodeLim), static_cast<long long>(
                      *std::max_element(dispatch_nodes.begin(), dispatch_nodes.end())));
    }
    if (primal_service) {
      primal_service->stop();
      service_failed |= primal_service->failed();
      joined_service_input = primal_service->joinedInput();
      service_had_owner = true;
      service_queries = primal_service->queries();
      service_notifications = primal_service->notifications();
    }
    if (primal_store) joined_primal = primal_store->latest();
    for (HighsInt i : indices) mipdata_->workers[i].attachPrimalService(nullptr);
  };

  auto syncSepaStats = [&](HighsMipWorker& worker) {
    mipdata_->cliquetable.getNumNeighbourhoodQueries() +=
        worker.getNumNeighbourhoodQueries();
    mipdata_->sepa_lp_iterations += worker.getSepaLpIterations();
    mipdata_->total_lp_iterations += worker.getSepaLpIterations();
    worker.resetSepaStats();
  };

  // Main solve loop
  std::vector<HighsInt> search_indices(1, 0);
  search_indices.reserve(max_num_workers);
  std::vector<RestartVote> workerRestartVotes(getMaxNumWorkers(),
                                              RestartVote::kNoCheck);
  bool root_node = true;  // Don't separate the root node again
  HighsInt nodeLim = max_num_workers > 1 ? 1 : kHighsIInf;  // for ramp up
  int64_t work_epoch = 0;
  while (!mipdata_->nodequeue.empty()) {
    // Possibly query existence of an external solution
    if (!submip)
      mipdata_->queryExternalSolution(
          solution_objective_, kExternalMipSolutionQueryOriginBeforeDive);

    // Update global pseudo-cost with worker information
    syncGlobalPseudoCost();

    // Get new candidate worker search indices
    getSearchIndicesWithNoNodes(search_indices, workerRestartVotes);

    // Only update worker's pseudo-costs that have been assigned a node
    resetWorkerPseudoCosts(search_indices);

    // Assign nodes to workers
    bool limit_reached = false;
    if (root_node) {
      master_worker.search_ptr_->installNode(
          mipdata_->nodequeue.popBestBoundNode());
    } else {
      installNodes(search_indices, limit_reached);
    }
    if (limit_reached) break;

    if (nodeLim != kHighsIInf) {
      if (num_workers >= max_num_workers) {
        nodeLim = std::max(HighsInt{20}, 2 * nodeLim);
      }
      if (boundedIncumbentEpochs()) {
        nodeLim = std::min(nodeLim,
                           options_mip_->mip_incumbent_epoch_nodes);
      } else if (nodeLim > 100) {
        nodeLim = kHighsIInf;
      }
    }

    for (HighsInt i : search_indices)
      mipdata_->workers[i].setWorkEpoch(work_epoch);
    ++work_epoch;

    // Process nodes (separation / heuristics / dives)
    processNodes(search_indices, workerRestartVotes, root_node, nodeLim, 100,
                 mipdata_->getLp().getAvgSolveIters());

    // Read worker-owned receipts only after runTask has joined the batch.
    if (!submip && options_mip_->mip_candidate_checkpoint)
      for (HighsInt i : search_indices) {
        const auto& work = mipdata_->workers[i].search_work;
        if (work.firstCheckpointSeconds() < 0 || checkpoint_receipts >= 128) continue;
        ++checkpoint_receipts;
        highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
          "MIP-CandidateReturn: schema=rost/highs-candidate-return/v1 worker=%d "
          "batch_sequence=%llu observed_s=%.17g returned_s=%.17g observed_phase=%s\n",
          int(i), static_cast<unsigned long long>(work.batchSequence()),
          work.firstCheckpointSeconds(), work.taskReturnSeconds(),
          highsTaskPhaseName(work.checkpointPhase()));
      }

    root_node = false;

    // Sync statistics, check infeasibility, and flush nodes from worker queues
    bool infeasible = false;
    bool lp_recovery_failed = false;
    for (HighsInt i : search_indices) {
      HighsMipWorker& worker = mipdata_->workers[i];
      lp_recovery_failed |= worker.search_ptr_->lpRecoveryFailed();
      if (worker.getGlobalDomain().infeasible()) {
        infeasible = true;
      }
      profiling_->start(kMipClockOpenNodesToQueue0);
      worker.search_ptr_->openNodesToQueue(mipdata_->nodequeue);
      while (worker.nodequeue.numNodes() > 0) {
        HighsNodeQueue::OpenNode node =
            std::move(worker.nodequeue.popBestNode());
        mipdata_->nodequeue.emplaceNode(
            std::move(node.domchgstack), std::move(node.branchings),
            node.lower_bound, node.estimate, node.depth, std::move(node.basis),
            node.yield_count);
      }
      profiling_->stop(kMipClockOpenNodesToQueue0);
      worker.search_ptr_->flushStatistics(*this);
      syncSepaStats(worker);
      mipdata_->heuristics.flushStatistics(*this, worker);
    }

    // Workers may prove their improving search empty after publishing its
    // final incumbent. Admit joined packets before any terminal proof exit.
    const int64_t admissions_before_sync = candidate_admitted;
    if (!syncSolutions()) break;

    if (lp_recovery_failed) {
      highsLogUser(options_mip_->log_options, HighsLogType::kWarning,
                   "Node LP recovery failed; unresolved search is not an "
                   "infeasibility or optimality proof.\n");
      modelstatus_ = HighsModelStatus::kSolveError;
      break;
    }

    if (infeasible) {
      mipdata_->nodequeue.clear();
      mipdata_->pruned_treeweight = 1.0;
      mipdata_->updateLowerBound(std::min(kHighsInf, mipdata_->upper_bound));
      break;
    }

    mipdata_->updateLowerBound(std::min(
        mipdata_->upper_bound, mipdata_->nodequeue.getBestLowerBound()));

    limit_reached = mipdata_->checkLimits();
    if (limit_reached) {
      mipdata_->printDisplayLine();
      break;
    }

    if (!submip && options_mip_->mip_candidate_checkpoint &&
        candidate_admitted > admissions_before_sync) {
      ++post_admission_queries;
      mipdata_->queryExternalSolution(
          solution_objective_, kExternalMipSolutionQueryOriginBeforeDive);
      if (mipdata_->checkLimits()) break;
    }

    assert(!nodesInstalled());

    // Sync global information
    profiling_->start(kMipClockDomainPropgate);
    syncPools(search_indices);
    syncGlobalDomain(search_indices);
    mipdata_->getDomain().propagate();
    profiling_->stop(kMipClockDomainPropgate);

    profiling_->start(kMipClockPruneInfeasibleNodes);
    mipdata_->pruned_treeweight += mipdata_->nodequeue.pruneInfeasibleNodes(
        mipdata_->getDomain(), mipdata_->feastol);
    profiling_->stop(kMipClockPruneInfeasibleNodes);

    if (mipdata_->getDomain().infeasible()) {
      mipdata_->nodequeue.clear();
      mipdata_->pruned_treeweight = 1.0;
      mipdata_->updateLowerBound(std::min(kHighsInf, mipdata_->upper_bound));
      mipdata_->printDisplayLine();
      break;
    }

    mipdata_->updateLowerBound(std::min(
        mipdata_->upper_bound, mipdata_->nodequeue.getBestLowerBound()));
    mipdata_->printDisplayLine();

    if (mipdata_->nodequeue.empty()) break;

    // Reset global domain and sync worker's global domains.
    bool spawn_more_workers = num_workers < max_num_workers &&
                              mipdata_->nodequeue.numNodes() > num_workers;
    if (!resetGlobalDomain(spawn_more_workers, mipdata_->hasMultipleWorkers())) {
      // Candidate admission precedes reconciliation, so retain the admitted
      // incumbent while closing the now-infeasible improving search domain.
      mipdata_->nodequeue.clear();
      mipdata_->pruned_treeweight = 1.0;
      mipdata_->updateLowerBound(std::min(kHighsInf, mipdata_->upper_bound));
      mipdata_->printDisplayLine();
      break;
    }

    if ((nodeLim == kHighsIInf || boundedIncumbentEpochs()) &&
        checkWorkerRestartVotes(workerRestartVotes))
      goto restart;

    if (spawn_more_workers) {
      HighsInt new_max_num_workers =
          std::min(static_cast<HighsInt>(mipdata_->nodequeue.numNodes()),
                   max_num_workers);
      mipdata_->getPseudoCost().removeChanged();
      if (num_workers == 1) {
        constructAdditionalWorkerData(master_worker);
      }
      createNewWorkers(new_max_num_workers - num_workers);
      num_workers = new_max_num_workers;
    }
  }
  if (modelstatus_ != HighsModelStatus::kSolveError) syncSolutions();
  if (opportunisticIncumbentExchange()) {
    uint64_t resident_boundaries = 0;
    HighsNodeLpEffortSummary resident;
    for (const auto& worker : mipdata_->workers) {
      resident_boundaries += worker.residentBoundaries();
      const auto& observed = worker.residentLpSummary();
      resident.attempts += observed.attempts;
      resident.completed += observed.completed;
      resident.yielded += observed.yielded;
      resident.limit_stops += observed.limit_stops;
      resident.numerical_failures += observed.numerical_failures;
      resident.total_iterations += observed.total_iterations;
      resident.maximum_iterations = std::max(resident.maximum_iterations,
                                             observed.maximum_iterations);
    }
    highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
        "MIP-PrimalService: schema=rost/highs-primal-service/v2 "
        "queries=%llu notifications=%llu resident_boundaries=%llu "
        "resident_attempts=%lld resident_completed=%lld resident_yielded=%lld "
        "resident_limit_stops=%lld resident_failures=%lld "
        "resident_lp_iterations=%lld resident_lp_max=%lld "
        "failed=%d admission=checked_record proof_updates=joined\n",
        static_cast<unsigned long long>(service_queries),
        static_cast<unsigned long long>(service_notifications),
        static_cast<unsigned long long>(resident_boundaries),
        static_cast<long long>(resident.attempts),
        static_cast<long long>(resident.completed),
        static_cast<long long>(resident.yielded),
        static_cast<long long>(resident.limit_stops),
        static_cast<long long>(resident.numerical_failures),
        static_cast<long long>(resident.total_iterations),
        static_cast<long long>(resident.maximum_iterations), int(service_failed));
  }
  profiling_->stop(kMipClockSearch);

  if (options_mip_->mip_node_lp_effort && !submip) {
    HighsNodeLpEffortSummary aggregate;
    for (const HighsMipWorker& worker : mipdata_->workers) {
      const HighsNodeLpEffortSummary& summary = worker.nodeLpEffortSummary();
      aggregate.attempts += summary.attempts;
      aggregate.completed += summary.completed;
      aggregate.yielded += summary.yielded;
      aggregate.serial_completions += summary.serial_completions;
      aggregate.limit_stops += summary.limit_stops;
      aggregate.numerical_failures += summary.numerical_failures;
      aggregate.total_iterations += summary.total_iterations;
      aggregate.completed_iterations += summary.completed_iterations;
      aggregate.yielded_iterations += summary.yielded_iterations;
      aggregate.maximum_iterations =
          std::max(aggregate.maximum_iterations, summary.maximum_iterations);
      aggregate.task_leases += summary.task_leases;
      aggregate.task_lease_total_seconds +=
          summary.task_lease_total_seconds;
      aggregate.task_lease_max_seconds = std::max(
          aggregate.task_lease_max_seconds, summary.task_lease_max_seconds);
      for (size_t i = 0; i != aggregate.histogram.size(); ++i)
        aggregate.histogram[i] += summary.histogram[i];
      for (size_t i = 0; i != aggregate.yielded_histogram.size(); ++i)
        aggregate.yielded_histogram[i] += summary.yielded_histogram[i];
      for (size_t i = 0;
           i != aggregate.task_lease_microseconds_histogram.size(); ++i)
        aggregate.task_lease_microseconds_histogram[i] +=
            summary.task_lease_microseconds_histogram[i];
    }
    using IterationHistogram = std::array<
        int64_t, HighsNodeLpEffortSummary::kHistogramBuckets>;
    auto iteration_percentile = [&](const IterationHistogram& histogram,
                                    double quantile) {
      int64_t samples = 0;
      for (int64_t count : histogram) samples += count;
      if (samples == 0) return HighsInt{0};
      const int64_t target = std::max<int64_t>(
          1, static_cast<int64_t>(std::ceil(quantile * samples)));
      int64_t cumulative = 0;
      for (size_t bucket = 0; bucket != histogram.size(); ++bucket) {
        cumulative += histogram[bucket];
        if (cumulative >= target) {
          if (bucket + 1 >= histogram.size()) return kHighsIInf;
          return static_cast<HighsInt>(uint64_t{1} << (bucket + 1));
        }
      }
      return aggregate.maximum_iterations;
    };
    const double task_lease_p50_seconds =
        HighsNodeLpEffortController::taskLeasePercentileUpperSeconds(
            aggregate.task_lease_microseconds_histogram, 0.50);
    const double task_lease_p95_seconds =
        HighsNodeLpEffortController::taskLeasePercentileUpperSeconds(
            aggregate.task_lease_microseconds_histogram, 0.95);
    highsLogUser(
        options_mip_->log_options, HighsLogType::kInfo,
        "MIP-NodeLpEffort: schema=rost/highs-node-lp-effort/v3 "
        "event=summary attempts=%lld completed=%lld yielded=%lld "
        "serial_completions=%lld limit_stops=%lld numerical_failures=%lld "
        "lp_iterations=%lld completed_lp_iterations=%lld "
        "yielded_lp_iterations=%lld completed_p50=%" HIGHSINT_FORMAT
        " completed_p95=%" HIGHSINT_FORMAT
        " yielded_p50=%" HIGHSINT_FORMAT
        " yielded_p95=%" HIGHSINT_FORMAT " max=%" HIGHSINT_FORMAT
        " task_leases=%lld task_lease_total=%.17g "
        "task_lease_p50=%.17g task_lease_p95=%.17g "
        "task_lease_max=%.17g "
        "task_lease_quantiles=histogram_upper_bounds\n",
        static_cast<long long>(aggregate.attempts),
        static_cast<long long>(aggregate.completed),
        static_cast<long long>(aggregate.yielded),
        static_cast<long long>(aggregate.serial_completions),
        static_cast<long long>(aggregate.limit_stops),
        static_cast<long long>(aggregate.numerical_failures),
        static_cast<long long>(aggregate.total_iterations),
        static_cast<long long>(aggregate.completed_iterations),
        static_cast<long long>(aggregate.yielded_iterations),
        iteration_percentile(aggregate.histogram, 0.50),
        iteration_percentile(aggregate.histogram, 0.95),
        iteration_percentile(aggregate.yielded_histogram, 0.50),
        iteration_percentile(aggregate.yielded_histogram, 0.95),
        aggregate.maximum_iterations,
        static_cast<long long>(aggregate.task_leases),
        aggregate.task_lease_total_seconds, task_lease_p50_seconds,
        task_lease_p95_seconds,
        aggregate.task_lease_max_seconds);
  }

  if (!submip) {
    HighsSearchWorkSummary sum;
    for (const HighsMipWorker& worker : mipdata_->workers) {
      const auto& s = worker.search_work.summary();
      if (options_mip_->mip_root_work_budget) {
        highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
            "MIP-StrongProbe: version=1 event=summary worker=%d "
            "scope=surviving_worker_tree_tasks calls=%lld certified=%lld "
            "unavailable=%lld dropped=%lld retained=%llu\n", int(worker.workerId()),
            static_cast<long long>(worker.search_work.probeCalls()),
            static_cast<long long>(worker.search_work.probeCertified()),
            static_cast<long long>(worker.search_work.probeUnavailable()),
            static_cast<long long>(worker.search_work.probeDropped()),
            static_cast<unsigned long long>(worker.search_work.strongProbes().size()));
        for (const auto& probe : worker.search_work.strongProbes())
          highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
              "MIP-StrongProbe: version=1 event=sample worker=%d lease=%lld "
              "node=%lld depth=%lld column=%lld candidates=%lld reliable_directions=%lld "
              "up=%d lp_iterations=%lld seconds=%.17g status=%d certified=%d "
              "scope=optional_probe_only\n", int(worker.workerId()),
              static_cast<long long>(probe.lease), static_cast<long long>(probe.node),
              static_cast<long long>(probe.depth), static_cast<long long>(probe.column),
              static_cast<long long>(probe.candidates),
              static_cast<long long>(probe.reliable_directions), int(probe.up),
              static_cast<long long>(probe.iterations), probe.seconds,
              probe.status, int(probe.certified));
      }
      for (const auto& task : worker.search_work.tasks())
        highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
          "MIP-TaskWork: worker=%d scope=local_completed_relaxations work=%lld "
          "started_s=%.17g seconds=%.17g restoration_s=%.17g local_bound_gain=%.17g "
          "completion_exemptions=%lld\n",
          int(worker.workerId()), static_cast<long long>(task.work), task.started_seconds, task.seconds,
          task.restoration_seconds, task.local_bound_gain,
          static_cast<long long>(task.completion_exemptions));
      sum.leases += s.leases;
      sum.exhausted_leases += s.exhausted_leases;
      sum.yields += s.yields;
      sum.skipped_optional += s.skipped_optional;
      sum.submip_reported_iterations += s.submip_reported_iterations;
      sum.max_lease_work = std::max(sum.max_lease_work, s.max_lease_work);
      sum.restoration_calls += s.restoration_calls;
      sum.deferred_restorations += s.deferred_restorations;
      sum.restoration_seconds += s.restoration_seconds;
      sum.local_bound_observations += s.local_bound_observations;
      sum.local_bound_gain += s.local_bound_gain;
      sum.completion_exemptions += s.completion_exemptions;
      sum.task_records_dropped += s.task_records_dropped;
      sum.repair_reservations += s.repair_reservations;
      sum.repair_iterations += s.repair_iterations;
      for (size_t i = 0; i < sum.direct_iterations.size(); ++i)
        sum.direct_iterations[i] += s.direct_iterations[i];
      for (size_t i = 0; i < sum.lp_calls.size(); ++i)
        sum.lp_calls[i].merge(s.lp_calls[i]);
    }
    for (size_t i = 0; i < sum.lp_calls.size(); ++i) {
      const auto& work = sum.lp_calls[i];
      highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
          "MIP-LpWork: schema=rost/highs-lp-work/v1 event=summary "
          "scope=surviving_worker_tree_tasks kind=%s calls=%lld "
          "simplex_iterations=%lld ipm_iterations=%lld "
          "unavailable_iteration_calls=%lld unavailable_seconds_calls=%lld "
          "seconds=%.17g\n",
          highsLpWorkKindName(static_cast<HighsLpWorkKind>(i)),
          static_cast<long long>(work.calls),
          static_cast<long long>(work.simplex_iterations),
          static_cast<long long>(work.ipm_iterations),
          static_cast<long long>(work.unavailable_iteration_calls),
          static_cast<long long>(work.unavailable_seconds_calls), work.seconds);
    }
    highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
        "MIP-SearchWork: schema=rost/highs-search-work/v1 event=summary "
        "scope=surviving_worker_tree_tasks enforce=%d limit=%lld leases=%lld exhausted=%lld "
        "yields=%lld skipped_optional=%lld evaluation_lp=%lld "
        "separation_lp=%lld heuristic_lp=%lld branch_lp=%lld "
        "submip_reported_lp=%lld max_lease_work=%lld restoration_calls=%lld "
        "restoration_us=%lld local_bound_observations=%lld "
        "completion_exemptions=%lld task_records_dropped=%lld deferred_restorations=%lld "
        "repair_reservations=%lld repair_iterations=%lld\n",
        int(options_mip_->mip_search_work_enforce && options_mip_->mip_search_work_lp_limit > 0),
        static_cast<long long>(options_mip_->mip_search_work_lp_limit),
        static_cast<long long>(sum.leases),
        static_cast<long long>(sum.exhausted_leases),
        static_cast<long long>(sum.yields),
        static_cast<long long>(sum.skipped_optional),
        static_cast<long long>(sum.direct_iterations[0]),
        static_cast<long long>(sum.direct_iterations[1]),
        static_cast<long long>(sum.direct_iterations[2]),
        static_cast<long long>(sum.direct_iterations[3]),
        static_cast<long long>(sum.submip_reported_iterations),
        static_cast<long long>(sum.max_lease_work),
        static_cast<long long>(sum.restoration_calls),
        static_cast<long long>(sum.restoration_seconds * 1e6),
        static_cast<long long>(sum.local_bound_observations),
        static_cast<long long>(sum.completion_exemptions),
        static_cast<long long>(sum.task_records_dropped),
        static_cast<long long>(sum.deferred_restorations),
        static_cast<long long>(sum.repair_reservations),
        static_cast<long long>(sum.repair_iterations));
    highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
        "MIP-TaskGain: scope=local_completed_relaxations "
        "bound_gain=%.17g leases=%lld restoration_s=%.17g "
        "global_bound_before_cleanup=%.17g\n",
        sum.local_bound_gain, static_cast<long long>(sum.leases),
        sum.restoration_seconds, mipdata_->lower_bound);
  }
  if (!submip && options_mip_->mip_candidate_checkpoint) {
    int64_t yields = 0;
    for (const HighsMipWorker& worker : mipdata_->workers)
      yields += worker.search_work.summary().candidate_yields;
    highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
      "MIP-CandidateCheckpoint: schema=rost/highs-candidate-checkpoint/v1 "
      "requests=%llu yields=%lld checked=%lld admitted=%lld "
      "post_admission_queries=%lld publication_wait_max_s=%.17g "
      "admission_call_s=%.17g\n",
      static_cast<unsigned long long>(candidate_checkpoint.sequence()),
      static_cast<long long>(yields), static_cast<long long>(candidate_checked),
      static_cast<long long>(candidate_admitted),
      static_cast<long long>(post_admission_queries), candidate_wait_max,
      candidate_check_seconds);
  }
  if (opportunisticIncumbentExchange()) {
    int64_t imports = 0;
    double wait = 0;
    for (const auto& worker : mipdata_->workers) {
      imports += worker.peerImports();
      wait = std::max(wait, worker.peerWaitMax());
    }
    highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
        "MIP-PeerPrimal: schema=rost/highs-peer-primal/v1 "
        "mode=opportunistic scope=surviving_workers peer_imports=%lld "
        "peer_wait_max_s=%.17g authority=checked_record_joined_proof\n",
        static_cast<long long>(imports), wait);
  }
  cleanupSolve();
}

void HighsMipSolver::cleanupSolve() {
  for (HighsMipWorker& worker : mipdata_->workers) {
    assert(worker.incumbent_mailbox_.size() == 0);
  }
  if (mipdata_->terminatorActive()) {
    if (mipdata_->terminatorTerminated()) {
      // Indicate that this instance has been interrupted
      modelstatus_ = HighsModelStatus::kHighsInterrupt;
    } else if (!submip) {
      // When sub-MIPs call cleanupSolve(), they generally don't have
      // a termination criterion for the whole MIP solver
      //
      // Possibly allow sub-MIPs to terminate if the time limit is
      // reached
      //
      // No other instance has terminated, so terminate
      mipdata_->terminatorTerminate();
    }
  }

  // Force a final logging line
  mipdata_->printDisplayLine(kSolutionSourceCleanup);
  mipdata_->logPrimalHeuristicSummary();
  // Stop the solve clock - which won't be running if presolve
  // determines the model status
  if (profiling_->running(kSolveTime)) profiling_->stop(kSolveTime);
  // Need to complete the calculation of P-D integral, checking for NO
  // gap change
  mipdata_->updatePrimalDualIntegral(
      mipdata_->lower_bound, mipdata_->lower_bound, mipdata_->upper_bound,
      mipdata_->upper_bound, false);
  profiling_->start(kPostsolveTime);

  bool havesolution = solution_objective_ != kHighsInf;
  bool feasible;
  if (havesolution)
    feasible =
        bound_violation_ <= options_mip_->mip_feasibility_tolerance &&
        integrality_violation_ <= options_mip_->mip_feasibility_tolerance &&
        row_violation_ <= options_mip_->mip_feasibility_tolerance;
  else
    feasible = false;

  dual_bound_ = mipdata_->lower_bound;
  if (mipdata_->objectiveFunction.isIntegral()) {
    double rounded_lower_bound =
        std::ceil(mipdata_->lower_bound *
                      mipdata_->objectiveFunction.integralScale() -
                  mipdata_->feastol) /
        mipdata_->objectiveFunction.integralScale();
    dual_bound_ = std::max(dual_bound_, rounded_lower_bound);
  }
  dual_bound_ += model_->offset_;
  primal_bound_ = mipdata_->upper_bound + model_->offset_;
  node_count_ = mipdata_->num_nodes;
  total_lp_iterations_ = mipdata_->total_lp_iterations;
  dual_bound_ = std::min(dual_bound_, primal_bound_);
  primal_dual_integral_ = mipdata_->primal_dual_integral.value;

  // adjust objective sense in case of maximization problem
  if (orig_model_->sense_ == ObjSense::kMaximize) {
    dual_bound_ = -dual_bound_;
    primal_bound_ = -primal_bound_;
  }

  if (modelstatus_ == HighsModelStatus::kNotset ||
      modelstatus_ == HighsModelStatus::kInfeasible) {
    if (feasible && havesolution)
      modelstatus_ = HighsModelStatus::kOptimal;
    else
      modelstatus_ = HighsModelStatus::kInfeasible;
  }

  profiling_->stop(kPostsolveTime);
  timer_.stop();

  gap_ = fabs(primal_bound_ - dual_bound_);
  if (primal_bound_ == 0.0)
    gap_ = dual_bound_ == 0.0 ? 0.0 : kHighsInf;
  else if (primal_bound_ != kHighsInf)
    gap_ = fabs(primal_bound_ - dual_bound_) / fabs(primal_bound_);
  else
    gap_ = kHighsInf;

  //
  if (options_mip_->output_flag) {
    // solutionstatus is only used in solvingReport, but depends on
    // values of havesolution and feasible that are local to
    // HighsMipSolver::cleanupSolve
    const std::string solutionstatus =
        havesolution ? (feasible ? "feasible" : "infeasible") : "-";
    solvingReport(solutionstatus);
  }

  assert(modelstatus_ != HighsModelStatus::kNotset);

  if (improving_solution_file_ != nullptr) fclose(improving_solution_file_);
}

void HighsMipSolver::solvingReport(const std::string& solutionstatus) const {
  std::array<char, 128> gapString =
      getGapString(gap_, primal_bound_, options_mip_);

  bool timeless_log = options_mip_->timeless_log;
  highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
               "\nSolving report\n");
  if (this->orig_model_->model_name_.length())
    highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
                 "  Model             %s\n",
                 this->orig_model_->model_name_.c_str());
  highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
               "  Status            %s\n"
               "  Primal bound      %.12g\n"
               "  Dual bound        %.12g\n"
               "  Gap               %s\n",
               utilModelStatusToString(modelstatus_).c_str(), primal_bound_,
               dual_bound_, gapString.data());
  if (!timeless_log)
    highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
                 "  P-D integral      %.12g\n",
                 mipdata_->primal_dual_integral.value);
  highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
               "  Solution status   %s\n", solutionstatus.c_str());
  if (solutionstatus != "-")
    highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
                 "                    %.12g (objective)\n"
                 "                    %.12g (bound viol.)\n"
                 "                    %.12g (int. viol.)\n"
                 "                    %.12g (row viol.)\n",
                 solution_objective_, bound_violation_, integrality_violation_,
                 row_violation_);
  if (!timeless_log) {
    auto callRecord = [&](HighsInt clock) {
      double mip_time = profiling_->read(clock, kMipRecord);
      double submip_time = profiling_->read(clock, kSubMipRecord);
      HighsInt mip_calls = profiling_->numCall(clock, kMipRecord);
      HighsInt submip_calls = profiling_->numCall(clock, kSubMipRecord);
      double total_time = mip_time + submip_time;
      // Only log postsolve if it's written as nonzero
      highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
                   "                    %.2f (%s)\n", total_time,
                   profiling_->name[clock].c_str());
      if (mip_calls > 1 || submip_calls > 0) {
        highsLogUser(
            options_mip_->log_options, HighsLogType::kInfo,
            "                        MIP    time [calls] = %.2f [%d]\n",
            mip_time, int(mip_calls));
        if (submip_calls > 0)
          highsLogUser(
              options_mip_->log_options, HighsLogType::kInfo,
              "                        subMIP time [calls] = %.2f [%d]\n",
              submip_time, int(submip_calls));
      }
    };
    double total = timer_.read();
    highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
                 "  Timing            %.2f\n", total);
    callRecord(kPresolveTime);
    callRecord(kSolveTime);
    callRecord(kPostsolveTime);
  }
  highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
               "  Max sub-MIP depth %d\n"
               "  Nodes             %llu\n",
               int(max_submip_level), (long long unsigned)mipdata_->num_nodes);
  if (mipdata_->total_repair_lp) {
    highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
                 "  Repair LPs        %llu (%llu feasible; %llu iterations)\n",
                 (long long unsigned)mipdata_->total_repair_lp,
                 (long long unsigned)mipdata_->total_repair_lp_feasible,
                 (long long unsigned)mipdata_->total_repair_lp_iterations);
  } else {
    highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
                 "  Repair LPs        0\n");
  }
  highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
               "  LP iterations     %llu\n",
               (long long unsigned)mipdata_->total_lp_iterations);
  if (mipdata_->total_lp_iterations)
    highsLogUser(options_mip_->log_options, HighsLogType::kInfo,
                 "                    %llu (strong br.)\n"
                 "                    %llu (separation)\n"
                 "                    %llu (heuristics)\n",
                 (long long unsigned)mipdata_->sb_lp_iterations,
                 (long long unsigned)mipdata_->sepa_lp_iterations,
                 (long long unsigned)mipdata_->heuristic_lp_iterations);
}

// Only called in Highs::runPresolve
void HighsMipSolver::runMipPresolve(const HighsInt presolve_reduction_limit) {
  mipdata_ = decltype(mipdata_)(new HighsMipSolverData(*this));
  mipdata_->init();
  mipdata_->runMipPresolve(presolve_reduction_limit);
}

const HighsLp& HighsMipSolver::getPresolvedModel() const {
  return mipdata_->presolvedModel;
}

HighsPresolveStatus HighsMipSolver::getPresolveStatus() const {
  return mipdata_->presolve_status;
}

presolve::HighsPostsolveStack HighsMipSolver::getPostsolveStack() const {
  return mipdata_->postSolveStack;
}

const HighsPresolveLog& HighsMipSolver::getPresolveLog() const {
  return mipdata_->presolve_log;
}

void HighsMipSolver::callbackGetCutPool() const {
  assert(callback_->user_callback);
  assert(callback_->callbackActive(kCallbackMipGetCutPool));
  callback_->clearHighsCallbackOutput();
  HighsCallbackOutput& data_out = callback_->data_out;

  HighsSparseMatrix cut_matrix;
  mipdata_->getLp().getCutPool(data_out.cutpool_num_col,
                               data_out.cutpool_num_cut, data_out.cutpool_lower,
                               data_out.cutpool_upper, cut_matrix);

  // take ownership
  data_out.cutpool_start = std::move(cut_matrix.start_);
  data_out.cutpool_index = std::move(cut_matrix.index_);
  data_out.cutpool_value = std::move(cut_matrix.value_);

  const bool interrupt = mipdata_->interruptFromCallbackWithData(
      kCallbackMipGetCutPool, solution_objective_, "MIP cut pool");
  assert(!interrupt);
}

std::array<char, 128> getGapString(const double gap_,
                                   const double primal_bound_,
                                   const HighsOptions* options_mip_) {
  std::array<char, 128> gapString = {};
  if (gap_ == kHighsInf)
    std::strcpy(gapString.data(), "inf");
  else {
    double printTol = std::max(std::min(1e-2, 1e-1 * gap_), 1e-6);
    auto gapValString = highsDoubleToString(100.0 * gap_, printTol);
    double gapTol = options_mip_->mip_rel_gap;

    if (options_mip_->mip_abs_gap > options_mip_->mip_feasibility_tolerance) {
      gapTol = primal_bound_ == 0.0
                   ? kHighsInf
                   : std::max(gapTol,
                              options_mip_->mip_abs_gap / fabs(primal_bound_));
    }

    if (gapTol == 0.0)
      std::snprintf(gapString.data(), gapString.size(), "%s%%",
                    gapValString.data());
    else if (gapTol != kHighsInf) {
      printTol = std::max(std::min(1e-2, 1e-1 * gapTol), 1e-6);
      auto gapTolString = highsDoubleToString(100.0 * gapTol, printTol);
      std::snprintf(gapString.data(), gapString.size(),
                    "%s%% (tolerance: %s%%)", gapValString.data(),
                    gapTolString.data());
    } else
      std::snprintf(gapString.data(), gapString.size(), "%s%% (tolerance: inf)",
                    gapValString.data());
  }

  return gapString;
}

bool HighsMipSolver::solutionFeasible(const HighsLp* lp,
                                      const std::vector<double>& col_value,
                                      const std::vector<double>* pass_row_value,
                                      double& bound_violation,
                                      double& row_violation,
                                      double& integrality_violation,
                                      HighsCDouble& obj) const {
  const auto checked = highsVerifyCandidate(*lp, col_value, pass_row_value,
                                            options_mip_->mip_feasibility_tolerance);
  bound_violation = checked.bound_violation;
  row_violation = checked.row_violation;
  integrality_violation = checked.integrality_violation;
  obj = checked.objective;
  return checked.feasible;
}

std::vector<HighsModelStatus> HighsMipSolver::initialiseTerminatorRecord(
    HighsInt num_instance) const {
  std::vector<HighsModelStatus> record(num_instance, HighsModelStatus::kNotset);
  return record;
}

void HighsMipSolver::initialiseTerminator(HighsInt num_instance_,
                                          HighsInt my_instance_,
                                          HighsModelStatus* record_) {
  this->termination_status_ = HighsModelStatus::kNotset;
  this->terminator_.clear();
  this->terminator_.initialise(num_instance_, my_instance_, record_);
}

void HighsMipSolver::initialiseTerminator(const HighsMipSolver& mip_solver) {
  this->terminator_.clear();
  if (!mip_solver.mipdata_->terminatorActive()) return;
  assert(mip_solver.mipdata_->terminatorConcurrency() > 0);
  this->initialiseTerminator(mip_solver.mipdata_->terminatorConcurrency(),
                             mip_solver.mipdata_->terminatorMyInstance(),
                             mip_solver.terminator_.record);
}

void HighsMipSolver::setParallelLock(bool lock) const {
  if (!mipdata_->hasMultipleWorkers()) return;
  mipdata_->parallel_lock = lock;
  for (HighsConflictPool& conflictpool : mipdata_->conflictpools) {
    conflictpool.setAgeLock(lock);
  }
  mipdata_->cliquetable.setAllowParallel(!lock && !submip);
}

void HighsMipSolver::setProfiling(HighsProfiling* profiling) {
  assert(profiling);
  this->profiling_ = profiling;
}
