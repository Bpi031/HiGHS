#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <thread>

#include "HCheckConfig.h"
#include "Highs.h"
#include "catch.hpp"
#include "mip/HighsNodeLpEffort.h"
#include "mip/HighsNodeQueue.h"
#include "mip/HighsSearch.h"
#include "mip/HighsSearchWork.h"
#include "mip/HighsReplayCapture.h"
#include "mip/HighsCandidateCheckpoint.h"
#include "mip/HighsMipSolverData.h"
#include "lp_data/HighsRefinementBudget.h"

TEST_CASE("node-lp-recovery-failure-keeps-unresolved-work-local",
          "[highs_node_lp_effort]") {
  Highs owner;
  owner.setOptionValue("output_flag", false);
  REQUIRE(owner.addVar(0, 1) == HighsStatus::kOk);
  HighsCallback callback(&owner);
  HighsSolution solution;
  HighsMipSolver mip(callback, owner.getOptions(), owner.getLp(), solution);
  mip.modelstatus_ = HighsModelStatus::kNotset;
  mip.mipdata_.reset(new HighsMipSolverData(mip));
  auto& data = *mip.mipdata_;
  HighsMipWorker worker(mip, &data.getLp(), &data.getDomain(),
                        &data.getCutPool(), &data.getConflictPool(),
                        &data.getPseudoCost(), 0);
  auto& search = *worker.search_ptr_;
  search.createNewNode();
  REQUIRE_FALSE(search.currentNodePruned());
  REQUIRE(search.failLpRecovery() == HighsSearch::NodeResult::kLpSolveError);
  REQUIRE(search.lpRecoveryFailed());
  REQUIRE_FALSE(search.currentNodePruned());
  REQUIRE_FALSE(search.getLocalDomain().infeasible());
  REQUIRE(search.checkLocalLimits());
  REQUIRE(search.checkLimits());
  REQUIRE(search.nnodes == 0);
  REQUIRE(search.nleaves == 0);
  REQUIRE(search.treeweight == 0);
  REQUIRE(mip.modelstatus_ == HighsModelStatus::kNotset);
  REQUIRE(data.lower_bound == -kHighsInf);

  worker.resetSearch();
  HighsSearch helper(worker, data.getPseudoCost());
  helper.setHeuristic(true);
  helper.createNewNode();
  REQUIRE(helper.failLpRecovery() == HighsSearch::NodeResult::kLpSolveError);
  REQUIRE(helper.lpRecoveryFailed());
  REQUIRE_FALSE(worker.search_ptr_->lpRecoveryFailed());
  REQUIRE(mip.modelstatus_ == HighsModelStatus::kNotset);
}

TEST_CASE("iteration-only-helper-cannot-enter-unbounded-numerical-recovery",
          "[highs_node_lp_effort]") {
  Highs owner;
  HighsCallback callback(&owner);
  HighsOptions options;
  HighsLp lp;
  HighsSolution solution;
  HighsMipSolver child(callback, options, lp, solution, true, 1);
  HighsLpRelaxation relaxation(child);
  REQUIRE_FALSE(relaxation.isBoundedHelper());
  relaxation.setHelperIterationLimit(10);
  REQUIRE(relaxation.isBoundedHelper());
  relaxation.setIterationLimit();
  REQUIRE(relaxation.getLpSolver().getOptions().simplex_iteration_limit == 10);
  relaxation.setHelperIterationLimit(kHighsIInf);
  REQUIRE_FALSE(relaxation.isBoundedHelper());
  relaxation.setHelperDeadline(0.0);
  REQUIRE(relaxation.isBoundedHelper());
}

TEST_CASE("bounded-helper-stops-without-checkpoint-or-IPM-retry",
          "[highs_node_lp_effort]") {
  Highs owner;
  owner.resetGlobalScheduler(true);
  owner.setOptionValue("output_flag", false);
  owner.setOptionValue("threads", 1);
  highs::parallel::initialize_scheduler(1);
  REQUIRE(owner.readModel(std::string(HIGHS_DIR) + "/check/instances/bell5.mps")
          == HighsStatus::kOk);
  HighsCallback callback(&owner);
  HighsSolution solution;
  HighsMipSolver mip(callback, owner.getOptions(), owner.getLp(), solution);
  mip.modelstatus_ = HighsModelStatus::kNotset;
  mip.mipdata_.reset(new HighsMipSolverData(mip));
  mip.mipdata_->feastol = owner.getOptions().mip_feasibility_tolerance;
  HighsProfiling profiling;
  profiling.initialize(mip.timer_, false);
  mip.setProfiling(&profiling);
  HighsLpRelaxation relaxation(mip);
  relaxation.setProfiling(&profiling);
  relaxation.loadModel();
  relaxation.getLpSolver().setOptionValue("presolve", "off");
  SECTION("iteration-only budget") {
    relaxation.setHelperIterationLimit(1);
    REQUIRE(relaxation.run() == HighsLpRelaxation::Status::kIterationLimit);
    REQUIRE(relaxation.getNumLpIterations() <= 1);
    REQUIRE(relaxation.getLpSolver().getInfo().ipm_iteration_count == 0);
    std::vector<HighsInt> indices;
    std::vector<double> values;
    double rhs = 0;
    REQUIRE_FALSE(relaxation.computeDualInfProof(mip.mipdata_->getDomain(),
                                                indices, values, rhs));
  }
  SECTION("expired deadline is not infeasibility") {
    relaxation.setHelperDeadline(0);
    REQUIRE(relaxation.run() == HighsLpRelaxation::Status::kError);
    REQUIRE(relaxation.getLpSolver().getModelStatus() == HighsModelStatus::kTimeLimit);
    REQUIRE(relaxation.getNumLpIterations() == 0);
  }
  SECTION("callback budget follows root and copied helper LPs") {
    HighsRefinementBudget budget(1, HighsRefinementBudget::Clock::now() +
        std::chrono::seconds(10), nullptr);
    callback.refinement_budget = &budget;
    REQUIRE(relaxation.isBoundedHelper());
    REQUIRE(relaxation.run() == HighsLpRelaxation::Status::kIterationLimit);
    REQUIRE(budget.simplex_iterations == 1);
    REQUIRE(budget.ipm_iterations == 0);
    HighsLpRelaxation copied(relaxation);
    REQUIRE(copied.isBoundedHelper());
    REQUIRE(copied.run() == HighsLpRelaxation::Status::kIterationLimit);
    REQUIRE(budget.lp_runs == 1);
    REQUIRE(mip.mipdata_->checkLimits());
    REQUIRE(mip.modelstatus_ == HighsModelStatus::kIterationLimit);
    std::vector<HighsInt> indices;
    std::vector<double> values;
    double rhs = 0;
    REQUIRE_FALSE(copied.computeDualInfProof(mip.mipdata_->getDomain(),
                                            indices, values, rhs));
    callback.refinement_budget = nullptr;
  }
  SECTION("callback cancellation before root does not create a proof") {
    std::atomic<bool> cancel{true};
    HighsRefinementBudget budget(1000, HighsRefinementBudget::Clock::now() +
        std::chrono::seconds(10), &cancel);
    callback.refinement_budget = &budget;
    REQUIRE(relaxation.run() == HighsLpRelaxation::Status::kIterationLimit);
    REQUIRE(budget.lp_runs == 0);
    REQUIRE(mip.mipdata_->checkLimits());
    REQUIRE(mip.modelstatus_ == HighsModelStatus::kInterrupt);
    REQUIRE_FALSE(mip.mipdata_->getDomain().infeasible());
    callback.refinement_budget = nullptr;
  }
  SECTION("cancelled private repair does not start feasibility jump") {
    std::atomic<bool> cancel{true};
    HighsRefinementBudget budget(1000, HighsRefinementBudget::Clock::now() +
        std::chrono::seconds(10), &cancel);
    callback.refinement_budget = &budget;
    REQUIRE(mip.mipdata_->feasibilityJump() == HighsModelStatus::kInterrupt);
    REQUIRE(mip.solution_.empty());
    REQUIRE(mip.modelstatus_ == HighsModelStatus::kInterrupt);
    callback.refinement_budget = nullptr;
  }
  SECTION("callback IPX work consumes the same finite allowance") {
    HighsRefinementBudget budget(1, HighsRefinementBudget::Clock::now() +
        std::chrono::seconds(10), nullptr);
    callback.refinement_budget = &budget;
    relaxation.useIpxForRecovery();
    REQUIRE(relaxation.run() == HighsLpRelaxation::Status::kIterationLimit);
    REQUIRE(budget.ipm_iterations == 1);
    REQUIRE(budget.simplex_iterations == 0);
    REQUIRE(budget.lp_runs == 1);
    REQUIRE(mip.mipdata_->checkLimits());
    REQUIRE(mip.modelstatus_ == HighsModelStatus::kIterationLimit);
    callback.refinement_budget = nullptr;
  }
  SECTION("explicit recovery uses IPX without changing the parent default") {
    const auto original_solver = owner.getOptions().mip_lp_solver;
    relaxation.useIpxForRecovery();
    REQUIRE(relaxation.run() == HighsLpRelaxation::Status::kOptimal);
    REQUIRE(relaxation.getLpSolver().getInfo().ipm_iteration_count > 0);
    REQUIRE(owner.getOptions().mip_lp_solver == original_solver);
  }
}

TEST_CASE("deferred-cut-cleanup-invalidates-evidence-and-reoptimizes-on-demand",
          "[highs_node_lp_effort]") {
  Highs owner;
  owner.resetGlobalScheduler(true);
  owner.setOptionValue("output_flag", false);
  owner.setOptionValue("threads", 1);
  highs::parallel::initialize_scheduler(1);
  REQUIRE(owner.readModel(std::string(HIGHS_DIR) + "/check/instances/bell5.mps")
          == HighsStatus::kOk);
  HighsCallback callback(&owner);
  HighsSolution solution;
  HighsMipSolver mip(callback, owner.getOptions(), owner.getLp(), solution);
  mip.mipdata_.reset(new HighsMipSolverData(mip));
  mip.mipdata_->feastol = 1e-6;
  HighsProfiling profiling;
  profiling.initialize(mip.timer_, false);
  mip.setProfiling(&profiling);
  HighsLpRelaxation relaxation(mip);
  relaxation.setProfiling(&profiling);
  relaxation.loadModel();
  REQUIRE(relaxation.run() == HighsLpRelaxation::Status::kOptimal);
  const double original_objective = relaxation.getObjective();
  const HighsInt original_rows = relaxation.numRows();
  HighsCutSet cut;
  cut.cutindices = {0};
  cut.cutpools = {0};
  cut.lower_ = {-kHighsInf};
  cut.upper_ = {1};
  cut.ARstart_ = {0}; // A redundant zero row with a basic slack.
  relaxation.addCuts(cut);
  REQUIRE(relaxation.run() == HighsLpRelaxation::Status::kOptimal);
  REQUIRE(relaxation.numRows() == original_rows + 1);
  relaxation.storeBasis();
  REQUIRE(relaxation.getStoredBasis());
  relaxation.removeObsoleteRows(false,
      HighsLpRelaxation::CutRemovalMode::kModelAndBasisOnly);
  REQUIRE(relaxation.numRows() == original_rows);
  REQUIRE(relaxation.getStatus() == HighsLpRelaxation::Status::kNotSet);
  REQUIRE(relaxation.getObjective() == -kHighsInf);
  REQUIRE_FALSE(relaxation.getStoredBasis());
  REQUIRE_FALSE(relaxation.getLpSolver().getInfo().valid);
  REQUIRE(relaxation.getFractionalIntegers().empty());
  std::vector<HighsInt> indices;
  std::vector<double> values;
  double rhs = 0;
  REQUIRE_FALSE(relaxation.computeDualInfProof(mip.mipdata_->getDomain(),
                                              indices, values, rhs));
  REQUIRE(relaxation.run() == HighsLpRelaxation::Status::kOptimal);
  REQUIRE(relaxation.getObjective() == Approx(original_objective));
}

TEST_CASE("unlimited-work-observation-does-not-enforce-a-budget",
          "[highs_node_lp_effort]") {
  HighsSearchWork work;
  HighsCandidateCheckpoint checkpoint;
  work.begin(0, true, 10.0);
  work.attachCheckpoint(&checkpoint, 0);
  work.chargeLp(120);
  work.chargeSubMip(50);
  work.recordRestoration(0.25);
  work.recordLocalBound(2.0, 5.0);
  REQUIRE_FALSE(work.exhausted());
  REQUIRE(work.remaining() == std::numeric_limits<int64_t>::max());
  {
    HighsTaskPhaseScope phase(work, HighsTaskPhase::kChildMip);
    REQUIRE(checkpoint.request());
    REQUIRE(work.checkpointRequested());
  }
  REQUIRE(work.phase() == HighsTaskPhase::kEvaluation);
  REQUIRE(work.checkpointPhase() == HighsTaskPhase::kChildMip);
  work.finish();
  REQUIRE(work.summary().leases == 1);
  REQUIRE(work.summary().exhausted_leases == 0);
  REQUIRE(work.summary().max_lease_work == 170);
  REQUIRE(work.tasks().size() == 1);
  REQUIRE(work.tasks()[0].restoration_seconds == 0.25);
  REQUIRE(work.tasks()[0].local_bound_gain == 3.0);
  REQUIRE(work.taskReturnSeconds() >= work.firstCheckpointSeconds());
  for (int i = 0; i < 130; ++i) {
    work.begin(0, false);
    work.finish();
  }
  REQUIRE(work.tasks().size() == 128);
  REQUIRE(work.summary().task_records_dropped == 3);
}

TEST_CASE("optional-lp-interrupt-observes-only-actual-candidate-requests",
          "[highs_node_lp_effort]") {
  HighsCandidateCheckpoint checkpoint;
  HighsCandidateLpInterrupt interrupt;
  interrupt.begin(nullptr, 0);
  REQUIRE_FALSE(interrupt.poll());
  interrupt.begin(&checkpoint, checkpoint.sequence());
  for (int i = 0; i < 100; ++i) REQUIRE_FALSE(interrupt.poll());
  REQUIRE(checkpoint.request());
  REQUIRE(interrupt.poll());
  REQUIRE(interrupt.interrupted());
  interrupt.begin(&checkpoint, checkpoint.sequence());
  REQUIRE_FALSE(interrupt.poll());
  REQUIRE_FALSE(interrupt.interrupted());

  // Simulate another owner publishing during a simplex interrupt poll. No
  // payload is read; resuming the unchanged LP must still reach its optimum.
  Highs highs;
  highs.resetGlobalScheduler(true);
  REQUIRE(highs.setOptionValue("output_flag", false) == HighsStatus::kOk);
  REQUIRE(highs.setOptionValue("threads", 1) == HighsStatus::kOk);
  REQUIRE(highs.setOptionValue("presolve", "off") == HighsStatus::kOk);
  REQUIRE(highs.setOptionValue("solver", "simplex") == HighsStatus::kOk);
  REQUIRE(highs.readModel(std::string(HIGHS_DIR) + "/check/instances/bell5.mps") == HighsStatus::kOk);
  HighsLp relaxation = highs.getLp();
  relaxation.integrality_.clear();
  REQUIRE(highs.passModel(relaxation) == HighsStatus::kOk);
  int polls = 0;
  highs.setCallback([&](int, const std::string&, const HighsCallbackOutput*,
                        HighsCallbackInput* input, void*) {
    ++polls;
    checkpoint.request();
    input->user_interrupt = interrupt.poll();
  });
  REQUIRE(highs.startCallback(kCallbackSimplexInterrupt) == HighsStatus::kOk);
  REQUIRE(highs.run() == HighsStatus::kWarning);
  REQUIRE(polls > 0);
  REQUIRE(highs.getModelStatus() == HighsModelStatus::kInterrupt);
  REQUIRE(highs.getLp().col_cost_ == relaxation.col_cost_);
  REQUIRE(highs.getLp().col_lower_ == relaxation.col_lower_);
  REQUIRE(highs.getLp().col_upper_ == relaxation.col_upper_);
  REQUIRE(highs.getLp().a_matrix_.value_ == relaxation.a_matrix_.value_);
  REQUIRE(highs.stopCallback(kCallbackSimplexInterrupt) == HighsStatus::kOk);
  REQUIRE(highs.run() == HighsStatus::kOk);
  REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
  const double optimum = highs.getInfo().objective_function_value;
  Highs control;
  control.setOptionValue("output_flag", false);
  control.setOptionValue("threads", 1);
  control.setOptionValue("presolve", "off");
  control.setOptionValue("solver", "simplex");
  REQUIRE(control.passModel(relaxation) == HighsStatus::kOk);
  REQUIRE(control.run() == HighsStatus::kOk);
  REQUIRE(control.getInfo().objective_function_value == Approx(optimum));
}

TEST_CASE("pending-child-lp-request-is-a-yield-not-an-infeasibility-proof",
          "[highs_node_lp_effort]") {
  HighsCandidateCheckpoint checkpoint;
  Highs owner;
  HighsCallback callback(&owner);
  HighsOptions options;
  HighsLp lp;
  HighsSolution solution;
  HighsMipSolver child(callback, options, lp, solution, true, 1);
  child.parent_candidate_checkpoint = &checkpoint;
  child.parent_candidate_sequence = checkpoint.sequence();
  REQUIRE(checkpoint.request());
  HighsLpRelaxation relaxation(child);
  REQUIRE(relaxation.run() == HighsLpRelaxation::Status::kIterationLimit);
  std::vector<HighsInt> indices;
  std::vector<double> values;
  double rhs = 0;
  HighsDomain domain(child);
  REQUIRE_FALSE(relaxation.computeDualInfProof(domain, indices, values, rhs));
}

TEST_CASE("candidate-checkpoint-lp-chunks-have-an-eventual-completion-escape",
          "[highs_node_lp_effort]") {
  REQUIRE(highsCandidateLpIterationLimit(kHighsIInf, 0) == 16384);
  REQUIRE(highsCandidateLpIterationLimit(kHighsIInf, 1) == 32768);
  REQUIRE(highsCandidateLpIterationLimit(kHighsIInf, 2) == 65536);
  REQUIRE(highsCandidateLpIterationLimit(kHighsIInf, 3) == kHighsIInf);
  REQUIRE(highsCandidateLpIterationLimit(1, 3) == kHighsIInf);
  REQUIRE(highsCandidateLpIterationLimit(100, 0) == 100);
  REQUIRE(highsCandidateLpIterationLimit(100, 2) == 100);
  REQUIRE(highsCandidateLpIterationLimit(kHighsIInf, kHighsIInf) == kHighsIInf);
}

TEST_CASE("candidate-checkpoint-retains-requests-for-late-starting-workers",
          "[highs_node_lp_effort]") {
  HighsCandidateCheckpoint checkpoint;
  const uint64_t batch = checkpoint.sequence();
  REQUIRE_FALSE(checkpoint.requestedSince(batch));
  REQUIRE(checkpoint.request());
  HighsSearchWork late_worker;
  late_worker.begin(0, false);
  late_worker.attachCheckpoint(&checkpoint, batch);
  REQUIRE(late_worker.checkpointRequested());
  REQUIRE(late_worker.skipOptional());
  REQUIRE_FALSE(late_worker.exhausted());
  late_worker.recordCandidateYield();
  late_worker.finish();
  REQUIRE_FALSE(late_worker.checkpointRequested());
  REQUIRE(late_worker.summary().candidate_yields == 1);
  late_worker.begin(0, false);
  late_worker.attachCheckpoint(&checkpoint, checkpoint.sequence());
  REQUIRE_FALSE(late_worker.checkpointRequested());
  late_worker.requestCheckpoint();
  REQUIRE(late_worker.checkpointRequested());
  REQUIRE(checkpoint.sequence() == 2);
}

TEST_CASE("candidate-checkpoint-concurrent-requests-do-not-lose-wakeups",
          "[highs_node_lp_effort]") {
  HighsCandidateCheckpoint checkpoint;
  std::vector<std::thread> workers;
  for (int i = 0; i < 4; ++i)
    workers.emplace_back([&]() {
      for (int j = 0; j < 1000; ++j) checkpoint.request();
    });
  for (auto& worker : workers) worker.join();
  REQUIRE(checkpoint.sequence() == 4000);
  REQUIRE(checkpoint.requestedSince(0));
  REQUIRE_FALSE(checkpoint.requestedSince(4000));
  REQUIRE_FALSE(checkpoint.overflowed());
  HighsCandidateCheckpoint exhausted(std::numeric_limits<uint64_t>::max());
  REQUIRE_FALSE(exhausted.request());
  REQUIRE(exhausted.overflowed());
  REQUIRE(exhausted.sequence() == std::numeric_limits<uint64_t>::max());
}

TEST_CASE("candidate-checkpoints-preserve-original-model-optimality",
          "[highs_node_lp_effort]") {
  const std::string exchange = GENERATE(std::string("off"),
      std::string("deterministic"), std::string("opportunistic"));
  for (HighsInt threads : {1, 4}) {
    for (HighsInt seed : {0, 42, 43}) {
      CAPTURE(threads, seed, exchange);
      Highs highs;
      highs.resetGlobalScheduler(true);
      REQUIRE(highs.setOptionValue("threads", threads) == HighsStatus::kOk);
      REQUIRE(highs.setOptionValue("random_seed", seed) == HighsStatus::kOk);
      REQUIRE(highs.setOptionValue("time_limit", 20.0) == HighsStatus::kOk);
      REQUIRE(highs.setOptionValue("parallel", threads == 1 ? "off" : "on") == HighsStatus::kOk);
      REQUIRE(highs.setOptionValue("mip_candidate_checkpoint", true) == HighsStatus::kOk);
      REQUIRE(highs.setOptionValue("mip_incumbent_exchange", exchange) == HighsStatus::kOk);
      REQUIRE(highs.setOptionValue("mip_node_lp_effort", true) == HighsStatus::kOk);
      REQUIRE(highs.setOptionValue("mip_node_lp_effort_min_iterations", 1) == HighsStatus::kOk);
      REQUIRE(highs.setOptionValue("mip_node_lp_effort_max_iterations", 1) == HighsStatus::kOk);
      REQUIRE(highs.setOptionValue("mip_rel_gap", 0.0) == HighsStatus::kOk);
      REQUIRE(highs.setOptionValue("mip_abs_gap", 0.0) == HighsStatus::kOk);
      REQUIRE(highs.setOptionValue("mip_allow_restart", false) == HighsStatus::kOk);
      REQUIRE(highs.setOptionValue("log_to_console", false) == HighsStatus::kOk);
      int summaries = 0;
      int requests = 0;
      highs.setCallback([&](int kind, const std::string& message,
                            const HighsCallbackOutput*, HighsCallbackInput*, void*) {
        if (kind != kCallbackLogging || message.find("MIP-CandidateCheckpoint:") == std::string::npos)
          return;
        ++summaries;
        std::istringstream tokens(message);
        std::string token;
        while (tokens >> token)
          if (token.compare(0, 9, "requests=") == 0) requests += std::stoi(token.substr(9));
      });
      REQUIRE(highs.startCallback(kCallbackLogging) == HighsStatus::kOk);
      REQUIRE(highs.readModel(std::string(HIGHS_DIR) + "/check/instances/bell5.mps") == HighsStatus::kOk);
      REQUIRE(highs.run() == HighsStatus::kOk);
      REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
      REQUIRE(highs.getInfo().objective_function_value == Approx(8966406.491519));
      REQUIRE(highs.getInfo().mip_dual_bound == Approx(highs.getInfo().objective_function_value));
      REQUIRE(highs.getInfo().max_primal_infeasibility <= 1e-6);
      REQUIRE(highs.getInfo().max_integrality_violation <= 1e-6);
      REQUIRE(summaries >= 1);
      if (threads == 4) REQUIRE(requests > 0);
    }
  }
}

TEST_CASE("candidate-checkpoints-retain-terminal-worker-incumbents",
          "[highs_node_lp_effort]") {
  const std::string exchange = GENERATE(std::string("off"),
      std::string("deterministic"), std::string("opportunistic"));
  // An odd cycle needs ceil(n/2) selected vertices; every feasible subset is
  // checked here independently of presolve, cuts and incumbent admission.
  constexpr HighsInt cycle_size = 9;
  constexpr HighsInt cycles = 3;
  constexpr HighsInt n = cycle_size * cycles;
  HighsInt component_optimum = cycle_size;
  for (unsigned mask = 0; mask != (1u << cycle_size); ++mask) {
    HighsInt selected = 0;
    bool feasible = true;
    for (HighsInt col = 0; col != cycle_size; ++col) {
      selected += (mask >> col) & 1u;
      feasible &= ((mask >> col) & 1u) ||
                  ((mask >> ((col + 1) % cycle_size)) & 1u);
    }
    if (feasible) component_optimum = std::min(component_optimum, selected);
  }
  REQUIRE(component_optimum == 5);
  const HighsInt optimum = cycles * component_optimum;
  for (HighsInt threads : {1, 4}) {
    Highs highs;
    highs.resetGlobalScheduler(true);
    REQUIRE(highs.setOptionValue("threads", threads) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("parallel", threads == 1 ? "off" : "on") == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("presolve", "off") == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_candidate_checkpoint", true) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_incumbent_exchange", exchange) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_rel_gap", 0.0) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_abs_gap", 0.0) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_allow_restart", false) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_allow_cut_separation_at_nodes", false) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_root_max_separation_rounds", 0) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_heuristic_effort", 0.0) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_heuristic_run_feasibility_jump", false) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_heuristic_run_rens", false) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_heuristic_run_rins", false) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_heuristic_run_root_reduced_cost", false) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_detect_symmetry", false) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_pscost_minreliable", 0) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("log_to_console", false) == HighsStatus::kOk);
    HighsModel model;
    HighsLp& lp = model.lp_;
    lp.num_col_ = n;
    lp.num_row_ = n;
    lp.col_cost_.assign(n, 1);
    lp.col_lower_.assign(n, 0);
    lp.col_upper_.assign(n, 1);
    lp.integrality_.assign(n, HighsVarType::kInteger);
    lp.row_lower_.assign(n, 1);
    lp.row_upper_.assign(n, kHighsInf);
    lp.a_matrix_.format_ = MatrixFormat::kRowwise;
    lp.a_matrix_.start_.clear();
    for (HighsInt row = 0; row != n; ++row) {
      lp.a_matrix_.start_.push_back(2 * row);
      lp.a_matrix_.index_.push_back(row);
      lp.a_matrix_.index_.push_back(row / cycle_size * cycle_size +
                                   (row + 1) % cycle_size);
      lp.a_matrix_.value_.push_back(1);
      lp.a_matrix_.value_.push_back(1);
    }
    lp.a_matrix_.start_.push_back(2 * n);
    int admitted = 0;
    std::string log;
    highs.setCallback([&](int kind, const std::string& message,
                          const HighsCallbackOutput*, HighsCallbackInput*, void*) {
      if (kind == kCallbackLogging) log += message;
      if (kind == kCallbackLogging &&
          message.find("MIP-CandidateAdmission:") != std::string::npos &&
          message.find("accepted=1") != std::string::npos)
        ++admitted;
    });
    REQUIRE(highs.startCallback(kCallbackLogging) == HighsStatus::kOk);
    REQUIRE(highs.passModel(model) == HighsStatus::kOk);
    REQUIRE(highs.getLp().a_matrix_.value_.size() == size_t(2 * n));
    REQUIRE(highs.run() == HighsStatus::kOk);
    INFO(log);
    REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
    REQUIRE(highs.getInfo().objective_function_value == Approx(optimum));
    REQUIRE(highs.getInfo().mip_dual_bound == Approx(optimum));
    if (threads == 4) REQUIRE(admitted > 0);
    const auto& solution = highs.getSolution().col_value;
    REQUIRE(solution.size() == size_t(n));
    for (HighsInt col = 0; col != n; ++col) {
      REQUIRE(std::abs(solution[col] - std::round(solution[col])) <= 1e-6);
      const HighsInt next = col / cycle_size * cycle_size +
                            (col + 1) % cycle_size;
      REQUIRE(solution[col] + solution[next] >= 1 - 1e-6);
    }
  }
}

TEST_CASE("local-replay-snapshot-preserves-matrix-domains-and-offset",
          "[highs_replay_capture]") {
  HighsLp lp;
  lp.num_col_ = 2;
  lp.num_row_ = 1;
  lp.col_cost_ = {0, 2};
  lp.col_lower_ = {0, 0};
  lp.col_upper_ = {1, 10};
  lp.row_lower_ = {1};
  lp.row_upper_ = {kHighsInf};
  lp.offset_ = 3;
  lp.col_names_ = {"c0", "c1"};
  lp.row_names_ = {"r0"};
  lp.integrality_ = {HighsVarType::kInteger, HighsVarType::kContinuous};
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.num_col_ = 2;
  lp.a_matrix_.num_row_ = 1;
  lp.a_matrix_.start_ = {0, 1, 2};
  lp.a_matrix_.index_ = {0, 0};
  lp.a_matrix_.value_ = {1, 1};
  HighsBasis basis;
  basis.valid = true;
  basis.col_status = {HighsBasisStatus::kLower, HighsBasisStatus::kLower};
  basis.row_status = {HighsBasisStatus::kBasic};
  HighsReplayCapture::capture(lp, "node_lp", 1, &basis);
  HighsReplayCapture::capture(lp, "sub_mip", 0, &basis);
  REQUIRE(lp.integrality_.size() == 2);
  REQUIRE(lp.offset_ == 3);
  REQUIRE(lp.col_lower_[0] == 0);
}

TEST_CASE("local-replay-canonical-names-preserve-indexed-model",
          "[highs_replay_capture][replay_name_contract]") {
  const std::vector<std::vector<std::string>> names = {
      {}, {""}, {"original"}, {"duplicate", "duplicate"},
      {"space name", ""}, {"r0", "extra", "surplus"}};
  for (const auto& col_names : names) {
    for (const auto& row_names : names) {
      for (bool node : {false, true}) {
        HighsLp original;
        original.num_col_ = 2;
        original.num_row_ = 2;
        original.col_cost_ = {-2, 3};
        original.col_lower_ = {-1, 0};
        original.col_upper_ = {1, 10};
        original.row_lower_ = {1, -kHighsInf};
        original.row_upper_ = {kHighsInf, 5};
        original.offset_ = -7;
        original.sense_ = ObjSense::kMaximize;
        original.col_names_ = col_names;
        original.row_names_ = row_names;
        original.objective_name_ = "r0";
        original.integrality_ = {HighsVarType::kInteger,
                                 HighsVarType::kContinuous};
        original.a_matrix_.format_ = MatrixFormat::kColwise;
        original.a_matrix_.num_col_ = 2;
        original.a_matrix_.num_row_ = 2;
        original.a_matrix_.start_ = {0, 1, 3};
        original.a_matrix_.index_ = {0, 0, 1};
        original.a_matrix_.value_ = {1, 1, -1};
        original.col_hash_.name2index["old"] = 1;
        original.row_hash_.name2index["old"] = 1;
        const HighsLp replay = HighsReplayCapture::modelForReplay(original, node);
        REQUIRE(replay.col_names_ == std::vector<std::string>{"c0", "c1"});
        REQUIRE(replay.row_names_ == std::vector<std::string>{"r0", "r1"});
        REQUIRE(replay.objective_name_ == "OBJ");
        REQUIRE(replay.col_hash_.name2index.empty());
        REQUIRE(replay.row_hash_.name2index.empty());
        REQUIRE(replay.num_col_ == original.num_col_);
        REQUIRE(replay.num_row_ == original.num_row_);
        REQUIRE(replay.col_cost_ == original.col_cost_);
        REQUIRE(replay.col_lower_ == original.col_lower_);
        REQUIRE(replay.col_upper_ == original.col_upper_);
        REQUIRE(replay.row_lower_ == original.row_lower_);
        REQUIRE(replay.row_upper_ == original.row_upper_);
        REQUIRE(replay.offset_ == original.offset_);
        REQUIRE(replay.sense_ == original.sense_);
        REQUIRE(replay.a_matrix_.format_ == original.a_matrix_.format_);
        REQUIRE(replay.a_matrix_.start_ == original.a_matrix_.start_);
        REQUIRE(replay.a_matrix_.index_ == original.a_matrix_.index_);
        REQUIRE(replay.a_matrix_.value_ == original.a_matrix_.value_);
        if (node)
          REQUIRE(replay.integrality_.empty());
        else
          REQUIRE(replay.integrality_ == original.integrality_);
        REQUIRE(original.col_names_ == col_names);
        REQUIRE(original.row_names_ == row_names);
        REQUIRE(original.objective_name_ == "r0");
        REQUIRE(original.col_hash_.name2index.at("old") == 1);
        REQUIRE(original.row_hash_.name2index.at("old") == 1);
        REQUIRE(original.integrality_.size() == 2);
      }
    }
  }
}

TEST_CASE("construction-replay-filter-estimates-and-context-fail-closed",
          "[highs_replay_capture]") {
  REQUIRE(HighsReplayCapture::selected("", "node_lp"));
  REQUIRE(HighsReplayCapture::selected("all", "sub_mip"));
  REQUIRE(HighsReplayCapture::selected("rens", "sub_mip", "rens"));
  REQUIRE_FALSE(HighsReplayCapture::selected("rens", "node_lp", "rens"));
  REQUIRE_FALSE(HighsReplayCapture::selected("rens", "sub_mip", "rins"));
  REQUIRE_FALSE(HighsReplayCapture::selected("invalid", "sub_mip", "invalid"));
  REQUIRE_FALSE(HighsReplayCapture::selected("all", "unknown"));
  HighsLp lp;
  lp.num_col_ = 2;
  lp.num_row_ = 1;
  REQUIRE(HighsReplayCapture::captureEstimate(lp, true) >
          HighsReplayCapture::captureEstimate(lp, false));
  lp.num_col_ = std::numeric_limits<HighsInt>::max();
  REQUIRE(HighsReplayCapture::captureEstimate(lp, true) > 64 * 1024 * 1024);
  lp.num_col_ = -1;
  REQUIRE(HighsReplayCapture::captureEstimate(lp, false) > 64 * 1024 * 1024);
  lp.num_col_ = 2;
  HighsOptions options;
  std::vector<double> values = {0.5, 1};
  std::vector<double> lower = {0, -kHighsInf}, upper = {1, kHighsInf};
  std::vector<double> seed;
  HighsReplayContext context;
  context.method = "rens";
  context.parent_phase = "tree";
  context.decision = 1;
  context.worker = context.restart = 0;
  context.parent_time = 10;
  context.helper_lp_limit = 100;
  context.lp_solution_status = kSolutionStatusFeasible;
  context.relaxation = &values;
  context.base_lower = &lower;
  context.base_upper = &upper;
  context.checked_seed = &seed;
  context.options = &options;
  REQUIRE(HighsReplayCapture::validContext(lp, context));
  for (const char* method : {"all", "", "node_lp", "quote\""}) {
    context.method = method;
    REQUIRE_FALSE(HighsReplayCapture::validContext(lp, context));
  }
  context.method = "rens";
  context.decision = 0;
  REQUIRE_FALSE(HighsReplayCapture::validContext(lp, context));
  context.decision = 1;
  values[0] = std::numeric_limits<double>::quiet_NaN();
  REQUIRE_FALSE(HighsReplayCapture::validContext(lp, context));
  values[0] = 0.5;
  lower[0] = 2;
  REQUIRE_FALSE(HighsReplayCapture::validContext(lp, context));
  lower[0] = 0;
  seed = {1};
  REQUIRE_FALSE(HighsReplayCapture::validContext(lp, context));
  seed = {1, 1};
  REQUIRE(HighsReplayCapture::validContext(lp, context));
  lp.integrality_ = {HighsVarType::kInteger};
  REQUIRE_FALSE(HighsReplayCapture::validContext(lp, context));
  lp.integrality_.clear();
  options.parallel = "quote\"";
  REQUIRE_FALSE(HighsReplayCapture::validContext(lp, context));
  options.parallel = "off";
  options.mip_abs_gap = kHighsInf;
  REQUIRE_FALSE(HighsReplayCapture::validContext(lp, context));
  options.mip_abs_gap = 0;
  context.helper_lp_limit = -1;
  REQUIRE_FALSE(HighsReplayCapture::validContext(lp, context));
  context.helper_lp_limit = 100;
  context.original_space = true;
  context.method = "local_branching";
  REQUIRE_FALSE(HighsReplayCapture::validContext(lp, context));
  context.relaxation = nullptr;
  REQUIRE(HighsReplayCapture::validContext(lp, context));
  seed.clear();
  REQUIRE_FALSE(HighsReplayCapture::validContext(lp, context));
  seed = {1, 1};
  context.checked_seed = nullptr;
  REQUIRE_FALSE(HighsReplayCapture::validContext(lp, context));
}

// The gate and CTest run this explicitly in a fresh process with an owned empty
// replay directory. The process-wide capture cache cannot join the aggregate suite.
TEST_CASE("construction-replay-files-read-back-and-never-clobber",
          "[construction_capture_io][.]") {
  const char* directory = std::getenv("ROST_HIGHS_REPLAY_DIR");
  REQUIRE(directory != nullptr);
  const char* method = std::getenv("ROST_HIGHS_REPLAY_METHOD");
  REQUIRE(method != nullptr);
  REQUIRE(std::string(method) == "rens");
  const std::string base = std::string(directory) + "/sub_mip-";
  Highs owner;
  owner.setOptionValue("output_flag", false);
  REQUIRE(owner.addVar(0, 1) == HighsStatus::kOk);
  REQUIRE(owner.addVar(-kHighsInf, kHighsInf) == HighsStatus::kOk);
  REQUIRE(owner.changeColIntegrality(0, HighsVarType::kInteger) == HighsStatus::kOk);
  REQUIRE(owner.changeColCost(1, 2) == HighsStatus::kOk);
  REQUIRE(owner.changeObjectiveOffset(3) == HighsStatus::kOk);
  const HighsInt indices[] = {0, 1};
  const double coefficients[] = {1, -1};
  REQUIRE(owner.addRow(-1, kHighsInf, 2, indices, coefficients) == HighsStatus::kOk);
  HighsLp before = owner.getLp();
  before.a_matrix_.ensureColwise();
  HighsBasis basis;
  basis.valid = true;
  basis.col_status = {HighsBasisStatus::kLower, HighsBasisStatus::kZero};
  basis.row_status = {HighsBasisStatus::kBasic};
  std::vector<double> relaxation = {0.5, 1}, seed;
  HighsReplayContext context;
  context.method = "rens";
  context.parent_phase = "tree";
  context.decision = 1;
  context.worker = context.restart = 0;
  context.parent_time = 10;
  context.helper_lp_limit = 100;
  context.lp_solution_status = kSolutionStatusFeasible;
  context.relaxation = &relaxation;
  context.base_lower = &before.col_lower_;
  context.base_upper = &before.col_upper_;
  context.checked_seed = &seed;
  context.options = &owner.getOptions();
  REQUIRE_FALSE(HighsReplayCapture::capture(before, "node_lp", 0, &basis));
  {
    std::ofstream guard(base + "0.json");
    guard << "owned guard";
  }
  REQUIRE_FALSE(HighsReplayCapture::capture(before, "sub_mip", 0, &basis, &context));
  {
    std::ifstream guard(base + "0.json");
    std::string text;
    std::getline(guard, text);
    REQUIRE(text == "owned guard");
  }
  REQUIRE(std::remove((base + "0.json").c_str()) == 0);
  std::vector<std::thread> threads;
  std::vector<int> accepted(6, 0);
  for (size_t i = 0; i < 6; ++i)
    threads.emplace_back([&, i]() {
      accepted[i] = HighsReplayCapture::capture(before, "sub_mip", 0, &basis, &context);
    });
  for (auto& thread : threads) thread.join();
  for (int value : accepted) REQUIRE(value == 1);
  REQUIRE_FALSE(HighsReplayCapture::capture(before, "sub_mip", 0, &basis, &context));
  for (size_t i = 0; i < 6; ++i) {
    Highs readback;
    readback.setOptionValue("output_flag", false);
    REQUIRE(readback.readModel(base + std::to_string(i) + ".mps") == HighsStatus::kOk);
    const auto& lp = readback.getLp();
    REQUIRE(lp.col_cost_ == before.col_cost_);
    REQUIRE(lp.col_lower_ == before.col_lower_);
    REQUIRE(lp.col_upper_ == before.col_upper_);
    REQUIRE(lp.row_lower_ == before.row_lower_);
    REQUIRE(lp.row_upper_ == before.row_upper_);
    REQUIRE(lp.integrality_ == before.integrality_);
    REQUIRE(lp.offset_ == before.offset_);
    REQUIRE(lp.a_matrix_.start_ == before.a_matrix_.start_);
    REQUIRE(lp.a_matrix_.index_ == before.a_matrix_.index_);
    REQUIRE(lp.a_matrix_.value_ == before.a_matrix_.value_);
    std::ifstream input(base + std::to_string(i) + ".json");
    const std::string text((std::istreambuf_iterator<char>(input)), {});
    REQUIRE(text.find("\"method\":\"rens\"") != std::string::npos);
    REQUIRE(text.find("\"basis_available\":true") != std::string::npos);
    REQUIRE(text.find("\"construction_context\"") != std::string::npos);
  }
  REQUIRE(owner.getLp().col_lower_ == before.col_lower_);
  REQUIRE(owner.getLp().integrality_ == before.integrality_);
  REQUIRE(relaxation == std::vector<double>{0.5, 1});
}

TEST_CASE("search-work-is-worker-owned-scoped-and-non-destructive",
          "[highs_node_lp_effort]") {
  HighsSearchWork work;
  work.chargeLp(100);
  REQUIRE(work.summary().direct_iterations[0] == 0);
  work.begin(10, false);
  work.chargeLp(11);
  REQUIRE_FALSE(work.exhausted());
  REQUIRE_FALSE(work.skipOptional());
  work.finish();
  REQUIRE(work.summary().exhausted_leases == 1);
  work.begin(10, true, 2.0);
  REQUIRE(work.remaining() == 10);
  work.chargeLp(2);
  REQUIRE(work.remaining() == 8);
  work.recordRestoration(0.25);
  work.recordLocalBound(1.0, 3.0);
  work.recordLocalBound(-kHighsInf, 5.0);
  work.recordCompletionExemption();
  {
    HighsSearchWorkScope scope(work, HighsSearchWorkRole::kHeuristic);
    work.chargeLp(3);
    {
      HighsSearchWorkScope inner(work, HighsSearchWorkRole::kBranch);
      work.chargeLp(1);
    }
    REQUIRE(work.role() == HighsSearchWorkRole::kHeuristic);
  }
  work.chargeSubMip(4);
  REQUIRE(work.exhausted());
  REQUIRE(work.remaining() == 0);
  REQUIRE(work.skipOptional());
  work.recordYield();
  work.finish();
  REQUIRE_FALSE(work.exhausted());
  REQUIRE(work.summary().leases == 2);
  REQUIRE(work.summary().yields == 1);
  REQUIRE(work.summary().skipped_optional == 1);
  REQUIRE(work.summary().direct_iterations[0] == 13);
  REQUIRE(work.summary().direct_iterations[2] == 3);
  REQUIRE(work.summary().direct_iterations[3] == 1);
  REQUIRE(work.summary().submip_reported_iterations == 4);
  REQUIRE(work.summary().max_lease_work == 11);
  REQUIRE(work.summary().restoration_calls == 1);
  REQUIRE(work.summary().restoration_seconds == 0.25);
  REQUIRE(work.summary().local_bound_gain == 2.0);
  REQUIRE(work.summary().completion_exemptions == 1);
  REQUIRE(work.tasks().size() == 2);
  REQUIRE(work.tasks()[1].started_seconds == 2.0);
  REQUIRE(work.tasks()[1].work == 10);
  REQUIRE(work.tasks()[1].restoration_seconds == 0.25);
  work.begin(1, true);
  work.chargeLp(std::numeric_limits<int64_t>::max());
  work.chargeLp(1);
  work.finish();
  REQUIRE(work.summary().direct_iterations[0] ==
          std::numeric_limits<int64_t>::max());
}

TEST_CASE("node-lp-effort-applies-only-to-authoritative-tree-search",
          "[highs_node_lp_effort]") {
  REQUIRE(HighsSearch::boundedNodeLpEffortApplies(false, true));
  REQUIRE_FALSE(HighsSearch::boundedNodeLpEffortApplies(true, true));
  REQUIRE_FALSE(HighsSearch::boundedNodeLpEffortApplies(false, false));
}

TEST_CASE("node-lp-effort-resume-basis-requires-complete-dimensions",
          "[highs_node_lp_effort]") {
  HighsBasis basis;
  basis.valid = true;
  basis.row_status.resize(2, HighsBasisStatus::kBasic);
  basis.col_status.resize(3, HighsBasisStatus::kLower);
  REQUIRE(HighsSearch::resumableBasisMatches(basis, 2, 3));
  REQUIRE_FALSE(HighsSearch::resumableBasisMatches(basis, 1, 3));
  REQUIRE_FALSE(HighsSearch::resumableBasisMatches(basis, 2, 2));
  basis.valid = false;
  REQUIRE_FALSE(HighsSearch::resumableBasisMatches(basis, 2, 3));
}

TEST_CASE("node-lp-effort-disabled-is-an-identity",
          "[highs_node_lp_effort]") {
  HighsNodeLpEffortController controller;
  REQUIRE(controller.iterationLimit(1000000, 1, 0) == kHighsIInf);
}

TEST_CASE("node-lp-effort-root-seed-is-bounded",
          "[highs_node_lp_effort]") {
  HighsNodeLpEffortConfig config;
  config.enabled = true;
  config.minimum_iterations = 10000;
  config.maximum_iterations = 250000;
  config.quantile_multiplier = 2.0;
  HighsNodeLpEffortController controller;
  controller.configure(config);

  REQUIRE(controller.iterationLimit(10000000, 20, 0) == 250000);
  REQUIRE(controller.iterationLimit(40000, 20, 0) == 20000);
}

TEST_CASE("node-lp-effort-learns-and-retries-deterministically",
          "[highs_node_lp_effort]") {
  HighsNodeLpEffortConfig config;
  config.enabled = true;
  config.minimum_iterations = 1000;
  config.maximum_iterations = 100000;
  config.quantile_multiplier = 2.0;
  config.retry_multiplier = 2;
  config.maximum_yields = 3;
  HighsNodeLpEffortController controller;
  controller.configure(config);

  for (HighsInt iterations : {1000, 1500, 2000, 3000}) {
    HighsNodeLpEffortObservation observation;
    observation.depth = 4;
    observation.iteration_count = iterations;
    controller.recordCompleted(observation);
  }

  const HighsInt first = controller.iterationLimit(1000000, 20, 0);
  const HighsInt retry = controller.iterationLimit(1000000, 20, 1);
  REQUIRE(first == 4000);
  REQUIRE(retry == 8000);
  REQUIRE(controller.iterationLimit(1000000, 20, 3) == kHighsIInf);
  REQUIRE(controller.percentileIterations(0.75) == 2048);
}

TEST_CASE("node-lp-effort-telemetry-is-bounded",
          "[highs_node_lp_effort]") {
  HighsNodeLpEffortController controller;
  HighsNodeLpEffortConfig config;
  config.enabled = true;
  controller.configure(config);
  controller.recordAttempt();
  controller.recordYield(12000);
  controller.recordAttempt();
  HighsNodeLpEffortObservation observation;
  observation.iteration_count = 8000;
  controller.recordCompleted(observation);
  controller.recordSerialCompletion();
  controller.recordTaskLease(1.5);
  controller.recordTaskLease(0.5);
  controller.recordTaskLease(1100.0);
  controller.recordTaskLease(2200.0);
  controller.recordTaskLease(-1.0);

  const HighsNodeLpEffortSummary& summary = controller.summary();
  REQUIRE(summary.attempts == 2);
  REQUIRE(summary.yielded == 1);
  REQUIRE(summary.completed == 1);
  REQUIRE(summary.serial_completions == 1);
  REQUIRE(summary.total_iterations == 20000);
  REQUIRE(summary.completed_iterations == 8000);
  REQUIRE(summary.yielded_iterations == 12000);
  REQUIRE(summary.maximum_iterations == 12000);
  REQUIRE(summary.histogram[12] == 1);
  REQUIRE(summary.yielded_histogram[13] == 1);
  REQUIRE(summary.task_leases == 4);
  REQUIRE(summary.task_lease_total_seconds == 3302.0);
  REQUIRE(summary.task_lease_max_seconds == 2200.0);
  REQUIRE(summary.task_lease_microseconds_histogram[18] == 1);
  REQUIRE(summary.task_lease_microseconds_histogram[20] == 1);
  REQUIRE(summary.task_lease_microseconds_histogram[30] == 1);
  REQUIRE(summary.task_lease_microseconds_histogram[31] == 1);
  const double median_upper =
      HighsNodeLpEffortController::taskLeasePercentileUpperSeconds(
          summary.task_lease_microseconds_histogram, 0.50);
  const double p95_upper =
      HighsNodeLpEffortController::taskLeasePercentileUpperSeconds(
          summary.task_lease_microseconds_histogram, 0.95);
  REQUIRE(median_upper >= 1.5);
  REQUIRE(median_upper < 2.1);
  REQUIRE(p95_upper >= 2200.0);
  REQUIRE(std::isfinite(p95_upper));
}

TEST_CASE("node-lp-effort-learns-only-from-recent-completions",
          "[highs_node_lp_effort]") {
  HighsNodeLpEffortConfig config;
  config.enabled = true;
  config.minimum_iterations = 1;
  config.maximum_iterations = 1000000;
  config.quantile_multiplier = 1.0;
  HighsNodeLpEffortController controller;
  controller.configure(config);

  for (HighsInt iterations = 100; iterations <= 800; iterations += 100) {
    HighsNodeLpEffortObservation observation;
    observation.depth = 40;
    observation.iteration_count = iterations;
    controller.recordCompleted(observation);
  }
  controller.recordYield(100000);
  HighsNodeLpEffortObservation failure;
  failure.depth = 40;
  failure.iteration_count = 200000;
  failure.outcome = HighsNodeLpEffortOutcome::kNumericalFailure;
  controller.recordCompleted(failure);
  HighsNodeLpEffortObservation limit;
  limit.depth = 40;
  limit.iteration_count = 50000;
  limit.outcome = HighsNodeLpEffortOutcome::kLimitReached;
  controller.recordCompleted(limit);

  REQUIRE(controller.iterationLimit(1000000, 40, 0) == 600);
  REQUIRE(controller.summary().completed == 8);
  REQUIRE(controller.summary().yielded == 1);
  REQUIRE(controller.summary().numerical_failures == 1);
  REQUIRE(controller.summary().limit_stops == 1);
  REQUIRE(controller.summary().total_iterations == 353600);
  REQUIRE(controller.summary().completed_iterations == 3600);
  REQUIRE(controller.summary().yielded_iterations == 100000);
}

TEST_CASE("node-lp-effort-small-depth-window-seeds-from-parent",
          "[highs_node_lp_effort]") {
  HighsNodeLpEffortConfig config;
  config.enabled = true;
  config.minimum_iterations = 1;
  config.maximum_iterations = 1000000;
  config.quantile_multiplier = 1.0;
  HighsNodeLpEffortController controller;
  controller.configure(config);

  for (HighsInt iterations : {1000, 2000, 3000, 4000}) {
    HighsNodeLpEffortObservation observation;
    observation.depth = 20;
    observation.iteration_count = iterations;
    controller.recordCompleted(observation);
  }
  for (HighsInt iterations : {10000, 20000}) {
    HighsNodeLpEffortObservation observation;
    observation.depth = 60;
    observation.iteration_count = iterations;
    controller.recordCompleted(observation);
  }

  REQUIRE(controller.iterationLimit(1000000, 60, 0) == 3000);
}

TEST_CASE("yielded-node-resume-envelope-preserves-proof-state",
          "[highs_node_lp_effort]") {
  HighsNodeQueue queue;
  queue.setNumCol(0);
  auto basis = std::make_shared<HighsBasis>();

  queue.emplaceNode({}, {}, 17.0, 19.0, 4, basis, 2);
  HighsNodeQueue::OpenNode node = std::move(queue.popBestNode());

  REQUIRE(node.lower_bound == 17.0);
  REQUIRE(node.estimate == 19.0);
  REQUIRE(node.depth == 4);
  REQUIRE(node.basis == basis);
  REQUIRE(node.yield_count == 2);
  REQUIRE(queue.empty());
}

TEST_CASE("forced-node-lp-yields-preserve-terminal-proof",
          "[highs_node_lp_effort]") {
  struct Outcome {
    HighsModelStatus status;
    double objective;
    double dual_bound;
  };
  const std::string filename =
      std::string(HIGHS_DIR) + "/check/instances/bell5.mps";

  auto solve = [&](bool bounded) {
    Highs highs;
    highs.resetGlobalScheduler(true);
    REQUIRE(highs.setOptionValue("output_flag", false) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("threads", 1) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("parallel", kHighsOffString) ==
            HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_rel_gap", 0.0) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_abs_gap", 0.0) == HighsStatus::kOk);
    if (bounded) {
      REQUIRE(highs.setOptionValue("mip_node_lp_effort", true) ==
              HighsStatus::kOk);
      REQUIRE(highs.setOptionValue("mip_node_lp_effort_min_iterations", 1) ==
              HighsStatus::kOk);
      REQUIRE(highs.setOptionValue("mip_node_lp_effort_max_iterations", 1) ==
              HighsStatus::kOk);
      REQUIRE(highs.setOptionValue("mip_node_lp_effort_max_yields", 3) ==
              HighsStatus::kOk);
    }
    REQUIRE(highs.readModel(filename) == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    const HighsInfo& info = highs.getInfo();
    return Outcome{highs.getModelStatus(), info.objective_function_value,
                   info.mip_dual_bound};
  };

  const Outcome baseline = solve(false);
  const Outcome yielded = solve(true);
  REQUIRE(baseline.status == HighsModelStatus::kOptimal);
  REQUIRE(yielded.status == baseline.status);
  REQUIRE(std::abs(yielded.objective - baseline.objective) <= 1e-6);
  REQUIRE(std::abs(yielded.dual_bound - baseline.dual_bound) <= 1e-6);
}

TEST_CASE("cooperative-task-yields-preserve-tree-proof-and-shadow-identity",
          "[highs_node_lp_effort]") {
  struct Result {
    double objective;
    double bound;
    int64_t nodes;
    int64_t iterations;
    int64_t yields;
  };
  auto solve = [&](HighsInt limit, bool enforce, HighsInt threads) {
    Highs highs;
    highs.resetGlobalScheduler(true);
    highs.setOptionValue("log_to_console", false);
    highs.setOptionValue("threads", threads);
    highs.setOptionValue("parallel", threads == 1 ? "off" : "on");
    highs.setOptionValue("mip_rel_gap", 0.0);
    highs.setOptionValue("mip_abs_gap", 0.0);
    highs.setOptionValue("mip_allow_restart", false);
    REQUIRE(highs.setOptionValue("mip_search_work_lp_limit", limit) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_search_work_enforce", enforce) == HighsStatus::kOk);
    int64_t yields = 0;
    highs.setCallback([&](int kind, const std::string& message,
                          const HighsCallbackOutput*, HighsCallbackInput*, void*) {
      if (kind != kCallbackLogging || message.find("MIP-SearchWork:") == std::string::npos)
        return;
      std::istringstream tokens(message);
      std::string token;
      while (tokens >> token)
        if (token.compare(0, 7, "yields=") == 0) yields += std::stoll(token.substr(7));
    });
    REQUIRE(highs.startCallback(kCallbackLogging) == HighsStatus::kOk);
    REQUIRE(highs.readModel(std::string(HIGHS_DIR) + "/check/instances/bell5.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
    const auto& info = highs.getInfo();
    return Result{info.objective_function_value, info.mip_dual_bound,
                  info.mip_node_count, info.simplex_iteration_count, yields};
  };
  const Result baseline = solve(0, false, 1);
  const Result shadow = solve(1, false, 1);
  REQUIRE(shadow.nodes == baseline.nodes);
  REQUIRE(shadow.iterations == baseline.iterations);
  REQUIRE(shadow.yields == 0);
  for (HighsInt threads : {1, 4}) {
    const Result bounded = solve(1, true, threads);
    REQUIRE(bounded.yields > 0);
    REQUIRE(bounded.objective == Approx(baseline.objective));
    REQUIRE(bounded.bound == Approx(baseline.bound));
  }
}

TEST_CASE("exclusive-lp-call-integration-emits-bounded-final-observations",
          "[highs_node_lp_effort][highs_lp_work_integration]") {
  Highs highs;
  highs.resetGlobalScheduler(true);
  highs.setOptionValue("log_to_console", false);
  highs.setOptionValue("threads", 1);
  highs.setOptionValue("mip_rel_gap", 0.0);
  highs.setOptionValue("mip_abs_gap", 0.0);
  highs.setOptionValue("mip_allow_restart", false);
  highs.setOptionValue("mip_search_work_lp_limit", 1);
  highs.setOptionValue("mip_search_work_enforce", false);
  std::array<bool, 4> seen{};
  int64_t calls = 0;
  highs.setCallback([&](int kind, const std::string& message,
                        const HighsCallbackOutput*, HighsCallbackInput*, void*) {
    if (kind != kCallbackLogging || message.find("MIP-LpWork:") == std::string::npos)
      return;
    REQUIRE(message.find("schema=rost/highs-lp-work/v1") != std::string::npos);
    REQUIRE(message.find("scope=surviving_worker_tree_tasks") != std::string::npos);
    for (size_t i = 0; i < seen.size(); ++i) {
      const std::string name = highsLpWorkKindName(static_cast<HighsLpWorkKind>(i));
      if (message.find("kind=" + name + " ") != std::string::npos) {
        REQUIRE_FALSE(seen[i]);
        seen[i] = true;
      }
    }
    std::istringstream tokens(message);
    std::string token;
    while (tokens >> token) {
      if (token.compare(0, 6, "calls=") == 0) calls += std::stoll(token.substr(6));
      if (token.compare(0, 8, "seconds=") == 0) {
        const double seconds = std::stod(token.substr(8));
        REQUIRE(std::isfinite(seconds));
        REQUIRE(seconds >= 0);
      }
    }
  });
  REQUIRE(highs.startCallback(kCallbackLogging) == HighsStatus::kOk);
  REQUIRE(highs.readModel(std::string(HIGHS_DIR) + "/check/instances/bell5.mps") == HighsStatus::kOk);
  REQUIRE(highs.run() == HighsStatus::kOk);
  REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
  for (bool found : seen) REQUIRE(found);
  REQUIRE(calls > 0);
}

TEST_CASE("managed-repair-preserves-exact-optimum-with-depleted-task-budgets",
          "[highs_node_lp_effort][highs_primal_heuristic_manager]") {
  for (HighsInt threads : {1, 4}) {
    Highs highs;
    highs.resetGlobalScheduler(true);
    highs.setOptionValue("log_to_console", false);
    highs.setOptionValue("threads", threads);
    highs.setOptionValue("parallel", threads == 1 ? "off" : "on");
    highs.setOptionValue("mip_rel_gap", 0.0);
    highs.setOptionValue("mip_abs_gap", 0.0);
    highs.setOptionValue("mip_allow_restart", false);
    highs.setOptionValue("mip_root_max_separation_rounds", 0);
    highs.setOptionValue("mip_heuristic_run_root_reduced_cost", false);
    highs.setOptionValue("mip_incumbent_exchange", "opportunistic");
    highs.setOptionValue("mip_primal_heuristic_schedule", "managed-rins");
    highs.setOptionValue("mip_rins_max_calls", 8);
    highs.setOptionValue("mip_rins_max_lp_iterations", 200000);
    highs.setOptionValue("mip_rins_max_nodes", 200);
    highs.setOptionValue("mip_search_work_lp_limit", 1);
    highs.setOptionValue("mip_search_work_enforce", true);
    highs.setOptionValue("time_limit", 30.0);
    int64_t reservations = 0;
    highs.setCallback([&](int kind, const std::string& message,
                          const HighsCallbackOutput*, HighsCallbackInput*, void*) {
      if (kind != kCallbackLogging || message.find("MIP-SearchWork:") == std::string::npos) return;
      REQUIRE(message.find("scope=surviving_worker_tree_tasks") != std::string::npos);
      std::istringstream tokens(message);
      std::string token;
      while (tokens >> token)
        if (token.compare(0, 20, "repair_reservations=") == 0)
          reservations += std::stoll(token.substr(20));
    });
    highs.startCallback(kCallbackLogging);
    REQUIRE(highs.readModel(std::string(HIGHS_DIR) + "/check/instances/bell5.mps") == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
    REQUIRE(highs.getInfo().objective_function_value == Approx(8966406.491519));
    REQUIRE(highs.getInfo().mip_dual_bound == Approx(8966406.491519));
    REQUIRE(reservations > 0);
  }
}

TEST_CASE("tree-lease-and-native-branch-probe-budget-boundaries",
          "[highs_node_lp_effort][highs_branch_lease]") {
  const HighsInt columns = GENERATE(5, 7, 9);
  Highs owner;
  owner.resetGlobalScheduler(true);
  REQUIRE(owner.setOptionValue("output_flag", false) == HighsStatus::kOk);
  REQUIRE(owner.setOptionValue("threads", 1) == HighsStatus::kOk);
  highs::parallel::initialize_scheduler(1);
  for (HighsInt col = 0; col < columns; ++col) {
    REQUIRE(owner.addCol(1, 0, 1, 0, nullptr, nullptr) == HighsStatus::kOk);
    REQUIRE(owner.changeColIntegrality(col, HighsVarType::kInteger) == HighsStatus::kOk);
  }
  for (HighsInt row = 0; row < columns; ++row) {
    const HighsInt indices[] = {row, (row + 1) % columns};
    const double values[] = {1, 1};
    REQUIRE(owner.addRow(1, kHighsInf, 2, indices, values) == HighsStatus::kOk);
  }
  REQUIRE(owner.ensureColwise() == HighsStatus::kOk);
  REQUIRE(owner.getLp().a_matrix_.isColwise());

  HighsCallback callback(&owner);
  HighsSolution solution;
  HighsMipSolver mip(callback, owner.getOptions(), owner.getLp(), solution);
  mip.modelstatus_ = HighsModelStatus::kNotset;
  mip.improving_solution_file_ = nullptr;
  mip.timer_.start();
  HighsProfiling profiling;
  profiling.initialize(mip.timer_, false);
  mip.setProfiling(&profiling);
  mip.mipdata_.reset(new HighsMipSolverData(mip));
  auto& data = *mip.mipdata_;
  data.init();
  data.presolvedModel = owner.getLp();
  mip.setModel(data.presolvedModel);
  data.runSetup();
  data.workers.emplace_back(mip, &data.getLp(), &data.getDomain(),
                            &data.getCutPool(), &data.getConflictPool(),
                            &data.getPseudoCost(), 0);
  auto& worker = data.workers.front();
  auto& relaxation = worker.getLpRelaxation();
  relaxation.setProfiling(&profiling);
  relaxation.setMipWorker(worker);
  relaxation.loadModel();
  auto& search = *worker.search_ptr_;
  search.createNewNode();
  REQUIRE(search.evaluateNode() == HighsSearch::NodeResult::kOpen);
  REQUIRE(relaxation.getFractionalIntegers().size() == static_cast<size_t>(columns));
  REQUIRE(relaxation.getObjective() == Approx(0.5 * columns));
  search.setMinReliable(5);

  bool no_probe = false;
  int64_t strong_branch_limit = std::numeric_limits<int64_t>::max();
  SECTION("exhausted enforced tree lease") {
    worker.search_work.begin(1, true);
    worker.search_work.chargeLp(1);
    // The rejected hard-stop experiment did not qualify: completing selection
    // still uses the native probe budget, rather than treating a lease as proof.
  }
  SECTION("advisory lease does not change selection") {
    worker.search_work.begin(1, false);
    worker.search_work.chargeLp(1);
  }
  SECTION("heuristics retain their separate budget") {
    worker.search_work.begin(1, true);
    worker.search_work.chargeLp(1);
    search.setHeuristic(true);
  }
  SECTION("inactive lease does not change selection") {}
  SECTION("native strong-branch limit still stops probes") {
    worker.search_work.begin(1, false);
    strong_branch_limit = 0;
    no_probe = true;
  }

  const double parent_bound = search.getCurrentLowerBound();
  const auto lower = search.localdom.col_lower_;
  const auto upper = search.localdom.col_upper_;
  const int64_t before = relaxation.getNumLpIterations();
  double down_bound = -kHighsInf, up_bound = -kHighsInf;
  const HighsInt candidate = search.selectBranchingCandidate(
      strong_branch_limit, down_bound, up_bound);
  CAPTURE(columns, no_probe, candidate, before,
          relaxation.getNumLpIterations(), search.sblpiterations);
  if (no_probe) {
    REQUIRE(relaxation.getNumLpIterations() == before);
    REQUIRE(candidate >= 0);
    REQUIRE(search.sblpiterations == 0);
    REQUIRE(down_bound == parent_bound);
    REQUIRE(up_bound == parent_bound);
    REQUIRE(search.localdom.col_lower_ == lower);
    REQUIRE(search.localdom.col_upper_ == upper);
    REQUIRE_FALSE(search.currentNodePruned());
    for (const auto& account : worker.search_work.summary().lp_calls)
      REQUIRE(account.calls == 0);
  } else {
    REQUIRE(relaxation.getNumLpIterations() > before);
  }
  REQUIRE_FALSE(search.localdom.infeasible());
  REQUIRE(mip.modelstatus_ == HighsModelStatus::kNotset);
  worker.search_work.finish();
}

TEST_CASE("tree-lease-selection-preserves-independently-enumerated-optima",
          "[highs_node_lp_effort][highs_branch_lease]") {
  constexpr HighsInt columns = 11;
  for (unsigned seed = 0; seed < 12; ++seed) {
    HighsModel model;
    auto& lp = model.lp_;
    lp.num_col_ = columns;
    lp.num_row_ = columns + 3;
    lp.col_lower_.assign(columns, 0);
    lp.col_upper_.assign(columns, 1);
    lp.integrality_.assign(columns, HighsVarType::kInteger);
    lp.sense_ = seed % 2 ? ObjSense::kMaximize : ObjSense::kMinimize;
    lp.offset_ = seed % 2 ? -13.25 : 7.5;
    std::vector<std::vector<double>> rows(lp.num_row_,
                                          std::vector<double>(columns, 0));
    lp.row_lower_.assign(lp.num_row_, -kHighsInf);
    lp.row_upper_.assign(lp.num_row_, kHighsInf);
    for (HighsInt row = 0; row < columns; ++row) {
      rows[row][row] = rows[row][(row + 1) % columns] = 1;
      lp.row_lower_[row] = 1;
    }
    unsigned state = seed + 1;
    auto next = [&]() { state = state * 1664525U + 1013904223U; return state; };
    for (HighsInt col = 0; col < columns; ++col)
      lp.col_cost_.push_back((static_cast<int>(next() % 11) - 3) * 0.25);
    for (HighsInt row = columns; row < lp.num_row_; ++row) {
      double planted = 0;
      for (HighsInt col = 0; col < columns; ++col) {
        rows[row][col] = static_cast<int>(next() % 7) - 3;
        if (col % 2 == 0) planted += rows[row][col];
      }
      lp.row_lower_[row] = planted - 2;
      lp.row_upper_[row] = planted + 2;
    }
    if (seed % 4 == 3) {
      rows.back().assign(columns, 1);
      lp.row_lower_.back() = -kHighsInf;
      lp.row_upper_.back() = 0;
    }
    lp.a_matrix_.format_ = MatrixFormat::kRowwise;
    lp.a_matrix_.num_col_ = lp.num_col_;
    lp.a_matrix_.num_row_ = lp.num_row_;
    lp.a_matrix_.start_.assign(1, 0);
    for (const auto& row : rows) {
      for (HighsInt col = 0; col < columns; ++col) {
        if (row[col] == 0) continue;
        lp.a_matrix_.index_.push_back(col);
        lp.a_matrix_.value_.push_back(row[col]);
      }
      lp.a_matrix_.start_.push_back(lp.a_matrix_.index_.size());
    }
    REQUIRE(lp.a_matrix_.start_.size() == static_cast<size_t>(lp.num_row_ + 1));
    for (HighsInt row = 0; row < lp.num_row_; ++row) {
      std::vector<double> decoded(columns, 0);
      for (HighsInt k = lp.a_matrix_.start_[row]; k < lp.a_matrix_.start_[row + 1]; ++k)
        decoded[lp.a_matrix_.index_[k]] += lp.a_matrix_.value_[k];
      REQUIRE(decoded == rows[row]);
    }

    bool feasible = false;
    double best = lp.sense_ == ObjSense::kMinimize ? kHighsInf : -kHighsInf;
    for (unsigned bits = 0; bits < (1U << columns); ++bits) {
      bool satisfies = true;
      for (HighsInt row = 0; row < lp.num_row_; ++row) {
        double activity = 0;
        for (HighsInt col = 0; col < columns; ++col)
          if (bits & (1U << col)) activity += rows[row][col];
        if (activity < lp.row_lower_[row] || activity > lp.row_upper_[row]) {
          satisfies = false;
          break;
        }
      }
      if (!satisfies) continue;
      feasible = true;
      double objective = lp.offset_;
      for (HighsInt col = 0; col < columns; ++col)
        if (bits & (1U << col)) objective += lp.col_cost_[col];
      best = lp.sense_ == ObjSense::kMinimize ? std::min(best, objective)
                                             : std::max(best, objective);
    }
    for (HighsInt threads : {1, 4}) {
      for (HighsInt limit : {0, 1, 8, 64}) {
        CAPTURE(seed, threads, limit, feasible, best);
        Highs highs;
        highs.resetGlobalScheduler(true);
        REQUIRE(highs.setOptionValue("log_to_console", false) == HighsStatus::kOk);
        std::string log;
        highs.setCallback([&](int type, const std::string& message,
                              const HighsCallbackOutput*, HighsCallbackInput*, void*) {
          if (type == kCallbackLogging && log.size() < 16384) log += message;
        });
        REQUIRE(highs.startCallback(kCallbackLogging) == HighsStatus::kOk);
        REQUIRE(highs.setOptionValue("threads", threads) == HighsStatus::kOk);
        REQUIRE(highs.setOptionValue("parallel", "on") == HighsStatus::kOk);
        REQUIRE(highs.setOptionValue("mip_rel_gap", 0.0) == HighsStatus::kOk);
        REQUIRE(highs.setOptionValue("mip_abs_gap", 0.0) == HighsStatus::kOk);
        REQUIRE(highs.setOptionValue("mip_allow_restart", false) == HighsStatus::kOk);
        REQUIRE(highs.setOptionValue("mip_min_cliquetable_entries_for_parallelism", 0) == HighsStatus::kOk);
        REQUIRE(highs.setOptionValue("mip_root_max_separation_rounds", 0) == HighsStatus::kOk);
        REQUIRE(highs.setOptionValue("mip_allow_cut_separation_at_nodes", false) == HighsStatus::kOk);
        REQUIRE(highs.setOptionValue("mip_search_work_lp_limit", limit) == HighsStatus::kOk);
        REQUIRE(highs.setOptionValue("mip_search_work_enforce", limit > 0) == HighsStatus::kOk);
        REQUIRE(highs.passModel(model) == HighsStatus::kOk);
        REQUIRE(highs.run() == HighsStatus::kOk);
        INFO(log);
        if (!feasible) {
          REQUIRE(highs.getModelStatus() == HighsModelStatus::kInfeasible);
          continue;
        }
        REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
        REQUIRE(highs.getInfo().objective_function_value == Approx(best));
        REQUIRE(highs.getInfo().mip_dual_bound == Approx(best));
        const auto& answer = highs.getSolution().col_value;
        double objective = lp.offset_;
        for (HighsInt col = 0; col < columns; ++col) {
          REQUIRE(answer[col] == Approx(std::round(answer[col])).margin(1e-7));
          REQUIRE(answer[col] >= -1e-7);
          REQUIRE(answer[col] <= 1 + 1e-7);
          objective += lp.col_cost_[col] * answer[col];
        }
        REQUIRE(objective == Approx(best));
        for (HighsInt row = 0; row < lp.num_row_; ++row) {
          double activity = 0;
          for (HighsInt col = 0; col < columns; ++col)
            activity += rows[row][col] * answer[col];
          REQUIRE(activity >= lp.row_lower_[row] - 1e-7);
          REQUIRE(activity <= lp.row_upper_[row] + 1e-7);
        }
      }
    }
  }
}
