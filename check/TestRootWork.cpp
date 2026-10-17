#include <limits>
#include <sstream>
#include <map>
#include <future>
#include <random>
#include <thread>
#include <vector>

#include "HCheckConfig.h"
#include "Highs.h"
#include "catch.hpp"
#include "lp_data/HighsOptions.h"
#include "mip/HighsIncumbentMailbox.h"
#include "mip/HighsPrimalSnapshot.h"
#include "mip/HighsPrimalService.h"
#include "interfaces/highs_c_api.h"
#include "mip/HighsMipSolverData.h"
#include "mip/HighsConflictPool.h"
#include "mip/HighsPrimalHeuristics.h"
#include "mip/HighsRootWork.h"
#include "mip/MipTimer.h"
#include "parallel/HighsHardwareTopology.h"

TEST_CASE("sub-MIP provenance names are exact", "[highs_incumbent]") {
  REQUIRE(std::string(HighsPrimalHeuristics::subMipMethodName(
              HighsSubMipOrigin::kRootReducedCost)) == "root_reduced_cost");
  REQUIRE(std::string(HighsPrimalHeuristics::subMipMethodName(
              HighsSubMipOrigin::kRootRens)) == "rens");
  REQUIRE(std::string(HighsPrimalHeuristics::subMipMethodName(
              HighsSubMipOrigin::kTreeRins)) == "rins");
  REQUIRE(std::string(HighsPrimalHeuristics::subMipParentPhaseName(
              HighsSubMipOrigin::kRootRens)) == "root");
  REQUIRE(std::string(HighsPrimalHeuristics::subMipParentPhaseName(
              HighsSubMipOrigin::kTreeRens)) == "tree");
}

TEST_CASE("managed RINS reaches child search without a preparation LP",
          "[highs_incumbent][highs_rins_agreement]") {
  Highs::resetGlobalScheduler(true);
  highs::parallel::initialize_scheduler(1);
  Highs owner;
  owner.setOptionValue("output_flag", false);
  owner.setOptionValue("threads", 1);
  owner.setOptionValue("presolve", "off");
  owner.setOptionValue("mip_detect_symmetry", false);
  for (double cost : {5.0, 2.0, 3.0}) {
    REQUIRE(owner.addCol(cost, 0, 1, 0, nullptr, nullptr) == HighsStatus::kOk);
    REQUIRE(owner.changeColIntegrality(owner.getNumCol() - 1,
                                       HighsVarType::kInteger) == HighsStatus::kOk);
  }
  const HighsInt indices[] = {0, 1, 2};
  const double coefficients[] = {1, 1, 1};
  REQUIRE(owner.addRow(1, kHighsInf, 3, indices, coefficients) == HighsStatus::kOk);
  HighsLp original = owner.getLp();
  original.ensureColwise();
  HighsCallback callback(&owner);
  HighsSolution solution;
  HighsMipSolver mip(callback, owner.getOptions(), original, solution);
  mip.modelstatus_ = HighsModelStatus::kNotset;
  mip.improving_solution_file_ = nullptr;
  mip.timer_.start();
  HighsProfiling profiling;
  profiling.initialize(mip.timer_, false);
  mip.setProfiling(&profiling);
  mip.mipdata_.reset(new HighsMipSolverData(mip));
  auto& data = *mip.mipdata_;
  data.init();
  data.presolvedModel = original;
  mip.setModel(data.presolvedModel);
  data.runSetup();
  data.getLp().loadModel();
  data.workers.emplace_back(mip, &data.getLp(), &data.getDomain(),
      &data.getCutPool(), &data.getConflictPool(), &data.getPseudoCost(), 0);
  auto& worker = data.workers.front();
  REQUIRE(data.addIncumbent({1, 1, 1}, 10, kSolutionSourceRens));
  worker.upper_bound = data.upper_bound;
  worker.upper_limit = data.upper_limit;
  worker.optimality_limit = data.optimality_limit;
  const double before_bound = data.lower_bound;
  const auto before_lower = data.getDomain().col_lower_;
  const auto before_upper = data.getDomain().col_upper_;
  HighsSubMipCallBudget budget;
  budget.max_time = 2;
  budget.max_nodes = 20;
  budget.max_lp_iterations = 1000;
  budget.min_fixing_rate = 0.25;
  for (int exhausted = 0; exhausted < 3; ++exhausted) {
    auto empty = budget;
    if (exhausted == 0) empty.max_time = 0;
    if (exhausted == 1) empty.max_nodes = 0;
    if (exhausted == 2) empty.max_lp_iterations = 0;
    HighsPrimalHeuristicOutcome skipped;
    data.heuristics.RINS(worker, {1, 0.5, 0.5}, &empty, &skipped);
    REQUIRE_FALSE(skipped.started);
    REQUIRE(skipped.child_calls == 0);
    REQUIRE(skipped.preparation_exhaustions == 1);
    REQUIRE(data.upper_bound == 10);
    REQUIRE_FALSE(profiling.running(kMipClockSubMipSolve));
  }
  HighsPrimalHeuristicOutcome outcome;
  data.heuristics.RINS(worker, {1, 0.5, 0.5}, &budget, &outcome);
  REQUIRE(outcome.started);
  REQUIRE(outcome.completed);
  REQUIRE(outcome.phase_measured);
  REQUIRE(outcome.preparation_lp_iterations == 0);
  REQUIRE(outcome.child_calls == 1);
  REQUIRE(outcome.lp_iterations == outcome.child_lp_iterations);
  REQUIRE(outcome.preparation_exhaustions == 0);
  CAPTURE(outcome.candidate_proposed, outcome.candidate_accepted,
          outcome.candidate_improved, data.upper_bound, worker.upper_bound,
          mip.solution_);
  REQUIRE(outcome.candidate_improved);
  REQUIRE(data.upper_bound == 5);  // Exhaustive restricted optimum is (1,0,0).
  REQUIRE(highsVerifyCandidate(original, mip.solution_, nullptr, data.feastol).feasible);
  REQUIRE(data.lower_bound == before_bound);
  REQUIRE(data.num_nodes == 0);
  for (size_t col = 0; col < before_lower.size(); ++col) {
    REQUIRE(data.getDomain().col_lower_[col] >= before_lower[col]);
    REQUIRE(data.getDomain().col_upper_[col] <= before_upper[col]);
  }
  // Owner admission may propagate its new objective cutoff, but child-only
  // agreement fixings must not remove any remaining parent improvement.
  for (int mask = 1; mask < 8; ++mask) {
    const double cost = 5 * (mask & 1) + 2 * ((mask >> 1) & 1) +
                        3 * ((mask >> 2) & 1);
    if (cost >= data.upper_bound) continue;
    for (int col = 0; col < 3; ++col) {
      const double value = (mask >> col) & 1;
      REQUIRE(value >= data.getDomain().col_lower_[col]);
      REQUIRE(value <= data.getDomain().col_upper_[col]);
    }
  }
  REQUIRE_FALSE(profiling.running(kMipClockSubMipSolve));
}

TEST_CASE("zero cut age supports immediate and buffered LP removal",
          "[highs_robustness][highs_root_work]") {
  for (HighsInt initial_limit : {0, 30})
    for (bool buffered : {false, true}) {
      CAPTURE(initial_limit, buffered);
      Highs owner;
      REQUIRE(owner.setOptionValue("output_flag", false) == HighsStatus::kOk);
      REQUIRE(owner.setOptionValue("mip_pool_age_limit", initial_limit) == HighsStatus::kOk);
      REQUIRE(owner.addVar(0, 1) == HighsStatus::kOk);
      REQUIRE(owner.addVar(0, 1) == HighsStatus::kOk);
      HighsInt index[] = {0, 1};
      double value[] = {1, 1};
      REQUIRE(owner.addRow(-kHighsInf, 2, 2, index, value) == HighsStatus::kOk);
      HighsCallback callback(&owner);
      HighsSolution solution;
      HighsMipSolver mip(callback, owner.getOptions(), owner.getLp(), solution);
      mip.modelstatus_ = HighsModelStatus::kNotset;
      mip.mipdata_.reset(new HighsMipSolverData(mip));
      auto& pool = mip.mipdata_->getCutPool();
      const HighsInt cut = pool.addCut(mip, index, value, 2, 1.5, false, false, false);
      REQUIRE(cut >= 0);
      REQUIRE(pool.getNumCuts() == 1);
      HighsCutSet set;
      pool.separateLpCutsAfterRestart(set);
      REQUIRE(set.numCuts() == 1);
      REQUIRE(pool.getNumAvailableCuts() == 0);
      pool.setAgeLimit(0);
      pool.lpCutRemoved(cut, buffered);
      pool.performAging();
      pool.performAging();
      REQUIRE(pool.getNumCuts() == 0);
      REQUIRE(pool.getNumAvailableCuts() == 0);
      REQUIRE(mip.modelstatus_ == HighsModelStatus::kNotset);
    }
}

TEST_CASE("cold feasibility seeds retain one finite diversification opportunity",
          "[highs_root_work][highs_incumbent]") {
  HighsRootWorkConfig config;
  config.enabled = true;
  config.max_heuristic_time = 30;
  config.max_single_helper_time = 10;
  config.tree_reserve_time = 30;
  for (double seed : {-100.0, 0.0, 100.0}) {
    for (bool cold : {false, true}) {
      for (bool weighted : {false, true}) {
        for (double incumbent : {seed - 1, seed}) {
          HighsRootWorkController control;
          control.beginStage(config, 0, 300);
          control.beginEpoch();
          const bool eligible = cold && !weighted && incumbent == seed;
          REQUIRE(control.coldSeedDiversificationAllowance(
                      50, cold, weighted, seed, incumbent) ==
                  (eligible ? 10 : 0));
          control.beginEpoch();
          if (cold)
            REQUIRE(control.coldSeedDiversificationAllowance(
                        60, cold, false, seed, seed) == 0);
        }
      }
    }
  }
  SECTION("same total heuristic allowance") {
    HighsRootWorkController control;
    control.beginStage(config, 0, 300);
    control.recordCompleted(HighsRootWorkPhase::kRens, 25);
    REQUIRE(control.coldSeedDiversificationAllowance(50, true, false, 10, 10) == 5);
    REQUIRE(control.coldSeedDiversificationAllowance(51, true, false, 10, 10) == 0);
  }
  for (int exhausted = 0; exhausted < 5; ++exhausted) {
    auto limited = config;
    if (exhausted == 0) limited.enabled = false;
    if (exhausted == 1) limited.max_single_helper_time = 0;
    if (exhausted == 2) limited.max_heuristic_time = 0;
    if (exhausted == 3) limited.max_single_helper_time = -1;
    if (exhausted == 3) limited.max_heuristic_time = -1;
    HighsRootWorkController control;
    control.beginStage(limited, 0, exhausted == 3 ? kHighsInf : 100);
    REQUIRE(control.coldSeedDiversificationAllowance(
                exhausted == 4 ? 71 : 50, true, false, 10, 10) == 0);
  }
  for (double invalid : {-1.0, kHighsInf,
                         std::numeric_limits<double>::quiet_NaN()}) {
    HighsRootWorkController control;
    control.beginStage(config, 0, 300);
    REQUIRE(control.coldSeedDiversificationAllowance(invalid, true, false, 10, 10) == 0);
  }
  for (double invalid : {kHighsInf, -kHighsInf,
                         std::numeric_limits<double>::quiet_NaN()}) {
    HighsRootWorkController control;
    control.beginStage(config, 0, 300);
    REQUIRE(control.coldSeedDiversificationAllowance(50, true, false, invalid, 10) == 0);
    REQUIRE(control.coldSeedDiversificationAllowance(50, true, false, 10, invalid) == 0);
  }
}

TEST_CASE("relaxation-based RENS can improve an existing feasibility seed",
          "[highs_root_work][highs_incumbent]") {
  Highs::resetGlobalScheduler(true);
  highs::parallel::initialize_scheduler(1);
  Highs owner;
  owner.setOptionValue("output_flag", false);
  owner.setOptionValue("threads", 1);
  owner.setOptionValue("presolve", "off");
  owner.setOptionValue("mip_detect_symmetry", false);
  for (double cost : {5.0, 2.0, 3.0}) {
    REQUIRE(owner.addCol(cost, 0, 1, 0, nullptr, nullptr) == HighsStatus::kOk);
    REQUIRE(owner.changeColIntegrality(owner.getNumCol() - 1,
                                       HighsVarType::kInteger) == HighsStatus::kOk);
  }
  const HighsInt indices[] = {0, 1, 2};
  const double coefficients[] = {1, 1, 1};
  REQUIRE(owner.addRow(1, kHighsInf, 3, indices, coefficients) == HighsStatus::kOk);
  HighsLp original = owner.getLp();
  original.ensureColwise();
  HighsCallback callback(&owner);
  HighsSolution solution;
  HighsMipSolver mip(callback, owner.getOptions(), original, solution);
  mip.modelstatus_ = HighsModelStatus::kNotset;
  mip.improving_solution_file_ = nullptr;
  mip.timer_.start();
  HighsProfiling profiling;
  profiling.initialize(mip.timer_, false);
  mip.setProfiling(&profiling);
  mip.mipdata_.reset(new HighsMipSolverData(mip));
  auto& data = *mip.mipdata_;
  data.init();
  data.presolvedModel = original;
  mip.setModel(data.presolvedModel);
  data.runSetup();
  data.getLp().loadModel();
  data.workers.emplace_back(mip, &data.getLp(), &data.getDomain(),
      &data.getCutPool(), &data.getConflictPool(), &data.getPseudoCost(), 0);
  auto& worker = data.workers.front();
  REQUIRE(data.addIncumbent({1, 1, 1}, 10, kSolutionSourceSubMip));
  worker.upper_bound = data.upper_bound;
  worker.upper_limit = data.upper_limit;
  worker.optimality_limit = data.optimality_limit;
  const double parent_bound = data.lower_bound;
  const auto parent_nodes = data.num_nodes;
  HighsRootWorkConfig config;
  config.enabled = true;
  config.max_heuristic_time = 2;
  config.max_single_helper_time = 2;
  HighsRootWorkController control;
  control.beginStage(config, 0, 30);
  const double allowance = control.coldSeedDiversificationAllowance(
      mip.timer_.read(), true, false, 10, data.upper_bound);
  REQUIRE(allowance > 0);
  data.heuristics.RENS(worker, {0.5, 0.5, 0.5}, allowance,
                       HighsSubMipOrigin::kRootRens);
  REQUIRE(data.upper_bound < 10);
  REQUIRE(highsVerifyCandidate(original, mip.solution_, nullptr, data.feastol).feasible);
  double oracle = kHighsInf;
  for (int mask = 1; mask < 8; ++mask) {
    const double objective = 5 * (mask & 1) + 2 * ((mask >> 1) & 1) +
                             3 * ((mask >> 2) & 1);
    oracle = std::min(oracle, objective);
    if (objective < data.upper_bound)
      for (int col = 0; col < 3; ++col) {
        const double value = (mask >> col) & 1;
        REQUIRE(value >= data.getDomain().col_lower_[col]);
        REQUIRE(value <= data.getDomain().col_upper_[col]);
      }
  }
  REQUIRE(data.upper_bound >= oracle);
  REQUIRE(data.lower_bound == parent_bound);
  REQUIRE(data.num_nodes == parent_nodes);
  REQUIRE(mip.modelstatus_ == HighsModelStatus::kNotset);
  REQUIRE_FALSE(profiling.running(kMipClockSubMipSolve));
  REQUIRE(control.coldSeedDiversificationAllowance(mip.timer_.read(), true, false, 10,
                                                  data.upper_bound) == 0);
}

TEST_CASE("lowering cut age retains live histogram buckets until deletion",
          "[highs_robustness][highs_root_work]") {
  Highs owner;
  REQUIRE(owner.setOptionValue("output_flag", false) == HighsStatus::kOk);
  REQUIRE(owner.setOptionValue("mip_pool_age_limit", 30) == HighsStatus::kOk);
  REQUIRE(owner.addVar(0, 1) == HighsStatus::kOk);
  HighsInt index = 0;
  double value = 1;
  REQUIRE(owner.addRow(-kHighsInf, 1, 1, &index, &value) == HighsStatus::kOk);
  HighsCallback callback(&owner);
  HighsSolution solution;
  HighsMipSolver mip(callback, owner.getOptions(), owner.getLp(), solution);
  mip.modelstatus_ = HighsModelStatus::kNotset;
  mip.mipdata_.reset(new HighsMipSolverData(mip));
  auto& pool = mip.mipdata_->getCutPool();
  REQUIRE(pool.addCut(mip, &index, &value, 1, 0.5, false, false, false) >= 0);
  REQUIRE(pool.getNumAvailableCuts() == 1);
  pool.setAgeLimit(0);
  pool.performAging();
  REQUIRE(pool.getNumAvailableCuts() == 0);
  REQUIRE(pool.getNumCuts() == 0);
}

TEST_CASE("bounded root helper sub-MIPs cannot recurse",
          "[highs_incumbent]") {
  HighsOptions options;
  options.mip_heuristic_run_rens = true;
  options.mip_heuristic_run_rins = true;
  options.mip_heuristic_run_root_reduced_cost = true;

  HighsPrimalHeuristics::configureSubMipOptions(
      options, HighsSubMipOrigin::kRootRens);

  REQUIRE_FALSE(options.mip_heuristic_run_rens);
  REQUIRE_FALSE(options.mip_heuristic_run_rins);
  REQUIRE_FALSE(options.mip_heuristic_run_root_reduced_cost);
}

TEST_CASE("lowering conflict age preserves live buckets until retirement",
          "[highs_robustness][highs_root_work]") {
  for (int operation = 0; operation < 5; ++operation) {
    CAPTURE(operation);
    HighsConflictPool pool(30, 100);
    const HighsDomainChange change{1.0, 0, HighsBoundType::kLower};
    pool.addConflictFromOtherPool(&change, 1);
    for (int age = 0; age < 12; ++age) pool.performAging();
    REQUIRE(pool.getNumConflicts() == 1);
    pool.setAgeLimit(0);
    REQUIRE(pool.getNumConflicts() == 1);
    if (operation == 0) {
      pool.resetAge(0);
    } else if (operation == 1) {
      pool.setAgeLock(true);
      pool.resetAge(0);
      pool.performAging(true);
      REQUIRE(pool.getNumConflicts() == 1);
      pool.setAgeLock(false);
    } else if (operation == 2) {
      pool.removeConflict(0);
    } else if (operation == 3) {
      HighsConflictPool destination(30, 100);
      pool.syncConflictPool(destination);
      REQUIRE(destination.getNumConflicts() == 1);
      REQUIRE(destination.getConflictEntryVector()[0] == change);
    }
    pool.performAging(true);
    REQUIRE(pool.getNumConflicts() == 0);
    // Retirement and slot reuse must not leave negative histogram counts.
    pool.setAgeLimit(2);
    pool.addConflictFromOtherPool(&change, 1);
    for (int age = 1; age <= 3; ++age) {
      pool.performAging();
      REQUIRE(pool.getNumConflicts() == (age <= 2 ? 1 : 0));
    }
  }
}

TEST_CASE("conflict age transitions match an independent lifecycle model",
          "[highs_robustness][highs_root_work]") {
  for (unsigned seed : {1u, 42u, 103u, 2026u}) {
    std::minstd_rand random(seed);
    HighsConflictPool pool(7, 1000);
    std::map<HighsInt, int> ages;
    HighsInt next_id = 0;
    int limit = 7;
    for (int step = 0; step < 200; ++step) {
      CAPTURE(seed, step, limit);
      const int operation = random() % 5;
      if (operation == 0 || ages.empty()) {
        const HighsDomainChange change{1.0, next_id, HighsBoundType::kLower};
        pool.addConflictFromOtherPool(&change, 1);
        ages[next_id++] = 0;
      } else if (operation == 1) {
        limit = random() % 10;
        pool.setAgeLimit(limit);
      } else if (operation == 2) {
        pool.performAging();
        for (auto it = ages.begin(); it != ages.end();) {
          if (++it->second > limit) it = ages.erase(it);
          else ++it;
        }
      } else {
        auto chosen = ages.begin();
        std::advance(chosen, random() % ages.size());
        const HighsInt id = chosen->first;
        const auto& ranges = pool.getConflictRanges();
        const auto& entries = pool.getConflictEntryVector();
        HighsInt index = 0;
        while (index < static_cast<HighsInt>(ranges.size()) &&
               (ranges[index].first < 0 || entries[ranges[index].first].column != id))
          ++index;
        REQUIRE(index < static_cast<HighsInt>(ranges.size()));
        if (operation == 3) {
          pool.resetAge(index);
          chosen->second = 0;
        } else {
          pool.removeConflict(index);
          ages.erase(chosen);
        }
      }
      REQUIRE(pool.getNumConflicts() == static_cast<HighsInt>(ages.size()));
      std::map<HighsInt, int> actual;
      for (const auto& range : pool.getConflictRanges()) {
        if (range.first < 0) {
          REQUIRE(range.second == -1);
          continue;
        }
        REQUIRE(range.second == range.first + 1);
        REQUIRE(range.second <= static_cast<HighsInt>(pool.getConflictEntryVector().size()));
        const auto& entry = pool.getConflictEntryVector()[range.first];
        REQUIRE(entry.boundval == 1.0);
        REQUIRE(entry.boundtype == HighsBoundType::kLower);
        REQUIRE(ages.count(entry.column) == 1);
        REQUIRE(actual.emplace(entry.column, 0).second);
      }
      REQUIRE(actual.size() == ages.size());
    }
  }
}

TEST_CASE("tree heuristic sub-MIPs are terminal and single threaded",
          "[highs_incumbent]") {
  HighsOptions options;
  options.mip_heuristic_run_rens = true;
  options.mip_heuristic_run_rins = true;
  options.mip_heuristic_run_root_reduced_cost = true;
  options.parallel = kHighsOnString;
  options.threads = 4;
  options.mip_detect_symmetry = true;
  options.mip_heuristic_effort = 0.8;

  HighsPrimalHeuristics::configureSubMipOptions(
      options, HighsSubMipOrigin::kTreeRins);

  REQUIRE_FALSE(options.mip_heuristic_run_rens);
  REQUIRE_FALSE(options.mip_heuristic_run_rins);
  REQUIRE_FALSE(options.mip_heuristic_run_root_reduced_cost);
  REQUIRE(options.parallel == kHighsOffString);
  REQUIRE(options.threads == 1);
  REQUIRE_FALSE(options.mip_detect_symmetry);
  REQUIRE(options.mip_heuristic_effort == 0.05);
}

TEST_CASE("nested heuristic sub-MIP launch is rejected",
          "[highs_incumbent]") {
  REQUIRE(HighsPrimalHeuristics::subMipLaunchAllowed(false, 0));
  REQUIRE_FALSE(HighsPrimalHeuristics::subMipLaunchAllowed(true, 1));
  REQUIRE_FALSE(HighsPrimalHeuristics::subMipLaunchAllowed(false, 1));
}

TEST_CASE("incumbent mailbox is bounded and preserves publication order",
          "[highs_incumbent]") {
  HighsIncumbentMailbox mailbox;
  for (size_t i = 0; i != HighsIncumbentMailbox::kCapacity; ++i) {
    HighsWorkerCandidate candidate;
    candidate.native_objective = static_cast<double>(i);
    candidate.metadata.publication_sequence = static_cast<int64_t>(i);
    REQUIRE(mailbox.publish(std::move(candidate)));
  }
  HighsWorkerCandidate overflow;
  REQUIRE_FALSE(mailbox.publish(std::move(overflow)));
  REQUIRE(mailbox.overflowed());

  std::vector<HighsWorkerCandidate> drained = mailbox.drain();
  REQUIRE(drained.size() == HighsIncumbentMailbox::kCapacity);
  for (size_t i = 0; i != drained.size(); ++i)
    REQUIRE(drained[i].metadata.publication_sequence ==
            static_cast<int64_t>(i));
  REQUIRE(mailbox.size() == 0);
}

TEST_CASE("peer witness verifies vectors and isolates generations",
          "[highs_incumbent]") {
  Highs model;
  model.setOptionValue("output_flag", false);
  REQUIRE(model.addVar(0, 10) == HighsStatus::kOk);
  REQUIRE(model.changeColCost(0, 1) == HighsStatus::kOk);
  REQUIRE(model.changeColIntegrality(0, HighsVarType::kInteger) == HighsStatus::kOk);
  HighsInt col = 0;
  double value = 1;
  REQUIRE(model.addRow(2, 10, 1, &col, &value) == HighsStatus::kOk);
  HighsLp lp = model.getLp();
  lp.ensureColwise();
  HighsPrimalSnapshotStore store(lp, lp, 1e-6, 9, 3);
  HighsWorkerCandidate packet;
  packet.solution = {4};
  packet.native_objective = -123456; // Never trusted as a pruning cutoff.
  packet.metadata.admission_generation = 9;
  packet.metadata.work_epoch = 3;
  REQUIRE(store.publish(packet, {4}));
  const auto retained = store.latest();
  REQUIRE(retained->upper_bound == 4);
  SECTION("stale generation") {
    packet.metadata.admission_generation = 8;
  }
  SECTION("stale mapping epoch") { packet.metadata.work_epoch = 2; }
  SECTION("missing reduced column") { packet.solution.clear(); }
  SECTION("nonfinite reduced column") {
    packet.solution = {std::numeric_limits<double>::quiet_NaN()};
  }
  SECTION("fractional integer") { packet.solution = {2.5}; }
  SECTION("violated row") { packet.solution = {1}; }
  SECTION("violated bound") { packet.solution = {11}; }
  SECTION("unsupported row orientation fails closed") {
    HighsLp rowwise = lp;
    rowwise.ensureRowwise();
    REQUIRE_FALSE(highsVerifyCandidate(rowwise, {3}, nullptr, 1e-6).feasible);
    packet.metadata.admission_generation = 8;
  }
  SECTION("invalid original witness") {
    packet.solution = {3};
    REQUIRE_FALSE(store.publish(packet, {0}));
    packet.metadata.admission_generation = 8;
  }
  SECTION("old readers retain immutable witness") {
    packet.solution = {3};
    REQUIRE(store.publish(packet, {3}));
    REQUIRE(store.latest()->upper_bound == 3);
    REQUIRE(retained->upper_bound == 4);
    REQUIRE(retained->candidate.solution == std::vector<double>{4});
    packet.metadata.admission_generation = 8;
  }
  REQUIRE_FALSE(store.publish(packet, {3}));
  REQUIRE(retained->original_columns == std::vector<double>{4});
}

TEST_CASE("peer witness handles maximization offset and concurrent publishers",
          "[highs_incumbent]") {
  Highs model;
  model.setOptionValue("output_flag", false);
  REQUIRE(model.addVar(0, 10) == HighsStatus::kOk);
  REQUIRE(model.changeColCost(0, 1) == HighsStatus::kOk);
  HighsLp original = model.getLp();
  original.sense_ = ObjSense::kMaximize;
  original.offset_ = 7;
  HighsLp reduced = original;
  reduced.sense_ = ObjSense::kMinimize;
  reduced.col_cost_[0] = -1;
  reduced.offset_ = -7;
  HighsPrimalSnapshotStore store(original, reduced, 1e-6, 9, 3);
  std::promise<void> release_slow_worker;
  auto wait = release_slow_worker.get_future();
  std::thread slow_worker([&]() { wait.wait(); });
  std::vector<std::thread> publishers;
  for (int i = 3; i <= 8; ++i)
    publishers.emplace_back([&, i]() {
      HighsWorkerCandidate packet;
      packet.solution = {static_cast<double>(i)};
      packet.metadata.admission_generation = 9;
      packet.metadata.work_epoch = 3;
      store.publish(packet, packet.solution);
    });
  for (auto& publisher : publishers) publisher.join();
  // Read while the unrelated worker is still deliberately blocked. No batch
  // join or master callback is necessary to make this checked witness visible.
  const auto snapshot = store.latest();
  release_slow_worker.set_value();
  slow_worker.join();
  REQUIRE(snapshot);
  REQUIRE(snapshot->upper_bound == -8);
  REQUIRE(snapshot->original_columns == std::vector<double>{8});
}

TEST_CASE("peer witnesses support implied columns without relaxing checks",
          "[highs_incumbent][implicit_integer_snapshot]") {
  Highs model;
  model.setOptionValue("output_flag", false);
  REQUIRE(model.addVar(0, 3) == HighsStatus::kOk);
  REQUIRE(model.addVar(0, 3) == HighsStatus::kOk);
  REQUIRE(model.changeColCost(0, 1) == HighsStatus::kOk);
  REQUIRE(model.changeColCost(1, 2) == HighsStatus::kOk);
  REQUIRE(model.changeColIntegrality(0, HighsVarType::kInteger) == HighsStatus::kOk);
  HighsInt indices[] = {0, 1};
  double values[] = {1, 1};
  REQUIRE(model.addRow(3, 3, 2, indices, values) == HighsStatus::kOk);
  HighsLp original = model.getLp();
  original.ensureColwise();
  original.offset_ = 7;
  HighsLp reduced = original;
  // x is declared integer; x+z=3 implies integer z at every feasible point.
  reduced.integrality_[1] = HighsVarType::kImplicitInteger;
  SECTION("presolve-only implied column") {}
  SECTION("original and reduced implied columns") {
    original.integrality_[1] = HighsVarType::kImplicitInteger;
  }
  HighsPrimalSnapshotStore store(original, reduced, 1e-6, 9, 3);
  HighsWorkerCandidate packet;
  packet.metadata.admission_generation = 9;
  packet.metadata.work_epoch = 3;
  packet.solution = {1, 2};
  packet.native_objective = -123456;
  REQUIRE(store.publish(packet, packet.solution));
  const auto retained = store.latest();
  REQUIRE(retained);
  REQUIRE(retained->upper_bound == 5);
  for (const std::vector<double>& invalid : std::vector<std::vector<double>>{
           {1}, {0.5, 2.5}, {2, 2}, {4, -1},
           {1, std::numeric_limits<double>::quiet_NaN()}}) {
    packet.solution = invalid;
    REQUIRE(store.publishChecked(packet, invalid) == HighsPrimalPublication::kInfeasible);
    REQUIRE(store.latest()->sequence == 1);
  }
  packet.solution = {3, 0};
  REQUIRE(store.publishChecked(packet, {1, 3}) == HighsPrimalPublication::kInfeasible);
  packet.metadata.work_epoch = 2;
  REQUIRE(store.publishChecked(packet, packet.solution) == HighsPrimalPublication::kStale);
  packet.metadata.work_epoch = 3;
  REQUIRE(store.publish(packet, packet.solution));
  REQUIRE(store.latest()->upper_bound == 3);
  REQUIRE(store.latest()->sequence == 2);
  REQUIRE(retained->upper_bound == 5);
}

TEST_CASE("weakly implied columns keep native candidate feasibility semantics",
          "[highs_incumbent][implicit_integer_snapshot]") {
  Highs model;
  model.setOptionValue("output_flag", false);
  for (int i = 0; i < 3; ++i) REQUIRE(model.addVar(0, 4) == HighsStatus::kOk);
  for (int i = 0; i < 2; ++i) {
    REQUIRE(model.changeColIntegrality(i, HighsVarType::kInteger) == HighsStatus::kOk);
    REQUIRE(model.changeColCost(i, 10) == HighsStatus::kOk);
  }
  REQUIRE(model.changeColCost(2, 1) == HighsStatus::kOk);
  HighsInt indices[] = {0, 1, 2};
  double first[] = {3, 2, 1};
  double second[] = {1, 3, -1};
  REQUIRE(model.addRow(-kHighsInf, 4, 3, indices, first) == HighsStatus::kOk);
  REQUIRE(model.addRow(-kHighsInf, 3, 3, indices, second) == HighsStatus::kOk);
  HighsLp original = model.getLp();
  original.ensureColwise();
  original.sense_ = ObjSense::kMaximize;
  original.offset_ = 7;
  HighsLp reduced = original;
  reduced.sense_ = ObjSense::kMinimize;
  reduced.offset_ = -7;
  for (double& cost : reduced.col_cost_) cost = -cost;
  reduced.integrality_[2] = HighsVarType::kImplicitInteger;
  // At fixed integer x,y an integer optimal z exists, but nonoptimal feasible
  // z can be fractional. Do not add an explicit integrality requirement for z.
  HighsPrimalSnapshotStore store(original, reduced, 1e-6, 9, 3);
  HighsWorkerCandidate packet;
  packet.metadata.admission_generation = 9;
  packet.metadata.work_epoch = 3;
  packet.solution = {0, 1, 0.5};
  REQUIRE(highsVerifyCandidate(original, packet.solution, nullptr, 1e-6).feasible);
  REQUIRE(store.publish(packet, packet.solution));
  REQUIRE(store.latest()->upper_bound == -10.5);
  packet.solution = {0, 1, 1.5};
  REQUIRE(store.publish(packet, packet.solution));
  REQUIRE(store.latest()->upper_bound == -11.5);
  packet.solution = {0.5, 1, 0.5};
  REQUIRE(store.publishChecked(packet, packet.solution) == HighsPrimalPublication::kInfeasible);
  packet.solution = {0, 1, 0.5};
  for (HighsVarType unsupported : {HighsVarType::kSemiContinuous,
                                  HighsVarType::kSemiInteger,
                                  static_cast<HighsVarType>(99)}) {
    HighsLp invalid_original = original;
    invalid_original.integrality_[2] = unsupported;
    HighsPrimalSnapshotStore original_guard(invalid_original, reduced, 1e-6, 9, 3);
    REQUIRE(original_guard.publishChecked(packet, packet.solution) == HighsPrimalPublication::kInfeasible);
    REQUIRE_FALSE(original_guard.latest());
    HighsLp invalid_reduced = reduced;
    invalid_reduced.integrality_[2] = unsupported;
    HighsPrimalSnapshotStore reduced_guard(original, invalid_reduced, 1e-6, 9, 3);
    REQUIRE(reduced_guard.publishChecked(packet, packet.solution) == HighsPrimalPublication::kInfeasible);
    REQUIRE_FALSE(reduced_guard.latest());
  }
}

TEST_CASE("peer repair seed changes only worker state before joined admission",
          "[highs_incumbent]") {
  Highs owner;
  owner.setOptionValue("output_flag", false);
  REQUIRE(owner.addVar(0, 10) == HighsStatus::kOk);
  REQUIRE(owner.changeColCost(0, 1) == HighsStatus::kOk);
  REQUIRE(owner.changeColIntegrality(0, HighsVarType::kInteger) == HighsStatus::kOk);
  SECTION("explicit integer") {}
  SECTION("implied integer") {
    REQUIRE(owner.changeColIntegrality(0, HighsVarType::kImplicitInteger) == HighsStatus::kOk);
  }
  HighsCallback callback(&owner);
  HighsSolution solution;
  HighsMipSolver mip(callback, owner.getOptions(), owner.getLp(), solution);
  mip.mipdata_.reset(new HighsMipSolverData(mip));
  auto& data = *mip.mipdata_;
  data.feastol = 1e-6;
  HighsMipWorker worker(mip, &data.getLp(), &data.getDomain(),
                        &data.getCutPool(), &data.getConflictPool(),
                        &data.getPseudoCost(), 0);
  auto store = std::make_shared<HighsPrimalSnapshotStore>(
      owner.getLp(), owner.getLp(), 1e-6, 9, 3);
  worker.attachPrimalStore(store);
  HighsWorkerCandidate packet;
  packet.solution = {4};
  packet.metadata.admission_generation = 9;
  packet.metadata.work_epoch = 3;
  packet.metadata.worker_id = 1;
  REQUIRE(store->publish(packet, {4}));
  worker.importPrimalSnapshot();
  REQUIRE(worker.upper_bound == 4);
  REQUIRE(worker.peerImports() == 1);
  REQUIRE(worker.repairIncumbent() == std::vector<double>{4});
  REQUIRE(data.incumbent.empty());
  REQUIRE(data.upper_bound == kHighsInf);
  REQUIRE(data.lower_bound == -kHighsInf);
  worker.importPrimalSnapshot();
  REQUIRE(worker.peerImports() == 1);
  worker.attachPrimalStore(nullptr);
  REQUIRE(worker.repairIncumbent().empty());
}

TEST_CASE("checked worker publication wakes checkpoints with peer exchange",
          "[highs_incumbent][highs_node_lp_effort][checkpoint_exchange]") {
  const bool exchange = GENERATE(false, true);
  const bool enabled = GENERATE(false, true);
  const bool maximize = GENERATE(false, true);
  CAPTURE(exchange, enabled, maximize);
  Highs::resetGlobalScheduler(true);
  highs::parallel::initialize_scheduler(1);
  Highs owner;
  REQUIRE(owner.setOptionValue("output_flag", false) == HighsStatus::kOk);
  REQUIRE(owner.addVar(0, 10) == HighsStatus::kOk);
  REQUIRE(owner.changeColCost(0, 1) == HighsStatus::kOk);
  REQUIRE(owner.changeColIntegrality(0, HighsVarType::kInteger) == HighsStatus::kOk);
  REQUIRE(owner.changeObjectiveOffset(7) == HighsStatus::kOk);
  REQUIRE(owner.changeObjectiveSense(maximize ? ObjSense::kMaximize
                                             : ObjSense::kMinimize) == HighsStatus::kOk);
  HighsCallback callback(&owner);
  callback.data_in.admission_generation = 9;
  HighsSolution solution;
  HighsMipSolver mip(callback, owner.getOptions(), owner.getLp(), solution);
  mip.modelstatus_ = HighsModelStatus::kNotset;
  mip.improving_solution_file_ = nullptr;
  mip.timer_.start();
  mip.mipdata_.reset(new HighsMipSolverData(mip));
  auto& data = *mip.mipdata_;
  data.init();
  data.presolvedModel = owner.getLp();
  if (maximize) {
    data.presolvedModel.sense_ = ObjSense::kMinimize;
    data.presolvedModel.col_cost_[0] = -1;
    data.presolvedModel.offset_ = -7;
  }
  mip.setModel(data.presolvedModel);
  data.runSetup();
  data.workers.emplace_back(mip, &data.getLp(), &data.getDomain(),
                            &data.getCutPool(), &data.getConflictPool(),
                            &data.getPseudoCost(), 0);
  auto& worker = data.workers.front();
  worker.setWorkEpoch(3);
  HighsCandidateCheckpoint checkpoint;
  worker.search_work.begin(0, false);
  worker.search_work.attachCheckpoint(enabled ? &checkpoint : nullptr,
                                      checkpoint.sequence());
  const auto store = exchange ? std::make_shared<HighsPrimalSnapshotStore>(
      owner.getLp(), *mip.model_, data.feastol, 9, 3) : nullptr;
  worker.attachPrimalStore(store);
  const double sign = maximize ? -1 : 1;
  const double first = maximize ? 6 : 4;
  REQUIRE(data.addIncumbent({first}, sign * first, kSolutionSourceBranching));
  // A master seed is already globally admitted, not a new worker discovery.
  REQUIRE(checkpoint.sequence() == 0);
  worker.upper_bound = data.upper_bound;
  worker.upper_limit = data.upper_limit;
  worker.optimality_limit = data.optimality_limit;
  REQUIRE(worker.trySolution({first}, kSolutionSourceRins));
  REQUIRE_FALSE(worker.trySolution({11}, kSolutionSourceRins));
  REQUIRE_FALSE(worker.trySolution({1.5}, kSolutionSourceRins));
  REQUIRE_FALSE(worker.trySolution({std::numeric_limits<double>::quiet_NaN()},
                                   kSolutionSourceRins));
  REQUIRE(checkpoint.sequence() == 0);
  const double better = maximize ? 8 : 2;
  REQUIRE(worker.trySolution({better}, kSolutionSourceRins));
  REQUIRE(checkpoint.sequence() == (enabled ? 1 : 0));
  REQUIRE(worker.search_work.checkpointRequested() == enabled);
  REQUIRE(worker.incumbent_mailbox_.size() == 1);
  REQUIRE(data.upper_bound == sign * first);
  REQUIRE(data.incumbent == std::vector<double>{first});
  REQUIRE(data.lower_bound == -kHighsInf);
  if (store) {
    REQUIRE(store->latest()->upper_bound == sign * better);
    REQUIRE(store->latest()->original_columns == std::vector<double>{better});
    REQUIRE(highsVerifyCandidate(owner.getLp(), store->latest()->original_columns,
                                 nullptr, data.feastol).feasible);
  }
  REQUIRE(worker.trySolution({better}, kSolutionSourceRins));
  REQUIRE(checkpoint.sequence() == (enabled ? 1 : 0));
  worker.search_work.finish();
  worker.search_work.begin(0, false);
  worker.search_work.attachCheckpoint(enabled ? &checkpoint : nullptr,
                                      checkpoint.sequence());
  REQUIRE_FALSE(worker.search_work.checkpointRequested());
  const double best = maximize ? 10 : 0;
  REQUIRE(worker.trySolution({best}, kSolutionSourceRins));
  REQUIRE(checkpoint.sequence() == (enabled ? 2 : 0));
  REQUIRE(data.upper_bound == sign * first);
  REQUIRE(data.lower_bound == -kHighsInf);
  worker.search_work.finish();
  worker.attachPrimalStore(nullptr);
}

TEST_CASE("stale or superseded peer publications do not wake checkpoints",
          "[highs_incumbent][highs_node_lp_effort][checkpoint_exchange]") {
  Highs::resetGlobalScheduler(true);
  highs::parallel::initialize_scheduler(1);
  Highs owner;
  REQUIRE(owner.setOptionValue("output_flag", false) == HighsStatus::kOk);
  REQUIRE(owner.addVar(0, 10) == HighsStatus::kOk);
  REQUIRE(owner.changeColCost(0, 1) == HighsStatus::kOk);
  REQUIRE(owner.changeColIntegrality(0, HighsVarType::kInteger) == HighsStatus::kOk);
  HighsCallback callback(&owner);
  callback.data_in.admission_generation = 9;
  HighsSolution solution;
  HighsMipSolver mip(callback, owner.getOptions(), owner.getLp(), solution);
  mip.modelstatus_ = HighsModelStatus::kNotset;
  mip.improving_solution_file_ = nullptr;
  mip.timer_.start();
  mip.mipdata_.reset(new HighsMipSolverData(mip));
  auto& data = *mip.mipdata_;
  data.init();
  data.presolvedModel = owner.getLp();
  mip.setModel(data.presolvedModel);
  data.runSetup();
  HighsMipWorker worker(mip, &data.getLp(), &data.getDomain(),
                        &data.getCutPool(), &data.getConflictPool(),
                        &data.getPseudoCost(), 0);
  worker.setWorkEpoch(3);
  auto store = std::make_shared<HighsPrimalSnapshotStore>(
      owner.getLp(), owner.getLp(), data.feastol, 9, 3);
  worker.attachPrimalStore(store);
  HighsCandidateCheckpoint checkpoint;
  worker.search_work.begin(0, false);
  worker.search_work.attachCheckpoint(&checkpoint, checkpoint.sequence());
  SECTION("stale generation") { callback.data_in.admission_generation = 10; }
  SECTION("stale epoch") { worker.setWorkEpoch(4); }
  SECTION("already superseded by a checked peer") {
    HighsWorkerCandidate peer;
    peer.solution = {1};
    peer.metadata.admission_generation = 9;
    peer.metadata.work_epoch = 3;
    peer.metadata.worker_id = 1;
    REQUIRE(store->publish(peer, peer.solution));
    // Publication races with a worker that has not imported the peer yet.
    REQUIRE(worker.upper_bound == kHighsInf);
  }
  REQUIRE(worker.trySolution({2}, kSolutionSourceRins));
  REQUIRE(checkpoint.sequence() == 0);
  REQUIRE_FALSE(worker.search_work.checkpointRequested());
  REQUIRE(data.upper_bound == kHighsInf);
  REQUIRE(data.lower_bound == -kHighsInf);
  worker.search_work.finish();
  worker.attachPrimalStore(nullptr);
}

TEST_CASE("serial master admission seeds an already running private service",
          "[highs_incumbent][serial_primal_handoff]") {
  const bool maximize = GENERATE(false, true);
  const bool implied = GENERATE(false, true);
  const bool use_try_solution = GENERATE(false, true);
  CAPTURE(maximize, implied, use_try_solution);
  highs::parallel::initialize_scheduler(1);
  Highs owner;
  REQUIRE(owner.setOptionValue("output_flag", false) == HighsStatus::kOk);
  REQUIRE(owner.addVar(0, 10) == HighsStatus::kOk);
  REQUIRE(owner.changeColCost(0, 1) == HighsStatus::kOk);
  REQUIRE(owner.changeColIntegrality(0, implied ? HighsVarType::kImplicitInteger
                                              : HighsVarType::kInteger) == HighsStatus::kOk);
  REQUIRE(owner.changeObjectiveOffset(7) == HighsStatus::kOk);
  REQUIRE(owner.changeObjectiveSense(maximize ? ObjSense::kMaximize
                                             : ObjSense::kMinimize) == HighsStatus::kOk);
  HighsCallback callback(&owner);
  callback.data_in.admission_generation = 9;
  HighsSolution solution;
  HighsMipSolver mip(callback, owner.getOptions(), owner.getLp(), solution);
  mip.modelstatus_ = HighsModelStatus::kNotset;
  mip.improving_solution_file_ = nullptr;
  mip.timer_.start();
  mip.mipdata_.reset(new HighsMipSolverData(mip));
  auto& data = *mip.mipdata_;
  data.init();
  data.presolvedModel = owner.getLp();
  if (maximize) {
    data.presolvedModel.sense_ = ObjSense::kMinimize;
    data.presolvedModel.col_cost_[0] = -1;
    data.presolvedModel.offset_ = -7;
  }
  mip.setModel(data.presolvedModel);
  data.runSetup();
  data.workers.emplace_back(mip, &data.getLp(), &data.getDomain(),
                            &data.getCutPool(), &data.getConflictPool(),
                            &data.getPseudoCost(), 0);
  auto& worker = data.workers.front();
  worker.setWorkEpoch(3);
  auto store = std::make_shared<HighsPrimalSnapshotStore>(
      owner.getLp(), *mip.model_, data.feastol, 9, 3);
  worker.attachPrimalStore(store);
  struct Context {
    int notifications = 0, queries = 0;
    std::vector<double> witness;
    double objective = kHighsInf;
  } context;
  HighsPrimalService service(store, owner.getLp(), *mip.model_, owner.getOptions(),
      data.postSolveStack, callback.data_in,
      [](int type, const char*, const HighsCallbackDataOut* out,
         HighsCallbackDataIn*, void* raw) {
        auto& context = *static_cast<Context*>(raw);
        if (type == kCallbackMipImprovingSolution) ++context.notifications;
        if (type == kCallbackMipUserSolution) ++context.queries;
        context.witness.assign(out->mip_solution, out->mip_solution + 1);
        context.objective = out->mip_primal_bound;
      }, &context, data.lower_bound, 3, false);
  worker.attachPrimalService(&service);
  service.start();
  worker.residentLpBoundary();
  REQUIRE(context.queries == 0);
  REQUIRE_FALSE(store->latest());
  const double first = maximize ? 6 : 4;
  const double sign = maximize ? -1 : 1;
  const auto admit = [&](double value) {
    return use_try_solution
        ? data.heuristics.trySolution({value}, kSolutionSourceRens, worker)
        : data.addIncumbent({value}, sign * value, kSolutionSourceBranching);
  };
  REQUIRE(admit(first));
  REQUIRE(data.incumbent == std::vector<double>{first});
  REQUIRE(mip.solution_ == std::vector<double>{first});
  REQUIRE(data.upper_bound == sign * first);
  const auto retained = store->latest();
  REQUIRE(retained);
  REQUIRE(retained->original_columns == mip.solution_);
  REQUIRE(retained->candidate.solution == data.incumbent);
  REQUIRE(retained->candidate.metadata.admission_generation == 9);
  REQUIRE(retained->candidate.metadata.work_epoch == 3);
  REQUIRE(retained->upper_bound == data.upper_bound);
  REQUIRE(worker.incumbent_mailbox_.size() == 0);
  worker.residentLpBoundary();
  REQUIRE(context.notifications == 1);
  REQUIRE(context.queries == 1);
  REQUIRE(context.witness == mip.solution_);
  REQUIRE(context.objective == first + 7);
  REQUIRE(data.lower_bound == -kHighsInf);
  REQUIRE(admit(5));
  REQUIRE(store->latest()->sequence == retained->sequence);
  REQUIRE_FALSE(data.trySolution({11}, kSolutionSourceHeuristic));
  REQUIRE(store->latest()->sequence == retained->sequence);
  const double better = maximize ? 8 : 2;
  REQUIRE(admit(better));
  REQUIRE(store->latest()->sequence == retained->sequence + 1);
  REQUIRE(retained->original_columns == std::vector<double>{first});
  worker.residentLpBoundary();
  REQUIRE(context.notifications == 2);
  REQUIRE(context.witness == std::vector<double>{better});
  REQUIRE(context.objective == better + 7);
  service.stop();
  worker.attachPrimalService(nullptr);
  worker.attachPrimalStore(nullptr);
  REQUIRE_FALSE(service.failed());
}

TEST_CASE("root completion without analytic centre uses retained root points",
          "[highs_root_work][root_cold_start]") {
  Highs::resetGlobalScheduler(true);
  highs::parallel::initialize_scheduler(1);
  Highs owner;
  owner.setOptionValue("output_flag", false);
  owner.addVar(0, 1);
  owner.changeColCost(0, 7);
  owner.changeColIntegrality(0, HighsVarType::kInteger);
  HighsCallback callback(&owner);
  HighsSolution solution;
  HighsMipSolver mip(callback, owner.getOptions(), owner.getLp(), solution);
  mip.modelstatus_ = HighsModelStatus::kNotset;
  mip.improving_solution_file_ = nullptr;
  mip.timer_.start();
  mip.mipdata_.reset(new HighsMipSolverData(mip));
  auto& data = *mip.mipdata_;
  data.init();
  data.presolvedModel = owner.getLp();
  mip.setModel(data.presolvedModel);
  data.runSetup();
  data.workers.emplace_back(mip, &data.getLp(), &data.getDomain(),
                            &data.getCutPool(), &data.getConflictPool(),
                            &data.getPseudoCost(), 0);
  REQUIRE(data.analyticCenter.empty());
  REQUIRE(data.incumbent.empty());
  SECTION("two retained points") {
    data.firstlpsol = {0.25};
    data.rootlpsol = {0.5};
  }
  SECTION("only first point") { data.firstlpsol = {0.25}; }
  SECTION("only root point") { data.rootlpsol = {0.5}; }
  data.heuristics.centralRounding(data.workers.front());
  REQUIRE(data.incumbent.size() == 1);
  REQUIRE(highsVerifyCandidate(owner.getLp(), data.incumbent, nullptr, 1e-6).feasible);
  REQUIRE(data.lower_bound == -kHighsInf);
}

TEST_CASE("serial master handoff unlocks background repair before batch join",
          "[highs_incumbent][serial_primal_handoff]") {
  highs::parallel::initialize_scheduler(1);
  Highs owner;
  REQUIRE(owner.setOptionValue("output_flag", false) == HighsStatus::kOk);
  REQUIRE(owner.addVar(2, 10) == HighsStatus::kOk);
  REQUIRE(owner.changeColCost(0, 1) == HighsStatus::kOk);
  REQUIRE(owner.changeColIntegrality(0, HighsVarType::kInteger) == HighsStatus::kOk);
  HighsCallback callback(&owner);
  callback.data_in.admission_generation = 9;
  callback.data_in.admission.generation = 9;
  HighsSolution solution;
  HighsMipSolver mip(callback, owner.getOptions(), owner.getLp(), solution);
  mip.modelstatus_ = HighsModelStatus::kNotset;
  mip.improving_solution_file_ = nullptr;
  mip.timer_.start();
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
  worker.setWorkEpoch(3);
  auto store = std::make_shared<HighsPrimalSnapshotStore>(
      owner.getLp(), *mip.model_, data.feastol, 9, 3);
  worker.attachPrimalStore(store);
  struct Context {
    int calls = 0, receipt = 0;
    bool invalid = false;
    std::promise<void> repaired;
  } context;
  auto repaired = context.repaired.get_future();
  HighsPrimalService service(store, owner.getLp(), *mip.model_, owner.getOptions(),
      data.postSolveStack, callback.data_in,
      [](int type, const char*, const HighsCallbackDataOut*,
         HighsCallbackDataIn* in, void* raw) {
        if (type != kCallbackMipUserSolution) return;
        auto& context = *static_cast<Context*>(raw);
        HighsCallbackSolutionOutcome receipt{};
        if (Highs_getCallbackSolutionOutcome(in, 1, &receipt) != 0)
          context.invalid = true;
        if (++context.calls == 1) {
          const double value = 2;
          if (Highs_setCallbackSolution(in, 1, &value) != 0 ||
              Highs_tagCallbackSolution(in, 1, receipt.generation) != 0)
            context.invalid = true;
        } else if (context.calls == 2) {
          context.receipt = receipt.status;
          context.repaired.set_value();
        }
      }, &context, data.lower_bound, 3, true);
  service.start();
  const bool admitted = data.heuristics.trySolution({4}, kSolutionSourceRens, worker);
  const auto ready = repaired.wait_for(std::chrono::seconds(5));
  service.stop();
  REQUIRE(admitted);
  REQUIRE(ready == std::future_status::ready);
  REQUIRE_FALSE(context.invalid);
  REQUIRE_FALSE(service.failed());
  REQUIRE(context.receipt == kHighsAdmissionImproved);
  REQUIRE(store->latest()->upper_bound == 2);
  REQUIRE(store->latest()->original_columns == std::vector<double>{2});
  // Candidate-only authority may improve, but global proof remains master-owned.
  REQUIRE(data.upper_bound == 4);
  REQUIRE(data.incumbent == std::vector<double>{4});
  REQUIRE(data.lower_bound == -kHighsInf);
  REQUIRE(worker.incumbent_mailbox_.size() == 0);
  const auto latest = store->latest();
  REQUIRE(data.addIncumbent(latest->candidate.solution, latest->upper_bound,
                            kSolutionSourceUserSolution));
  REQUIRE(data.upper_bound == 2);
  REQUIRE(mip.solution_ == std::vector<double>{2});
  REQUIRE(store->latest()->sequence == latest->sequence);
  worker.attachPrimalStore(nullptr);
}

TEST_CASE("master handoff publishes repaired continuous columns",
          "[highs_incumbent][serial_primal_handoff]") {
  highs::parallel::initialize_scheduler(1);
  Highs owner;
  REQUIRE(owner.setOptionValue("output_flag", false) == HighsStatus::kOk);
  REQUIRE(owner.addVar(0, 10) == HighsStatus::kOk);
  REQUIRE(owner.addVar(0, 10) == HighsStatus::kOk);
  REQUIRE(owner.changeColCost(0, 1) == HighsStatus::kOk);
  REQUIRE(owner.changeColCost(1, 1) == HighsStatus::kOk);
  REQUIRE(owner.changeColIntegrality(0, HighsVarType::kInteger) == HighsStatus::kOk);
  const HighsInt indices[] = {0, 1};
  const double coefficients[] = {1, -1};
  REQUIRE(owner.addRow(-kHighsInf, 0, 2, indices, coefficients) == HighsStatus::kOk);
  HighsLp original = owner.getLp();
  original.ensureColwise();
  HighsCallback callback(&owner);
  callback.data_in.admission_generation = 9;
  HighsSolution solution;
  HighsMipSolver mip(callback, owner.getOptions(), original, solution);
  mip.modelstatus_ = HighsModelStatus::kNotset;
  mip.improving_solution_file_ = nullptr;
  mip.timer_.start();
  HighsProfiling profiling;
  profiling.initialize(mip.timer_, false);
  mip.setProfiling(&profiling);
  mip.mipdata_.reset(new HighsMipSolverData(mip));
  auto& data = *mip.mipdata_;
  data.init();
  data.presolvedModel = original;
  mip.setModel(data.presolvedModel);
  data.runSetup();
  data.workers.emplace_back(mip, &data.getLp(), &data.getDomain(),
                            &data.getCutPool(), &data.getConflictPool(),
                            &data.getPseudoCost(), 0);
  auto& worker = data.workers.front();
  worker.setWorkEpoch(3);
  auto store = std::make_shared<HighsPrimalSnapshotStore>(
      original, *mip.model_, data.feastol, 9, 3);
  worker.attachPrimalStore(store);
  REQUIRE(data.addIncumbent({4, 3}, 7, kSolutionSourceRens));
  REQUIRE(data.total_repair_lp_feasible == 1);
  REQUIRE(mip.solution_ == std::vector<double>{4, 4});
  REQUIRE(data.upper_bound == 8);
  const auto snapshot = store->latest();
  REQUIRE(snapshot);
  REQUIRE(snapshot->original_columns == mip.solution_);
  REQUIRE(snapshot->candidate.solution == std::vector<double>{4, 4});
  REQUIRE(snapshot->upper_bound == 8);
  REQUIRE(highsVerifyCandidate(*mip.model_, snapshot->candidate.solution,
                               nullptr, data.feastol).feasible);
  worker.attachPrimalStore(nullptr);
}

TEST_CASE("peer snapshot version polling preserves every publication order",
          "[highs_incumbent]") {
  Highs model;
  model.setOptionValue("output_flag", false);
  REQUIRE(model.addVar(0, 10) == HighsStatus::kOk);
  REQUIRE(model.changeColCost(0, 1) == HighsStatus::kOk);
  const auto lp = model.getLp();
  std::vector<int> order{3, 5, 7};
  do {
    HighsPrimalSnapshotStore store(lp, lp, 1e-6, 9, 3);
    REQUIRE_FALSE(store.latestAfter(0));
    int64_t seen = 0;
    double best = kHighsInf;
    std::shared_ptr<const HighsPrimalSnapshot> retained;
    for (int value : order) {
      HighsWorkerCandidate candidate;
      candidate.solution = {static_cast<double>(value)};
      candidate.metadata.admission_generation = 9;
      candidate.metadata.work_epoch = 3;
      bool improved = false;
      std::thread publisher([&]() { improved = store.publish(candidate, candidate.solution); });
      publisher.join();
      auto next = store.latestAfter(seen);
      REQUIRE(improved == (value < best));
      REQUIRE(static_cast<bool>(next) == improved);
      if (improved) {
        if (retained) REQUIRE(retained->upper_bound == best);
        best = value;
        REQUIRE(next->sequence == seen + 1);
        REQUIRE(next->upper_bound == best);
        seen = next->sequence;
        retained = next;
      }
      for (int poll = 0; poll < 100; ++poll) REQUIRE_FALSE(store.latestAfter(seen));
      candidate.metadata.work_epoch = 2;
      REQUIRE_FALSE(store.publish(candidate, candidate.solution));
      REQUIRE_FALSE(store.latestAfter(seen));
    }
    REQUIRE(retained->upper_bound == 3);
  } while (std::next_permutation(order.begin(), order.end()));
}

TEST_CASE("peer snapshot idle polling microbenchmark",
          "[.highs_snapshot_poll_benchmark]") {
  Highs model;
  model.setOptionValue("output_flag", false);
  REQUIRE(model.addVar(0, 1) == HighsStatus::kOk);
  const auto lp = model.getLp();
  HighsPrimalSnapshotStore store(lp, lp, 1e-6, 1, 1);
  HighsWorkerCandidate seed;
  seed.solution = {0};
  seed.metadata.admission_generation = seed.metadata.work_epoch = 1;
  REQUIRE(store.publish(seed, seed.solution));
  constexpr int polls = 100000;
  for (int repeat = 0; repeat < 6; ++repeat) {
    for (int arm = 0; arm < 2; ++arm) {
      const bool versioned = (repeat + arm) % 2 != 0;
      std::promise<void> start;
      auto ready = start.get_future().share();
      std::vector<std::thread> readers;
      std::array<int, 4> observed{};
      for (int worker = 0; worker < 4; ++worker)
        readers.emplace_back([&, worker]() {
          ready.wait();
          for (int poll = 0; poll < polls; ++poll) {
            auto snapshot = versioned ? store.latestAfter(1) : store.latest();
            observed[worker] += static_cast<bool>(snapshot);
          }
        });
      const auto began = std::chrono::steady_clock::now();
      start.set_value();
      for (auto& reader : readers) reader.join();
      const double seconds = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - began).count();
      for (int count : observed) REQUIRE(count == (versioned ? 0 : polls));
      std::printf("SnapshotPoll repeat=%d mode=%s threads=4 polls=%d seconds=%.9f\n",
                  repeat, versioned ? "versioned" : "locked", 4 * polls, seconds);
    }
  }
}

TEST_CASE("native candidate paths reject nonfinite values before admission",
          "[highs_incumbent]") {
  Highs owner;
  owner.setOptionValue("output_flag", false);
  REQUIRE(owner.addVar(-kHighsInf, kHighsInf) == HighsStatus::kOk);
  REQUIRE(owner.changeColCost(0, 1) == HighsStatus::kOk);
  REQUIRE(owner.changeColIntegrality(0, HighsVarType::kInteger) == HighsStatus::kOk);
  HighsCallback callback(&owner);
  HighsSolution solution;
  HighsMipSolver mip(callback, owner.getOptions(), owner.getLp(), solution);
  mip.mipdata_.reset(new HighsMipSolverData(mip));
  auto& data = *mip.mipdata_;
  data.feastol = 1e-6;
  HighsMipWorker worker(mip, &data.getLp(), &data.getDomain(),
                        &data.getCutPool(), &data.getConflictPool(),
                        &data.getPseudoCost(), 0);
  for (double value : {std::numeric_limits<double>::quiet_NaN(), kHighsInf,
                       -kHighsInf}) {
    CHECK_FALSE(worker.trySolution({value}, kSolutionSourceUserSolution));
    CHECK_FALSE(data.trySolution({value}, kSolutionSourceUserSolution));
    CHECK_FALSE(data.solutionRowFeasible({value}));
    CHECK_FALSE(worker.addIncumbent({value}, 0, kSolutionSourceUserSolution));
    CHECK_FALSE(data.addIncumbent({value}, 0, kSolutionSourceUserSolution));
    CHECK_FALSE(worker.addIncumbent({0}, value, kSolutionSourceUserSolution));
    CHECK_FALSE(data.addIncumbent({0}, value, kSolutionSourceUserSolution));
    CHECK(data.incumbent.empty());
    CHECK(worker.incumbent_mailbox_.size() == 0);
    CHECK(data.upper_bound == kHighsInf);
    CHECK(worker.upper_bound == kHighsInf);
  }
  CHECK_FALSE(worker.addIncumbent({}, 0, kSolutionSourceUserSolution));
  CHECK_FALSE(data.addIncumbent({}, 0, kSolutionSourceUserSolution));
  CHECK_FALSE(data.solutionRowFeasible({}));
}

TEST_CASE("postsolve readers own cursors without copying reduction bytes",
          "[highs_incumbent]") {
  HighsDataStack bytes;
  const std::vector<double> payload{1, 2, 3};
  bytes.push(42);
  bytes.push(payload);
  bytes.push(std::vector<double>{});
  auto first = bytes.reader(), second = bytes.reader();
  static_assert(std::is_trivially_copy_constructible<HighsDataStack::Reader>::value,
                "a reader must not own a copied byte vector");
  std::vector<double> values{9};
  first.pop(values);
  REQUIRE(values.empty());
  first.pop(values);
  REQUIRE(values == payload);
  // Even the legacy cursor is independent of the two borrowed readers.
  bytes.resetPosition();
  bytes.pop(values);
  REQUIRE(values.empty());
  second.pop(values);
  REQUIRE(values.empty());
  second.pop(values);
  REQUIRE(values == payload);
  int value = 0;
  first.pop(value);
  REQUIRE(value == 42);
  second.pop(value);
  REQUIRE(value == 42);
  bytes.pop(values);
  REQUIRE(values == payload);

  presolve::HighsPostsolveStack mapping;
  mapping.initializeIndexMaps(1, 2);
  mapping.linearTransform(0, 2, 3);
  const HighsInt row = 0;
  const double coefficient = 3;
  mapping.fixedColAtLower(1, 5, 7,
      HighsMatrixSlice<HighsCompressedSlice>(&row, &coefficient, 1));
  mapping.compressIndexMaps({0}, {0, -1});
  HighsOptions options;
  auto check = [&](bool thread_safe) {
    for (int repeat = 0; repeat < 64; ++repeat) {
      HighsSolution solution;
      solution.value_valid = solution.dual_valid = true;
      solution.col_value = {4};
      solution.row_value = {0};
      solution.col_dual = {6};
      solution.row_dual = {2};
      HighsBasis basis;
      basis.valid = true;
      basis.col_status = {HighsBasisStatus::kBasic};
      basis.row_status = {HighsBasisStatus::kLower};
      mapping.undo(options, solution, basis, -1, thread_safe);
      if (solution.col_value != std::vector<double>{11, 5} ||
          solution.col_dual != std::vector<double>{3, 1} ||
          basis.col_status != std::vector<HighsBasisStatus>{
              HighsBasisStatus::kBasic, HighsBasisStatus::kLower}) return false;
      solution.col_value = {4};
      mapping.undoPrimal(options, solution, -1, thread_safe);
      if (solution.col_value != std::vector<double>{11, 5}) return false;
    }
    return true;
  };
  REQUIRE(check(false));
  std::vector<std::future<bool>> readers;
  for (int worker = 0; worker < 4; ++worker)
    readers.push_back(std::async(std::launch::async, [&] { return check(true); }));
  for (auto& reader : readers) REQUIRE(reader.get());
  REQUIRE(mapping.getReducedPrimalSolution({11, 5}) == std::vector<double>{4});
}

TEST_CASE("opportunistic exchange preserves a known MIP optimum",
          "[highs_incumbent]") {
  for (int configuration : {0, 1, 2}) {
    const int threads = configuration == 0 ? 1 : 4;
    const bool parallel = configuration != 2;
    Highs highs;
    highs.resetGlobalScheduler(true);
    REQUIRE(highs.setOptionValue("log_to_console", false) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("threads", threads) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("parallel", parallel ? "on" : "off") == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_incumbent_exchange", "opportunistic") == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_candidate_checkpoint", true) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_incumbent_epoch_nodes", 4) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_rel_gap", 0.0) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_abs_gap", 0.0) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("time_limit", 30.0) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("presolve", "off") == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_root_work_budget", true) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_root_work_max_separation_rounds", 0) == HighsStatus::kOk);
    REQUIRE(highs.getOptions().mip_root_work_budget);
    REQUIRE(highs.getOptions().mip_root_work_max_separation_rounds == 0);
    struct CallbackContext {
      const std::thread::id owner = std::this_thread::get_id();
      std::atomic<int> calls{0}, background_calls{0};
    } context;
    std::map<std::string, int64_t> resident;
    highs.setCallback([&](int kind, const std::string& message,
                          const HighsCallbackOutput*, HighsCallbackInput*, void*) {
      if (kind != kCallbackLogging || message.find("MIP-PrimalService:") != 0) return;
      std::istringstream tokens(message);
      std::string token;
      while (tokens >> token) {
        const auto separator = token.find('=');
        if (token.compare(0, 9, "resident_") == 0 && separator != std::string::npos)
          resident[token.substr(0, separator)] = std::stoll(token.substr(separator + 1));
      }
    });
    highs.startCallback(kCallbackLogging);
    highs.setPrimalServiceCallback([](int, const char*, const HighsCallbackDataOut*,
                                     HighsCallbackDataIn*, void* raw) {
      auto& context = *static_cast<CallbackContext*>(raw);
      ++context.calls;
      if (std::this_thread::get_id() != context.owner) ++context.background_calls;
    }, &context);
    REQUIRE(highs.readModel(std::string(HIGHS_DIR) + "/check/instances/rgn.mps")
            == HighsStatus::kOk);
    REQUIRE(highs.run() != HighsStatus::kError);
    REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
    REQUIRE(highs.getObjectiveValue() == Approx(82.19999924).margin(1e-5));
    REQUIRE(highsVerifyCandidate(highs.getLp(), highs.getSolution().col_value,
                                 nullptr, 1e-6).feasible);
    REQUIRE(context.calls.load() > 0);
    if (!parallel || threads == 1) CHECK(context.background_calls.load() == 0);
    REQUIRE(resident["resident_attempts"] > 0);
    REQUIRE(resident["resident_completed"] > 0);
    REQUIRE(resident["resident_lp_iterations"] > 0);
    REQUIRE(resident["resident_lp_max"] <= resident["resident_lp_iterations"]);
    REQUIRE(resident["resident_attempts"] == resident["resident_completed"] +
        resident["resident_yielded"] + resident["resident_limit_stops"] +
        resident["resident_failures"]);
    highs.resetGlobalScheduler(true);
  }
}

TEST_CASE("opportunistic exchange rejects dynamic lazy constraints",
          "[highs_incumbent]") {
  Highs model;
  model.setOptionValue("output_flag", false);
  model.setOptionValue("mip_incumbent_exchange", "opportunistic");
  REQUIRE(model.addVar(0, 1) == HighsStatus::kOk);
  REQUIRE(model.changeColIntegrality(0, HighsVarType::kInteger) == HighsStatus::kOk);
  model.setCallback([](int, const std::string&, const HighsCallbackOutput*,
                        HighsCallbackInput*, void*) {});
  model.startCallback(kCallbackMipDefineLazyConstraints);
  REQUIRE(model.run() == HighsStatus::kError);
  REQUIRE(model.getModelStatus() == HighsModelStatus::kModelError);
}

TEST_CASE("private primal owner admits and repairs while proof worker is blocked",
          "[highs_incumbent][implicit_integer_snapshot]") {
  Highs model;
  model.setOptionValue("output_flag", false);
  model.setOptionValue("threads", 1);
  REQUIRE(model.addVar(2, 10) == HighsStatus::kOk);
  REQUIRE(model.changeColCost(0, 1) == HighsStatus::kOk);
  REQUIRE(model.changeColIntegrality(0, HighsVarType::kInteger) == HighsStatus::kOk);
  auto lp = model.getLp();
  lp.ensureColwise();
  HighsLp reduced = lp;
  SECTION("explicit reduced column") {}
  SECTION("presolve implied reduced column") {
    reduced.integrality_[0] = HighsVarType::kImplicitInteger;
  }
  presolve::HighsPostsolveStack mapping;
  mapping.initializeIndexMaps(0, 1);
  HighsCallbackInput input;
  input.admission_generation = 9;
  input.admission.generation = 9;
  auto store = std::make_shared<HighsPrimalSnapshotStore>(lp, reduced, 1e-6, 9, 3);
  struct Context {
    std::atomic<int> calls{0}, receipt{0};
    std::atomic<bool> invalid{false};
    std::promise<void> repaired;
  } context;
  auto repaired = context.repaired.get_future();
  auto callback = [](int type, const char*, const HighsCallbackDataOut*,
                     HighsCallbackDataIn* in, void* raw) {
    if (type != kCallbackMipUserSolution) return;
    auto& ctx = *static_cast<Context*>(raw);
    HighsCallbackSolutionOutcome receipt{};
    if (Highs_getCallbackSolutionOutcome(in, 1, &receipt) != 0) ctx.invalid = true;
    if (++ctx.calls == 1) {
      const double value = 2;
      if (Highs_setCallbackSolution(in, 1, &value) != 0 ||
          Highs_refineCallbackSolution(in, 1.0, 20, 1000, 1) != 0 ||
          Highs_tagCallbackSolution(in, 1, receipt.generation) != 0)
        ctx.invalid = true;
    } else if (ctx.calls == 2) {
      ctx.receipt = receipt.status;
      ctx.repaired.set_value();
    }
  };
  HighsPrimalService service(store, lp, reduced, model.getOptions(), mapping,
                            input, callback, &context, 0, 3, true);
  std::promise<void> release;
  auto blocked = release.get_future();
  std::atomic<bool> proof_done{false};
  std::thread proof([&] { blocked.wait(); proof_done = true; });
  service.start();
  HighsWorkerCandidate candidate;
  candidate.solution = {4};
  candidate.metadata.admission_generation = 9;
  candidate.metadata.work_epoch = 3;
  const bool published = store->publish(candidate, {4});
  const auto ready = repaired.wait_for(std::chrono::seconds(5));
  const bool admitted_before_join = !proof_done.load();
  service.stop();
  release.set_value();
  proof.join();
  REQUIRE(published);
  REQUIRE(ready == std::future_status::ready);
  REQUIRE(admitted_before_join);
  REQUIRE_FALSE(context.invalid.load());
  REQUIRE_FALSE(service.failed());
  REQUIRE(context.receipt.load() == kHighsAdmissionImproved);
  REQUIRE(store->latest()->upper_bound == 2);
  REQUIRE(service.joinedInput().admission.objective_after == 2);
}

TEST_CASE("private primal service reseeding is not an improvement",
          "[highs_incumbent]") {
  Highs model;
  model.setOptionValue("output_flag", false);
  REQUIRE(model.addVar(0, 10) == HighsStatus::kOk);
  REQUIRE(model.changeColCost(0, 1) == HighsStatus::kOk);
  auto lp = model.getLp();
  lp.ensureColwise();
  presolve::HighsPostsolveStack mapping;
  mapping.initializeIndexMaps(0, 1);
  HighsCallbackInput input;
  input.admission_generation = 9;
  input.incumbent_sequence = 42;
  auto make_store = [&](int epoch) {
    auto store = std::make_shared<HighsPrimalSnapshotStore>(lp, lp, 1e-6, 9, epoch);
    HighsWorkerCandidate seed;
    seed.solution = {4};
    seed.metadata.admission_generation = 9;
    seed.metadata.work_epoch = epoch;
    REQUIRE(store->publish(seed, {4}));
    return store;
  };
  auto callback = [](int, const char*, const HighsCallbackDataOut*,
                     HighsCallbackDataIn*, void*) {};
  auto store = make_store(3);
  HighsPrimalService service(store, lp, lp, model.getOptions(), mapping,
                            input, callback, nullptr, 0, 3, false);
  service.start();
  service.atResidentBoundary();
  service.stop();
  REQUIRE(service.notifications() == 1);
  REQUIRE(service.joinedInput().incumbent_sequence == 42);
  store = make_store(4);
  service.nextBatch(store, input, 0, 4);
  service.start();
  service.atResidentBoundary();
  service.stop();
  REQUIRE_FALSE(service.failed());
  REQUIRE(service.notifications() == 1);
  REQUIRE(service.queries() == 2);
  REQUIRE(service.joinedInput().incumbent_sequence == 42);

  HighsPrimalService failing(store, lp, lp, model.getOptions(), mapping,
      input, [](int, const char*, const HighsCallbackDataOut*,
                HighsCallbackDataIn*, void*) { throw std::runtime_error("test failure"); },
      nullptr, 0, 4, false);
  failing.start();
  failing.atResidentBoundary();
  failing.stop();
  REQUIRE(failing.failed());
}

TEST_CASE("private primal receipt distinguishes mapping rejection from no improvement",
          "[highs_incumbent]") {
  Highs model;
  model.setOptionValue("output_flag", false);
  REQUIRE(model.addVar(0, 10) == HighsStatus::kOk);
  REQUIRE(model.changeColCost(0, 1) == HighsStatus::kOk);
  auto original = model.getLp(), reduced = original;
  original.ensureColwise();
  reduced.ensureColwise();
  // Simulate a corrupted/stale mapping boundary: the original candidate is
  // feasible but its reduced projection violates a tightened bound.
  reduced.col_lower_[0] = 3;
  auto store = std::make_shared<HighsPrimalSnapshotStore>(original, reduced, 1e-6, 9, 3);
  HighsWorkerCandidate seed;
  seed.solution = {4};
  seed.metadata.admission_generation = 9;
  seed.metadata.work_epoch = 3;
  REQUIRE(store->publish(seed, {4}));
  presolve::HighsPostsolveStack mapping;
  mapping.initializeIndexMaps(0, 1);
  HighsCallbackInput input;
  input.admission_generation = 9;
  struct Context { int queries = 0; std::vector<int> receipts; } context;
  auto callback = [](int type, const char*, const HighsCallbackDataOut*,
                     HighsCallbackDataIn* in, void* raw) {
    if (type != kCallbackMipUserSolution) return;
    auto& ctx = *static_cast<Context*>(raw);
    HighsCallbackSolutionOutcome receipt{};
    Highs_getCallbackSolutionOutcome(in, 1, &receipt);
    ctx.receipts.push_back(receipt.status);
    if (++ctx.queries > 2) return;
    const double value = ctx.queries == 1 ? 2.0 : 5.0;
    Highs_setCallbackSolution(in, 1, &value);
    Highs_tagCallbackSolution(in, ctx.queries, 9);
  };
  HighsPrimalService service(store, original, reduced, model.getOptions(), mapping,
                            input, callback, &context, 0, 3, false);
  service.start();
  for (int i = 0; i < 3; ++i) service.atResidentBoundary();
  service.stop();
  REQUIRE_FALSE(service.failed());
  REQUIRE(context.receipts.size() == 3);
  REQUIRE(context.receipts[1] == kHighsAdmissionInfeasible);
  REQUIRE(context.receipts[2] == kHighsAdmissionNoImprovement);
  REQUIRE(store->latest()->upper_bound == 4);
  REQUIRE(store->latest()->sequence == 1);
}

TEST_CASE("private primal repair checks generation mapping and cancellation",
          "[highs_incumbent]") {
  Highs original, reduced;
  original.setOptionValue("output_flag", false);
  reduced.setOptionValue("output_flag", false);
  REQUIRE(original.addVar(0, 10) == HighsStatus::kOk);
  REQUIRE(original.changeColCost(0, 1) == HighsStatus::kOk);
  REQUIRE(original.addVar(5, 5) == HighsStatus::kOk);
  REQUIRE(original.changeColCost(1, 2) == HighsStatus::kOk);
  REQUIRE(reduced.addVar(0, 10) == HighsStatus::kOk);
  REQUIRE(reduced.changeColCost(0, 1) == HighsStatus::kOk);
  REQUIRE(reduced.changeObjectiveOffset(10) == HighsStatus::kOk);
  const HighsInt index = 0;
  const double coefficient = 1;
  REQUIRE(original.addRow(1, kHighsInf, 1, &index, &coefficient) == HighsStatus::kOk);
  REQUIRE(reduced.addRow(1, kHighsInf, 1, &index, &coefficient) == HighsStatus::kOk);
  auto lp = original.getLp(), red = reduced.getLp();
  lp.ensureColwise();
  red.ensureColwise();
  presolve::HighsPostsolveStack mapping;
  mapping.initializeIndexMaps(1, 2);
  mapping.compressIndexMaps({0}, {0, -1});
  HighsCallbackInput input;
  input.admission_generation = 9;
  auto store = std::make_shared<HighsPrimalSnapshotStore>(lp, red, 1e-6, 9, 3);
  input.admission.generation = 9;
  HighsWorkerCandidate seed;
  seed.solution = {4};
  seed.metadata.admission_generation = 9;
  seed.metadata.work_epoch = 3;
  REQUIRE(store->publish(seed, {4, 5}));
  struct Context {
    int queries = 0;
    bool invalid = false;
    std::vector<int> receipts;
  } context;
  auto callback = [](int type, const char*, const HighsCallbackDataOut*,
                     HighsCallbackDataIn* in, void* raw) {
    if (type != kCallbackMipUserSolution) return;
    auto& ctx = *static_cast<Context*>(raw);
    HighsCallbackSolutionOutcome receipt{};
    if (Highs_getCallbackSolutionOutcome(in, 1, &receipt) != 0) ctx.invalid = true;
    ctx.receipts.push_back(receipt.status);
    ++ctx.queries;
    if (ctx.queries > 3) return;
    double values[] = {ctx.queries == 2 ? 0.0 : 2.0, 5};
    if (Highs_setCallbackSolution(in, 2, values) != 0 ||
        Highs_tagCallbackSolution(in, ctx.queries == 1 ? 1000 : ctx.queries,
                                  ctx.queries == 1 ? 8 : 9) != 0)
      ctx.invalid = true;
  };
  HighsPrimalService service(store, lp, red, original.getOptions(), mapping,
                            input, callback, &context, 0, 3, false);
  service.start();
  for (int i = 0; i < 4; ++i) service.atResidentBoundary();
  service.stop();
  REQUIRE_FALSE(context.invalid);
  REQUIRE_FALSE(service.failed());
  REQUIRE(context.queries == 4);
  REQUIRE(context.receipts[1] == kHighsAdmissionStale);
  REQUIRE(context.receipts[2] == kHighsAdmissionInfeasible);
  REQUIRE(context.receipts[3] == kHighsAdmissionImproved);
  REQUIRE(store->latest()->candidate.solution == std::vector<double>{2});
  REQUIRE(store->latest()->upper_bound == 2);
  REQUIRE(service.joinedInput().admission.objective_after == 12);

  std::atomic<bool> stop{true};
  HighsCallbackInput cancelled;
  cancelled.highs = &original;
  cancelled.user_has_solution = true;
  cancelled.user_solution = {2, 5};
  cancelled.refinement_cancel = &stop;
  REQUIRE(cancelled.refineSolution(1, 10, 100, true) == HighsStatus::kWarning);
  REQUIRE(cancelled.refinement_status == static_cast<HighsInt>(HighsModelStatus::kInterrupt));
  REQUIRE_FALSE(cancelled.user_has_solution);
}

TEST_CASE("resident simplex boundary pauses without rebuilding or publishing a proof",
          "[highs_incumbent]") {
  Highs reference;
  reference.resetGlobalScheduler(true);
  reference.setOptionValue("output_flag", false);
  reference.setOptionValue("threads", 1);
  reference.setOptionValue("solver", "simplex");
  reference.setOptionValue("presolve", "off");
  reference.setOptionValue("solve_relaxation", true);
  REQUIRE(reference.readModel(std::string(HIGHS_DIR) + "/check/instances/bell5.mps")
          == HighsStatus::kOk);
  REQUIRE(reference.run() == HighsStatus::kOk);
  const auto iterations = reference.getInfo().simplex_iteration_count;
  std::promise<void> paused, resume;
  auto paused_future = paused.get_future();
  auto resume_future = resume.get_future();
  std::atomic<bool> done{false};
  HighsStatus result = HighsStatus::kError;
  HighsModelStatus status = HighsModelStatus::kNotset;
  double objective = kHighsInf;
  HighsInt resumed_iterations = -1;
  // Pause inside the real simplex callback: returning preserves the same
  // stack, phase, perturbations, factors and pricing arrays, unlike a limit.
  std::thread numerical([&] {
    Highs solver;
    solver.passOptions(reference.getOptions());
    solver.passModel(reference.getModel());
    bool first = true;
    solver.setCallback([&](int, const std::string&, const HighsCallbackOutput* out,
                           HighsCallbackInput*, void*) {
      if (first && out->simplex_iteration_count > 0) {
        first = false;
        paused.set_value();
        resume_future.wait();
      }
    });
    solver.startCallback(kCallbackSimplexInterrupt);
    result = solver.run();
    status = solver.getModelStatus();
    objective = solver.getObjectiveValue();
    resumed_iterations = solver.getInfo().simplex_iteration_count;
    done = true;
  });
  const auto reached = paused_future.wait_for(std::chrono::seconds(5));
  const bool unfinished = !done.load();
  resume.set_value();
  numerical.join();
  REQUIRE(reached == std::future_status::ready);
  REQUIRE(unfinished);
  REQUIRE(result == HighsStatus::kOk);
  REQUIRE(status == HighsModelStatus::kOptimal);
  REQUIRE(objective == Approx(reference.getObjectiveValue()));
  REQUIRE(resumed_iterations == iterations);
}

TEST_CASE("worker allocation respects configured scheduler capacity",
          "[highs_incumbent][highs_worker_capacity]") {
  HighsHardwareTopology server;
  server.logical_cpus = 16;
  server.physical_cores = 8;
  server.numa_nodes = 1;
  server.last_level_cache_domains = 1;
  server.complete = true;
  REQUIRE(highsTopologyAwareWorkerCount(12, server) == 12);
  REQUIRE(highsTopologyAwareWorkerCount(32, server) == 16);

  HighsHardwareTopology heterogeneous = server;
  heterogeneous.logical_cpus = 10;
  heterogeneous.physical_cores = 10;
  heterogeneous.performance_cores = 4;
  heterogeneous.heterogeneous_core_classes = 2;
  REQUIRE(highsTopologyAwareWorkerCount(8, heterogeneous) == 8);

  HighsHardwareTopology unknown;
  REQUIRE(highsTopologyAwareWorkerCount(8, unknown) == 8);
  REQUIRE(highsTopologyAwareWorkerCount(1, server) == 1);

  const HighsHardwareTopology runtime = highsDetectHardwareTopology();
  REQUIRE(runtime.logical_cpus >= 1);
  REQUIRE(highsTopologyAwareWorkerCount(4, runtime) >= 1);
  REQUIRE(highsTopologyAwareWorkerCount(4, runtime) <= 4);
}

TEST_CASE("deterministic incumbent epochs repeat at fixed work",
          "[highs_incumbent]") {
  const std::string filename =
      std::string(HIGHS_DIR) + "/check/instances/rgn.mps";

  struct Evidence {
    HighsModelStatus status;
    double objective;
    int64_t nodes;
    HighsInt simplex_iterations;
    std::vector<double> solution;
  };

  auto solve = [&]() {
    Highs highs;
    highs.resetGlobalScheduler(true);
    REQUIRE(highs.setOptionValue("output_flag", false) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("parallel", kHighsOnString) ==
            HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("threads", 2) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("random_seed", 42) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_rel_gap", 0.0) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_abs_gap", 0.0) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_max_nodes", 25) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_incumbent_exchange", "deterministic") ==
            HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_incumbent_epoch_nodes", 4) ==
            HighsStatus::kOk);
    REQUIRE(highs.readModel(filename) == HighsStatus::kOk);
    REQUIRE(highs.run() != HighsStatus::kError);
    const HighsInfo& info = highs.getInfo();
    Evidence evidence{highs.getModelStatus(),
                      info.objective_function_value, info.mip_node_count,
                      info.simplex_iteration_count,
                      highs.getSolution().col_value};
    highs.resetGlobalScheduler(true);
    return evidence;
  };

  const Evidence first = solve();
  const Evidence second = solve();
  REQUIRE(first.status == second.status);
  REQUIRE(first.objective == second.objective);
  REQUIRE(first.nodes == second.nodes);
  REQUIRE(first.simplex_iterations == second.simplex_iterations);
  REQUIRE(first.solution == second.solution);
}

TEST_CASE("heuristic retries share one deadline", "[highs_incumbent]") {
  REQUIRE(HighsPrimalHeuristics::remainingHeuristicTime(10.0, 12.0, 7.0) ==
          Approx(5.0));
  REQUIRE(HighsPrimalHeuristics::remainingHeuristicTime(10.0, 20.0, 7.0) ==
          Approx(0.0));
  REQUIRE(HighsPrimalHeuristics::remainingHeuristicTime(
              10.0, 20.0, std::numeric_limits<double>::infinity()) ==
          kHighsInf);
}

TEST_CASE("root-work-disabled", "[highs_root_work]") {
  HighsRootWorkController controller;
  controller.beginStage(HighsRootWorkConfig{}, 10.0, 20.0);
  controller.beginEpoch();
  controller.recordCompleted(HighsRootWorkPhase::kSeparation, 50.0, true);

  const HighsRootWorkDecision decision =
      controller.beforeOptional(HighsRootWorkPhase::kSeparation, 100.0);
  REQUIRE(decision.action == HighsRootWorkAction::kContinue);
  REQUIRE(decision.reason == HighsRootWorkReason::kDisabled);
  REQUIRE(controller.snapshot().separation_rounds == 0);
}

TEST_CASE("root-work-cumulative-across-restarts", "[highs_root_work]") {
  HighsRootWorkConfig config;
  config.enabled = true;
  config.max_separation_rounds = 2;
  HighsRootWorkController controller;
  controller.beginStage(config, 0.0, 100.0);

  controller.beginEpoch();
  controller.recordCompleted(HighsRootWorkPhase::kSeparation, 1.0, true);
  controller.beginEpoch();
  controller.recordCompleted(HighsRootWorkPhase::kSeparation, 2.0, true);

  const HighsRootWorkDecision decision =
      controller.beforeOptional(HighsRootWorkPhase::kSeparation, 4.0);
  REQUIRE(controller.snapshot().epoch == 1);
  REQUIRE(controller.snapshot().separation_rounds == 2);
  REQUIRE(controller.snapshot().separation_time == 3.0);
  REQUIRE(decision.action == HighsRootWorkAction::kStopSeparation);
  REQUIRE(decision.reason == HighsRootWorkReason::kSeparationRoundLimit);
}

TEST_CASE("root-work-heuristic-budget-is-local-decision", "[highs_root_work]") {
  HighsRootWorkConfig config;
  config.enabled = true;
  config.max_heuristic_time = 5.0;
  HighsRootWorkController controller;
  controller.beginStage(config, 0.0, 100.0);
  controller.beginEpoch();
  controller.recordCompleted(HighsRootWorkPhase::kRens, 6.0);

  const HighsRootWorkDecision helper =
      controller.beforeOptional(HighsRootWorkPhase::kRens, 7.0);
  const HighsRootWorkDecision separation =
      controller.beforeOptional(HighsRootWorkPhase::kSeparation, 7.0);
  REQUIRE(helper.action == HighsRootWorkAction::kSkipOptional);
  REQUIRE(helper.reason == HighsRootWorkReason::kHeuristicTimeLimit);
  REQUIRE(separation.action == HighsRootWorkAction::kContinue);
}

TEST_CASE("root-work-tree-reserve-and-global-limit", "[highs_root_work]") {
  HighsRootWorkConfig config;
  config.enabled = true;
  config.tree_reserve_time = 10.0;
  HighsRootWorkController controller;
  controller.beginStage(config, 0.0, 100.0);
  controller.beginEpoch();

  HighsRootWorkDecision decision =
      controller.beforeOptional(HighsRootWorkPhase::kRens, 90.0);
  REQUIRE(decision.action == HighsRootWorkAction::kYieldToTree);
  REQUIRE(decision.reason == HighsRootWorkReason::kTreeReserve);

  decision = controller.beforeOptional(HighsRootWorkPhase::kRens, 100.0);
  REQUIRE(decision.action == HighsRootWorkAction::kStopGlobal);
  REQUIRE(decision.reason == HighsRootWorkReason::kGlobalLimit);
}

TEST_CASE("root-work-unlimited-and-nonfinite-accounting", "[highs_root_work]") {
  HighsRootWorkConfig config;
  config.enabled = true;
  HighsRootWorkController controller;
  controller.beginStage(config, 0.0, std::numeric_limits<double>::infinity());
  controller.beginEpoch();
  controller.recordCompleted(HighsRootWorkPhase::kSeparation,
                             std::numeric_limits<double>::infinity(), true);
  controller.recordCompleted(HighsRootWorkPhase::kRens, -5.0);

  REQUIRE(controller.snapshot().separation_time == 0.0);
  REQUIRE(controller.snapshot().heuristic_time == 0.0);
  REQUIRE(controller.beforeOptional(HighsRootWorkPhase::kSeparation, 1e100)
              .action == HighsRootWorkAction::kContinue);
}

TEST_CASE("root-work-helper-allowance-composes-remaining-budgets",
          "[highs_root_work]") {
  HighsRootWorkConfig config;
  config.enabled = true;
  config.max_heuristic_time = 12.0;
  config.max_single_helper_time = 7.0;
  config.tree_reserve_time = 10.0;
  HighsRootWorkController controller;
  controller.beginStage(config, 0.0, 100.0);
  controller.beginEpoch();

  REQUIRE(controller.optionalAllowance(HighsRootWorkPhase::kRens, 20.0) == 7.0);
  controller.recordCompleted(HighsRootWorkPhase::kRens, 9.0);
  REQUIRE(controller.optionalAllowance(HighsRootWorkPhase::kRens, 20.0) == 3.0);
  REQUIRE(controller.optionalAllowance(HighsRootWorkPhase::kAnalyticCenter,
                                       85.0) == 5.0);
  REQUIRE(controller.optionalAllowance(HighsRootWorkPhase::kRens, 90.0) == 0.0);
}

TEST_CASE("root-work-zero-budgets-yield-or-skip-without-global-stop",
          "[highs_root_work]") {
  HighsRootWorkConfig config;
  config.enabled = true;
  config.max_separation_rounds = 0;
  config.max_separation_time = 0.0;
  config.max_heuristic_time = 0.0;
  config.max_single_helper_time = 0.0;
  HighsRootWorkController controller;
  controller.beginStage(config, 0.0, 100.0);
  controller.beginEpoch();

  const HighsRootWorkDecision separation =
      controller.beforeOptional(HighsRootWorkPhase::kSeparation, 1.0);
  const HighsRootWorkDecision heuristic =
      controller.beforeOptional(HighsRootWorkPhase::kRens, 1.0);
  REQUIRE(separation.action == HighsRootWorkAction::kStopSeparation);
  REQUIRE(separation.reason == HighsRootWorkReason::kSeparationRoundLimit);
  REQUIRE(heuristic.action == HighsRootWorkAction::kSkipOptional);
  REQUIRE(heuristic.reason == HighsRootWorkReason::kHeuristicTimeLimit);
  REQUIRE(controller.optionalAllowance(HighsRootWorkPhase::kRens, 1.0) == 0.0);
}

TEST_CASE("root-work-low-marginal-separation-is-phase-local",
          "[highs_root_work]") {
  HighsRootWorkConfig config;
  config.enabled = true;
  HighsRootWorkController controller;
  controller.beginStage(config, 0.0, 100.0);
  controller.beginEpoch();

  auto record = [&](double start, double finish, double before_bound,
                    double after_bound) {
    HighsRootWorkObservation before;
    before.time = start;
    before.lp_iterations = static_cast<int64_t>(start * 1000.0);
    before.incumbent = 1000.0;
    before.dual_bound = before_bound;
    before.active_lp_rows = 100;
    before.active_lp_nonzeros = 1000;
    HighsRootWorkObservation after = before;
    after.time = finish;
    after.lp_iterations += 10000;
    after.dual_bound = after_bound;
    after.active_lp_rows += 100;
    after.active_lp_nonzeros += 1000;
    controller.recordCompleted(HighsRootWorkPhase::kSeparation, before, after,
                               10, true);
  };

  record(0.0, 1.0, 100.0, 110.0);
  record(1.0, 2.0, 110.0, 120.0);
  record(2.0, 3.0, 120.0, 120.0);
  REQUIRE_FALSE(controller.snapshot().separation_stopped);
  record(3.0, 4.0, 120.0, 120.0);
  REQUIRE(controller.snapshot().separation_stopped);

  const HighsRootWorkDecision separation =
      controller.beforeOptional(HighsRootWorkPhase::kSeparation, 5.0);
  const HighsRootWorkDecision reduced_cost =
      controller.beforeOptional(HighsRootWorkPhase::kReducedCostHeuristic, 5.0);
  const HighsRootWorkDecision completion =
      controller.beforeOptional(HighsRootWorkPhase::kIncumbentCompletion, 5.0);
  REQUIRE(separation.action == HighsRootWorkAction::kStopSeparation);
  REQUIRE(separation.reason ==
          HighsRootWorkReason::kSeparationLowMarginalValue);
  REQUIRE(reduced_cost.action == HighsRootWorkAction::kContinue);
  REQUIRE(completion.action == HighsRootWorkAction::kContinue);
}

TEST_CASE("root-work-new-stage-resets-cumulative-and-tree-state",
          "[highs_root_work]") {
  HighsRootWorkConfig config;
  config.enabled = true;
  config.max_separation_rounds = 1;
  HighsRootWorkController controller;
  controller.beginStage(config, 0.0, 100.0);
  controller.beginEpoch();
  controller.recordCompleted(HighsRootWorkPhase::kSeparation, 3.0, true);
  controller.recordTreeEntry();

  REQUIRE(controller.snapshot().separation_rounds == 1);
  REQUIRE(controller.snapshot().tree_entered);

  controller.beginStage(config, 20.0, 120.0);
  controller.beginEpoch();
  REQUIRE(controller.snapshot().epoch == 0);
  REQUIRE(controller.snapshot().separation_rounds == 0);
  REQUIRE(controller.snapshot().separation_time == 0.0);
  REQUIRE_FALSE(controller.snapshot().tree_entered);
  REQUIRE(
      controller.beforeOptional(HighsRootWorkPhase::kSeparation, 21.0).action ==
      HighsRootWorkAction::kContinue);
}

TEST_CASE("root-work-records-separate-activity-evidence", "[highs_root_work]") {
  HighsRootWorkConfig config;
  config.enabled = true;
  HighsRootWorkController controller;
  controller.beginStage(config, 0.0, 100.0);
  controller.beginEpoch();

  HighsRootWorkObservation before;
  before.time = 1.0;
  before.lp_iterations = 10;
  before.improving_solutions = 1;
  before.incumbent = 100.0;
  before.dual_bound = 20.0;
  before.active_lp_rows = 10;
  before.active_lp_nonzeros = 30;
  before.cut_pool_rows = 2;
  before.fractional_integers = 5;

  HighsRootWorkObservation after;
  after.time = 3.0;
  after.lp_iterations = 25;
  after.improving_solutions = 2;
  after.incumbent = 90.0;
  after.dual_bound = 26.0;
  after.active_lp_rows = 12;
  after.active_lp_nonzeros = 40;
  after.cut_pool_rows = 3;
  after.fractional_integers = 4;

  controller.recordCompleted(HighsRootWorkPhase::kSeparation, before, after, 3,
                             true);

  const HighsRootWorkActivityAccount& account =
      controller.activity(HighsRootWorkPhase::kSeparation);
  REQUIRE(account.calls == 1);
  REQUIRE(account.successes == 1);
  REQUIRE(account.wall_time == 2.0);
  REQUIRE(account.lp_iterations == 15);
  REQUIRE(account.accepted_incumbents == 1);
  REQUIRE(account.primal_gain == 10.0);
  REQUIRE(account.dual_gain == 6.0);
  REQUIRE(account.cuts_generated == 3);
  REQUIRE(account.cut_pool_rows_added == 1);
  REQUIRE(account.lp_rows_added == 2);
  REQUIRE(account.lp_nonzeros_added == 10);
  REQUIRE(account.last_fractional_integers == 4);
  REQUIRE(controller.snapshot().separation_rounds == 1);
  REQUIRE(controller.snapshot().separation_time == 2.0);
}

TEST_CASE("root-work-unlimited-telemetry-preserves-deterministic-search",
          "[highs_root_work]") {
  const std::string filename =
      std::string(HIGHS_DIR) + "/check/instances/rgn.mps";

  struct SolveEvidence {
    HighsModelStatus status;
    double objective;
    int64_t nodes;
    HighsInt simplex_iterations;
    std::vector<double> solution;
  };

  auto solve = [&](bool telemetry) {
    Highs highs;
    // The scheduler is process-global; prior shuffled tests may use four threads.
    highs.resetGlobalScheduler(true);
    REQUIRE(highs.setOptionValue("output_flag", false) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("threads", 1) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("random_seed", 42) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_rel_gap", 0.0) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_abs_gap", 0.0) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_max_nodes", 1) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_root_max_separation_rounds", 2) ==
            HighsStatus::kOk);
    REQUIRE(highs.readModel(filename) == HighsStatus::kOk);
    if (telemetry)
      REQUIRE(highs.setOptionValue("mip_root_work_budget", true) ==
              HighsStatus::kOk);
    // A deterministic node limit is expected to return warning/limited status;
    // only a native error invalidates the identity experiment.
    REQUIRE(highs.run() != HighsStatus::kError);
    const HighsInfo& info = highs.getInfo();
    SolveEvidence evidence{highs.getModelStatus(),
                           info.objective_function_value, info.mip_node_count,
                           info.simplex_iteration_count,
                           highs.getSolution().col_value};
    highs.resetGlobalScheduler(true);
    return evidence;
  };

  const SolveEvidence baseline = solve(false);
  const SolveEvidence telemetry = solve(true);
  REQUIRE(telemetry.status == baseline.status);
  REQUIRE(telemetry.objective == baseline.objective);
  REQUIRE(telemetry.nodes == baseline.nodes);
  REQUIRE(telemetry.simplex_iterations == baseline.simplex_iterations);
  REQUIRE(telemetry.solution == baseline.solution);
}

TEST_CASE("first-LP profiling preserves deterministic search evidence",
          "[highs_root_work]") {
  struct Evidence {
    HighsModelStatus status;
    double objective;
    int64_t nodes;
    HighsInt iterations;
    std::vector<double> solution;
    std::string log;
  };
  auto solve = [&](HighsInt profile) {
    Highs highs;
    highs.resetGlobalScheduler(true);
    std::string log;
    highs.setCallback([&](int type, const std::string& message,
                           const HighsCallbackOutput*, HighsCallbackInput*, void*) {
      if (type == kCallbackLogging) log += message;
    });
    REQUIRE(highs.startCallback(kCallbackLogging) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("log_to_console", false) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("log_dev_level", 1) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("threads", 1) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("presolve", "off") == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("random_seed", 42) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_rel_gap", 0.0) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_abs_gap", 0.0) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_max_nodes", 2) == HighsStatus::kOk);
    const HighsInt analysis = kHighsAnalysisLevelMipTime | profile;
    REQUIRE(highs.setOptionValue("highs_analysis_level", analysis) == HighsStatus::kOk);
    REQUIRE(highs.readModel(std::string(HIGHS_DIR) + "/check/instances/rgn.mps") ==
            HighsStatus::kOk);
    REQUIRE(highs.run() != HighsStatus::kError);
    REQUIRE(highs.getOptions().highs_analysis_level == analysis);
    Evidence evidence{highs.getModelStatus(), highs.getInfo().objective_function_value,
                      highs.getInfo().mip_node_count,
                      highs.getInfo().simplex_iteration_count,
                      highs.getSolution().col_value, log};
    highs.resetGlobalScheduler(true);
    return evidence;
  };
  const Evidence baseline = solve(0);
  REQUIRE(baseline.log.find("MIP-RootLpProfile:") == std::string::npos);
  const HighsInt clocks = kHighsAnalysisLevelSolverTime | kHighsAnalysisLevelNlaTime;
  for (HighsInt mask : {clocks, clocks | kHighsAnalysisLevelSolverSummaryData,
                        HighsInt{kHighsAnalysisLevelSolverSummaryData}}) {
    const Evidence profile = solve(mask);
    REQUIRE(profile.status == baseline.status);
    REQUIRE(profile.objective == baseline.objective);
    REQUIRE(profile.nodes == baseline.nodes);
    REQUIRE(profile.iterations == baseline.iterations);
    REQUIRE(profile.solution == baseline.solution);
    REQUIRE(profile.log.find("MIP-RootLpProfile: event=begin analysis_level=" +
                             std::to_string(mask) + "\n") != std::string::npos);
    REQUIRE(profile.log.find("MIP-RootLpProfile: event=end") != std::string::npos);
    REQUIRE(profile.log.find("MIP-RootLpWork:") != std::string::npos);
  }
}

TEST_CASE("root-work-active-path-preserves-tiny-mip-optimum",
          "[highs_root_work]") {
  const std::string filename =
      std::string(HIGHS_DIR) + "/check/instances/rgn.mps";

  auto solve = [&](bool enabled) {
    Highs highs;
    highs.resetGlobalScheduler(true);
    REQUIRE(highs.setOptionValue("output_flag", false) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("threads", 1) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_rel_gap", 0.0) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("mip_abs_gap", 0.0) == HighsStatus::kOk);
    REQUIRE(highs.readModel(filename) == HighsStatus::kOk);
    if (enabled) {
      REQUIRE(highs.setOptionValue("mip_root_work_budget", true) ==
              HighsStatus::kOk);
      REQUIRE(highs.setOptionValue("mip_root_work_max_separation_rounds", 0) ==
              HighsStatus::kOk);
      REQUIRE(highs.setOptionValue("mip_root_work_max_separation_time", 0.0) ==
              HighsStatus::kOk);
      REQUIRE(highs.setOptionValue("mip_root_work_max_heuristic_time", 0.0) ==
              HighsStatus::kOk);
      REQUIRE(highs.setOptionValue("mip_root_work_max_single_helper_time",
                                   0.0) == HighsStatus::kOk);
      REQUIRE(highs.setOptionValue("mip_root_work_tree_reserve_time", -1.0) ==
              HighsStatus::kOk);
    }
    REQUIRE(highs.run() == HighsStatus::kOk);
    const HighsModelStatus status = highs.getModelStatus();
    const double objective = highs.getInfo().objective_function_value;
    highs.resetGlobalScheduler(true);
    return std::make_pair(status, objective);
  };

  const auto upstream = solve(false);
  const auto budgeted = solve(true);
  REQUIRE(upstream.first == HighsModelStatus::kOptimal);
  REQUIRE(budgeted.first == HighsModelStatus::kOptimal);
  REQUIRE(budgeted.second == Approx(upstream.second).margin(1e-7));
}
