#include "catch.hpp"
#include "mip/HighsPrimalHeuristicManager.h"
#include "mip/HighsPrimalHeuristics.h"
#include "Highs.h"
#include <atomic>
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <thread>
#include <vector>
#include "mip/HighsPrimalSnapshot.h"
#include "mip/HighsSearchWork.h"
#include "presolve/HighsPostsolveStack.h"
#include "mip/HighsChildProposalSink.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"

static HighsLp slotFixture() {
  HighsLp model;
  model.num_col_ = 7;
  model.num_row_ = 4;
  model.offset_ = 7;
  model.col_names_ = {"a0", "a1", "b0", "b1", "c0", "c1", "penalty"};
  model.col_cost_ = {0, 0, 0, 0, 0, 0, 10};
  model.col_lower_.assign(7, 0);
  model.col_upper_ = {1, 1, 1, 1, 1, 1, 2};
  model.integrality_.assign(7, HighsVarType::kInteger);
  model.integrality_[6] = HighsVarType::kContinuous;
  model.row_lower_ = {1, 1, 1, -kHighsInf};
  model.row_upper_ = {1, 1, 1, 0};
  model.a_matrix_.num_col_ = 7;
  model.a_matrix_.num_row_ = 4;
  model.a_matrix_.format_ = MatrixFormat::kRowwise;
  model.a_matrix_.start_ = {0, 2, 4, 6, 9};
  model.a_matrix_.index_ = {0, 1, 2, 3, 4, 5, 0, 2, 6};
  model.a_matrix_.value_ = {1, 1, 1, 1, 1, 1, 1, 1, -1};
  return model;
}

TEST_CASE("coupled slots preserve seed, auxiliaries, proof model and exchange optimum",
          "[assignment_slots][highs_primal_heuristic_manager]") {
  auto model = slotFixture();
  HighsAssignmentSlots slots;
  slots.column_to_slot = {0, 0, 1, 1, 2, 2, -1};
  slots.count = 3;
  slots.limits.max_free_columns = 4;
  const std::vector<double> seed{1, 0, 1, 0, 0, 1, 2};
  HighsAssignmentSupportResult support;
  REQUIRE(HighsAssignmentSupport::select(model, seed, slots, 0, support));
  REQUIRE(support.free_columns == std::vector<HighsInt>{0, 1, 2, 3});
  REQUIRE(support.free_slots == 2);
  auto lower = model.col_lower_, upper = model.col_upper_;
  double rate = -1;
  REQUIRE(HighsPrimalHeuristics::buildBinaryPartitionNeighbourhood(
      model, seed, support.support, support.free_columns, 1e-6, lower, upper, rate));
  REQUIRE(rate == Approx(1.0 / 3));
  REQUIRE(lower[6] == 0);
  REQUIRE(upper[6] == 2);
  REQUIRE(model.col_lower_ == std::vector<double>(7, 0));
  REQUIRE(lower[4] == 0);
  REQUIRE(upper[5] == 1);
  REQUIRE(lower[5] == 1);
  // Independent exhaustive one-hot oracle, including the eliminated penalty.
  double best = 1e30;
  for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b) for (int c = 0; c < 2; ++c) {
    std::vector<double> x{double(a), double(1-a), double(b), double(1-b), double(c), double(1-c), double(a+b)};
    bool feasible = true;
    for (size_t col = 0; col < x.size(); ++col)
      feasible &= lower[col] <= x[col] && x[col] <= upper[col];
    if (feasible) best = std::min(best, model.offset_ + 10*x[6]);
  }
  REQUIRE(best == 7);
  model.col_lower_ = lower;
  model.col_upper_ = upper;
  Highs::resetGlobalScheduler(true);
  Highs solver;
  solver.setOptionValue("output_flag", false);
  solver.setOptionValue("threads", 1);
  REQUIRE(solver.passModel(model) == HighsStatus::kOk);
  REQUIRE(solver.run() == HighsStatus::kOk);
  REQUIRE(solver.getModelStatus() == HighsModelStatus::kOptimal);
  REQUIRE(solver.getObjectiveValue() == Approx(best));
  Highs::resetGlobalScheduler(true);
}

TEST_CASE("coupled slot caps and malformed contracts fail without partial mutation",
          "[assignment_slots][highs_primal_heuristic_manager]") {
  auto model = slotFixture();
  const std::string valid = "ROST_ASSIGNMENT_SLOTS_V1 7 3 6 4 128 3 100000 1048576\n"
      "0 0 a0\n1 0 a1\n2 1 b0\n3 1 b1\n4 2 c0\n5 2 c1\n";
  HighsAssignmentSlots slots;
  std::istringstream input(valid);
  REQUIRE(HighsAssignmentSupport::read(input, model, slots));
  for (const auto& bad : std::vector<std::string>{"", valid + "junk", valid.substr(0, valid.size()-8),
      "ROST_ASSIGNMENT_SLOTS_V1 7 3 6 4 128 3 100000 1\n",
      "ROST_ASSIGNMENT_SLOTS_V1 7 3 6 6 128 3 100000 1048576\n",
      "ROST_ASSIGNMENT_SLOTS_V1 7 3 6 4 128 3 100000 1048576\n0 0 wrong\n"}) {
    std::istringstream broken(bad);
    REQUIRE_FALSE(HighsAssignmentSupport::read(broken, model, slots));
    REQUIRE(slots.column_to_slot == std::vector<HighsInt>{0, 0, 1, 1, 2, 2, -1});
  }
  HighsAssignmentSupportResult result;
  result.free_columns = {99};
  const std::vector<double> seed{1, 0, 1, 0, 0, 1, 2};
  auto tiny = slots;
  tiny.limits.max_index_bytes = 1;
  REQUIRE_FALSE(HighsAssignmentSupport::select(model, seed, tiny, 0, result));
  REQUIRE(result.free_columns == std::vector<HighsInt>{99});
  tiny = slots;
  tiny.limits.max_edge_visits = 8;
  REQUIRE_FALSE(HighsAssignmentSupport::select(model, seed, tiny, 0, result));
  REQUIRE(result.free_columns == std::vector<HighsInt>{99});
  tiny = slots;
  tiny.limits.max_edge_visits = 9;
  REQUIRE(HighsAssignmentSupport::select(model, seed, tiny, 0, result));
  REQUIRE(result.work_cap_reached);
  REQUIRE(result.visited_edges == 9);
  for (int pattern = 0; pattern < 8; ++pattern) {
    const double a = pattern & 1, b = (pattern >> 1) & 1, c = (pattern >> 2) & 1;
    const std::vector<double> values{a, 1-a, b, 1-b, c, 1-c, a+b};
    if (a+b == 0) { REQUIRE_FALSE(HighsAssignmentSupport::select(model, values, slots, 0, result)); continue; }
    REQUIRE(HighsAssignmentSupport::select(model, values, slots, 0, result));
    const auto expected = result.free_columns;
    model.a_matrix_.ensureColwise();
    REQUIRE(HighsAssignmentSupport::select(model, values, slots, 0, result));
    REQUIRE(result.free_columns == expected);
    for (int col = 0; col < 6; col += 2)
      REQUIRE((std::count(expected.begin(), expected.end(), col) != 0) ==
              (std::count(expected.begin(), expected.end(), col+1) != 0));
  }
}

TEST_CASE("slot graph follows alternate duties through a three-slot exchange",
          "[assignment_slots][highs_primal_heuristic_manager]") {
  HighsLp model;
  model.num_col_ = 11;
  model.num_row_ = 8;
  model.col_cost_.assign(11, 0);
  model.col_cost_[10] = 10;
  model.col_lower_.assign(11, 0);
  model.col_upper_.assign(11, 1);
  model.integrality_.assign(11, HighsVarType::kInteger);
  model.integrality_[10] = HighsVarType::kContinuous;
  model.row_lower_ = {1, 1, 1, 1, 1, -kHighsInf, 1, 1};
  model.row_upper_ = {1, 1, 1, 1, 1, 0, 1, 1};
  model.a_matrix_.num_col_ = 11;
  model.a_matrix_.num_row_ = 8;
  model.a_matrix_.format_ = MatrixFormat::kRowwise;
  model.a_matrix_.start_ = {0, 2, 4, 6, 8, 10, 12, 14, 16};
  model.a_matrix_.index_ = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 10, 1, 2, 3, 4};
  model.a_matrix_.value_ = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1};
  const std::vector<double> seed{1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1};
  HighsAssignmentSlots slots;
  slots.count = 5;
  slots.column_to_slot = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, -1};
  slots.limits.max_free_columns = 6;
  HighsAssignmentSupportResult selected;
  REQUIRE(HighsAssignmentSupport::select(model, seed, slots, 0, selected));
  REQUIRE(selected.free_columns == std::vector<HighsInt>{0, 1, 2, 3, 4, 5});
  model.a_matrix_.ensureColwise();
  HighsAssignmentSupportResult transposed;
  REQUIRE(HighsAssignmentSupport::select(model, seed, slots, 0, transposed));
  REQUIRE(transposed.free_columns == selected.free_columns);
  // Slot IDs are opaque and need not match original column order.
  auto permuted = slots;
  for (auto& slot : permuted.column_to_slot) if (slot >= 0) slot = 4-slot;
  HighsAssignmentSupportResult other;
  REQUIRE(HighsAssignmentSupport::select(model, seed, permuted, 0, other));
  REQUIRE(other.free_columns == selected.free_columns);
  double rate;
  auto lower = model.col_lower_, upper = model.col_upper_;
  REQUIRE(HighsPrimalHeuristics::buildBinaryPartitionNeighbourhood(
      model, seed, selected.support, selected.free_columns, 1e-6, lower, upper, rate));
  // Independent exhaustive oracle: both coverage equalities must hold.
  double best = 10;
  for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b) for (int c = 0; c < 2; ++c) {
    if (1-a+b == 1 && 1-b+c == 1) best = std::min(best, 10.0*a);
  }
  REQUIRE(best == 0);
  REQUIRE(model.col_lower_ == std::vector<double>(11, 0));
  model.col_lower_ = lower;
  model.col_upper_ = upper;
  Highs::resetGlobalScheduler(true);
  Highs solver;
  solver.setOptionValue("output_flag", false);
  solver.setOptionValue("threads", 1);
  REQUIRE(solver.passModel(model) == HighsStatus::kOk);
  REQUIRE(solver.run() == HighsStatus::kOk);
  REQUIRE(solver.getModelStatus() == HighsModelStatus::kOptimal);
  REQUIRE(solver.getObjectiveValue() == Approx(best));
  Highs::resetGlobalScheduler(true);
}

TEST_CASE("duplicate slot attempts advance selection without renewing child credit",
          "[assignment_slots][highs_primal_heuristic_manager]") {
  HighsLp model;
  model.num_col_ = 11;
  model.num_row_ = 3;
  model.col_cost_ = {0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 9};
  model.col_lower_.assign(11, 0);
  model.col_upper_.assign(11, 1);
  model.integrality_.assign(11, HighsVarType::kInteger);
  for (int i = 8; i < 11; ++i) model.integrality_[i] = HighsVarType::kContinuous;
  model.a_matrix_.num_col_ = 11;
  model.a_matrix_.num_row_ = 3;
  model.a_matrix_.format_ = MatrixFormat::kRowwise;
  model.a_matrix_.start_ = {0, 2, 4, 6};
  model.a_matrix_.index_ = {0, 8, 0, 9, 2, 10};
  model.a_matrix_.value_ = {1, -1, 1, -1, 1, -1};
  HighsAssignmentSlots slots;
  slots.count = 4;
  slots.column_to_slot = {0, 0, 1, 1, 2, 2, 3, 3, -1, -1, -1};
  slots.limits.max_free_columns = 4;
  const std::vector<double> seed{1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1};
  HighsPrimalHeuristicManager manager;
  HighsPrimalHeuristicManagerConfig config;
  config.schedule = HighsPrimalHeuristicSchedule::kPortfolio;
  config.portfolio_max_calls = 2;
  manager.initialise(config);
  HighsPrimalHeuristicContext context;
  context.method = HighsPrimalHeuristicMethod::kLocalBranching;
  context.has_incumbent = context.has_lp_solution = true;
  context.neighbourhood = HighsPrimalNeighbourhoodKind::kCoupledSlots;
  for (int attempt = 0; attempt < 3; ++attempt) {
    const auto ordinal = manager.nextSlotSelectionOrdinal();
    REQUIRE(ordinal == attempt);
    HighsAssignmentSupportResult selected;
    REQUIRE(HighsAssignmentSupport::select(model, seed, slots, ordinal, selected));
    auto lower = model.col_lower_, upper = model.col_upper_;
    double rate;
    REQUIRE(HighsPrimalHeuristics::buildBinaryPartitionNeighbourhood(
        model, seed, selected.support, selected.free_columns, 1e-6, lower, upper, rate));
    context.restriction_identity = lower;
    context.restriction_identity.insert(context.restriction_identity.end(), upper.begin(), upper.end());
    const auto decision = manager.before(context);
    if (attempt == 1) {
      REQUIRE_FALSE(decision.run);
      REQUIRE(decision.reason == HighsPrimalHeuristicDecisionReason::kDuplicateNeighbourhood);
      REQUIRE(decision.reservation_id == 0);
      REQUIRE(manager.account(context.method).scheduled == 1);
    } else {
      REQUIRE(decision.run);
      manager.record(context, {}, decision.observation_id, decision.reservation_id);
    }
  }
  REQUIRE(manager.account(context.method).scheduled == 2);
  REQUIRE(manager.account(context.method).lp_iterations == 0);
  context.restriction_identity = {0, 0};
  REQUIRE(manager.before(context).reason == HighsPrimalHeuristicDecisionReason::kCallLimit);
  REQUIRE(manager.integrityErrors() == 0);
  // Attempts remain unique across worker callers; none is an issued reservation.
  std::array<std::vector<int64_t>, 4> attempts;
  std::array<std::thread, 4> workers;
  for (int w = 0; w < 4; ++w) workers[w] = std::thread([&, w]() {
    for (int i = 0; i < 64; ++i) attempts[w].push_back(manager.nextSlotSelectionOrdinal());
  });
  for (auto& worker : workers) worker.join();
  std::vector<int64_t> all;
  for (const auto& values : attempts) all.insert(all.end(), values.begin(), values.end());
  std::sort(all.begin(), all.end());
  for (size_t i = 0; i < all.size(); ++i) REQUIRE(all[i] == int64_t(i)+3);
  REQUIRE(manager.account(context.method).scheduled == 2);
  manager.initialise(config);
  REQUIRE(manager.nextSlotSelectionOrdinal() == 0);
  manager.initialise({});
  REQUIRE(manager.nextSlotSelectionOrdinal() == -1);
}

TEST_CASE("progressive schedule is distinct from late opportunity allocation",
          "[highs_primal_heuristic_manager][progressive_child]") {
  HighsPrimalHeuristicManager manager;
  HighsPrimalHeuristicManagerConfig config;
  config.schedule = HighsPrimalHeuristicSchedule::kPortfolioAssignmentProgressive;
  manager.initialise(config);
  REQUIRE(manager.progressiveEnabled());
  REQUIRE(manager.composedEnabled());
  REQUIRE_FALSE(manager.opportunityCoverageEnabled());
  REQUIRE(std::string(manager.scheduleName(config.schedule)) == "portfolio-assignment-progressive");
  REQUIRE(manager.config().portfolio_late_reserve_calls == 0);
}

TEST_CASE("progressive child checks are bounded and independent of owner proof",
          "[highs_primal_heuristic_manager][progressive_child]") {
  for (ObjSense sense : {ObjSense::kMinimize, ObjSense::kMaximize}) {
    HighsLp lp;
    lp.num_col_ = 2;
    lp.num_row_ = 1;
    lp.offset_ = -7;
    lp.sense_ = sense;
    lp.col_cost_ = {4, 1};
    lp.col_lower_ = {0, 0};
    lp.col_upper_ = {1, 1};
    lp.integrality_ = {HighsVarType::kInteger, HighsVarType::kContinuous};
    lp.row_lower_ = {-kHighsInf};
    lp.row_upper_ = {0};
    lp.a_matrix_.format_ = MatrixFormat::kColwise;
    lp.a_matrix_.num_col_ = 2;
    lp.a_matrix_.num_row_ = 1;
    lp.a_matrix_.start_ = {0, 1, 2};
    lp.a_matrix_.index_ = {0, 0};
    lp.a_matrix_.value_ = {-1, 1};
    int calls = 0;
    double owner_lower_bound = -100;
    HighsChildProposalSink sink(lp, 1e-6,
        [&](const std::vector<double>& values) {
          ++calls;
          REQUIRE(highsVerifyCandidate(lp, values, nullptr, 1e-6).feasible);
          return true;
        }, 1);
    REQUIRE_FALSE(sink.propose({}));
    REQUIRE_FALSE(sink.propose({1, std::numeric_limits<double>::quiet_NaN()}));
    REQUIRE_FALSE(sink.propose({0, 1})); // Negative row coefficient, violated row.
    REQUIRE_FALSE(sink.propose({0.5, 0}));
    const std::vector<double> first = sense == ObjSense::kMinimize
        ? std::vector<double>{1, 1} : std::vector<double>{0, 0};
    REQUIRE(sink.propose(first));
    REQUIRE_FALSE(sink.propose(first)); // Final readback duplicate.
    REQUIRE(sink.duplicates == 1);
    REQUIRE_FALSE(sink.propose(sense == ObjSense::kMinimize
        ? std::vector<double>{0, 0} : std::vector<double>{1, 1}));
    REQUIRE(sink.capacity_denied == 1);
    REQUIRE(calls == 1);
    REQUIRE(owner_lower_bound == -100);
  }
}

TEST_CASE("progressive rejected publication can retry without duplicate reward",
          "[highs_primal_heuristic_manager][progressive_child]") {
  HighsLp lp;
  lp.num_col_ = 1;
  lp.col_cost_ = {1};
  lp.col_lower_ = {0};
  lp.col_upper_ = {1};
  lp.integrality_ = {HighsVarType::kInteger};
  lp.a_matrix_.num_col_ = 1;
  lp.a_matrix_.start_ = {0, 0};
  int calls = 0;
  HighsChildProposalSink sink(lp, 1e-6,
      [&](const std::vector<double>&) { return ++calls > 1; }, 4);
  REQUIRE_FALSE(sink.propose({1}));
  REQUIRE(sink.propose({1}));
  REQUIRE_FALSE(sink.propose({1}));
  REQUIRE(sink.propose({0}));
  REQUIRE_FALSE(sink.propose({1}));
  REQUIRE(sink.proposed == 3);
  REQUIRE(sink.published == 2);
  REQUIRE(sink.rejected == 1);
  REQUIRE(sink.duplicates == 1);
  REQUIRE(sink.non_improving == 1);
  HighsChildProposalSink disabled(lp, 1e-6,
      [&](const std::vector<double>&) { ++calls; return true; }, 0);
  REQUIRE_FALSE(disabled.propose({0}));
  REQUIRE(disabled.capacity_denied == 1);
  REQUIRE(disabled.rejected == 0);
  for (double tolerance : {-1.0, std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity()}) {
    HighsChildProposalSink invalid(lp, tolerance,
        [&](const std::vector<double>&) { ++calls; return true; });
    REQUIRE_FALSE(invalid.propose({0}));
    REQUIRE(invalid.proposed == 0);
  }
  REQUIRE(calls == 3);
}

TEST_CASE("progressive child producer postsolves before termination without public solution callback",
          "[highs_primal_heuristic_manager][progressive_child]") {
  Highs::resetGlobalScheduler(true);
  highs::parallel::initialize_scheduler(1);
  for (ObjSense sense : {ObjSense::kMinimize, ObjSense::kMaximize}) {
    HighsLp lp;
    lp.num_col_ = 4;
    lp.num_row_ = 3;
    lp.offset_ = 7;
    lp.sense_ = sense;
    lp.col_cost_ = sense == ObjSense::kMinimize
        ? std::vector<double>{1, 1, 1, 2} : std::vector<double>{-1, -1, -1, -2};
    lp.col_lower_ = {0, 0, 0, 2};
    lp.col_upper_ = {1, 1, 1, 2};
    lp.integrality_.assign(4, HighsVarType::kInteger);
    lp.row_lower_.assign(3, 1);
    lp.row_upper_.assign(3, kHighsInf);
    lp.a_matrix_.format_ = MatrixFormat::kColwise;
    lp.a_matrix_.num_col_ = 4;
    lp.a_matrix_.num_row_ = 3;
    lp.a_matrix_.start_ = {0, 2, 4, 6, 6};
    lp.a_matrix_.index_ = {0, 1, 0, 2, 1, 2};
    lp.a_matrix_.value_.assign(6, 1);
    Highs holder;
    HighsOptions options = holder.getOptions();
    options.output_flag = false;
    options.parallel = "off";
    options.presolve = "on";
    options.time_limit = 5;
    options.mip_heuristic_run_rens = false;
    options.mip_heuristic_run_rins = false;
    options.mip_heuristic_run_root_reduced_cost = false;
    options.mip_primal_heuristic_schedule = "off";
    HighsCallback callback(&holder);
    int public_solutions = 0;
    callback.user_callback = [&](int type, const std::string&,
        const HighsCallbackOutput*, HighsCallbackInput*, void*) {
      if (type == kCallbackMipSolution || type == kCallbackMipImprovingSolution)
        ++public_solutions;
    };
    callback.active[kCallbackMipSolution] = true;
    callback.active[kCallbackMipImprovingSolution] = true;
    HighsMipSolver child(callback, options, lp, HighsSolution{}, true, 1);
    HighsProfiling profiling;
    profiling.initialize(child.timer_, false);
    child.setProfiling(&profiling);
    bool returned = false;
    int publications = 0;
    HighsChildProposalSink sink(lp, 1e-6,
        [&](const std::vector<double>& values) {
          REQUIRE_FALSE(returned);
          REQUIRE(values.size() == 4);
          REQUIRE(values[3] == 2); // A column removed by presolve was restored.
          REQUIRE(highsVerifyCandidate(lp, values, nullptr, 1e-6).feasible);
          ++publications;
          return true;
        });
    child.child_proposal_sink = &sink;
    child.run();
    returned = true;
    child.child_proposal_sink = nullptr;
    REQUIRE(child.modelstatus_ == HighsModelStatus::kOptimal);
    REQUIRE(child.solution_.size() == 4);
    REQUIRE(publications > 0);
    REQUIRE(public_solutions == 0);
    REQUIRE_FALSE(sink.propose(child.solution_));
    REQUIRE(sink.duplicates > 0);
    REQUIRE(child.solution_objective_ == Approx(sense == ObjSense::kMinimize ? 13 : 1));
  }
  Highs::resetGlobalScheduler(true);
}

TEST_CASE("finite late opportunities survive restart and charge consumed credit",
          "[highs_primal_heuristic_manager][late_opportunities]") {
  for (bool cancel : {false, true}) {
    HighsPrimalHeuristicManager manager;
    HighsPrimalHeuristicManagerConfig config;
    config.schedule = HighsPrimalHeuristicSchedule::kPortfolioAssignmentOpportunities;
    config.trace_decisions = true;
    config.portfolio_solve_calls = 5;
    config.portfolio_late_reserve_calls = 2;
    config.portfolio_late_proof_progress = 100;
    config.portfolio_cooldown_lp = 0;
    config.portfolio_lp_per_call = 10;
    config.portfolio_construction_lp_per_call = 10;
    manager.initialise(config);
    for (int call = 0; call < 5; ++call) {
      HighsPrimalHeuristicContext context;
      context.method = HighsPrimalHeuristicMethod::kRens;
      context.has_lp_solution = true;
      context.restriction_identity = {double(call), 1};
      context.restart = call;
      context.main_lp_iterations = call < 3 ? 1000 : 1100;
      context.time = call * 30;
      if (call == 3) {
        for (int attempt = 0; attempt < 8; ++attempt)
          REQUIRE_FALSE(manager.portfolioOpportunityAvailable(1099, attempt, attempt, 90));
        const auto held = manager.opportunityTrace().observations[7];
        REQUIRE(held.reason == HighsPrimalHeuristicDecisionReason::kProofReserve);
        REQUIRE(held.remaining_calls == 2);
        REQUIRE(held.late_calls_held == 2);
        REQUIRE(held.late_proof_needed == 1);
        REQUIRE(manager.opportunityAccount().reserved_calls == 3);
        REQUIRE(manager.opportunityAccount().charged_lp == (cancel ? 12 : 3));
        context.burst_id = manager.beginPortfolioBurst(0, context.restart, context.time);
        REQUIRE(context.burst_id != 0);
        context.main_lp_iterations = 1099;
        REQUIRE(manager.before(context).reason == HighsPrimalHeuristicDecisionReason::kProofReserve);
        manager.endPortfolioBurst(context.burst_id);
        context.main_lp_iterations = 1100;
      }
      REQUIRE(manager.portfolioOpportunityAvailable(context.main_lp_iterations, call, call, context.time));
      context.burst_id = manager.beginPortfolioBurst(0, context.restart, context.time);
      REQUIRE(context.burst_id != 0);
      const auto decision = manager.before(context);
      REQUIRE(decision.run);
      if (cancel && call == 2) {
        HighsPrimalHeuristicReservation abandoned(manager, context, decision);
      } else {
        HighsPrimalHeuristicOutcome outcome;
        outcome.started = outcome.completed = true;
        outcome.lp_iterations = 1;
        manager.record(context, outcome, decision.observation_id, decision.reservation_id);
      }
      manager.endPortfolioBurst(context.burst_id);
    }
    const auto credit = manager.opportunityAccount();
    REQUIRE(credit.reserved_calls == 5);
    REQUIRE(credit.charged_lp == (cancel ? 14 : 5));
    REQUIRE_FALSE(manager.portfolioOpportunityAvailable(1000000, 9, 9, 300));
    const auto released = manager.opportunityTrace().observations[12];
    REQUIRE(released.late_calls_held == 0);
    REQUIRE(released.late_proof_needed == 0);
    REQUIRE(manager.integrityErrors() == 0);
    manager.initialise(config);
    REQUIRE(manager.opportunityAccount().reserved_calls == 0);
  }
}

TEST_CASE("late opportunity configuration fails closed and last call is serialized",
          "[highs_primal_heuristic_manager][late_opportunities]") {
  for (int64_t held : {-1, 1, 5}) {
    HighsPrimalHeuristicManager manager;
    HighsPrimalHeuristicManagerConfig config;
    config.schedule = HighsPrimalHeuristicSchedule::kPortfolioAssignmentOpportunities;
    config.portfolio_solve_calls = 5;
    config.portfolio_late_reserve_calls = held;
    config.portfolio_late_proof_progress = 0;
    manager.initialise(config);
    REQUIRE_FALSE(manager.portfolioOpportunityAvailable(1000000, 0));
    HighsPrimalHeuristicContext context;
    context.method = HighsPrimalHeuristicMethod::kRens;
    context.has_lp_solution = true;
    context.restriction_identity = {0, 1};
    REQUIRE(manager.before(context).reason == HighsPrimalHeuristicDecisionReason::kInvalidRestriction);
    REQUIRE(manager.opportunityAccount().reserved_calls == 0);
  }
  HighsPrimalHeuristicManager manager;
  HighsPrimalHeuristicManagerConfig config;
  config.schedule = HighsPrimalHeuristicSchedule::kPortfolioAssignmentOpportunities;
  config.portfolio_solve_calls = 2;
  config.portfolio_late_reserve_calls = 1;
  config.portfolio_late_proof_progress = 1;
  config.portfolio_cooldown_lp = 0;
  manager.initialise(config);
  HighsPrimalHeuristicContext context;
  context.method = HighsPrimalHeuristicMethod::kRens;
  context.has_lp_solution = true;
  context.restriction_identity = {0, 1};
  context.burst_id = manager.beginPortfolioBurst(0, 0, 0);
  const auto first = manager.before(context);
  REQUIRE(first.run);
  HighsPrimalHeuristicOutcome outcome;
  outcome.started = outcome.completed = true;
  outcome.lp_iterations = 1;
  manager.record(context, outcome, first.observation_id, first.reservation_id);
  manager.endPortfolioBurst(context.burst_id);
  context.main_lp_iterations = 1;
  context.burst_id = manager.beginPortfolioBurst(0, 0, 30);
  context.time = 30;
  context.restriction_identity = {1, 0};
  std::atomic<int> allowed{0};
  std::vector<std::thread> workers;
  for (int i = 0; i < 8; ++i) workers.emplace_back([&] {
    auto last = manager.before(context);
    if (last.run) {
      ++allowed;
      manager.record(context, outcome, last.observation_id, last.reservation_id);
    }
  });
  for (auto& worker : workers) worker.join();
  manager.endPortfolioBurst(context.burst_id);
  REQUIRE(allowed == 1);
  REQUIRE(manager.opportunityAccount().reserved_calls == 2);
  REQUIRE(manager.opportunityAccount().charged_lp == 2);
  REQUIRE(manager.integrityErrors() == 0);
}

TEST_CASE("periodic opportunity exploration uses exercised recency not cheap work",
          "[highs_primal_heuristic_manager][late_opportunities]") {
  for (auto schedule : {HighsPrimalHeuristicSchedule::kPortfolioAssignmentComposed,
                        HighsPrimalHeuristicSchedule::kPortfolioAssignmentOpportunities}) {
    HighsPrimalHeuristicManager manager;
    HighsPrimalHeuristicManagerConfig config;
    config.schedule = schedule;
    config.portfolio_cooldown_lp = 0;
    manager.initialise(config);
    for (int call = 0; call < 9; ++call) {
      const int expected = call < 5 ? call : call < 8 ? 0 :
          schedule == HighsPrimalHeuristicSchedule::kPortfolioAssignmentComposed ? 2 : 1;
      REQUIRE(manager.selectPartitionVariant() == expected);
      REQUIRE(manager.selectPartitionVariant() == expected);
      HighsPrimalHeuristicContext context;
      context.method = HighsPrimalHeuristicMethod::kLocalBranching;
      context.has_incumbent = true;
      context.checked_seed = {1, 0};
      context.partition_variant = expected;
      context.partition_groups = expected < 3 ? 4 : 0;
      context.neighbourhood = expected < 3 ? HighsPrimalNeighbourhoodKind::kStaffPartition :
          expected == 3 ? HighsPrimalNeighbourhoodKind::kAssignmentLocalBranching :
          HighsPrimalNeighbourhoodKind::kReducedLocalBranching;
      context.time = call * 30;
      context.restriction_identity = {double(call), 1};
      context.burst_id = manager.beginPortfolioBurst(0, 0, context.time);
      const auto decision = manager.before(context);
      REQUIRE(decision.run);
      HighsPrimalHeuristicOutcome outcome;
      outcome.started = outcome.completed = true;
      outcome.elapsed = 0.1;
      outcome.lp_iterations = expected == 0 ? 50 : expected == 1 ? 10000 : 1;
      if (expected == 0) {
        outcome.candidate_proposed = outcome.candidate_accepted = outcome.candidate_improved = true;
        outcome.admission.measured = outcome.admission.improved = true;
        outcome.admission.before = 100;
        outcome.admission.after = outcome.admission.candidate = 80;
        outcome.parent_lower_bound = 0;
      }
      manager.record(context, outcome, decision.observation_id, decision.reservation_id);
      manager.endPortfolioBurst(context.burst_id);
      REQUIRE(manager.partitionAccounts()[expected].last_exercised_call == call + 1);
    }
    REQUIRE(manager.integrityErrors() == 0);
  }
}

TEST_CASE("original partition contract fails closed on malformed or drifting support",
          "[highs_primal_heuristic_manager][binary_partition]") {
  HighsLp model;
  model.num_col_ = 5;
  model.col_names_ = {"a", "b", "c", "d", "aux"};
  model.col_lower_.assign(5, 0);
  model.col_upper_ = {1, 1, 1, 1, 7};
  model.integrality_ = {HighsVarType::kInteger, HighsVarType::kInteger,
      HighsVarType::kInteger, HighsVarType::kInteger, HighsVarType::kContinuous};
  const std::string valid = "ROST_BINARY_PARTITION_V1 5 2 1 4\n0 1 a\n1 1 b\n2 2 c\n3 2 d\n";
  std::istringstream input(valid);
  std::vector<HighsInt> groups{9};
  HighsInt free_groups = 9;
  REQUIRE(HighsPrimalHeuristics::readBinaryPartitionContract(input, model, groups, free_groups));
  REQUIRE(groups == std::vector<HighsInt>{1, 1, 2, 2, 0});
  REQUIRE(free_groups == 1);
  for (const auto& bad : std::vector<std::string>{"", valid + "extra", valid.substr(0, valid.size() - 7),
           "ROST_BINARY_PARTITION_V1 6 2 1 4\n", "ROST_BINARY_PARTITION_V1 5 2 2 4\n",
           "ROST_BINARY_PARTITION_V1 5 2 1 2\n0 1 a\n0 2 a\n",
           "ROST_BINARY_PARTITION_V1 5 2 1 2\n0 1 changed\n2 2 c\n",
           "ROST_BINARY_PARTITION_V1 5 2 1 2\n0 1 a\n4 2 aux\n",
           "ROST_BINARY_PARTITION_V1 5 2 1 2\n0 1 a\n1 1 b\n"}) {
    std::istringstream broken(bad);
    const auto before = groups;
    REQUIRE_FALSE(HighsPrimalHeuristics::readBinaryPartitionContract(broken, model, groups, free_groups));
    REQUIRE(groups == before);
    REQUIRE(free_groups == 1);
  }
}

TEST_CASE("partition ranking is sparse, deterministic and rotates useful support",
          "[highs_primal_heuristic_manager][binary_partition]") {
  HighsLp model;
  model.num_col_ = 7;
  model.num_row_ = 2;
  model.col_cost_ = {0, 0, 0, 0, 0, 0, 10};
  model.col_lower_.assign(7, 0);
  model.col_upper_ = {1, 1, 1, 1, 1, 1, 7};
  model.integrality_.assign(7, HighsVarType::kInteger);
  model.integrality_[6] = HighsVarType::kContinuous;
  model.a_matrix_.num_col_ = 7;
  model.a_matrix_.num_row_ = 2;
  model.a_matrix_.format_ = MatrixFormat::kRowwise;
  model.a_matrix_.start_ = {0, 3, 10};
  model.a_matrix_.index_ = {0, 1, 6, 0, 1, 2, 3, 4, 5, 6};
  model.a_matrix_.value_ = {0, -1, 1, 1, 1, 1, 1, 1, 1, 1};
  const std::vector<double> seed{1, 1, 1, 1, 1, 1, 3};
  const std::vector<HighsInt> groups{1, 2, 3, 4, 5, 6, 0};
  std::vector<HighsInt> support, free;
  for (int ordinal = 0; ordinal < 6; ++ordinal) {
    REQUIRE(HighsPrimalHeuristics::selectBinaryPartition(model, seed, groups, 1, ordinal, support, free));
    REQUIRE(support == std::vector<HighsInt>{0, 1, 2, 3, 4, 5});
    REQUIRE(free == std::vector<HighsInt>{ordinal == 0 ? 1 : ordinal == 1 ? 0 : ordinal});
    auto lower = model.col_lower_, upper = model.col_upper_;
    double rate;
    REQUIRE(HighsPrimalHeuristics::buildBinaryPartitionNeighbourhood(
        model, seed, support, free, 1e-6, lower, upper, rate));
    REQUIRE(rate == Approx(5.0 / 6));
    REQUIRE(lower[6] == 0);
    REQUIRE(upper[6] == 7);
  }
  REQUIRE_FALSE(HighsPrimalHeuristics::selectBinaryPartition(model, seed, groups, 6, 0, support, free));
  REQUIRE_FALSE(HighsPrimalHeuristics::selectBinaryPartition(model, seed, groups, 1, -1, support, free));
}

TEST_CASE("partition forward mapping owns affine and merged-column state",
          "[highs_primal_heuristic_manager][binary_partition]") {
  presolve::HighsPostsolveStack parent;
  parent.initializeIndexMaps(0, 4);
  parent.linearTransform(0, -1, 1);
  REQUIRE(parent.duplicateColumn(1, 0, 1, 0, 1, 1, 2, true, true, 1e-6));
  parent.compressIndexMaps({}, {0, 1, -1, 2});
  const auto original = std::vector<double>{1, 0, 1, 0.5};
  auto owned = parent;
  const auto reduced = owned.getReducedPrimalSolution(original);
  REQUIRE(reduced == std::vector<double>{0, 1, 0.5});
  HighsSolution restored;
  restored.value_valid = true;
  restored.col_value = reduced;
  HighsOptions options;
  owned.undoPrimal(options, restored, -1, true);
  REQUIRE(restored.col_value[0] == 1);
  REQUIRE(restored.col_value[1] + restored.col_value[2] == 1);
  REQUIRE(restored.col_value[3] == 0.5);
  auto second = parent;
  REQUIRE(second.getReducedPrimalSolution(original) == reduced);
  REQUIRE(original == std::vector<double>{1, 0, 1, 0.5});
}

TEST_CASE("outer opportunity trace preserves binding credits without changing scheduling",
          "[highs_primal_heuristic_manager][opportunity_trace]") {
  HighsPrimalHeuristicManager plain, traced;
  HighsPrimalHeuristicManagerConfig config;
  config.schedule = HighsPrimalHeuristicSchedule::kPortfolioAssignmentComposed;
  config.portfolio_solve_calls = 1;
  config.portfolio_bootstrap_lp = 100;
  config.portfolio_lp_cap = 100;
  config.portfolio_construction_lp_per_call = 50;
  plain.initialise(config);
  config.trace_decisions = true;
  traced.initialise(config);
  REQUIRE(traced.portfolioOpportunityAvailable(0, 0, 1, 0) ==
          plain.portfolioOpportunityAvailable(0, 0, 1, 0));
  HighsPrimalHeuristicContext context;
  context.method = HighsPrimalHeuristicMethod::kRens;
  context.has_lp_solution = true;
  context.restriction_identity = {0, 1};
  context.neighbourhood = HighsPrimalNeighbourhoodKind::kRens;
  context.burst_id = traced.beginPortfolioBurst(0, 0, 0);
  REQUIRE(context.burst_id == plain.beginPortfolioBurst(0, 0, 0));
  const auto observed = traced.before(context);
  const auto unobserved = plain.before(context);
  REQUIRE(observed.run == unobserved.run);
  REQUIRE(observed.run);
  REQUIRE(observed.budget.max_lp_iterations == 50);
  REQUIRE(traced.portfolioOpportunityAvailable(0, 0, 2, 0) ==
          plain.portfolioOpportunityAvailable(0, 0, 2, 0));
  HighsPrimalHeuristicOutcome outcome;
  outcome.started = outcome.completed = true;
  outcome.lp_iterations = 17;
  traced.record(context, outcome, observed.observation_id, observed.reservation_id);
  plain.record(context, outcome, unobserved.observation_id, unobserved.reservation_id);
  traced.endPortfolioBurst(context.burst_id);
  plain.endPortfolioBurst(context.burst_id);
  REQUIRE(traced.portfolioOpportunityAvailable(100, 99, 3, 0) ==
          plain.portfolioOpportunityAvailable(100, 99, 3, 0));
  const auto trace = traced.opportunityTrace();
  REQUIRE(trace.retained == 3);
  REQUIRE(trace.considered == 3);
  REQUIRE(trace.observations[0].remaining_calls == 1);
  REQUIRE(trace.observations[0].remaining_lp == 100);
  REQUIRE(trace.observations[1].reason == HighsPrimalHeuristicDecisionReason::kPortfolioBusy);
  REQUIRE(trace.observations[1].outstanding == 1);
  REQUIRE(trace.observations[2].reason == HighsPrimalHeuristicDecisionReason::kCallLimit);
  REQUIRE(trace.observations[2].remaining_calls == 0);
  REQUIRE(trace.observations[2].remaining_lp == 83);
  REQUIRE(trace.observations[2].charged_lp == 17);
  REQUIRE(trace.observations[2].solve_scoped_calls);
  REQUIRE(trace.observations[2].restart == 99);
  REQUIRE(trace.observations[2].time == 3);
  REQUIRE(trace.observations[2].remaining_burst_seconds == -1);
  REQUIRE(plain.opportunityTrace().retained == 0);
  for (int query = 0; query < 200; ++query)
    REQUIRE_FALSE(traced.portfolioOpportunityAvailable(1000, 99, 4 + query, 0));
  REQUIRE(traced.opportunityTrace().retained == HighsPrimalOpportunityTrace::kCapacity);
  REQUIRE(traced.opportunityTrace().considered == 203);
  REQUIRE(traced.opportunityAccount().reserved_calls == 1);
  REQUIRE(traced.opportunityAccount().charged_lp == 17);
  traced.initialise(config);
  REQUIRE(traced.opportunityTrace().retained == 0);
  REQUIRE(traced.portfolioOpportunityAvailable(0, 0));
  REQUIRE(traced.opportunityTrace().observations[0].time == -1);
}

TEST_CASE("outer opportunity observations are serialized with racing queries",
          "[highs_primal_heuristic_manager][opportunity_trace]") {
  HighsPrimalHeuristicManager manager;
  HighsPrimalHeuristicManagerConfig config;
  config.schedule = HighsPrimalHeuristicSchedule::kPortfolioAssignmentComposed;
  config.trace_decisions = true;
  manager.initialise(config);
  std::vector<std::thread> threads;
  for (int worker = 0; worker < 4; ++worker)
    threads.emplace_back([&manager, worker] {
      for (int query = 0; query < 100; ++query)
        manager.portfolioOpportunityAvailable(query, worker, query, worker);
    });
  for (auto& thread : threads) thread.join();
  const auto trace = manager.opportunityTrace();
  REQUIRE(trace.considered == 400);
  REQUIRE(trace.retained == HighsPrimalOpportunityTrace::kCapacity);
  for (size_t i = 0; i < trace.retained; ++i) {
    REQUIRE(trace.observations[i].id == static_cast<int64_t>(i + 1));
    REQUIRE(trace.observations[i].remaining_calls == config.portfolio_solve_calls);
  }
  REQUIRE(manager.opportunityAccount().reserved_calls == 0);
}

TEST_CASE("adaptive portfolio reservations survive restart without renewing credit",
          "[highs_primal_heuristic_manager][assignment_adaptive]") {
  HighsPrimalHeuristicManager manager;
  HighsPrimalHeuristicManagerConfig config;
  config.schedule = HighsPrimalHeuristicSchedule::kPortfolioAssignmentAdaptive;
  config.portfolio_lp_per_call = 25000;
  config.portfolio_construction_lp_per_call = 25000;
  config.portfolio_bootstrap_lp = 400000;
  manager.initialise(config);
  REQUIRE(manager.portfolioEnabled());
  REQUIRE_FALSE(manager.burstEnabled());
  REQUIRE_FALSE(manager.deferredEnabled());
  HighsPrimalHeuristicContext malformed;
  malformed.method = HighsPrimalHeuristicMethod::kRens;
  malformed.has_lp_solution = true;
  malformed.restriction_identity = {0, 1};
  malformed.partition_variant = 3;
  REQUIRE_FALSE(manager.before(malformed).run);
  REQUIRE(manager.opportunityAccount().reserved_calls == 0);
  for (int call = 0; call < 16; ++call) {
    HighsPrimalHeuristicContext context;
    context.method = HighsPrimalHeuristicMethod::kRens;
    context.has_lp_solution = true;
    context.main_lp_iterations = call * 25000;
    context.restart = call;
    context.time = call * 2;
    context.restriction_identity = {double(call), 1};
    for (int query = 0; query < 3; ++query) {
      REQUIRE(manager.portfolioOpportunityAvailable(context.main_lp_iterations,
          context.restart, context.time, context.incumbent_sequence));
      REQUIRE(manager.selectPartitionVariant() == 0);
      REQUIRE(manager.opportunityAccount().reserved_calls == call);
    }
    const auto decision = manager.before(context);
    REQUIRE(decision.run);
    REQUIRE(decision.budget.max_lp_iterations == 25000);
    REQUIRE_FALSE(manager.portfolioOpportunityAvailable(context.main_lp_iterations,
        context.restart, context.time, context.incumbent_sequence));
    HighsPrimalHeuristicOutcome outcome;
    outcome.started = outcome.completed = true;
    outcome.lp_iterations = 100;
    outcome.elapsed = 0.1;
    manager.record(context, outcome, decision.observation_id, decision.reservation_id);
    REQUIRE(manager.account(context.method).selection_samples == 0);
    REQUIRE_FALSE(manager.portfolioOpportunityAvailable(context.main_lp_iterations,
        context.restart + 1, context.time + 1, context.incumbent_sequence));
  }
  REQUIRE_FALSE(manager.portfolioOpportunityAvailable(10000000, 99, 100, 100));
  auto account = manager.opportunityAccount();
  REQUIRE(account.reserved_calls == 16);
  REQUIRE(account.charged_lp == 1600);
  int64_t reasons = 0;
  for (auto count : account.reasons) reasons += count;
  REQUIRE(reasons == account.considered);
  REQUIRE(manager.integrityErrors() == 0);
}

TEST_CASE("composed continuations own exact seeds and one finite solve ledger",
          "[highs_primal_heuristic_manager][assignment_composed]") {
  HighsPrimalHeuristicManager manager;
  HighsPrimalHeuristicManagerConfig config;
  config.schedule = HighsPrimalHeuristicSchedule::kPortfolioAssignmentComposed;
  config.portfolio_cooldown_lp = 0;
  config.portfolio_bootstrap_lp = config.portfolio_lp_cap;
  manager.initialise(config);
  REQUIRE(manager.burstEnabled());
  REQUIRE(manager.deferredEnabled());
  REQUIRE(manager.partitionVariantCount() == 5);
  for (int call = 0; call < 16; ++call) {
    const auto burst = manager.beginPortfolioBurst(0, call, call * 30);
    REQUIRE(burst != 0);
    HighsPrimalHeuristicContext context;
    context.method = HighsPrimalHeuristicMethod::kRens;
    context.neighbourhood = HighsPrimalNeighbourhoodKind::kRens;
    context.burst_id = burst;
    context.restart = call;
    context.time = call * 30;
    context.has_incumbent = context.has_lp_solution = true;
    context.incumbent_sequence = call;
    context.seed_upper_bound = 100 - call;
    context.checked_seed = {double(call), 0};
    context.restriction_identity = {double(call), 1};
    const auto decision = manager.before(context);
    REQUIRE(decision.run);
    HighsPrimalHeuristicOutcome outcome;
    outcome.started = outcome.completed = outcome.candidate_accepted = true;
    outcome.admission.measured = outcome.admission.improved = true;
    outcome.admission.before = context.seed_upper_bound;
    outcome.admission.after = outcome.admission.candidate = context.seed_upper_bound - 1;
    outcome.admission.checked_seed = {double(call + 1), 0};
    outcome.lp_iterations = 10;
    manager.record(context, outcome, decision.observation_id, decision.reservation_id);
    REQUIRE_FALSE(manager.continuePortfolioBurst(burst, 0, call, call + 1,
        outcome.admission.after, context.time + 1, {double(call + 1), 1}));
    REQUIRE(manager.continuePortfolioBurst(burst, 0, call, call + 1,
        outcome.admission.after, context.time + 1, outcome.admission.checked_seed));
    auto stale = context;
    stale.time += 1;
    stale.incumbent_sequence += 1;
    stale.seed_upper_bound -= 1;
    stale.checked_seed = {double(call + 1), 1};
    stale.restriction_identity = {double(call + 1), 2};
    REQUIRE_FALSE(manager.before(stale).run);
    manager.endPortfolioBurst(burst);
  }
  REQUIRE(manager.opportunityAccount().reserved_calls == 16);
  REQUIRE(manager.opportunityAccount().charged_lp == 160);
  REQUIRE(manager.burstAccount().extra_reserved == 0);
  REQUIRE_FALSE(manager.portfolioOpportunityAvailable(1000000, 100, 1000, 100));
  REQUIRE(manager.integrityErrors() == 0);
}

TEST_CASE("composed checkpoint deferral preserves exact seed without extra credit",
          "[highs_primal_heuristic_manager][assignment_composed]") {
  HighsPrimalHeuristicManager manager;
  HighsPrimalHeuristicManagerConfig config;
  config.schedule = HighsPrimalHeuristicSchedule::kPortfolioAssignmentComposed;
  config.portfolio_cooldown_lp = 0;
  manager.initialise(config);
  auto burst = manager.beginPortfolioBurst(0, 0, 0);
  HighsPrimalHeuristicContext context;
  context.method = HighsPrimalHeuristicMethod::kRens;
  context.neighbourhood = HighsPrimalNeighbourhoodKind::kRens;
  context.burst_id = burst;
  context.has_incumbent = context.has_lp_solution = true;
  context.seed_upper_bound = 100;
  context.checked_seed = {1, 0};
  context.restriction_identity = {0, 1};
  auto decision = manager.before(context);
  REQUIRE(decision.run);
  HighsPrimalHeuristicOutcome gain;
  gain.started = gain.completed = gain.candidate_accepted = true;
  gain.admission.measured = gain.admission.improved = true;
  gain.admission.before = 100;
  gain.admission.after = gain.admission.candidate = 90;
  gain.admission.checked_seed = {0, 1};
  gain.lp_iterations = 10;
  manager.record(context, gain, decision.observation_id, decision.reservation_id);
  REQUIRE_FALSE(manager.deferPortfolioContinuation(burst, 0, 0, 1, 90, 1, {1, 0}, true));
  REQUIRE(manager.deferPortfolioContinuation(burst, 0, 0, 1, 90, 1, {0, 1}, true));
  manager.endPortfolioBurst(burst);
  burst = manager.beginDeferredPortfolioBurst(0, 0, 2, 90, {0, 1});
  REQUIRE(burst != 0);
  context.burst_id = burst;
  context.time = 2;
  context.incumbent_sequence = 1;
  context.seed_upper_bound = 90;
  context.checked_seed = {0, 1};
  context.restriction_identity = {1, 0};
  context.method = HighsPrimalHeuristicMethod::kRins;
  context.neighbourhood = HighsPrimalNeighbourhoodKind::kRins;
  decision = manager.before(context);
  REQUIRE(decision.run);
  REQUIRE(decision.deferred_continuation);
  { HighsPrimalHeuristicReservation cancelled(manager, context, decision); }
  REQUIRE(manager.opportunityAccount().reserved_calls == 2);
  REQUIRE(manager.opportunityAccount().charged_lp == 100010);
  REQUIRE(manager.burstAccount().extra_reserved == 0);
  REQUIRE(manager.burstAccount().deferred_reserved == 1);
  manager.endPortfolioBurst(burst);
  REQUIRE(manager.beginDeferredPortfolioBurst(0, 0, 30, 90, {0, 1}) == 0);
  REQUIRE(manager.integrityErrors() == 0);
}

TEST_CASE("assignment distance excludes auxiliaries and is permutation invariant",
          "[highs_primal_heuristic_manager][assignment_composed]") {
  HighsLp model;
  model.num_col_ = 7;
  model.col_lower_.assign(7, 0);
  model.col_upper_.assign(7, 1);
  model.integrality_.assign(7, HighsVarType::kInteger);
  const std::vector<double> seed{1, 0, 1, 0, 1, 0, 1};
  const std::vector<HighsInt> groups{1, 1, 2, 2, 3, 3, 0};
  for (HighsInt radius : {2, 4}) {
    HighsLocalBranchingRow row;
    REQUIRE(HighsPrimalHeuristics::buildAssignmentLocalBranchingRow(
        model, seed, groups, 1e-6, radius, row));
    for (unsigned mask = 0; mask < 128; ++mask) {
      double value = 0;
      int distance = 0;
      for (size_t j = 0; j < row.index.size(); ++j)
        value += row.value[j] * ((mask >> row.index[j]) & 1);
      for (int j = 0; j < 6; ++j) distance += int((mask >> j) & 1) != int(seed[j]);
      REQUIRE((value <= row.upper) == (distance <= radius));
    }
    auto permuted_seed = seed;
    auto permuted_groups = groups;
    std::reverse(permuted_seed.begin(), permuted_seed.end());
    std::reverse(permuted_groups.begin(), permuted_groups.end());
    HighsLocalBranchingRow permuted;
    REQUIRE(HighsPrimalHeuristics::buildAssignmentLocalBranchingRow(
        model, permuted_seed, permuted_groups, 1e-6, radius, permuted));
    REQUIRE(permuted.upper == row.upper);
    REQUIRE(permuted.index.size() == 6);
    REQUIRE(permuted.index.front() == 1);
    const auto before = row.index;
    auto malformed = groups;
    malformed[0] = -1;
    REQUIRE_FALSE(HighsPrimalHeuristics::buildAssignmentLocalBranchingRow(
        model, seed, malformed, 1e-6, radius, row));
    REQUIRE(row.index == before);
  }
}

TEST_CASE("composed distance radii distinguish exhaustion from unfinished search",
          "[highs_primal_heuristic_manager][assignment_composed]") {
  for (int variant = 0; variant < 5; ++variant) {
    HighsPrimalHeuristicManager manager;
    HighsPrimalHeuristicManagerConfig config;
    config.schedule = HighsPrimalHeuristicSchedule::kPortfolioAssignmentComposed;
    config.portfolio_cooldown_lp = 0;
    manager.initialise(config);
    for (int call = 0; call < 2; ++call) {
      const auto burst = manager.beginPortfolioBurst(0, 0, call * 30);
      HighsPrimalHeuristicContext context;
      context.method = HighsPrimalHeuristicMethod::kLocalBranching;
      context.neighbourhood = variant < 3 ? HighsPrimalNeighbourhoodKind::kStaffPartition :
          variant == 3 ? HighsPrimalNeighbourhoodKind::kAssignmentLocalBranching :
                         HighsPrimalNeighbourhoodKind::kReducedLocalBranching;
      context.partition_variant = variant;
      context.partition_groups = variant < 3 ? 4 : 0;
      context.has_incumbent = true;
      context.seed_upper_bound = 100;
      context.checked_seed = {0, 1};
      context.restriction_identity = {double(call), double(variant)};
      context.burst_id = burst;
      context.time = call * 30;
      auto malformed = context;
      malformed.neighbourhood = HighsPrimalNeighbourhoodKind::kRins;
      REQUIRE_FALSE(manager.before(malformed).run);
      const auto decision = manager.before(context);
      REQUIRE(decision.run);
      HighsPrimalHeuristicOutcome outcome;
      outcome.started = outcome.completed = true;
      outcome.elapsed = 0.1;
      outcome.lp_iterations = 10;
      HighsPrimalChildReceipt child;
      child.status = int(call == 0 ? HighsModelStatus::kOptimal : HighsModelStatus::kIterationLimit);
      outcome.children.append(child);
      manager.record(context, outcome, decision.observation_id, decision.reservation_id);
      REQUIRE(manager.partitionAccounts()[variant].completed == call + 1);
      REQUIRE(manager.partitionAccounts()[variant].improved == 0);
      if (variant >= 3)
        REQUIRE(manager.partitionAccounts()[variant].radius == (call == 0 ? 4 : 2));
      REQUIRE_FALSE(manager.continuePortfolioBurst(burst, 0, 0, 1, 90, context.time + 1, {1, 0}));
      manager.endPortfolioBurst(burst);
    }
    REQUIRE(manager.integrityErrors() == 0);
  }
}

TEST_CASE("assignment-distance tiny MIP agrees with exact offset completion oracle",
          "[highs_primal_heuristic_manager][assignment_composed]") {
  for (int radius : {2, 4}) {
    Highs::resetGlobalScheduler(true);
    HighsLp parent;
    parent.num_col_ = 5;
    parent.num_row_ = 1;
    parent.offset_ = 64;
    parent.col_cost_ = {7, -3, 5, -9, 0.5};
    parent.col_lower_.assign(5, 0);
    parent.col_upper_ = {1, 1, 1, 1, 2};
    parent.integrality_ = {HighsVarType::kInteger, HighsVarType::kInteger,
        HighsVarType::kInteger, HighsVarType::kInteger, HighsVarType::kContinuous};
    parent.row_lower_ = {0};
    parent.row_upper_ = {kHighsInf};
    parent.a_matrix_.format_ = MatrixFormat::kRowwise;
    parent.a_matrix_.num_col_ = 5;
    parent.a_matrix_.num_row_ = 1;
    parent.a_matrix_.start_ = {0, 5};
    parent.a_matrix_.index_ = {0, 1, 2, 3, 4};
    parent.a_matrix_.value_ = {-1, 1, -1, 1, 1};
    const std::vector<double> seed{1, 0, 1, 0, 2};
    HighsLocalBranchingRow row;
    REQUIRE(HighsPrimalHeuristics::buildAssignmentLocalBranchingRow(
        parent, seed, {1, 1, 2, 2, 0}, 1e-6, radius, row));
    double oracle = kHighsInf;
    for (int bits = 0; bits < 16; ++bits) {
      int distance = 0, aux = 0;
      double cost = 64;
      for (int col = 0; col < 4; ++col) {
        const int value = (bits >> col) & 1;
        distance += value != int(seed[col]);
        aux += (col % 2 == 0 ? 1 : -1) * value;
        cost += parent.col_cost_[col] * value;
      }
      if (distance <= radius) oracle = std::min(oracle, cost + 0.5 * std::max(0, aux));
    }
    Highs highs;
    REQUIRE(highs.setOptionValue("output_flag", false) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("threads", 1) == HighsStatus::kOk);
    REQUIRE(highs.passModel(parent) == HighsStatus::kOk);
    REQUIRE(highs.addRow(-kHighsInf, row.upper, row.index.size(), row.index.data(), row.value.data()) == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
    REQUIRE(highs.getInfo().objective_function_value == Approx(oracle));
    REQUIRE(parent.num_row_ == 1);
    REQUIRE(parent.col_upper_[4] == 2);
  }
  Highs::resetGlobalScheduler(true);
}

TEST_CASE("root repair owns an inactive lease without renewing existing tree work",
          "[highs_primal_heuristic_manager][assignment_adaptive]") {
  HighsSearchWork work;
  {
    HighsRepairReservationScope denied(work, 20);
    REQUIRE_FALSE(denied.granted());
    REQUIRE_FALSE(work.observingLpWork());
  }
  {
    HighsRepairReservationScope root(work, 20, true, 3);
    REQUIRE(root.granted());
    REQUIRE(work.observingLpWork());
    work.chargeSubMip(20);
    REQUIRE(work.exhausted());
  }
  REQUIRE_FALSE(work.observingLpWork());
  REQUIRE(work.summary().repair_iterations == 20);
  work.begin(10, true, 5);
  work.chargeLp(10);
  REQUIRE(work.exhausted());
  const auto leases = work.summary().leases;
  {
    HighsRepairReservationScope existing(work, 5, true, 6);
    REQUIRE(existing.granted());
    work.chargeSubMip(5);
  }
  REQUIRE(work.observingLpWork());
  REQUIRE(work.exhausted());
  REQUIRE(work.summary().leases == leases);
  work.finish();
  REQUIRE(work.tasks().back().work == 15);
}

TEST_CASE("construction and improvement spend different slices of the same allowance",
          "[highs_primal_heuristic_manager][assignment_adaptive]") {
  HighsPrimalHeuristicManager manager;
  HighsPrimalHeuristicManagerConfig config;
  config.schedule = HighsPrimalHeuristicSchedule::kPortfolioAssignmentAdaptive;
  config.portfolio_lp_per_call = 25000;
  config.portfolio_cooldown_lp = 0;
  manager.initialise(config);
  HighsPrimalHeuristicContext context;
  context.method = HighsPrimalHeuristicMethod::kRens;
  context.has_lp_solution = true;
  context.restriction_identity = {0, 1};
  auto decision = manager.before(context);
  REQUIRE(decision.run);
  REQUIRE(decision.budget.max_lp_iterations == 100000);
  HighsPrimalHeuristicOutcome outcome;
  outcome.started = outcome.completed = true;
  outcome.lp_iterations = 76152;
  outcome.elapsed = 14;
  manager.record(context, outcome, decision.observation_id, decision.reservation_id);
  context.has_incumbent = true;
  context.checked_seed = {1, 0};
  context.restriction_identity = {0, 2};
  context.restart = 1;
  decision = manager.before(context);
  REQUIRE(decision.run);
  REQUIRE(decision.budget.max_lp_iterations == 25000);
  outcome.lp_iterations = 25000;
  manager.record(context, outcome, decision.observation_id, decision.reservation_id);
  REQUIRE(manager.opportunityAccount().charged_lp == 101152);
  REQUIRE(manager.opportunityAccount().reserved_calls == 2);
  REQUIRE(manager.account(context.method).selection_samples == 1);
  REQUIRE(manager.integrityErrors() == 0);
}

TEST_CASE("adaptive partition outcomes separate context, own gain and complexity",
          "[highs_primal_heuristic_manager][assignment_adaptive]") {
  HighsPrimalHeuristicManager manager;
  HighsPrimalHeuristicManagerConfig config;
  config.schedule = HighsPrimalHeuristicSchedule::kPortfolioAssignmentAdaptive;
  config.portfolio_cooldown_lp = 0;
  manager.initialise(config);
  for (int variant = 0; variant < 3; ++variant) {
    REQUIRE(manager.selectPartitionVariant() == variant);
    HighsPrimalHeuristicContext context;
    context.method = HighsPrimalHeuristicMethod::kLocalBranching;
    context.has_incumbent = true;
    context.partition_variant = variant;
    context.partition_groups = 4;
    context.checked_seed = {1, 0};
    context.restriction_identity = {double(variant), 1};
    auto decision = manager.before(context);
    REQUIRE(decision.run);
    auto wrong = context;
    wrong.partition_variant = (variant + 1) % 3;
    HighsPrimalHeuristicOutcome outcome;
    outcome.started = outcome.completed = true;
    outcome.elapsed = 0.1;
    outcome.lp_iterations = 10 + variant;
    HighsPrimalChildReceipt child;
    child.status = int(variant == 0 ? HighsModelStatus::kOptimal : HighsModelStatus::kTimeLimit);
    outcome.children.append(child);
    if (variant == 2) outcome.candidate_proposed = true; // Rejection is not ordinary no-gain feedback.
    manager.record(wrong, outcome, decision.observation_id, decision.reservation_id);
    REQUIRE(manager.partitionAccounts()[variant].completed == 0);
    manager.record(context, outcome, decision.observation_id, decision.reservation_id);
    REQUIRE(manager.partitionAccounts()[variant].free_groups == (variant == 0 ? 5 : variant == 1 ? 3 : 4));
    REQUIRE(manager.partitionAccounts()[variant].reward == 0);
  }
  REQUIRE(manager.integrityErrors() == 3);
  REQUIRE(manager.opportunityAccount().charged_lp == 33);
}

TEST_CASE("assignment variants retain all original rows and free auxiliary domains",
          "[highs_primal_heuristic_manager][assignment_adaptive]") {
  HighsLp model;
  model.num_col_ = 7;
  model.num_row_ = 2;
  model.col_cost_ = {0, 0, 0, 0, 0, 0, 10};
  model.col_lower_.assign(7, 0);
  model.col_upper_.assign(7, 1);
  model.integrality_.assign(7, HighsVarType::kInteger);
  model.a_matrix_.num_col_ = 7;
  model.a_matrix_.num_row_ = 2;
  model.a_matrix_.format_ = MatrixFormat::kRowwise;
  model.a_matrix_.start_ = {0, 3, 5};
  model.a_matrix_.index_ = {3, 4, 6, 4, 5};
  model.a_matrix_.value_ = {-1, 1, 1, 1, -1};
  const std::vector<double> seed{1, 1, 1, 1, 1, 1, 1};
  const std::vector<HighsInt> groups{1, 2, 3, 4, 5, 6, 0};
  for (int variant = 0; variant < 3; ++variant) {
    std::vector<HighsInt> support, free, again, again_free;
    REQUIRE(HighsPrimalHeuristics::selectBinaryPartition(model, seed, groups, 2, 0, support, free, variant));
    REQUIRE(HighsPrimalHeuristics::selectBinaryPartition(model, seed, groups, 2, 0, again, again_free, variant));
    REQUIRE(support == again);
    REQUIRE(free == again_free);
    REQUIRE(free == (variant == 1 ? std::vector<HighsInt>{0, 1} : std::vector<HighsInt>{3, 4}));
    auto lower = model.col_lower_, upper = model.col_upper_;
    double fixing;
    REQUIRE(HighsPrimalHeuristics::buildBinaryPartitionNeighbourhood(
        model, seed, support, free, 1e-6, lower, upper, fixing));
    REQUIRE(lower[6] == 0);
    REQUIRE(upper[6] == 1);
    REQUIRE(model.a_matrix_.value_ == std::vector<double>{-1, 1, 1, 1, -1});
    for (int assignment = 0; assignment < 128; ++assignment) {
      bool child = true;
      for (int col = 0; col < 7; ++col) {
        const int value = (assignment >> col) & 1;
        child &= lower[col] <= value && value <= upper[col];
      }
      if (!child) continue;
      for (int col : support)
        if (std::find(free.begin(), free.end(), col) == free.end())
          REQUIRE(((assignment >> col) & 1) == 1);
    }
  }
  std::vector<HighsInt> support, free;
  REQUIRE_FALSE(HighsPrimalHeuristics::selectBinaryPartition(model, seed, groups, 1, 0, support, free, 3));
  model.col_lower_.clear();
  REQUIRE_FALSE(HighsPrimalHeuristics::selectBinaryPartition(model, seed, groups, 1, 0, support, free));
}

TEST_CASE("adaptive variants reward checked own gains and retain exploration",
          "[highs_primal_heuristic_manager][assignment_adaptive]") {
  HighsPrimalHeuristicManager manager;
  HighsPrimalHeuristicManagerConfig config;
  config.schedule = HighsPrimalHeuristicSchedule::kPortfolioAssignmentAdaptive;
  config.portfolio_cooldown_lp = 0;
  manager.initialise(config);
  for (int call = 0; call < 7; ++call) {
    const int variant = manager.selectPartitionVariant();
    const int expected = call < 3 ? call : call < 6 ? 0 : 1;
    REQUIRE(variant == expected);
    REQUIRE(manager.selectPartitionVariant() == variant);
    HighsPrimalHeuristicContext context;
    context.method = HighsPrimalHeuristicMethod::kLocalBranching;
    context.has_incumbent = true;
    context.checked_seed = {1, 0};
    context.partition_variant = variant;
    context.partition_groups = 4;
    context.restriction_identity = {double(call), 1};
    auto decision = manager.before(context);
    REQUIRE(decision.run);
    HighsPrimalHeuristicOutcome outcome;
    outcome.started = outcome.completed = true;
    outcome.candidate_proposed = outcome.candidate_accepted = outcome.candidate_improved = true;
    outcome.admission.measured = outcome.admission.improved = true;
    outcome.admission.before = 100;
    outcome.admission.after = outcome.admission.candidate = 80;
    outcome.parent_lower_bound = 0;
    outcome.elapsed = 0.1;
    outcome.lp_iterations = variant == 0 ? 50 : 10;
    // A peer's improvement must not become this variant's reward.
    if (variant != 0) outcome.admission.candidate = 90;
    manager.record(context, outcome, decision.observation_id, decision.reservation_id);
    REQUIRE(manager.partitionAccounts()[variant].improved == (variant == 0 ? call < 3 ? 1 : call - 1 : 0));
  }
  REQUIRE(manager.partitionAccounts()[0].reward > 0);
  REQUIRE(manager.partitionAccounts()[1].reward == 0);
  REQUIRE(manager.partitionAccounts()[2].reward == 0);
  REQUIRE(manager.integrityErrors() == 0);
}

TEST_CASE("owned child receipts retain separate calls and bounded coverage",
          "[highs_primal_heuristic_manager][owned_child_receipts]") {
  HighsPrimalChildReceipt first, second;
  first.invocation = 41;
  first.status = static_cast<int>(HighsModelStatus::kTimeLimit);
  first.lp_iterations = 250;
  first.fixing_rate = 0.6;
  second.invocation = 42;
  second.status = static_cast<int>(HighsModelStatus::kOptimal);
  second.lp_iterations = 100;
  second.fixing_rate = 0.3;
  HighsPrimalChildReceipts earlier, later;
  earlier.append(first);
  later.append(second);
  earlier.append(later);
  REQUIRE(earlier.retained == 2);
  REQUIRE(earlier.considered == 2);
  REQUIRE(earlier.items[0].invocation == 41);
  REQUIRE(earlier.items[1].invocation == 42);
  REQUIRE(earlier.items[0].status == 13);
  REQUIRE(earlier.items[1].status == 7);
  REQUIRE(earlier.items[0].fixing_rate == 0.6);
  REQUIRE(earlier.items[1].fixing_rate == 0.3);
  REQUIRE(later.retained == 1);
  earlier.append(earlier);
  REQUIRE(earlier.considered == 2);
  for (int i = 0; i < 5; ++i) later.append(second);
  earlier.append(later);
  REQUIRE(earlier.retained == 4);
  REQUIRE(earlier.considered == 8);
  REQUIRE(later.retained == 4);
  REQUIRE(later.considered == 6);
}

TEST_CASE("child observation does not change deterministic manager decisions",
          "[highs_primal_heuristic_manager][owned_child_receipts]") {
  std::vector<int> decisions;
  std::vector<HighsPrimalHeuristicAccount> accounts;
  for (bool trace : {false, true}) {
    HighsPrimalHeuristicManagerConfig config;
    config.schedule = HighsPrimalHeuristicSchedule::kPortfolio;
    config.trace_decisions = trace;
    config.portfolio_lp_cap = 1000;
    config.portfolio_proof_divisor = 1;
    HighsPrimalHeuristicManager manager;
    manager.initialise(config);
    HighsPrimalHeuristicContext context;
    context.method = HighsPrimalHeuristicMethod::kRens;
    context.has_lp_solution = true;
    context.main_lp_iterations = 1000;
    context.neighbourhood_signature = 11;
    context.restriction_identity = {0, 1};
    auto decision = manager.before(context);
    decisions.push_back(static_cast<int>(decision.reason));
    REQUIRE(decision.run);
    HighsPrimalHeuristicOutcome outcome;
    outcome.started = outcome.completed = true;
    outcome.lp_iterations = 10;
    outcome.elapsed = 1;
    HighsPrimalChildReceipt child;
    child.invocation = 17;
    child.status = static_cast<int>(HighsModelStatus::kIterationLimit);
    child.lp_iterations = 10;
    if (trace) outcome.children.append(child);
    manager.record(context, outcome, decision.observation_id, decision.reservation_id);
    decisions.push_back(static_cast<int>(manager.before(context).reason));
    accounts.push_back(manager.account(context.method));
    REQUIRE(manager.integrityErrors() == 0);
    if (trace) {
      const auto observed = manager.decisionTrace();
      REQUIRE(observed.observations[0].outcome.children.retained == 1);
      REQUIRE(observed.observations[0].outcome.children.items[0].status == 14);
    }
  }
  REQUIRE(decisions[0] == decisions[2]);
  REQUIRE(decisions[1] == decisions[3]);
  REQUIRE(accounts[0].lp_iterations == accounts[1].lp_iterations);
  REQUIRE(accounts[0].started == accounts[1].started);
  REQUIRE(accounts[0].scheduled == accounts[1].scheduled);
}

TEST_CASE("binary partition preserves seed, auxiliaries and caller bounds",
          "[highs_primal_heuristic_manager][binary_partition]") {
  HighsLp model;
  model.num_col_ = 5;
  model.integrality_ = {HighsVarType::kInteger, HighsVarType::kInteger,
      HighsVarType::kInteger, HighsVarType::kInteger, HighsVarType::kContinuous};
  model.col_lower_ = {0, 0, 0, 0, 0};
  model.col_upper_ = {1, 1, 1, 1, 5};
  const std::vector<double> seed{1, 0, 0, 1, 2.5};
  const std::vector<HighsInt> support{0, 1, 2}, free{1, 2};
  auto lower = model.col_lower_, upper = model.col_upper_;
  double rate = -1;
  REQUIRE(HighsPrimalHeuristics::buildBinaryPartitionNeighbourhood(
      model, seed, support, free, 1e-6, lower, upper, rate));
  REQUIRE(rate == Approx(1.0 / 3));
  REQUIRE(lower == std::vector<double>{1, 0, 0, 0, 0});
  REQUIRE(upper == std::vector<double>{1, 1, 1, 1, 5});
  for (int bits = 0; bits < 16; ++bits) {
    bool inside = true;
    for (int col = 0; col < 4; ++col) {
      const int bit = (bits >> col) & 1;
      inside &= lower[col] <= bit && bit <= upper[col];
    }
    REQUIRE(inside == ((bits & 1) == 1));
  }
  REQUIRE(model.col_lower_ == std::vector<double>{0, 0, 0, 0, 0});
  REQUIRE(model.col_upper_ == std::vector<double>{1, 1, 1, 1, 5});
  for (const auto& invalid : std::vector<std::vector<HighsInt>>{
           {}, {-1}, {5}, {0, 0}, {4}}) {
    auto lo = model.col_lower_, hi = model.col_upper_;
    REQUIRE_FALSE(HighsPrimalHeuristics::buildBinaryPartitionNeighbourhood(
        model, seed, invalid, free, 1e-6, lo, hi, rate));
    REQUIRE(lo == model.col_lower_);
    REQUIRE(hi == model.col_upper_);
    REQUIRE(rate == 0);
  }
  for (const auto& invalid : std::vector<std::vector<HighsInt>>{
           {}, {-1}, {5}, {1, 1}, {3}, {0, 1, 2}}) {
    auto lo = model.col_lower_, hi = model.col_upper_;
    REQUIRE_FALSE(HighsPrimalHeuristics::buildBinaryPartitionNeighbourhood(
        model, seed, support, invalid, 1e-6, lo, hi, rate));
    REQUIRE(lo == model.col_lower_);
    REQUIRE(hi == model.col_upper_);
  }
  for (double value : {0.5, 2.0, std::numeric_limits<double>::quiet_NaN()}) {
    auto bad = seed, lo = model.col_lower_, hi = model.col_upper_;
    bad[0] = value;
    REQUIRE_FALSE(HighsPrimalHeuristics::buildBinaryPartitionNeighbourhood(
        model, bad, support, free, 1e-6, lo, hi, rate));
    REQUIRE(lo == model.col_lower_);
    REQUIRE(hi == model.col_upper_);
  }
  auto lo = model.col_lower_, hi = model.col_upper_;
  hi[0] = 0;
  REQUIRE_FALSE(HighsPrimalHeuristics::buildBinaryPartitionNeighbourhood(
      model, seed, support, free, 1e-6, lo, hi, rate));
  REQUIRE(hi[0] == 0);
  lo = model.col_lower_;
  REQUIRE_FALSE(HighsPrimalHeuristics::buildBinaryPartitionNeighbourhood(
      model, seed, support, free, 1e-6, lo, lo, rate));
  REQUIRE(lo == model.col_lower_);
}

TEST_CASE("partition small MIP agrees with independent restricted enumeration",
          "[highs_primal_heuristic_manager][binary_partition]") {
  for (int fixed = 0; fixed < 4; ++fixed) {
    Highs::resetGlobalScheduler(true);
    HighsLp parent;
    parent.num_col_ = 5;
    parent.num_row_ = 2;
    parent.col_cost_ = {7, -3, 5, -9, 0.5};
    parent.col_lower_ = {0, 0, 0, 0, 0};
    parent.col_upper_ = {1, 1, 1, 1, 2};
    parent.integrality_ = {HighsVarType::kInteger, HighsVarType::kInteger,
        HighsVarType::kInteger, HighsVarType::kInteger, HighsVarType::kContinuous};
    parent.row_lower_ = {2, 0};
    parent.row_upper_ = {2, kHighsInf};
    parent.a_matrix_.format_ = MatrixFormat::kRowwise;
    parent.a_matrix_.num_col_ = 5;
    parent.a_matrix_.num_row_ = 2;
    parent.a_matrix_.start_ = {0, 4, 9};
    parent.a_matrix_.index_ = {0, 1, 2, 3, 0, 1, 2, 3, 4};
    parent.a_matrix_.value_ = {1, 1, 1, 1, -1, 1, -1, 1, 1};
    const std::vector<double> seed{1, 0, 1, 0, 2};
    std::vector<HighsInt> support{0, 1, 2, 3}, free;
    for (int col = 0; col < 4; ++col) if (col != fixed) free.push_back(col);
    auto child = parent;
    double rate;
    REQUIRE(HighsPrimalHeuristics::buildBinaryPartitionNeighbourhood(
        parent, seed, support, free, 1e-6, child.col_lower_, child.col_upper_, rate));
    double oracle = kHighsInf;
    for (int bits = 0; bits < 16; ++bits) {
      int count = 0, aux = 0;
      double cost = 0;
      for (int col = 0; col < 4; ++col) {
        const int value = (bits >> col) & 1;
        count += value;
        aux += (col % 2 == 0 ? 1 : -1) * value;
        cost += parent.col_cost_[col] * value;
      }
      if (count != 2 || ((bits >> fixed) & 1) != seed[fixed]) continue;
      oracle = std::min(oracle, cost + 0.5 * std::max(0, aux));
    }
    Highs highs;
    REQUIRE(highs.setOptionValue("output_flag", false) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("threads", 1) == HighsStatus::kOk);
    REQUIRE(highs.passModel(child) == HighsStatus::kOk);
    HighsSolution start;
    start.col_value = seed;
    start.value_valid = true;
    REQUIRE(highs.setSolution(start) == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
    REQUIRE(highs.getInfo().objective_function_value == Approx(oracle));
    REQUIRE(highs.getSolution().col_value[fixed] == seed[fixed]);
    REQUIRE(parent.col_upper_[4] == 2);
  }
  Highs::resetGlobalScheduler(true);
}

TEST_CASE("managed RINS agreement bounds preserve free and continuous choices",
          "[highs_primal_heuristic_manager][highs_rins_agreement]") {
  HighsLp model;
  model.num_col_ = 6;
  model.integrality_ = {HighsVarType::kInteger, HighsVarType::kInteger,
      HighsVarType::kInteger, HighsVarType::kInteger,
      HighsVarType::kContinuous, HighsVarType::kImplicitInteger};
  const std::vector<double> seed{1, 0, -2, 1, 0.5, 0.5};
  const std::vector<double> relaxation{1, 0.5, -2, 0.5, 0.5, 0.5};
  std::vector<double> lower{0, 0, -3, 0, 0, 0}, upper{1, 1, 2, 1, 1, 1};
  double rate;
  REQUIRE(HighsPrimalHeuristics::buildRinsNeighbourhood(
      model, relaxation, seed, 1e-6, 0.25, lower, upper, rate));
  REQUIRE(rate == 0.5);
  REQUIRE(lower == std::vector<double>{1, 0, -2, 0, 0, 0});
  REQUIRE(upper == std::vector<double>{1, 1, -2, 1, 1, 1});
  REQUIRE(model.integrality_[4] == HighsVarType::kContinuous);
}

TEST_CASE("RINS malformed or trivial neighborhoods cannot mutate bounds",
          "[highs_primal_heuristic_manager][highs_rins_agreement]") {
  for (int failure = 0; failure < 13; ++failure) {
    CAPTURE(failure);
    HighsLp model;
    model.num_col_ = 3;
    model.integrality_.assign(3, HighsVarType::kInteger);
    std::vector<double> lower(3, 0), upper(3, 1);
    std::vector<double> seed{1, 0, 1}, lp{1, 0.5, 0.5};
    double tolerance = 1e-6, minimum = 0.25;
    if (failure == 0) lp.pop_back();
    if (failure == 1) seed[2] = std::numeric_limits<double>::quiet_NaN();
    if (failure == 2) lp[2] = std::numeric_limits<double>::infinity();
    if (failure == 3) seed[2] = 0.75;
    if (failure == 4) lp = seed;
    if (failure == 5) lp.assign(3, 0.5);
    if (failure == 6) tolerance = -1;
    if (failure == 7) minimum = 0.5;
    if (failure == 8) model.integrality_[2] = HighsVarType::kSemiInteger;
    if (failure == 9) lower[2] = 2;
    if (failure == 10) model.integrality_.clear();
    if (failure == 11) lower[2] = std::numeric_limits<double>::quiet_NaN();
    if (failure == 12) minimum = std::numeric_limits<double>::quiet_NaN();
    const auto before_lower = lower, before_upper = upper;
    double rate;
    REQUIRE_FALSE(HighsPrimalHeuristics::buildRinsNeighbourhood(
        model, lp, seed, tolerance, minimum, lower, upper, rate));
    for (size_t i = 0; i < lower.size(); ++i)
      REQUIRE((lower[i] == before_lower[i] ||
               (std::isnan(lower[i]) && std::isnan(before_lower[i]))));
    REQUIRE(upper == before_upper);
  }
}

TEST_CASE("RINS agreement tolerance never rounds outside a strict child domain",
          "[highs_primal_heuristic_manager][highs_rins_agreement]") {
  HighsLp model;
  model.num_col_ = 4;
  model.integrality_.assign(4, HighsVarType::kInteger);
  std::vector<double> lower{0, 0, 0, 0}, upper{1, 1, 1 - 5e-7, 1};
  const std::vector<double> seed{1, 0, 1 - 5e-7, 1};
  const std::vector<double> lp{1 - 5e-7, 2e-6, 1 - 5e-7, 0.5};
  double rate;
  REQUIRE(HighsPrimalHeuristics::buildRinsNeighbourhood(
      model, lp, seed, 1e-6, 0.25, lower, upper, rate));
  REQUIRE(rate == 0.25);
  REQUIRE(lower[0] == upper[0]);
  REQUIRE(lower[1] == 0);
  REQUIRE(lower[2] == 0);
  REQUIRE(upper[2] == 1 - 5e-7);
}

TEST_CASE("RINS agreement exhaustive binary oracle and child optimization",
          "[highs_primal_heuristic_manager][highs_rins_agreement]") {
  for (unsigned agreement = 1; agreement < 63; ++agreement) {
    HighsLp model;
    model.num_col_ = 6;
    model.integrality_.assign(6, HighsVarType::kInteger);
    std::vector<double> seed(6), lp(6), lower(6, 0), upper(6, 1);
    for (unsigned col = 0; col < 6; ++col) {
      seed[col] = (col + agreement) % 2;
      lp[col] = agreement & (1u << col) ? seed[col] : 0.5;
    }
    double rate;
    REQUIRE(HighsPrimalHeuristics::buildRinsNeighbourhood(
        model, lp, seed, 1e-6, 0, lower, upper, rate));
    for (unsigned assignment = 0; assignment < 64; ++assignment) {
      bool permitted = true, expected = true;
      for (unsigned col = 0; col < 6; ++col) {
        const double value = (assignment >> col) & 1;
        permitted &= value >= lower[col] && value <= upper[col];
        if (agreement & (1u << col)) expected &= value == seed[col];
      }
      REQUIRE(permitted == expected);
    }
    if (agreement > 3) continue;
    for (const auto sense : {ObjSense::kMinimize, ObjSense::kMaximize}) {
      model.num_row_ = 1;
      model.col_cost_ = {-3, 4, 2, -1, 7, 5};
      model.offset_ = 123;
      model.sense_ = sense;
      model.col_lower_ = lower;
      model.col_upper_ = upper;
      model.row_lower_ = {2};
      model.row_upper_ = {4};
      model.a_matrix_.start_ = {0, 1, 2, 3, 4, 5, 6};
      model.a_matrix_.index_.assign(6, 0);
      model.a_matrix_.value_.assign(6, 1);
      double optimum = sense == ObjSense::kMinimize ? kHighsInf : -kHighsInf;
      for (unsigned assignment = 0; assignment < 64; ++assignment) {
        bool permitted = true;
        int activity = 0;
        double objective = 123;
        for (unsigned col = 0; col < 6; ++col) {
          const double value = (assignment >> col) & 1;
          permitted &= value >= lower[col] && value <= upper[col];
          activity += value;
          objective += value * model.col_cost_[col];
        }
        if (!permitted || activity < 2 || activity > 4) continue;
        optimum = sense == ObjSense::kMinimize ? std::min(optimum, objective)
                                                : std::max(optimum, objective);
      }
      Highs child;
      child.setOptionValue("output_flag", false);
      REQUIRE(child.passModel(model) == HighsStatus::kOk);
      HighsSolution start;
      start.col_value = seed;
      start.value_valid = true;
      REQUIRE(child.setSolution(start) == HighsStatus::kOk);
      REQUIRE(child.run() == HighsStatus::kOk);
      REQUIRE(child.getModelStatus() == HighsModelStatus::kOptimal);
      REQUIRE(child.getObjectiveValue() == Approx(optimum));
      REQUIRE(model.col_lower_ == lower);
    }
  }
}

TEST_CASE("helper deadlines fail closed for invalid clock data",
          "[highs_primal_heuristic_manager][highs_rins_agreement]") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  REQUIRE(HighsPrimalHeuristics::remainingHeuristicTime(1, 3, 5) == 3);
  REQUIRE(HighsPrimalHeuristics::remainingHeuristicTime(1, 9, 5) == 0);
  REQUIRE(HighsPrimalHeuristics::remainingHeuristicTime(1, 3, inf) == inf);
  REQUIRE(HighsPrimalHeuristics::remainingHeuristicTime(1, 3, nan) == 0);
  REQUIRE(HighsPrimalHeuristics::remainingHeuristicTime(nan, 3, 5) == 0);
  REQUIRE(HighsPrimalHeuristics::remainingHeuristicTime(1, inf, 5) == 0);
  REQUIRE(HighsPrimalHeuristics::remainingHeuristicTime(1, 3, -inf) == 0);
}

TEST_CASE("primal heuristic compatibility schedule never changes calls",
          "[highs_primal_heuristic_manager]") {
  HighsPrimalHeuristicManager manager;
  HighsPrimalHeuristicManagerConfig config;
  config.schedule = HighsPrimalHeuristicSchedule::kCompatibility;
  config.rins_max_calls = 0;
  config.rins_cooldown_nodes = 1000;
  manager.initialise(config);

  HighsPrimalHeuristicContext context;
  context.method = HighsPrimalHeuristicMethod::kRins;
  REQUIRE(manager.before(context).run);
  REQUIRE(manager.before(context).run);
  REQUIRE(manager.account(context.method).scheduled == 2);
}

TEST_CASE("managed RINS requires evidence and suppresses repeated work",
          "[highs_primal_heuristic_manager]") {
  HighsPrimalHeuristicManager manager;
  HighsPrimalHeuristicManagerConfig config;
  config.schedule = HighsPrimalHeuristicSchedule::kManagedRins;
  config.rins_max_calls = 2;
  config.rins_cooldown_nodes = 10;
  manager.initialise(config);

  HighsPrimalHeuristicContext context;
  context.method = HighsPrimalHeuristicMethod::kRins;
  context.has_lp_solution = true;
  context.neighbourhood_signature = 17;
  REQUIRE_FALSE(manager.before(context).run);
  REQUIRE(manager.account(context.method)
              .decisions[static_cast<size_t>(
                  HighsPrimalHeuristicDecisionReason::kNoIncumbent)] == 1);

  context.has_incumbent = true;
  REQUIRE(manager.before(context).run);
  context.node = 5;
  REQUIRE_FALSE(manager.before(context).run);
  context.node = 10;
  REQUIRE_FALSE(manager.before(context).run);
  context.neighbourhood_signature = 18;
  REQUIRE(manager.before(context).run);
  context.node = 30;
  context.neighbourhood_signature = 19;
  REQUIRE_FALSE(manager.before(context).run);
}

TEST_CASE("managed RINS accounts bounded work and protects proof share",
          "[highs_primal_heuristic_manager]") {
  HighsPrimalHeuristicManager manager;
  HighsPrimalHeuristicManagerConfig config;
  config.schedule = HighsPrimalHeuristicSchedule::kManagedRins;
  config.proof_work_reserve = 0.75;
  manager.initialise(config);

  HighsPrimalHeuristicContext context;
  context.method = HighsPrimalHeuristicMethod::kRins;
  context.has_incumbent = true;
  context.has_lp_solution = true;
  context.main_lp_iterations = 75;
  context.heuristic_lp_iterations = 25;
  REQUIRE_FALSE(manager.before(context).run);

  context.heuristic_lp_iterations = 24;
  REQUIRE(manager.before(context).run);
  HighsPrimalHeuristicOutcome outcome;
  outcome.started = true;
  outcome.completed = true;
  outcome.candidate_proposed = true;
  outcome.candidate_accepted = true;
  outcome.candidate_improved = true;
  outcome.nodes = 12;
  outcome.lp_iterations = 100;
  manager.record(context, outcome);
  const auto& account = manager.account(context.method);
  REQUIRE(account.started == 1);
  REQUIRE(account.improved == 1);
  REQUIRE(account.nodes == 12);
  REQUIRE(account.lp_iterations == 100);
  // Raw completed sub-MIP work dominates the size-adjusted native counter.
  context.heuristic_lp_iterations = 0;
  REQUIRE_FALSE(manager.before(context).run);
}

TEST_CASE("managed RINS grants only its remaining cumulative LP budget",
          "[highs_primal_heuristic_manager]") {
  HighsPrimalHeuristicManager manager;
  HighsPrimalHeuristicManagerConfig config;
  config.schedule = HighsPrimalHeuristicSchedule::kManagedRins;
  config.rins_budget.max_lp_iterations = 125;
  manager.initialise(config);

  HighsPrimalHeuristicContext context;
  context.method = HighsPrimalHeuristicMethod::kRins;
  context.has_incumbent = true;
  context.has_lp_solution = true;
  context.incumbent_sequence = 1;
  context.neighbourhood_signature = 11;
  REQUIRE(manager.before(context).budget.max_lp_iterations == 125);

  HighsPrimalHeuristicOutcome outcome;
  outcome.started = true;
  outcome.completed = true;
  outcome.lp_iterations = 100;
  manager.record(context, outcome);

  context.incumbent_sequence = 2;
  context.neighbourhood_signature = 12;
  REQUIRE(manager.before(context).budget.max_lp_iterations == 25);
  outcome.lp_iterations = 25;
  manager.record(context, outcome);

  context.incumbent_sequence = 3;
  context.neighbourhood_signature = 13;
  const auto denied = manager.before(context);
  REQUIRE_FALSE(denied.run);
  REQUIRE(denied.reason == HighsPrimalHeuristicDecisionReason::kLpWorkLimit);
}

TEST_CASE("RINS preparation and child phases conserve measured work",
          "[highs_primal_heuristic_manager]") {
  HighsPrimalHeuristicManager manager;
  HighsPrimalHeuristicManagerConfig config;
  config.schedule = HighsPrimalHeuristicSchedule::kCompatibility;
  manager.initialise(config);
  HighsPrimalHeuristicContext context;
  REQUIRE(manager.before(context).run);
  HighsPrimalHeuristicOutcome preparation_only;
  preparation_only.phase_measured = true;
  preparation_only.preparation_lp_iterations = 100;
  preparation_only.lp_iterations = 100;
  preparation_only.preparation_exhaustions = 1;
  preparation_only.preparation_seconds = preparation_only.elapsed = 2.0;
  manager.record(context, preparation_only);
  REQUIRE(manager.before(context).run);
  HighsPrimalHeuristicOutcome child;
  child.phase_measured = child.started = child.completed = true;
  child.preparation_lp_iterations = 20;
  child.child_lp_iterations = 80;
  child.lp_iterations = 100;
  child.child_calls = 2;
  child.preparation_seconds = 1.0;
  child.child_seconds = 4.0;
  child.elapsed = 5.0;
  manager.record(context, child);
  const auto account = manager.account(context.method);
  REQUIRE(account.phase_measured_calls == 2);
  REQUIRE(account.started == 1);
  REQUIRE(account.child_calls == 2);
  REQUIRE(account.preparation_exhaustions == 1);
  REQUIRE(account.preparation_lp_iterations + account.child_lp_iterations == account.lp_iterations);
  REQUIRE(account.preparation_seconds + account.child_seconds == account.elapsed);
  REQUIRE(account.proposed == 0);
}

TEST_CASE("parallel RINS reserves work before dispatch and releases unused work",
          "[highs_primal_heuristic_manager]") {
  HighsPrimalHeuristicManager manager;
  HighsPrimalHeuristicManagerConfig config;
  config.schedule = HighsPrimalHeuristicSchedule::kManagedRins;
  config.rins_budget.max_lp_iterations = 100;
  config.rins_lp_iterations_per_call = 25;
  config.trace_decisions = true;
  manager.initialise(config);
  std::atomic<int> ready{0}, granted{0};
  std::atomic<bool> finish{false};
  std::vector<std::thread> workers;
  for (int worker = 0; worker < 8; ++worker) {
    workers.emplace_back([&, worker]() {
      HighsPrimalHeuristicContext context;
      context.worker_id = worker;
      context.has_incumbent = context.has_lp_solution = true;
      context.incumbent_sequence = worker + 1;
      const auto decision = manager.before(context);
      if (decision.run) ++granted;
      ++ready;
      while (!finish.load()) std::this_thread::yield();
      if (decision.run) {
        HighsPrimalHeuristicOutcome outcome;
        outcome.lp_iterations = 10;
        manager.record(context, outcome, decision.observation_id);
      }
    });
  }
  while (ready.load() != 8) std::this_thread::yield();
  const int admissions = granted.load();
  finish = true;
  for (auto& worker : workers) worker.join();
  REQUIRE(admissions == 4);
  REQUIRE(manager.account(HighsPrimalHeuristicMethod::kRins).lp_iterations == 40);
  const auto trace = manager.decisionTrace();
  REQUIRE(trace.considered == 8);
  REQUIRE(trace.retained == 8);
  REQUIRE(trace.dropped == 0);
  int recorded = 0;
  for (size_t i = 0; i < trace.retained; ++i) {
    REQUIRE(trace.observations[i].id == i + 1);
    REQUIRE(trace.observations[i].outcome_recorded == trace.observations[i].decision.run);
    recorded += trace.observations[i].outcome_recorded ? 1 : 0;
  }
  REQUIRE(recorded == 4);
  HighsPrimalHeuristicContext retry;
  retry.has_incumbent = retry.has_lp_solution = true;
  retry.incumbent_sequence = 99;
  REQUIRE(manager.before(retry).budget.max_lp_iterations == 25);
  manager.record(retry, HighsPrimalHeuristicOutcome{});
  REQUIRE(manager.before(retry).run);
}

TEST_CASE("bounded heuristic trace cannot change decisions or work accounts",
          "[highs_primal_heuristic_manager]") {
  HighsPrimalHeuristicManager plain, observed;
  HighsPrimalHeuristicManagerConfig config;
  config.schedule = HighsPrimalHeuristicSchedule::kManagedRins;
  config.rins_max_calls = 3;
  config.rins_budget.max_lp_iterations = 100;
  config.rins_lp_iterations_per_call = 25;
  plain.initialise(config);
  config.trace_decisions = true;
  observed.initialise(config);
  HighsPrimalHeuristicContext context;
  context.has_incumbent = context.has_lp_solution = true;
  for (int call = 0; call != 160; ++call) {
    context.incumbent_sequence = call;
    context.time = 2.0 * call;
    const auto control = plain.before(context);
    const auto treatment = observed.before(context);
    REQUIRE(control.run == treatment.run);
    REQUIRE(control.reason == treatment.reason);
    REQUIRE(control.budget.max_lp_iterations == treatment.budget.max_lp_iterations);
    REQUIRE(control.observation_id == 0);
    REQUIRE(treatment.observation_id == static_cast<uint64_t>(call + 1));
    if (!control.run) continue;
    HighsPrimalHeuristicOutcome outcome;
    outcome.phase_measured = true;
    outcome.preparation_exhaustions = 1;
    outcome.lp_iterations = outcome.preparation_lp_iterations = 10;
    outcome.elapsed = outcome.preparation_seconds = 1.0;
    outcome.preparation_stop_mask = kHighsHeuristicStopDeadline;
    outcome.finished_time = context.time + outcome.elapsed;
    plain.record(context, outcome);
    observed.record(context, outcome, treatment.observation_id);
  }
  const auto trace = observed.decisionTrace();
  REQUIRE(trace.retained == kHighsPrimalHeuristicTraceCapacity);
  REQUIRE(trace.considered == 160);
  REQUIRE(trace.dropped == 32);
  REQUIRE(trace.observations[0].outcome_recorded);
  REQUIRE(trace.observations[0].outcome.preparation_stop_mask == kHighsHeuristicStopDeadline);
  REQUIRE_FALSE(trace.observations[3].decision.run);
  REQUIRE_FALSE(trace.observations[3].outcome_recorded);
  REQUIRE(plain.decisionTrace().considered == 0);
  const auto before = plain.account(context.method);
  const auto after = observed.account(context.method);
  REQUIRE(before.decisions == after.decisions);
  REQUIRE(before.lp_iterations == after.lp_iterations);
  REQUIRE(before.preparation_exhaustions == after.preparation_exhaustions);
  observed.initialise(config);
  REQUIRE(observed.decisionTrace().considered == 0);
}

TEST_CASE("heuristic observations join out of order by token and generation",
          "[highs_primal_heuristic_manager]") {
  HighsPrimalHeuristicManager manager;
  HighsPrimalHeuristicManagerConfig config;
  config.schedule = HighsPrimalHeuristicSchedule::kCompatibility;
  config.trace_decisions = true;
  manager.initialise(config);
  HighsPrimalHeuristicContext first, second;
  first.worker_id = 1;
  second.worker_id = 2;
  first.restart = second.restart = 3;
  const auto a = manager.before(first);
  const auto b = manager.before(second);
  HighsPrimalHeuristicOutcome outcome;
  outcome.started = outcome.completed = true;
  outcome.child_calls = 1;
  manager.record(second, outcome, a.observation_id);
  REQUIRE_FALSE(manager.decisionTrace().observations[0].outcome_recorded);
  manager.record(second, outcome, b.observation_id);
  manager.record(first, outcome, a.observation_id);
  auto trace = manager.decisionTrace();
  REQUIRE(trace.observations[0].context.worker_id == 1);
  REQUIRE(trace.observations[1].context.worker_id == 2);
  REQUIRE(trace.observations[0].outcome_recorded);
  REQUIRE(trace.observations[1].outcome_recorded);
  HighsPrimalHeuristicOutcome duplicate;
  manager.record(first, duplicate, a.observation_id);
  REQUIRE(manager.decisionTrace().observations[0].outcome.started);
}

TEST_CASE("portfolio reservations own release, charge overshoot and verify identity",
          "[highs_primal_heuristic_manager][primal_portfolio]") {
  HighsPrimalHeuristicManager manager;
  HighsPrimalHeuristicManagerConfig config;
  config.schedule = HighsPrimalHeuristicSchedule::kPortfolio;
  config.trace_decisions = true;
  config.portfolio_lp_cap = 100;
  config.portfolio_bootstrap_lp = 60;
  config.portfolio_lp_per_call = 40;
  manager.initialise(config);
  HighsPrimalHeuristicContext context;
  context.method = HighsPrimalHeuristicMethod::kRens;
  context.has_lp_solution = true;
  context.neighbourhood_signature = 3;
  context.restriction_identity = {0, 1};
  auto a = manager.before(context);
  REQUIRE(a.run);
  REQUIRE(a.reservation_id != 0);
  REQUIRE(a.budget.max_lp_iterations == 40);
  REQUIRE_FALSE(manager.before(context).run);
  HighsPrimalHeuristicOutcome outcome;
  outcome.lp_iterations = 50;
  manager.record(context, outcome, a.observation_id, a.reservation_id);
  manager.record(context, outcome, a.observation_id, a.reservation_id);
  REQUIRE(manager.integrityErrors() == 1);
  REQUIRE(manager.account(context.method).lp_iterations == 50);
  REQUIRE_FALSE(manager.before(context).run);
  // Same hash, different exact restriction must not be suppressed.
  context.restriction_identity = {1, 0};
  auto b = manager.before(context);
  REQUIRE(b.run);
  REQUIRE(b.budget.max_lp_iterations == 10);
  manager.record(context, outcome, b.observation_id, a.reservation_id);
  REQUIRE(manager.integrityErrors() == 2);
  REQUIRE(manager.before(context).reason == HighsPrimalHeuristicDecisionReason::kPortfolioBusy);
  manager.record(context, outcome, b.observation_id, b.reservation_id);
  context.restriction_identity = {0, 0};
  context.main_lp_iterations = std::numeric_limits<int64_t>::max();
  REQUIRE_FALSE(manager.before(context).run); // cumulative cap, not renewed per call
}

TEST_CASE("portfolio first sweep and abandoned tokens preserve accounting",
          "[highs_primal_heuristic_manager][primal_portfolio]") {
  HighsPrimalHeuristicManager manager;
  HighsPrimalHeuristicManagerConfig config;
  config.schedule = HighsPrimalHeuristicSchedule::kPortfolio;
  config.trace_decisions = true;
  manager.initialise(config);
  std::array<bool, kHighsPrimalHeuristicMethodCount> eligible{};
  for (auto method : {HighsPrimalHeuristicMethod::kRens, HighsPrimalHeuristicMethod::kRins,
                       HighsPrimalHeuristicMethod::kLocalBranching})
    eligible[static_cast<size_t>(method)] = true;
  HighsPrimalHeuristicContext context;
  context.has_incumbent = context.has_lp_solution = true;
  context.restriction_identity = {0, 1};
  for (auto method : {HighsPrimalHeuristicMethod::kRens, HighsPrimalHeuristicMethod::kRins,
                       HighsPrimalHeuristicMethod::kLocalBranching}) {
    REQUIRE(manager.selectPortfolio(eligible) == method);
    context.method = method;
    const auto decision = manager.before(context);
    REQUIRE(decision.run);
    HighsPrimalHeuristicReservation reservation(manager, context, decision);
    HighsPrimalHeuristicOutcome outcome;
    outcome.lp_iterations = 10;
    reservation.finish(outcome);
    reservation.finish(outcome);
  }
  REQUIRE(manager.integrityErrors() == 0);
  context.restriction_identity = {1, 1};
  const auto abandoned = manager.before(context);
  REQUIRE(abandoned.run);
  { HighsPrimalHeuristicReservation reservation(manager, context, abandoned); }
  REQUIRE(manager.account(context.method).lp_iterations == 100010);
  const auto trace = manager.decisionTrace();
  REQUIRE(trace.observations[trace.retained - 1].outcome_recorded);
  REQUIRE(trace.observations[trace.retained - 1].outcome.elapsed >= 0);
  REQUIRE(trace.observations[trace.retained - 1].outcome.finished_time >= context.time);
}

TEST_CASE("portfolio permits seeded construction and LP-free local branching",
          "[highs_primal_heuristic_manager][primal_portfolio]") {
  HighsPrimalHeuristicManager manager;
  HighsPrimalHeuristicManagerConfig config;
  config.schedule = HighsPrimalHeuristicSchedule::kPortfolio;
  manager.initialise(config);
  HighsPrimalHeuristicContext context;
  context.method = HighsPrimalHeuristicMethod::kRens;
  context.has_lp_solution = true;
  context.restriction_identity = {0, 1};
  auto decision = manager.before(context);
  REQUIRE(decision.run);
  manager.record(context, {}, 0, decision.reservation_id);
  context.method = HighsPrimalHeuristicMethod::kLocalBranching;
  context.has_lp_solution = false;
  REQUIRE_FALSE(manager.before(context).run);
  context.has_incumbent = true;
  decision = manager.before(context);
  REQUIRE(decision.run);
  manager.record(context, {}, 0, decision.reservation_id);
  context.recursion_depth = 1;
  REQUIRE_FALSE(manager.before(context).run);
}

TEST_CASE("binary local branching equals Hamming distance and does not mutate on failure",
          "[highs_primal_heuristic_manager][primal_portfolio]") {
  for (unsigned seed = 0; seed < 16; ++seed) {
    HighsLp model;
    model.num_col_ = 4;
    model.col_lower_.assign(4, 0);
    model.col_upper_.assign(4, 1);
    model.integrality_.assign(4, HighsVarType::kInteger);
    std::vector<double> incumbent(4);
    for (unsigned j = 0; j < 4; ++j) incumbent[j] = (seed >> j) & 1;
    for (int radius = 0; radius <= 4; ++radius) {
      HighsLocalBranchingRow row;
      REQUIRE(HighsPrimalHeuristics::buildLocalBranchingRow(model, incumbent, 1e-6, radius, row));
      for (unsigned point = 0; point < 16; ++point) {
        double activity = 0;
        unsigned distance = 0;
        for (unsigned j = 0; j < 4; ++j) {
          activity += row.value[j] * ((point >> j) & 1);
          distance += ((point >> j) & 1) != incumbent[j];
        }
        REQUIRE((activity <= row.upper) == (distance <= static_cast<unsigned>(radius)));
      }
      const auto before = row;
      incumbent[0] = std::numeric_limits<double>::quiet_NaN();
      REQUIRE_FALSE(HighsPrimalHeuristics::buildLocalBranchingRow(model, incumbent, 1e-6, radius, row));
      REQUIRE(row.index == before.index);
      REQUIRE(row.value == before.value);
      REQUIRE(row.upper == before.upper);
      incumbent[0] = seed & 1;
    }
  }
}

TEST_CASE("LP round boxes intersect original integer domains and leave auxiliaries free",
          "[highs_primal_heuristic_manager][primal_portfolio]") {
  HighsLp model;
  model.num_col_ = 4;
  model.integrality_ = {HighsVarType::kInteger, HighsVarType::kInteger,
                       HighsVarType::kContinuous, HighsVarType::kImplicitInteger};
  std::vector<double> lower{-3, 0, 0, 0}, upper{3, 1, 100, 100};
  double rate;
  REQUIRE(HighsPrimalHeuristics::buildRensNeighbourhood(
      model, {-1.25, 1, 4.5, 0.5}, 1e-6, lower, upper, rate));
  REQUIRE(lower == std::vector<double>{-2, 1, 0, 0});
  REQUIRE(upper == std::vector<double>{-1, 1, 100, 100});
  REQUIRE(rate == 0.5);
  auto lo = lower, hi = upper;
  REQUIRE_FALSE(HighsPrimalHeuristics::buildRensNeighbourhood(
      model, {-1.25, 2, 4.5, 0.5}, 1e-6, lower, upper, rate));
  REQUIRE(lower == lo);
  REQUIRE(upper == hi);
}

TEST_CASE("local branching preserves an improving point excluded by LP agreement",
          "[highs_primal_heuristic_manager][primal_portfolio]") {
  HighsLp model;
  model.num_col_ = 2;
  model.num_row_ = 1;
  model.col_lower_ = {0, 0};
  model.col_upper_ = {1, 1};
  model.col_cost_ = {1, 0.75};
  model.integrality_.assign(2, HighsVarType::kInteger);
  model.row_lower_ = {1};
  model.row_upper_ = {kHighsInf};
  model.a_matrix_.num_col_ = 2;
  model.a_matrix_.num_row_ = 1;
  model.a_matrix_.start_ = {0, 1, 2};
  model.a_matrix_.index_ = {0, 0};
  model.a_matrix_.value_ = {2, 1};
  auto lower = model.col_lower_, upper = model.col_upper_;
  double rate;
  REQUIRE(HighsPrimalHeuristics::buildRinsNeighbourhood(model, {0.5, 0}, {1, 0},
      1e-6, 0, lower, upper, rate));
  REQUIRE(upper[1] == 0);
  Highs restricted;
  restricted.setOptionValue("output_flag", false);
  auto rins_model = model;
  rins_model.col_lower_ = lower;
  rins_model.col_upper_ = upper;
  REQUIRE(restricted.passModel(rins_model) == HighsStatus::kOk);
  REQUIRE(restricted.run() == HighsStatus::kOk);
  REQUIRE(restricted.getObjectiveValue() == Approx(1));
  HighsLocalBranchingRow row;
  REQUIRE(HighsPrimalHeuristics::buildLocalBranchingRow(model, {1, 0}, 1e-6, 2, row));
  HighsSparseMatrix added;
  added.format_ = MatrixFormat::kRowwise;
  added.num_row_ = 1;
  added.num_col_ = 2;
  added.start_ = {0, 2};
  added.index_ = row.index;
  added.value_ = row.value;
  model.a_matrix_.addRows(added);
  ++model.num_row_;
  model.row_lower_.push_back(-kHighsInf);
  model.row_upper_.push_back(row.upper);
  Highs child;
  child.setOptionValue("output_flag", false);
  REQUIRE(child.passModel(model) == HighsStatus::kOk);
  REQUIRE(child.run() == HighsStatus::kOk);
  REQUIRE(child.getObjectiveValue() == Approx(0.75));
  REQUIRE(child.getSolution().col_value == std::vector<double>{0, 1});
}

TEST_CASE("attribution does not reward a peer or unqualified bound",
          "[highs_primal_heuristic_manager][primal_portfolio]") {
  HighsPrimalHeuristicOutcome outcome;
  outcome.admission.measured = outcome.admission.improved = true;
  outcome.admission.before = 100;
  outcome.admission.after = 60;
  outcome.admission.candidate = 90;
  outcome.parent_lower_bound = 50;
  REQUIRE(HighsPrimalHeuristicManager::attributedReward(outcome) == Approx(0.2));
  outcome.lp_iterations = 100000;
  REQUIRE(HighsPrimalHeuristicManager::attributedReward(outcome) == Approx(0.1));
  outcome.admission.improved = false;
  REQUIRE(HighsPrimalHeuristicManager::attributedReward(outcome) == 0);
  outcome.parent_lower_bound = -kHighsInf;
  REQUIRE(HighsPrimalHeuristicManager::attributedReward(outcome) == -1);
}

TEST_CASE("checked snapshot receipt uses the exact publication not a later peer",
          "[highs_primal_heuristic_manager][primal_portfolio]") {
  HighsLp model;
  model.num_col_ = 2;
  model.num_row_ = 0;
  model.col_cost_ = {1, 0.75};
  model.col_lower_ = {0, 0};
  model.col_upper_ = {1, 1};
  model.integrality_.assign(2, HighsVarType::kInteger);
  model.a_matrix_.num_col_ = 2;
  model.a_matrix_.start_ = {0, 0, 0};
  HighsPrimalSnapshotStore store(model, model, 1e-6, 1, 2);
  HighsWorkerCandidate candidate;
  candidate.metadata.admission_generation = 1;
  candidate.metadata.work_epoch = 2;
  candidate.solution = {1, 0};
  HighsPrimalAdmissionReceipt receipt;
  REQUIRE(store.publishChecked(candidate, candidate.solution, &receipt) == HighsPrimalPublication::kImproved);
  candidate.solution = {0, 1};
  REQUIRE(store.publishChecked(candidate, candidate.solution, &receipt) == HighsPrimalPublication::kImproved);
  REQUIRE(receipt.before == 1);
  REQUIRE(receipt.after == 0.75);
  candidate.solution = {1, 0};
  REQUIRE(store.publishChecked(candidate, candidate.solution, &receipt) == HighsPrimalPublication::kNoImprovement);
  REQUIRE_FALSE(receipt.improved);
  REQUIRE(receipt.before == 0.75);
  REQUIRE(receipt.after == 0.75);
  candidate.metadata.work_epoch = 3;
  REQUIRE(store.publishChecked(candidate, candidate.solution, &receipt) == HighsPrimalPublication::kStale);
  REQUIRE_FALSE(receipt.measured);
}

TEST_CASE("portfolio concurrent callers cannot double reserve a helper",
          "[highs_primal_heuristic_manager][primal_portfolio]") {
  HighsPrimalHeuristicManager manager;
  HighsPrimalHeuristicManagerConfig config;
  config.schedule = HighsPrimalHeuristicSchedule::kPortfolio;
  manager.initialise(config);
  std::atomic<int> ready{0}, granted{0};
  std::atomic<bool> release{false};
  std::vector<std::thread> threads;
  for (int worker = 0; worker < 8; ++worker) {
    threads.emplace_back([&, worker]() {
      HighsPrimalHeuristicContext context;
      context.method = HighsPrimalHeuristicMethod::kRens;
      context.has_lp_solution = true;
      context.worker_id = worker;
      context.restriction_identity = {double(worker), 1};
      const auto decision = manager.before(context);
      if (decision.run) ++granted;
      ++ready;
      while (!release.load()) std::this_thread::yield();
      if (decision.run) {
        HighsPrimalHeuristicReservation reservation(manager, context, decision);
        HighsPrimalHeuristicOutcome outcome;
        outcome.lp_iterations = 42;
        reservation.finish(outcome);
      }
    });
  }
  while (ready.load() != 8) std::this_thread::yield();
  const int grants = granted.load();
  release = true;
  for (auto& thread : threads) thread.join();
  REQUIRE(grants == 1);
  REQUIRE(manager.integrityErrors() == 0);
  REQUIRE(manager.account(HighsPrimalHeuristicMethod::kRens).lp_iterations == 42);
}

TEST_CASE("portfolio end-to-end small MIP preserves the independent optimum",
          "[highs_primal_heuristic_manager][primal_portfolio]") {
  // Three odd-cycle vertex covers; exact enumeration is independent of LP/MIP.
  HighsLp model;
  model.num_col_ = model.num_row_ = 15;
  model.col_lower_.assign(15, 0);
  model.col_upper_.assign(15, 1);
  model.col_cost_.assign(15, 1);
  model.integrality_.assign(15, HighsVarType::kInteger);
  model.row_lower_.assign(15, 1);
  model.row_upper_.assign(15, kHighsInf);
  model.a_matrix_.num_col_ = model.a_matrix_.num_row_ = 15;
  model.a_matrix_.start_ = {0};
  for (int col = 0; col < 15; ++col) {
    model.a_matrix_.index_.push_back(col);
    model.a_matrix_.index_.push_back(5 * (col / 5) + (col + 4) % 5);
    model.a_matrix_.value_.insert(model.a_matrix_.value_.end(), {1, 1});
    model.a_matrix_.start_.push_back(2 * (col + 1));
  }
  double best = kHighsInf;
  for (unsigned point = 0; point < (1u << 15); ++point) {
    bool feasible = true;
    unsigned cost = 0;
    for (unsigned col = 0; col < 15; ++col) {
      const unsigned next = 5 * (col / 5) + (col + 1) % 5;
      cost += (point >> col) & 1;
      feasible &= ((point >> col) & 1) || ((point >> next) & 1);
    }
    if (feasible) best = std::min(best, double(cost));
  }
  for (const char* schedule : {"portfolio", "portfolio-burst", "portfolio-continuation",
                               "portfolio-deferred-continuation", "portfolio-cost-aware",
                               "portfolio-ucb-success", "portfolio-ucb-cost"}) {
  for (int threads : {1, 4}) {
    Highs::resetGlobalScheduler(true);
    Highs solver;
    std::string log;
    std::mutex log_mutex;
    solver.setCallback([&](int type, const std::string& message,
                          const HighsCallbackOutput*, HighsCallbackInput*, void*) {
      if (type == kCallbackLogging) {
        std::lock_guard<std::mutex> lock(log_mutex);
        log += message;
      }
    });
    REQUIRE(solver.startCallback(kCallbackLogging) == HighsStatus::kOk);
    solver.setOptionValue("log_to_console", false);
    solver.setOptionValue("threads", threads);
    solver.setOptionValue("parallel", "on");
    solver.setOptionValue("presolve", "off");
    REQUIRE(solver.setOptionValue("mip_primal_heuristic_schedule", schedule) == HighsStatus::kOk);
    solver.setOptionValue("mip_root_work_budget", true);
    solver.setOptionValue("mip_root_work_max_separation_rounds", 0);
    solver.setOptionValue("mip_heuristic_run_root_reduced_cost", false);
    solver.setOptionValue("mip_heuristic_run_feasibility_jump", false);
    solver.setOptionValue("mip_heuristic_run_rens", false);
    solver.setOptionValue("mip_heuristic_run_rins", false);
    REQUIRE(solver.passModel(model) == HighsStatus::kOk);
    REQUIRE(solver.getLp().a_matrix_.start_.size() == 16);
    REQUIRE(solver.getLp().a_matrix_.start_.back() == 30);
    HighsSolution seed;
    seed.value_valid = true;
    seed.col_value.assign(15, 1);
    REQUIRE(solver.setSolution(seed) == HighsStatus::kOk);
    REQUIRE(solver.run() == HighsStatus::kOk);
    REQUIRE(solver.getModelStatus() == HighsModelStatus::kOptimal);
    REQUIRE(solver.getObjectiveValue() == Approx(best));
    REQUIRE(highsVerifyCandidate(model, solver.getSolution().col_value, nullptr, 1e-6).feasible);
    REQUIRE(model.num_row_ == 15);
    if (std::string(schedule) != "portfolio") {
      const auto marker = log.find("MIP-Portfolio-Burst:");
      REQUIRE(marker != std::string::npos);
      const auto count = log.find("continuations=", marker);
      REQUIRE(count != std::string::npos);
      REQUIRE(std::stoll(log.substr(count + 14)) > 0);
      if (std::string(schedule) == "portfolio-continuation" ||
          std::string(schedule) == "portfolio-deferred-continuation" ||
          std::string(schedule) == "portfolio-cost-aware" ||
          std::string(schedule).find("portfolio-ucb-") == 0)
        REQUIRE(log.find("MIP-Portfolio-Continuation:") != std::string::npos);
      if (std::string(schedule) == "portfolio-deferred-continuation" ||
          std::string(schedule) == "portfolio-cost-aware" ||
          std::string(schedule).find("portfolio-ucb-") == 0)
        REQUIRE(log.find("MIP-Portfolio-Deferred:") != std::string::npos);
      if (std::string(schedule) == "portfolio-cost-aware")
        REQUIRE(log.find("MIP-Portfolio-Selection:") != std::string::npos);
      if (std::string(schedule).find("portfolio-ucb-") == 0)
        REQUIRE(log.find("MIP-Portfolio-UCB:") != std::string::npos);
    }
    solver.clear();
  }
  }
  Highs::resetGlobalScheduler(true);
}

namespace {
HighsPrimalHeuristicOutcome selectionGain(int64_t lp = 10) {
  HighsPrimalHeuristicOutcome outcome;
  outcome.started = outcome.completed = outcome.candidate_proposed = true;
  outcome.candidate_accepted = outcome.candidate_improved = true;
  outcome.lp_iterations = lp;
  outcome.elapsed = 1;
  outcome.parent_lower_bound = 0;
  outcome.admission.measured = outcome.admission.improved = true;
  outcome.admission.before = 100;
  outcome.admission.after = outcome.admission.candidate = 80;
  return outcome;
}

void recordSelection(HighsPrimalHeuristicManager& manager,
                     HighsPrimalHeuristicMethod method,
                     const HighsPrimalHeuristicOutcome& outcome, int64_t restart,
                     HighsPrimalPortfolioChoice choice = HighsPrimalPortfolioChoice::kUnspecified) {
  HighsPrimalPortfolioBurstScope burst(manager, 0, restart, 0);
  REQUIRE(burst.id() != 0);
  HighsPrimalHeuristicContext context;
  context.method = method;
  context.portfolio_choice = choice;
  context.restart = restart;
  context.burst_id = burst.id();
  context.main_lp_iterations = 1000000;
  context.has_incumbent = context.has_lp_solution = true;
  context.seed_upper_bound = 100;
  context.checked_seed = {1, 0};
  context.restriction_identity = {double(restart), 1};
  const auto decision = manager.before(context);
  REQUIRE(decision.run);
  manager.record(context, outcome, decision.observation_id, decision.reservation_id);
}
}

TEST_CASE("cost selector explores first then favors own measured gain",
          "[highs_primal_heuristic_manager][primal_portfolio_selection]") {
  HighsPrimalHeuristicManager manager;
  HighsPrimalHeuristicManagerConfig config;
  config.schedule = HighsPrimalHeuristicSchedule::kPortfolioCostAware;
  manager.initialise(config);
  std::array<bool, kHighsPrimalHeuristicMethodCount> eligible{};
  for (auto method : {HighsPrimalHeuristicMethod::kRens, HighsPrimalHeuristicMethod::kRins,
                     HighsPrimalHeuristicMethod::kLocalBranching})
    eligible[static_cast<size_t>(method)] = true;
  HighsPrimalPortfolioChoice choice;
  for (auto method : {HighsPrimalHeuristicMethod::kRens, HighsPrimalHeuristicMethod::kRins,
                     HighsPrimalHeuristicMethod::kLocalBranching}) {
    REQUIRE(manager.selectPortfolio(eligible, &choice) == method);
    REQUIRE(choice == HighsPrimalPortfolioChoice::kFirstSweep);
    auto outcome = selectionGain(method == HighsPrimalHeuristicMethod::kLocalBranching ? 50 : 10);
    if (method != HighsPrimalHeuristicMethod::kLocalBranching) {
      outcome.candidate_improved = outcome.admission.improved = false;
      outcome.admission.after = outcome.admission.candidate = 100;
    }
    recordSelection(manager, method, outcome, int64_t(method), choice);
  }
  for (int call = 0; call < 8; ++call) {
    const bool exploration = call % 4 == 3;
    const auto expected = exploration
        ? (call == 3 ? HighsPrimalHeuristicMethod::kRens : HighsPrimalHeuristicMethod::kRins)
        : HighsPrimalHeuristicMethod::kLocalBranching;
    // Querying without a reservation cannot advance the exploration cadence.
    for (int query = 0; query < 3; ++query)
      REQUIRE(manager.selectPortfolio(eligible, &choice) == expected);
    REQUIRE(choice == (exploration ? HighsPrimalPortfolioChoice::kExplore
                                   : HighsPrimalPortfolioChoice::kOwnGain));
    auto outcome = selectionGain(exploration ? 0 : 50);
    if (exploration) {
      outcome.candidate_improved = outcome.admission.improved = false;
      outcome.admission.after = outcome.admission.candidate = 100;
    }
    recordSelection(manager, expected, outcome, 10 + call, choice);
    // Forced continuation records work but consumes no competitive slot.
    auto forced = eligible;
    forced[static_cast<size_t>(HighsPrimalHeuristicMethod::kRens)] = false;
    forced[static_cast<size_t>(HighsPrimalHeuristicMethod::kRins)] = false;
    REQUIRE(manager.selectPortfolio(forced, &choice) == HighsPrimalHeuristicMethod::kLocalBranching);
    REQUIRE(choice == HighsPrimalPortfolioChoice::kForced);
    recordSelection(manager, HighsPrimalHeuristicMethod::kLocalBranching,
                    selectionGain(), 30 + call, choice);
  }
  REQUIRE(manager.account(HighsPrimalHeuristicMethod::kRens).selection_explorations == 1);
  REQUIRE(manager.account(HighsPrimalHeuristicMethod::kRins).selection_explorations == 1);
  REQUIRE(manager.account(HighsPrimalHeuristicMethod::kLocalBranching).selection_reuses == 6);
  REQUIRE(manager.integrityErrors() == 0);
  manager.initialise(config);
  REQUIRE(manager.account(HighsPrimalHeuristicMethod::kLocalBranching).selection_samples == 0);
}

TEST_CASE("UCB selection matches an independent bounded-reward oracle",
          "[highs_primal_heuristic_manager][primal_portfolio_ucb]") {
  const std::array<HighsPrimalHeuristicMethod, 3> methods = {
      HighsPrimalHeuristicMethod::kRens, HighsPrimalHeuristicMethod::kRins,
      HighsPrimalHeuristicMethod::kLocalBranching};
  for (const auto schedule : {HighsPrimalHeuristicSchedule::kPortfolioUcbSuccess,
                              HighsPrimalHeuristicSchedule::kPortfolioUcbCost}) {
    HighsPrimalHeuristicManager manager;
    HighsPrimalHeuristicManagerConfig config;
    config.schedule = schedule;
    manager.initialise(config);
    std::array<bool, kHighsPrimalHeuristicMethodCount> eligible{};
    std::array<int64_t, 3> counts{};
    std::array<double, 3> rewards{};
    for (auto method : methods) eligible[size_t(method)] = true;
    for (int call = 0; call < 60; ++call) {
      size_t expected = 0;
      HighsPrimalPortfolioChoice expected_choice = HighsPrimalPortfolioChoice::kFirstSweep;
      double maximum = -1;
      int64_t total = counts[0] + counts[1] + counts[2];
      for (size_t index = 0; index < 3; ++index) {
        if (!counts[index]) { expected = index; break; }
        const double score = rewards[index] / counts[index] +
            std::sqrt(2.0 * std::log(double(total)) / counts[index]);
        if (score > maximum) { maximum = score; expected = index; }
        if (index == 2) expected_choice = HighsPrimalPortfolioChoice::kUcb;
      }
      for (int query = 0; query < 3; ++query) {
        HighsPrimalPortfolioChoice choice;
        REQUIRE(manager.selectPortfolio(eligible, &choice) == methods[expected]);
        REQUIRE(choice == expected_choice);
      }
      auto outcome = selectionGain();
      const bool improved = expected == 2 && call % 3 != 0;
      if (!improved) {
        outcome.candidate_improved = outcome.admission.improved = false;
        outcome.admission.after = outcome.admission.candidate = 100;
      }
      recordSelection(manager, methods[expected], outcome, call, expected_choice);
      ++counts[expected];
      if (improved) rewards[expected] +=
          schedule == HighsPrimalHeuristicSchedule::kPortfolioUcbSuccess ? 1 : 0.2 / 1.05;
      const auto account = manager.account(methods[expected]);
      REQUIRE(account.selection_samples == counts[expected]);
      REQUIRE(account.selection_reward == Approx(rewards[expected]));
      REQUIRE(account.selection_explorations == 0);
    }
    // Forced continuation cannot advance the competitive reuse counter.
    eligible[size_t(methods[0])] = eligible[size_t(methods[1])] = false;
    HighsPrimalPortfolioChoice choice;
    REQUIRE(manager.selectPortfolio(eligible, &choice) == methods[2]);
    REQUIRE(choice == HighsPrimalPortfolioChoice::kForced);
    const auto before = manager.account(methods[2]);
    recordSelection(manager, methods[2], selectionGain(), 100, choice);
    REQUIRE(manager.account(methods[2]).selection_reuses == before.selection_reuses);
    eligible.fill(false);
    REQUIRE(manager.selectPortfolio(eligible) == HighsPrimalHeuristicMethod::kCount);
    REQUIRE(manager.integrityErrors() == 0);
    manager.initialise(config);
    REQUIRE(manager.account(methods[2]).selection_samples == 0);
  }
}

TEST_CASE("UCB trials reject unstarted unfinished and unrelated rewards",
          "[highs_primal_heuristic_manager][primal_portfolio_ucb]") {
  for (const auto schedule : {HighsPrimalHeuristicSchedule::kPortfolioUcbSuccess,
                              HighsPrimalHeuristicSchedule::kPortfolioUcbCost}) {
    for (int invalid = 0; invalid < 11; ++invalid) {
      CAPTURE(invalid, int(schedule));
      HighsPrimalHeuristicManager manager;
      HighsPrimalHeuristicManagerConfig config;
      config.schedule = schedule;
      manager.initialise(config);
      auto outcome = selectionGain();
      if (invalid == 0) outcome.started = false;
      if (invalid == 1) outcome.completed = false;
      if (invalid == 2) outcome.candidate_accepted = false;
      if (invalid == 3) outcome.admission.candidate = 90;
      if (invalid == 4) outcome.admission.before = kHighsInf;
      if (invalid == 5) outcome.admission.after = std::numeric_limits<double>::quiet_NaN();
      if (invalid == 6) outcome.preparation_stop_mask = kHighsHeuristicStopUnfinishedLp;
      if (invalid == 7) outcome.elapsed = -1;
      if (invalid == 8) outcome.lp_iterations = -1;
      if (invalid == 9) outcome.parent_lower_bound = 90;
      if (invalid == 10) outcome.admission.measured = false;
      recordSelection(manager, HighsPrimalHeuristicMethod::kLocalBranching, outcome, 0);
      const auto account = manager.account(HighsPrimalHeuristicMethod::kLocalBranching);
      REQUIRE(account.selection_reward == 0);
      REQUIRE(account.selection_samples == ((invalid < 2 || invalid == 7 || invalid == 8) ? 0 : 1));
      REQUIRE(manager.integrityErrors() == ((invalid == 7 || invalid == 8) ? 1 : 0));
    }
    HighsPrimalHeuristicManager manager;
    HighsPrimalHeuristicManagerConfig config;
    config.schedule = schedule;
    manager.initialise(config);
    for (int call = 0; call < 4; ++call)
      recordSelection(manager, HighsPrimalHeuristicMethod::kLocalBranching,
                      selectionGain(100000), call, HighsPrimalPortfolioChoice::kUcb);
    REQUIRE_FALSE(manager.portfolioOpportunityAvailable(1000000, 100));
    REQUIRE(manager.account(HighsPrimalHeuristicMethod::kLocalBranching).lp_iterations == 400000);
  }
}

TEST_CASE("selection gives no credit for ties peer gains or invalid admission",
          "[highs_primal_heuristic_manager][primal_portfolio_selection]") {
  for (int invalid = 0; invalid < 14; ++invalid) {
    CAPTURE(invalid);
    HighsPrimalHeuristicManager manager;
    HighsPrimalHeuristicManagerConfig config;
    config.schedule = HighsPrimalHeuristicSchedule::kPortfolioCostAware;
    manager.initialise(config);
    auto outcome = selectionGain();
    if (invalid == 0) outcome.started = false;
    if (invalid == 1) outcome.completed = false;
    if (invalid == 2) outcome.candidate_accepted = false;
    if (invalid == 3) outcome.candidate_proposed = false;
    if (invalid == 4) outcome.admission.measured = false;
    if (invalid == 5) outcome.admission.improved = false;
    if (invalid == 6) outcome.admission.candidate = 90;
    if (invalid == 7) outcome.admission.before = kHighsInf;
    if (invalid == 8) outcome.admission.after = std::numeric_limits<double>::quiet_NaN();
    if (invalid == 9) outcome.parent_lower_bound = 100;
    if (invalid == 10) outcome.parent_lower_bound = 90;
    if (invalid == 11) outcome.candidate_improved = false;
    if (invalid == 12) outcome.preparation_stop_mask = kHighsHeuristicStopCheckpoint;
    if (invalid == 13) outcome.preparation_stop_mask = kHighsHeuristicStopUnfinishedLp;
    recordSelection(manager, HighsPrimalHeuristicMethod::kLocalBranching, outcome, 0);
    const auto account = manager.account(HighsPrimalHeuristicMethod::kLocalBranching);
    REQUIRE(account.selection_reward == 0);
    REQUIRE(account.selection_samples == 1);
    REQUIRE(account.lp_iterations == 10);
  }
}

TEST_CASE("cost selection cannot renew cumulative work after restart or denial",
          "[highs_primal_heuristic_manager][primal_portfolio_selection]") {
  HighsPrimalHeuristicManager manager;
  HighsPrimalHeuristicManagerConfig config;
  config.schedule = HighsPrimalHeuristicSchedule::kPortfolioCostAware;
  manager.initialise(config);
  for (int call = 0; call < 4; ++call)
    recordSelection(manager, HighsPrimalHeuristicMethod::kLocalBranching,
                    selectionGain(100000), call, HighsPrimalPortfolioChoice::kOwnGain);
  REQUIRE_FALSE(manager.portfolioOpportunityAvailable(1000000, 10));
  HighsPrimalPortfolioBurstScope burst(manager, 0, 10, 0);
  HighsPrimalHeuristicContext context;
  context.method = HighsPrimalHeuristicMethod::kLocalBranching;
  context.has_incumbent = true;
  context.burst_id = burst.id();
  context.restart = 10;
  context.main_lp_iterations = 1000000;
  context.restriction_identity = {0, 1};
  context.portfolio_choice = HighsPrimalPortfolioChoice::kExplore;
  REQUIRE(manager.before(context).reason == HighsPrimalHeuristicDecisionReason::kLpWorkLimit);
  const auto account = manager.account(context.method);
  REQUIRE(account.lp_iterations == 400000);
  REQUIRE(account.selection_reuses == 4);
  REQUIRE(account.selection_explorations == 0);
}

TEST_CASE("legacy selection ignores reward and invalid work cannot earn credit",
          "[highs_primal_heuristic_manager][primal_portfolio_selection]") {
  HighsPrimalHeuristicManager manager;
  HighsPrimalHeuristicManagerConfig config;
  config.schedule = HighsPrimalHeuristicSchedule::kPortfolioDeferredContinuation;
  manager.initialise(config);
  std::array<bool, kHighsPrimalHeuristicMethodCount> eligible{};
  for (auto method : {HighsPrimalHeuristicMethod::kRens, HighsPrimalHeuristicMethod::kRins,
                     HighsPrimalHeuristicMethod::kLocalBranching}) {
    eligible[static_cast<size_t>(method)] = true;
    recordSelection(manager, method, selectionGain(int64_t(method) * 10), int64_t(method));
  }
  REQUIRE(manager.selectPortfolio(eligible) == HighsPrimalHeuristicMethod::kRens);
  REQUIRE(manager.account(HighsPrimalHeuristicMethod::kLocalBranching).reward > 0);
  REQUIRE(manager.account(HighsPrimalHeuristicMethod::kLocalBranching).selection_samples == 0);
  for (int invalid = 0; invalid < 4; ++invalid) {
    CAPTURE(invalid);
    config.schedule = HighsPrimalHeuristicSchedule::kPortfolioCostAware;
    manager.initialise(config);
    auto outcome = selectionGain();
    if (invalid == 0) outcome.lp_iterations = -1;
    if (invalid == 1) outcome.elapsed = -1;
    if (invalid == 2) outcome.elapsed = std::numeric_limits<double>::infinity();
    if (invalid == 3) outcome.elapsed = std::numeric_limits<double>::quiet_NaN();
    recordSelection(manager, HighsPrimalHeuristicMethod::kLocalBranching, outcome, 0);
    const auto account = manager.account(HighsPrimalHeuristicMethod::kLocalBranching);
    REQUIRE(account.selection_reward == 0);
    REQUIRE(account.selection_samples == 0);
    REQUIRE(manager.integrityErrors() == 1);
  }
}

TEST_CASE("selection reward is invariant under objective units and offset",
          "[highs_primal_heuristic_manager][primal_portfolio_selection]") {
  double reference = -1;
  for (double scale : {0.125, 1.0, 16.0}) {
    for (double offset : {-1000.0, 0.0, 1000.0}) {
      HighsPrimalHeuristicManager manager;
      HighsPrimalHeuristicManagerConfig config;
      config.schedule = HighsPrimalHeuristicSchedule::kPortfolioCostAware;
      manager.initialise(config);
      auto outcome = selectionGain();
      outcome.admission.before = 100 * scale + offset;
      outcome.admission.after = outcome.admission.candidate = 80 * scale + offset;
      outcome.parent_lower_bound = offset;
      recordSelection(manager, HighsPrimalHeuristicMethod::kLocalBranching, outcome, 0);
      const double reward = manager.account(HighsPrimalHeuristicMethod::kLocalBranching).selection_reward;
      REQUIRE(reward > 0);
      if (reference < 0) reference = reward;
      REQUIRE(reward == Approx(reference));
    }
  }
}

TEST_CASE("portfolio burst owns finite cumulative work and refreshed checked gains",
          "[highs_primal_heuristic_manager][primal_portfolio_burst]") {
  HighsPrimalHeuristicManager manager;
  HighsPrimalHeuristicManagerConfig config;
  config.schedule = HighsPrimalHeuristicSchedule::kPortfolioBurst;
  config.portfolio_burst_calls = 3;
  config.portfolio_burst_lp = 100;
  config.portfolio_burst_time = 10;
  manager.initialise(config);
  const auto id = manager.beginPortfolioBurst(1, 2, 5);
  REQUIRE(id != 0);
  REQUIRE(manager.beginPortfolioBurst(3, 2, 5) == 0);
  HighsPrimalHeuristicContext context;
  context.method = HighsPrimalHeuristicMethod::kRens;
  context.has_lp_solution = true;
  context.worker_id = 1;
  context.restart = 2;
  context.burst_id = id;
  context.time = 5;
  context.restriction_identity = {0, 1};
  auto decision = manager.before(context);
  REQUIRE(decision.run);
  REQUIRE(decision.budget.max_time == 10);
  REQUIRE(decision.budget.max_lp_iterations == 100);
  HighsPrimalHeuristicOutcome gain;
  gain.started = gain.candidate_accepted = true;
  gain.admission.measured = gain.admission.improved = true;
  gain.admission.before = kHighsInf;
  gain.admission.after = gain.admission.candidate = 20;
  gain.lp_iterations = 40;
  manager.record(context, gain, 0, decision.reservation_id);
  REQUIRE_FALSE(manager.before(context).run); // no new child without a gain permit
  auto other = context;
  other.worker_id = 3;
  REQUIRE_FALSE(manager.before(other).run);
  REQUIRE_FALSE(manager.continuePortfolioBurst(id, 1, 3, 1, 20, 7));
  REQUIRE_FALSE(manager.continuePortfolioBurst(id, 1, 2, 0, 20, 7));
  REQUIRE_FALSE(manager.continuePortfolioBurst(id, 1, 2, 1, 19, 7)); // peer's better seed
  REQUIRE(manager.continuePortfolioBurst(id, 1, 2, 1, 20, 7));
  REQUIRE_FALSE(manager.continuePortfolioBurst(id, 1, 2, 1, 20, 7)); // consumed permit
  context.method = HighsPrimalHeuristicMethod::kLocalBranching;
  context.has_incumbent = true;
  context.incumbent_sequence = 1;
  context.seed_upper_bound = 20;
  context.time = 7;
  context.restriction_identity = {1, 0};
  other = context;
  other.incumbent_sequence = 2;
  REQUIRE_FALSE(manager.before(other).run); // changed between permit and launch
  decision = manager.before(context);
  REQUIRE(decision.run);
  REQUIRE(decision.budget.max_time == 8); // shared deadline, not renewed
  REQUIRE(decision.budget.max_lp_iterations == 60);
  gain.admission.before = 20;
  gain.admission.after = gain.admission.candidate = 18;
  gain.lp_iterations = 70; // a cooperative overshoot consumes the whole remaining burst
  manager.record(context, gain, 0, decision.reservation_id);
  REQUIRE_FALSE(manager.continuePortfolioBurst(id, 1, 2, 2, 18, 8));
  REQUIRE(manager.burstAccount().budget_exhausted == 1);
  manager.endPortfolioBurst(id);
  REQUIRE(manager.integrityErrors() == 0);
  const auto next = manager.beginPortfolioBurst(3, 2, 20);
  REQUIRE(next != id);
  manager.endPortfolioBurst(id); // stale token cannot release another owner's burst
  REQUIRE(manager.beginPortfolioBurst(4, 2, 20) == 0);
  manager.endPortfolioBurst(next);
  REQUIRE(manager.integrityErrors() == 1);
}

TEST_CASE("portfolio burst fails closed on no own gain, deadline, call cap and cancellation",
          "[highs_primal_heuristic_manager][primal_portfolio_burst]") {
  for (int variant = 0; variant < 9; ++variant) {
    HighsPrimalHeuristicManager manager;
    HighsPrimalHeuristicManagerConfig config;
    config.schedule = HighsPrimalHeuristicSchedule::kPortfolioBurst;
    if (variant == 7) config.portfolio_burst_calls = 1;
    manager.initialise(config);
    HighsPrimalPortfolioBurstScope burst(manager, 0, 0, 0);
    HighsPrimalHeuristicContext context;
    context.burst_id = burst.id();
    context.method = HighsPrimalHeuristicMethod::kLocalBranching;
    context.has_incumbent = true;
    context.restriction_identity = {0, 1};
    const auto decision = manager.before(context);
    REQUIRE(decision.run);
    {
      HighsPrimalHeuristicReservation reservation(manager, context, decision);
      if (variant != 8) {
        HighsPrimalHeuristicOutcome gain;
        gain.started = gain.candidate_accepted = true;
        gain.admission.measured = gain.admission.improved = true;
        gain.admission.before = 20;
        gain.admission.after = gain.admission.candidate = 18;
        if (variant == 0) gain.admission.improved = false;
        if (variant == 1) gain.admission.measured = false;
        if (variant == 2) gain.candidate_accepted = false;
        if (variant == 3) gain.admission.candidate = 19;
        if (variant == 4) gain.admission.after = std::numeric_limits<double>::quiet_NaN();
        if (variant == 5) gain.preparation_stop_mask = kHighsHeuristicStopCheckpoint;
        reservation.finish(gain);
      }
    }
    REQUIRE_FALSE(manager.continuePortfolioBurst(burst.id(), 0, 0, 1, 18,
                                                 variant == 6 ? 20 : 1));
    REQUIRE(manager.integrityErrors() == 0);
  }
}

TEST_CASE("sixth checked gain can reserve a finite experimental seventh child",
          "[highs_primal_heuristic_manager][primal_portfolio_continuation]") {
  for (auto schedule : {HighsPrimalHeuristicSchedule::kPortfolioBurst,
                        HighsPrimalHeuristicSchedule::kPortfolioContinuation}) {
    HighsPrimalHeuristicManager manager;
    HighsPrimalHeuristicManagerConfig config;
    config.schedule = schedule;
    config.portfolio_extra_continuations = 3;
    config.trace_decisions = true;
    manager.initialise(config);
    HighsPrimalHeuristicContext context;
    context.method = HighsPrimalHeuristicMethod::kLocalBranching;
    context.has_incumbent = true;
    context.worker_id = 2;
    context.main_lp_iterations = 1000000;
    for (int call = 0; call < 6; ++call) {
      context.time = 100 + 30 * call;
      context.burst_id = manager.beginPortfolioBurst(2, 0, context.time);
      context.incumbent_sequence = call;
      context.seed_upper_bound = 625805630 + (6 - call) * 1000000;
      context.restriction_identity = {double(call), 1};
      REQUIRE(context.burst_id != 0);
      const auto decision = manager.before(context);
      REQUIRE(decision.run);
      REQUIRE_FALSE(decision.productive_continuation);
      HighsPrimalHeuristicOutcome gain;
      gain.started = gain.completed = gain.candidate_accepted = true;
      gain.admission.measured = gain.admission.improved = true;
      gain.admission.before = context.seed_upper_bound;
      gain.admission.after = gain.admission.candidate = context.seed_upper_bound - 1000000;
      gain.lp_iterations = call == 5 ? 10810 : 50000;
      gain.elapsed = 1;
      manager.record(context, gain, decision.observation_id, decision.reservation_id);
      if (call != 5) manager.endPortfolioBurst(context.burst_id);
    }
    REQUIRE(manager.account(context.method).lp_iterations == 260810);
    REQUIRE(manager.continuePortfolioBurst(context.burst_id, 2, 0, 6, 625805630, 251));
    context.time = 251;
    context.incumbent_sequence = 6;
    context.seed_upper_bound = 625805630;
    context.restriction_identity = {6, 1};
    const auto seventh = manager.before(context);
    if (schedule == HighsPrimalHeuristicSchedule::kPortfolioBurst) {
      REQUIRE_FALSE(seventh.run);
      REQUIRE(seventh.reason == HighsPrimalHeuristicDecisionReason::kCallLimit);
      REQUIRE(manager.burstAccount().extra_reserved == 0);
    } else {
      REQUIRE(seventh.run);
      REQUIRE(seventh.productive_continuation);
      REQUIRE(seventh.budget.max_lp_iterations == 89190);
      REQUIRE(seventh.budget.max_time == 19);
      REQUIRE(manager.burstAccount().extra_reserved == 1);
      REQUIRE(manager.burstAccount().extra_started == 0);
      auto other = context;
      other.worker_id = 3;
      REQUIRE_FALSE(manager.before(other).run);
      HighsPrimalHeuristicOutcome gain;
      gain.started = gain.completed = gain.candidate_accepted = true;
      gain.admission.measured = gain.admission.improved = true;
      gain.admission.before = 625805630;
      gain.admission.after = gain.admission.candidate = 610481930;
      gain.lp_iterations = 15935;
      gain.elapsed = 14.3;
      manager.record(context, gain, seventh.observation_id, seventh.reservation_id);
      REQUIRE(manager.account(context.method).lp_iterations == 276745);
      REQUIRE(manager.burstAccount().extra_started == 1);
      REQUIRE(manager.burstAccount().extra_completed == 1);
      REQUIRE(manager.burstAccount().extra_lp_iterations == 15935);
      const auto errors = manager.integrityErrors();
      manager.record(context, gain, seventh.observation_id, seventh.reservation_id);
      REQUIRE(manager.integrityErrors() == errors + 1);
      REQUIRE(manager.burstAccount().extra_lp_iterations == 15935);
    }
    manager.endPortfolioBurst(context.burst_id);
  }
}

TEST_CASE("finite continuation state machine preserves caps across restarts",
          "[highs_primal_heuristic_manager][primal_portfolio_continuation]") {
  for (int ordinary = 1; ordinary <= 6; ++ordinary) {
  for (int extra = 0; extra <= 3; ++extra) {
    CAPTURE(ordinary, extra);
    HighsPrimalHeuristicManager manager;
    HighsPrimalHeuristicManagerConfig config;
    config.schedule = HighsPrimalHeuristicSchedule::kPortfolioContinuation;
    config.portfolio_max_calls = ordinary;
    config.portfolio_extra_continuations = extra;
    manager.initialise(config);
    int64_t sequence = 0, ordinary_reserved = 0;
    double upper = 1000;
    for (int restart = 0; restart < 3; ++restart) {
      for (int wave = 0; wave < ordinary; ++wave) {
        HighsPrimalPortfolioBurstScope burst(manager, 0, restart, sequence * 30);
        REQUIRE(burst.id() != 0);
        const double begin = sequence * 30;
        // Leave the final ordinary call inside a live burst; a fully ended
        // burst intentionally cannot carry a gain permit into a new wave.
        const int steps = wave == ordinary - 1 ? 3 : 1;
        for (int step = 0; step < steps; ++step) {
          HighsPrimalHeuristicContext context;
          context.method = HighsPrimalHeuristicMethod::kLocalBranching;
          context.has_incumbent = true;
          context.burst_id = burst.id();
          context.restart = restart;
          context.incumbent_sequence = sequence;
          context.seed_upper_bound = upper;
          context.time = begin + step;
          context.main_lp_iterations = 1000000;
          context.restriction_identity = {double(sequence), 1};
          const auto decision = manager.before(context);
          if (!decision.run) break;
          ordinary_reserved += !decision.productive_continuation;
          HighsPrimalHeuristicOutcome gain;
          gain.started = gain.completed = gain.candidate_accepted = true;
          gain.admission.measured = gain.admission.improved = true;
          gain.admission.before = upper;
          gain.admission.after = gain.admission.candidate = --upper;
          gain.lp_iterations = 10;
          manager.record(context, gain, 0, decision.reservation_id);
          ++sequence;
          REQUIRE(manager.burstAccount().extra_reserved <= extra);
          if (!manager.continuePortfolioBurst(burst.id(), 0, restart,
                                               sequence, upper, begin + step + 1)) break;
        }
      }
    }
    const auto ledger = manager.burstAccount();
    REQUIRE(ordinary_reserved <= 3 * ordinary);
    REQUIRE(ledger.extra_reserved == extra);
    REQUIRE(ledger.extra_started == extra);
    REQUIRE(ledger.extra_completed == extra);
    REQUIRE(ledger.extra_lp_iterations == 10 * extra);
    REQUIRE(manager.account(HighsPrimalHeuristicMethod::kLocalBranching).lp_iterations == 10 * sequence);
    REQUIRE(manager.integrityErrors() == 0);
  }
  }
}

TEST_CASE("continuation does not renew work after cancellation or overshoot",
          "[highs_primal_heuristic_manager][primal_portfolio_continuation]") {
  for (bool overshoot : {false, true}) {
    HighsPrimalHeuristicManager manager;
    HighsPrimalHeuristicManagerConfig config;
    config.schedule = HighsPrimalHeuristicSchedule::kPortfolioContinuation;
    config.portfolio_max_calls = 1;
    config.portfolio_extra_continuations = 3;
    config.portfolio_lp_cap = config.portfolio_bootstrap_lp = 100;
    config.portfolio_lp_per_call = 100;
    config.portfolio_burst_lp = 100;
    manager.initialise(config);
    HighsPrimalHeuristicContext context;
    context.method = HighsPrimalHeuristicMethod::kLocalBranching;
    context.has_incumbent = true;
    context.seed_upper_bound = 20;
    context.restriction_identity = {0, 1};
    context.burst_id = manager.beginPortfolioBurst(0, 0, 0);
    auto decision = manager.before(context);
    REQUIRE(decision.run);
    HighsPrimalHeuristicOutcome gain;
    gain.started = gain.completed = gain.candidate_accepted = true;
    gain.admission.measured = gain.admission.improved = true;
    gain.admission.before = 20;
    gain.admission.after = gain.admission.candidate = 18;
    gain.lp_iterations = 40;
    manager.record(context, gain, 0, decision.reservation_id);
    REQUIRE(manager.continuePortfolioBurst(context.burst_id, 0, 0, 1, 18, 1));
    context.incumbent_sequence = 1;
    context.seed_upper_bound = 18;
    context.time = 1;
    context.restriction_identity = {1, 0};
    decision = manager.before(context);
    REQUIRE(decision.productive_continuation);
    REQUIRE(decision.budget.max_lp_iterations == 60);
    {
      HighsPrimalHeuristicReservation reservation(manager, context, decision);
      if (overshoot) {
        gain.admission.before = 18;
        gain.admission.after = gain.admission.candidate = 16;
        gain.lp_iterations = 70;
        reservation.finish(gain);
      }
    }
    REQUIRE(manager.burstAccount().extra_reserved == 1);
    REQUIRE(manager.burstAccount().extra_started == (overshoot ? 1 : 0));
    REQUIRE(manager.burstAccount().extra_lp_iterations == (overshoot ? 70 : 60));
    REQUIRE_FALSE(manager.continuePortfolioBurst(context.burst_id, 0, 0, 2, 16, 2));
    manager.endPortfolioBurst(context.burst_id);
    REQUIRE_FALSE(manager.portfolioOpportunityAvailable(1000000, 1));
    context.burst_id = manager.beginPortfolioBurst(0, 1, 30);
    context.restart = 1;
    context.time = 30;
    context.restriction_identity = {2, 0};
    REQUIRE_FALSE(manager.before(context).run);
    manager.endPortfolioBurst(context.burst_id);
    REQUIRE(manager.integrityErrors() == 0);
  }
}

TEST_CASE("experimental continuation rejects stale ownership and unowned gain",
          "[highs_primal_heuristic_manager][primal_portfolio_continuation]") {
  for (int variant = 0; variant < 11; ++variant) {
    CAPTURE(variant);
    HighsPrimalHeuristicManager manager;
    HighsPrimalHeuristicManagerConfig config;
    config.schedule = HighsPrimalHeuristicSchedule::kPortfolioContinuation;
    config.portfolio_max_calls = 1;
    config.portfolio_extra_continuations = 1;
    manager.initialise(config);
    HighsPrimalPortfolioBurstScope burst(manager, 0, 0, 0);
    HighsPrimalHeuristicContext context;
    context.method = HighsPrimalHeuristicMethod::kLocalBranching;
    context.has_incumbent = true;
    context.seed_upper_bound = 20;
    context.restriction_identity = {0, 1};
    context.burst_id = burst.id();
    const auto decision = manager.before(context);
    REQUIRE(decision.run);
    HighsPrimalHeuristicOutcome gain;
    gain.started = gain.candidate_accepted = true;
    gain.admission.measured = gain.admission.improved = true;
    gain.admission.before = 20;
    gain.admission.after = gain.admission.candidate = 18;
    if (variant == 0) gain.admission.improved = false;
    if (variant == 1) gain.admission.measured = false;
    if (variant == 2) gain.candidate_accepted = false;
    if (variant == 3) gain.admission.candidate = 19;
    if (variant == 4) gain.admission.before = kHighsInf;
    if (variant == 5) gain.preparation_stop_mask = kHighsHeuristicStopUnfinishedLp;
    manager.record(context, gain, 0, decision.reservation_id);
    REQUIRE_FALSE(manager.continuePortfolioBurst(burst.id(), variant == 6 ? 1 : 0,
        variant == 7 ? 1 : 0, variant == 8 ? 0 : 1, variant == 9 ? 17 : 18,
        variant == 10 ? 20 : 1));
    REQUIRE(manager.burstAccount().extra_reserved == 0);
    REQUIRE(manager.integrityErrors() == 0);
  }
}

TEST_CASE("extra continuation reservations are atomic and overflow charges fail closed",
          "[highs_primal_heuristic_manager][primal_portfolio_continuation]") {
  HighsPrimalHeuristicManager manager;
  HighsPrimalHeuristicManagerConfig config;
  config.schedule = HighsPrimalHeuristicSchedule::kPortfolioContinuation;
  config.portfolio_max_calls = config.portfolio_extra_continuations = 1;
  manager.initialise(config);
  HighsPrimalPortfolioBurstScope burst(manager, 0, 0, 0);
  HighsPrimalHeuristicContext context;
  context.method = HighsPrimalHeuristicMethod::kLocalBranching;
  context.has_incumbent = true;
  context.seed_upper_bound = 20;
  context.restriction_identity = {0, 1};
  context.burst_id = burst.id();
  auto decision = manager.before(context);
  REQUIRE(decision.run);
  HighsPrimalHeuristicOutcome gain;
  gain.started = gain.completed = gain.candidate_accepted = true;
  gain.admission.measured = gain.admission.improved = true;
  gain.admission.before = 20;
  gain.admission.after = gain.admission.candidate = 18;
  gain.lp_iterations = 1;
  manager.record(context, gain, 0, decision.reservation_id);
  REQUIRE(manager.continuePortfolioBurst(burst.id(), 0, 0, 1, 18, 1));
  context.incumbent_sequence = 1;
  context.seed_upper_bound = 18;
  context.time = 1;
  context.restriction_identity = {1, 0};
  std::atomic<int> ready{0}, granted{0};
  std::atomic<bool> release{false};
  std::vector<std::thread> threads;
  for (int thread = 0; thread < 8; ++thread) {
    threads.emplace_back([&]() {
      const auto permit = manager.before(context);
      if (permit.run) ++granted;
      ++ready;
      while (!release.load()) std::this_thread::yield();
      if (permit.run) {
        HighsPrimalHeuristicOutcome overflow;
        overflow.started = true;
        overflow.lp_iterations = std::numeric_limits<int64_t>::max();
        manager.record(context, overflow, 0, permit.reservation_id);
      }
    });
  }
  while (ready.load() != 8) std::this_thread::yield();
  const int grants = granted.load();
  release = true;
  for (auto& thread : threads) thread.join();
  REQUIRE(grants == 1);
  REQUIRE(manager.burstAccount().extra_reserved == 1);
  REQUIRE(manager.burstAccount().extra_started == 1);
  REQUIRE(manager.burstAccount().extra_completed == 0);
  REQUIRE(manager.burstAccount().extra_lp_iterations == std::numeric_limits<int64_t>::max());
  REQUIRE(manager.account(context.method).lp_iterations == std::numeric_limits<int64_t>::max());
  REQUIRE_FALSE(manager.continuePortfolioBurst(burst.id(), 0, 0, 2, 16, 2));
  REQUIRE(manager.integrityErrors() == 0);
}

TEST_CASE("settlement cannot substitute a different checked seed bound",
          "[highs_primal_heuristic_manager][primal_portfolio_continuation]") {
  HighsPrimalHeuristicManager manager;
  HighsPrimalHeuristicManagerConfig config;
  config.schedule = HighsPrimalHeuristicSchedule::kPortfolioContinuation;
  config.portfolio_extra_continuations = 3;
  manager.initialise(config);
  HighsPrimalPortfolioBurstScope burst(manager, 0, 0, 0);
  HighsPrimalHeuristicContext context;
  context.method = HighsPrimalHeuristicMethod::kLocalBranching;
  context.has_incumbent = true;
  context.seed_upper_bound = 20;
  context.restriction_identity = {0, 1};
  context.burst_id = burst.id();
  const auto decision = manager.before(context);
  REQUIRE(decision.run);
  auto changed = context;
  changed.seed_upper_bound = 19;
  HighsPrimalHeuristicOutcome outcome;
  outcome.lp_iterations = 42;
  manager.record(changed, outcome, 0, decision.reservation_id);
  REQUIRE(manager.integrityErrors() == 1);
  REQUIRE(manager.before(context).reason == HighsPrimalHeuristicDecisionReason::kPortfolioBusy);
  manager.record(context, outcome, 0, decision.reservation_id);
  REQUIRE(manager.account(context.method).lp_iterations == 42);
  REQUIRE_FALSE(manager.continuePortfolioBurst(burst.id(), 0, 0, 1, 18, 1));
}

namespace {
HighsPrimalHeuristicManagerConfig deferredConfig() {
  HighsPrimalHeuristicManagerConfig config;
  config.schedule = HighsPrimalHeuristicSchedule::kPortfolioDeferredContinuation;
  config.portfolio_max_calls = 1;
  config.portfolio_extra_continuations = 3;
  config.portfolio_burst_time = 1;
  config.trace_decisions = true;
  return config;
}

HighsPrimalHeuristicContext deferredContext(uint64_t burst, int64_t sequence,
                                           double upper, double time) {
  HighsPrimalHeuristicContext context;
  context.method = HighsPrimalHeuristicMethod::kLocalBranching;
  context.burst_id = burst;
  context.has_incumbent = true;
  context.main_lp_iterations = 1000000;
  context.incumbent_sequence = sequence;
  context.seed_upper_bound = upper;
  context.time = time;
  context.checked_seed = {double(sequence), 1};
  context.restriction_identity = context.checked_seed;
  return context;
}

HighsPrimalHeuristicOutcome deferredGain(double upper, int64_t lp = 10) {
  HighsPrimalHeuristicOutcome gain;
  gain.started = gain.completed = gain.candidate_accepted = true;
  gain.admission.measured = gain.admission.improved = true;
  gain.admission.before = upper;
  gain.admission.after = gain.admission.candidate = upper - 2;
  gain.elapsed = 1.1;
  gain.lp_iterations = lp;
  return gain;
}
}

TEST_CASE("deferred own-gain tokens survive a burst but never renew search credit",
          "[highs_primal_heuristic_manager][primal_portfolio_deferred]") {
  for (auto schedule : {HighsPrimalHeuristicSchedule::kPortfolioDeferredContinuation,
                       HighsPrimalHeuristicSchedule::kPortfolioCostAware}) {
  for (int extra = 0; extra <= 3; ++extra) {
    CAPTURE(extra);
    HighsPrimalHeuristicManager manager;
    auto config = deferredConfig();
    config.schedule = schedule;
    config.portfolio_extra_continuations = extra;
    manager.initialise(config);
    uint64_t burst = manager.beginPortfolioBurst(0, 0, 0);
    double upper = 20;
    for (int call = 0; call <= extra; ++call) {
      const double time = call * 30;
      auto context = deferredContext(burst, call, upper, time);
      auto decision = manager.before(context);
      REQUIRE(decision.run);
      REQUIRE(decision.productive_continuation == (call > 0));
      REQUIRE(decision.deferred_continuation == (call > 0));
      manager.record(context, deferredGain(upper), decision.observation_id,
                     decision.reservation_id);
      upper -= 2;
      REQUIRE_FALSE(manager.continuePortfolioBurst(burst, 0, 0, call + 1, upper, time + 1.1));
      const auto seed = std::vector<double>{double(call + 1), 1};
      REQUIRE(manager.deferPortfolioContinuation(burst, 0, 0, call + 1,
              upper, time + 1.1, seed) == (call < extra));
      REQUIRE_FALSE(manager.deferPortfolioContinuation(burst, 0, 0, call + 1,
                                                       upper, time + 1.1, seed));
      manager.endPortfolioBurst(burst);
      if (call == extra) break;
      REQUIRE(manager.portfolioOpportunityAvailable(1000000, 0));
      REQUIRE(manager.beginPortfolioBurst(0, 0, time + 30) == 0);
      // An unrelated worker cannot claim or invalidate the witness.
      REQUIRE(manager.beginDeferredPortfolioBurst(1, 0, time + 30, upper, seed) == 0);
      burst = manager.beginDeferredPortfolioBurst(0, 0, time + 30, upper, seed);
      REQUIRE(burst != 0);
      REQUIRE(manager.beginDeferredPortfolioBurst(0, 0, time + 30, upper, seed) == 0);
    }
    const auto account = manager.burstAccount();
    REQUIRE(account.deferred_issued == extra);
    REQUIRE(account.deferred_claimed == extra);
    REQUIRE(account.deferred_reserved == extra);
    REQUIRE(account.deferred_started == extra);
    REQUIRE(account.deferred_completed == extra);
    REQUIRE(account.deferred_invalidated == 0);
    REQUIRE(account.extra_reserved == extra);
    REQUIRE(account.deferred_lp_iterations == extra * 10);
    REQUIRE(manager.account(HighsPrimalHeuristicMethod::kLocalBranching).lp_iterations == (extra + 1) * 10);
    REQUIRE_FALSE(manager.portfolioOpportunityAvailable(1000000, 0));
    REQUIRE(manager.integrityErrors() == 0);
  }
  }
}

TEST_CASE("deferred ownership rejects stale witnesses, malformed gains and budget renewal",
          "[highs_primal_heuristic_manager][primal_portfolio_deferred]") {
  for (int variant = 0; variant < 18; ++variant) {
    CAPTURE(variant);
    HighsPrimalHeuristicManager manager;
    auto config = deferredConfig();
    if (variant == 15) config.portfolio_lp_cap = config.portfolio_bootstrap_lp = 10;
    if (variant == 16) config.schedule = HighsPrimalHeuristicSchedule::kPortfolioContinuation;
    manager.initialise(config);
    const uint64_t burst = manager.beginPortfolioBurst(0, 0, 0);
    auto context = deferredContext(burst, 0, 20, 0);
    const auto decision = manager.before(context);
    REQUIRE(decision.run);
    auto gain = deferredGain(20);
    if (variant == 0) gain.completed = false;
    if (variant == 1) gain.started = false;
    if (variant == 2) gain.candidate_accepted = false;
    if (variant == 3) gain.admission.measured = false;
    if (variant == 4) gain.admission.improved = false;
    if (variant == 5) gain.admission.before = kHighsInf;
    if (variant == 6) gain.admission.after = std::numeric_limits<double>::quiet_NaN();
    if (variant == 7) gain.admission.candidate = 19;
    if (variant == 8) gain.preparation_stop_mask = kHighsHeuristicStopCheckpoint;
    if (variant == 9) gain.preparation_stop_mask = kHighsHeuristicStopUnfinishedLp;
    if (variant == 17) gain.elapsed = -1;
    manager.record(context, gain, decision.observation_id, decision.reservation_id);
    std::vector<double> seed{1, 1};
    if (variant == 13) seed.clear();
    if (variant == 14) seed[0] = kHighsInf;
    REQUIRE_FALSE(manager.deferPortfolioContinuation(burst, 0, 0,
        variant == 10 ? 0 : 1, variant == 11 ? 17 : 18,
        variant == 12 ? 0.5 : 1.1, seed));
    manager.endPortfolioBurst(burst);
    REQUIRE(manager.burstAccount().deferred_issued == 0);
    REQUIRE(manager.burstAccount().extra_reserved == 0);
    REQUIRE(manager.integrityErrors() == (variant == 17 ? 1 : 0));
  }
  for (int variant = 0; variant < 6; ++variant) {
    HighsPrimalHeuristicManager manager;
    manager.initialise(deferredConfig());
    const uint64_t burst = manager.beginPortfolioBurst(0, 0, 0);
    auto context = deferredContext(burst, 0, 20, 0);
    const auto decision = manager.before(context);
    manager.record(context, deferredGain(20), 0, decision.reservation_id);
    REQUIRE(manager.deferPortfolioContinuation(burst, 0, 0, 1, 18, 1.1, {1, 1}));
    manager.endPortfolioBurst(burst);
    std::vector<double> seed{1, 1};
    if (variant == 0) seed[0] = 0;
    if (variant == 1) seed.pop_back();
    if (variant == 5) seed[0] = std::numeric_limits<double>::quiet_NaN();
    REQUIRE(manager.beginDeferredPortfolioBurst(0, variant == 2 ? 1 : 0,
        variant == 3 ? std::numeric_limits<double>::quiet_NaN() : 30,
        variant == 4 ? 17 : 18, seed) == 0);
    REQUIRE(manager.burstAccount().deferred_invalidated == 1);
    REQUIRE(manager.beginDeferredPortfolioBurst(0, 0, 30, 18, {1, 1}) == 0);
    REQUIRE(manager.burstAccount().extra_reserved == 0);
  }
}

TEST_CASE("deferred canceled reservations charge once and cannot mint another token",
          "[highs_primal_heuristic_manager][primal_portfolio_deferred]") {
  HighsPrimalHeuristicManager manager;
  manager.initialise(deferredConfig());
  uint64_t burst = manager.beginPortfolioBurst(0, 0, 0);
  auto context = deferredContext(burst, 0, 20, 0);
  auto decision = manager.before(context);
  manager.record(context, deferredGain(20), 0, decision.reservation_id);
  REQUIRE(manager.deferPortfolioContinuation(burst, 0, 0, 1, 18, 1.1, {1, 1}));
  manager.endPortfolioBurst(burst);
  burst = manager.beginDeferredPortfolioBurst(0, 0, 30, 18, {1, 1});
  REQUIRE(burst != 0);
  context = deferredContext(burst, 1, 18, 30);
  auto changed = context;
  changed.checked_seed[0] = 0;
  REQUIRE_FALSE(manager.before(changed).run);
  decision = manager.before(context);
  REQUIRE(decision.run);
  {
    HighsPrimalHeuristicReservation abandoned(manager, context, decision);
  }
  REQUIRE_FALSE(manager.deferPortfolioContinuation(burst, 0, 0, 2, 16, 32, {2, 1}));
  manager.endPortfolioBurst(burst);
  const auto ledger = manager.burstAccount();
  REQUIRE(ledger.extra_reserved == 1);
  REQUIRE(ledger.deferred_reserved == 1);
  REQUIRE(ledger.deferred_started == 0);
  REQUIRE(ledger.deferred_completed == 0);
  REQUIRE(ledger.deferred_lp_iterations == 100000);
  REQUIRE(manager.account(context.method).lp_iterations == 100010);
  REQUIRE_FALSE(manager.portfolioOpportunityAvailable(1000000, 0));
  REQUIRE(manager.integrityErrors() == 0);
}

TEST_CASE("deferred claims are atomic and preserve the cumulative LP ceiling",
          "[highs_primal_heuristic_manager][primal_portfolio_deferred]") {
  for (int64_t cap : {10, 11, 15, 20, 30, 40}) {
    HighsPrimalHeuristicManager manager;
    auto config = deferredConfig();
    config.portfolio_lp_cap = config.portfolio_bootstrap_lp = cap;
    manager.initialise(config);
    auto burst = manager.beginPortfolioBurst(0, 0, 0);
    auto context = deferredContext(burst, 0, 20, 0);
    const auto decision = manager.before(context);
    REQUIRE(decision.run);
    manager.record(context, deferredGain(20), 0, decision.reservation_id);
    const bool deferred = manager.deferPortfolioContinuation(burst, 0, 0, 1, 18, 1.1, {1, 1});
    manager.endPortfolioBurst(burst);
    if (cap == 10) { REQUIRE_FALSE(deferred); continue; }
    REQUIRE(deferred);
    std::atomic<int> claimed{0};
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i) threads.emplace_back([&]() {
      if (manager.beginDeferredPortfolioBurst(0, 0, 30, 18, {1, 1})) ++claimed;
    });
    for (auto& thread : threads) thread.join();
    REQUIRE(claimed == 1);
    // Only two bursts exist; IDs are manager-owned and never reused here.
    context = deferredContext(burst + 1, 1, 18, 30);
    const auto extra = manager.before(context);
    REQUIRE(extra.run);
    REQUIRE(extra.budget.max_lp_iterations == cap - 10);
    manager.record(context, deferredGain(18, cap - 10), 0, extra.reservation_id);
    REQUIRE_FALSE(manager.deferPortfolioContinuation(burst + 1, 0, 0, 2, 16, 31.1, {2, 1}));
    manager.endPortfolioBurst(burst + 1);
    REQUIRE(manager.account(context.method).lp_iterations == cap);
    REQUIRE(manager.burstAccount().deferred_reserved == 1);
    REQUIRE(manager.integrityErrors() == 0);
  }
}
