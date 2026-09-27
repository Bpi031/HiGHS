#include "catch.hpp"
#include "mip/HighsPrimalHeuristicManager.h"

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
