#include <algorithm>
#include <limits>
#include <random>

#include "catch.hpp"
#include "mip/HighsProofFrontier.h"

TEST_CASE("frontier transfers retain bounds and invalidate old ownership",
          "[proof_frontier]") {
  using State = HighsProofFrontier::State;
  HighsProofFrontier ledger({"model", "mapping", 1, 0, 0});
  REQUIRE_FALSE(ledger.reconciled());
  REQUIRE(ledger.lowerBound() == -std::numeric_limits<double>::infinity());
  REQUIRE(ledger.initialize(1, 3));
  REQUIRE_FALSE(ledger.initialize(2, 9));
  HighsProofFrontier::Token ready, running, transit, received;
  REQUIRE(ledger.token(1, ready));
  REQUIRE(ledger.transition(ready, State::Ready, State::Running, running));
  REQUIRE_FALSE(ledger.strengthen(ready, 100));
  REQUIRE(ledger.strengthen(running, 4));
  REQUIRE(
      ledger.transition(running, State::Running, State::InTransit, transit));
  REQUIRE_FALSE(ledger.closeInfeasible(running));
  REQUIRE_FALSE(ledger.closeInfeasible(transit));
  REQUIRE_FALSE(ledger.reconciled());
  REQUIRE(ledger.lowerBound() == 4);
  REQUIRE(ledger.transition(transit, State::InTransit, State::Ready, received));
  REQUIRE_FALSE(
      ledger.transition(transit, State::InTransit, State::Ready, ready));
  REQUIRE(ledger.transition(received, State::Ready, State::Running, running));
  REQUIRE_FALSE(
      ledger.strengthen(running, std::numeric_limits<double>::quiet_NaN()));
  REQUIRE_FALSE(ledger.strengthen(running, 3));
  REQUIRE_FALSE(ledger.replaceWithChildren(running, {{2, 5}, {2, 6}}));
  REQUIRE_FALSE(ledger.replaceWithChildren(running, {{2, 5}, {3, 2}}));
  REQUIRE(ledger.size() == 1);
  REQUIRE(ledger.replaceWithChildren(running, {{2, 5}, {3, 6}}));
  REQUIRE(ledger.lowerBound() == 5);
  REQUIRE_FALSE(ledger.closeInfeasible(running));
  REQUIRE(ledger.token(2, ready));
  REQUIRE(ledger.transition(ready, State::Ready, State::Running, running));
  REQUIRE(ledger.closeWithBound(running, 5.25));
  REQUIRE(ledger.lowerBound() == 5.25);
  REQUIRE(ledger.token(3, ready));
  REQUIRE(ledger.transition(ready, State::Ready, State::Running, running));
  REQUIRE(ledger.closeInfeasible(running));
  REQUIRE(ledger.reconciled());
  REQUIRE(ledger.lowerBound() ==
          5.25);  // tolerance pruning must retain its bound
}

TEST_CASE("frontier schedule permutations preserve exhaustive leaf oracle",
          "[proof_frontier]") {
  using State = HighsProofFrontier::State;
  std::mt19937 random(7301);
  for (int repeat = 0; repeat != 200; ++repeat) {
    HighsProofFrontier ledger({"model", "mapping", 1, 0, 0});
    REQUIRE(ledger.initialize(0, -1));
    HighsProofFrontier::Token root, active;
    REQUIRE(ledger.token(0, root));
    REQUIRE(ledger.transition(root, State::Ready, State::Running, active));
    std::vector<HighsProofFrontier::Child> children;
    for (uint64_t i = 1; i <= 16; ++i) children.push_back({i, double(i % 4)});
    REQUIRE(ledger.replaceWithChildren(active, children));
    std::shuffle(children.begin(), children.end(), random);
    double oracle = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < children.size(); ++i) {
      HighsProofFrontier::Token token, next;
      REQUIRE(ledger.token(children[i].id, token));
      REQUIRE(ledger.transition(token, State::Ready, State::Running, next));
      token = next;
      if (i % 2 == 0) {
        REQUIRE(
            ledger.transition(token, State::Running, State::Suspended, next));
        token = next;
        REQUIRE(
            ledger.transition(token, State::Suspended, State::Running, next));
        token = next;
      }
      const double leaf = children[i].bound + 0.5;
      oracle = std::min(oracle, leaf);
      REQUIRE(ledger.closeWithBound(token, leaf));
      double expected = oracle;
      for (size_t j = i + 1; j < children.size(); ++j)
        expected = std::min(expected, children[j].bound);
      REQUIRE(ledger.lowerBound() == expected);
    }
    REQUIRE(ledger.reconciled());
    REQUIRE(ledger.lowerBound() == 0.5);
  }
}

TEST_CASE("frontier receipts cannot cross model stage restart or worker",
          "[proof_frontier]") {
  using State = HighsProofFrontier::State;
  const HighsProofFrontier::Generation generation{"model", "mapping", 12, 3, 4};
  HighsProofFrontier ledger(generation);
  REQUIRE(ledger.initialize(1, 2));
  HighsProofFrontier::Token ready, running;
  REQUIRE(ledger.token(1, ready));
  REQUIRE(ledger.transition(ready, State::Ready, State::Running, running, 17));
  REQUIRE(running.worker == 17);
  for (int mutation = 0; mutation != 8; ++mutation) {
    CAPTURE(mutation);
    auto stale = running;
    if (mutation == 0) stale.generation.model = "different model";
    if (mutation == 1) stale.generation.mapping = "permuted columns";
    if (mutation == 2) ++stale.generation.solve;
    if (mutation == 3) ++stale.generation.stage;
    if (mutation == 4) ++stale.generation.restart;
    if (mutation == 5) ++stale.worker;
    if (mutation == 6) ++stale.version;
    if (mutation == 7) ++stale.id;
    REQUIRE_FALSE(ledger.strengthen(stale, 90));
    REQUIRE_FALSE(ledger.replaceWithChildren(stale, {{2, 90}}));
    REQUIRE_FALSE(ledger.closeInfeasible(stale));
    REQUIRE_FALSE(ledger.closeWithBound(stale, 90));
    REQUIRE(ledger.lowerBound() == 2);
    REQUIRE(ledger.size() == 1);
    REQUIRE_FALSE(ledger.reconciled());
  }
  HighsProofFrontier restarted({"model", "mapping", 12, 3, 5});
  REQUIRE(restarted.initialize(1, 2));
  HighsProofFrontier::Token new_ready, new_running;
  REQUIRE(restarted.token(1, new_ready));
  REQUIRE(restarted.transition(new_ready, State::Ready, State::Running,
                               new_running, 17));
  REQUIRE_FALSE(restarted.closeInfeasible(running));
  REQUIRE_FALSE(ledger.closeInfeasible(new_running));
  REQUIRE(ledger.closeWithBound(running, 2.5));
  REQUIRE(ledger.lowerBound() == 2.5);
  REQUIRE(restarted.lowerBound() == 2);
}
