#include <atomic>
#include <thread>

#include "Highs.h"
#include "catch.hpp"
#include "mip/HighsSearchCoordinator.h"

namespace {
HighsDetachedNode descriptor(int id) {
  HighsDetachedNode n;
  n.changes = {{double(id % 2), 0, HighsBoundType::kLower}};
  n.branchings = {0};
  n.bound = id;
  n.estimate = id + 1;
  n.depth = 2;
  return n;
}
}  // namespace

TEST_CASE("detached node receipts retain in-flight coverage",
          "[search_coordinator]") {
  HighsSearchCoordinator::Generation g{"model", "mapping", 1, 2, 3};
  HighsSearchCoordinator c(g, 1, 2, 4096, -2);
  REQUIRE(c.offer(g, descriptor(3)));
  HighsSearchCoordinator::Receipt r;
  REQUIRE(c.take(0, r, false));
  REQUIRE(c.conservativeBound() == -2);
  auto stale = r;
  ++stale.generation.restart;
  REQUIRE_FALSE(c.installed(stale));
  stale = r;
  ++stale.worker;
  REQUIRE_FALSE(c.installed(stale));
  REQUIRE(c.drain().size() == 1);  // install failure cannot lose the packet
  REQUIRE_FALSE(c.installed(r));
  REQUIRE(c.conservativeBound() == -2);
  auto stale_generation = g;
  ++stale_generation.restart;
  REQUIRE_FALSE(c.offer(stale_generation, descriptor(4)));
  auto bad = descriptor(4);
  bad.changes[0].column = 1;
  REQUIRE_FALSE(c.offer(g, bad));
  bad = descriptor(4);
  bad.branchings = {1};
  REQUIRE_FALSE(c.offer(g, bad));
  HighsSearchCoordinator tiny(g, 1, 1, 0, -2);
  REQUIRE_FALSE(tiny.offer(g, descriptor(1)));
  REQUIRE_FALSE(tiny.failed());
}

TEST_CASE("idle workers take each immutable donation exactly once",
          "[search_coordinator]") {
  for (int repeat = 0; repeat != 20; ++repeat) {
    HighsSearchCoordinator::Generation g{"model", "mapping", 1, 0, 0};
    HighsSearchCoordinator c(g, 1, 3, 1024 * 1024, 0);
    std::atomic<int> accepted{0}, errors{0};
    auto consumer = [&](int w) {
      HighsSearchCoordinator::Receipt r;
      while (c.take(w, r, true)) {
        auto n = r.node->node();
        if (n.domchgstack.size() != 1 || !c.installed(r) || c.installed(r))
          ++errors;
        else
          ++accepted;
      }
    };
    std::thread a([&] { consumer(0); }), b([&] { consumer(1); });
    for (int i = 0; i != 100; ++i) REQUIRE(c.offer(g, descriptor(i)));
    c.retire(2);
    a.join();
    b.join();
    REQUIRE(errors == 0);
    REQUIRE(accepted == 100);
    REQUIRE(c.offered() == 100);
    REQUIRE(c.installed() == 100);
    REQUIRE(c.drain().empty());
    REQUIRE(c.conservativeBound() == 0);
  }
}

TEST_CASE("coordinator cancellation wakes receivers without proof",
          "[search_coordinator]") {
  HighsSearchCoordinator c({"m", "p", 0, 0, 0}, 1, 2, 4096, -9);
  std::atomic<bool> done{false};
  std::thread a([&] {
    HighsSearchCoordinator::Receipt r;
    done = !c.take(0, r, true);
  });
  c.fail();
  a.join();
  REQUIRE(done);
  REQUIRE(c.failed());
  REQUIRE(c.conservativeBound() == -9);
}

TEST_CASE("flexible dispatch preserves enumerated binary optima",
          "[search_coordinator][search_dispatch_integration]") {
  for (int threads : {1, 2, 4}) {
    Highs::resetGlobalScheduler(true);
    for (int family = 0; family != 6; ++family) {
      HighsLp lp;
      lp.num_col_ = 12;
      lp.num_row_ = 12;
      lp.col_lower_.assign(12, 0);
      lp.col_upper_.assign(12, 1);
      lp.integrality_.assign(12, HighsVarType::kInteger);
      lp.row_lower_.assign(12, 1);
      lp.row_upper_.assign(12, kHighsInf);
      lp.a_matrix_.start_ = {0};
      for (int j = 0; j != 12; ++j) {
        lp.col_cost_.push_back(1 + ((j * 5 + family * 7) % 9));
        for (int i = 0; i != 12; ++i)
          if (j == i || j == (i + 1 + family) % 12 || j == (i + 5) % 12) {
            lp.a_matrix_.index_.push_back(i);
            lp.a_matrix_.value_.push_back(1);
          }
        lp.a_matrix_.start_.push_back(lp.a_matrix_.index_.size());
      }
      double oracle = kHighsInf;
      for (int mask = 0; mask != 4096; ++mask) {
        double cost = 0;
        std::vector<int> rows(12, 0);
        for (int j = 0; j != 12; ++j)
          if (mask & (1 << j)) {
            cost += lp.col_cost_[j];
            for (int p = lp.a_matrix_.start_[j]; p < lp.a_matrix_.start_[j + 1];
                 ++p)
              ++rows[lp.a_matrix_.index_[p]];
          }
        if (*std::min_element(rows.begin(), rows.end()) >= 1)
          oracle = std::min(oracle, cost);
      }
      for (bool enabled : {false, true}) {
        CAPTURE(threads, family, enabled);
        Highs h;
        h.setOptionValue("output_flag", false);
        h.setOptionValue("threads", threads);
        h.setOptionValue("parallel", "on");
        h.setOptionValue("presolve", "off");
        h.setOptionValue("mip_rel_gap", 0.0);
        h.setOptionValue("time_limit", 10.0);
        REQUIRE(h.setOptionValue("mip_flexible_node_dispatch", enabled) ==
                HighsStatus::kOk);
        REQUIRE(h.passModel(lp) == HighsStatus::kOk);
        REQUIRE(h.run() == HighsStatus::kOk);
        REQUIRE(h.getModelStatus() == HighsModelStatus::kOptimal);
        REQUIRE(h.getObjectiveValue() == Approx(oracle));
        REQUIRE(h.getInfo().mip_dual_bound == Approx(oracle));
      }
    }
  }
  Highs::resetGlobalScheduler(true);
}
