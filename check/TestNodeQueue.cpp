#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "catch.hpp"
#include "mip/HighsNodeQueue.h"

TEST_CASE("node-queue-resize-preserves-outstanding-proof-work",
          "[highs_node_queue][highs_queue_lifecycle]") {
  for (bool parked : {false, true}) {
    HighsNodeQueue queue;
    queue.setNumCol(1);
    if (parked) queue.setOptimalityLimit(5);
    queue.emplaceNode({{1, 0, HighsBoundType::kLower}}, {0}, 7, 8, 3);
    REQUIRE(queue.empty() == parked);
    REQUIRE_NOTHROW(queue.setNumCol(1));
    REQUIRE_THROWS_AS(queue.setNumCol(2), std::logic_error);
    REQUIRE_THROWS_AS(queue.setNumCol(0), std::logic_error);
    REQUIRE_THROWS_AS(queue.setNumCol(-1), std::invalid_argument);
    REQUIRE(queue.numNodes() == 1);
    REQUIRE(queue.numNodesUp(0) == 1);
    REQUIRE(queue.getBestLowerBound() == 7);
    if (parked) {
      REQUIRE(queue.performBounding(7) == 0);
    } else {
      HighsNodeQueue::OpenNode node = std::move(queue.popBestBoundNode());
      REQUIRE(node.domchgstack.size() == 1);
      REQUIRE(node.domchgstack[0].boundval == 1);
      REQUIRE(node.branchings == std::vector<HighsInt>{0});
    }
    REQUIRE(queue.numNodes() == 0);
    REQUIRE(queue.getUpNodes(0).empty());
    REQUIRE_NOTHROW(queue.setNumCol(2));
    queue.emplaceNode({{2, 1, HighsBoundType::kUpper}}, {}, 4, 5, 2);
    REQUIRE(queue.numNodesDown(1) == 1);
    REQUIRE(queue.popBestNode().lower_bound == 4);
  }
}

TEST_CASE("node-queue-empty-resize-clear-and-move-lifecycle",
          "[highs_node_queue][highs_queue_lifecycle]") {
  for (HighsInt columns : {0, 1, 17, 256}) {
    HighsNodeQueue queue;
    REQUIRE_THROWS_AS(queue.setNumCol(-1), std::invalid_argument);
    for (int cycle = 0; cycle < 4; ++cycle) {
      queue.setNumCol(columns);
      queue.setOptimalityLimit(0);
      queue.clear();
      queue.clear();
      // Clear resets the optimality limit, even for a never-populated queue.
      std::vector<HighsDomainChange> changes;
      if (columns) changes.push_back({2, columns - 1, HighsBoundType::kUpper});
      queue.emplaceNode(std::move(changes), {}, 1, 2, 3);
      REQUIRE_FALSE(queue.empty());
      HighsNodeQueue moved(std::move(queue));
      REQUIRE(moved.getBestLowerBound() == 1);
      queue.clear();
      REQUIRE(queue.numNodes() == 0);
      REQUIRE(queue.getBestLowerBound() == kHighsInf);
      queue.emplaceNode({}, {}, -1, 0, 1);
      REQUIRE(queue.popBestBoundNode().lower_bound == -1);
      REQUIRE(moved.popBestBoundNode().depth == 3);
      moved.clear();
      moved.setNumCol(0);
      moved.emplaceNode({}, {}, 0, 1, 1);
      REQUIRE(moved.popBestNode().lower_bound == 0);
      queue.clear();
      queue.setNumCol(0);
    }
  }
}

TEST_CASE("node-queue-bounds-parked-nodes-with-no-active-frontier",
          "[highs_node_queue][highs_node_lp_effort]") {
  HighsNodeQueue queue;
  queue.setNumCol(0);
  queue.setOptimalityLimit(5);
  queue.emplaceNode({}, {}, 7, 8, 3);
  REQUIRE(queue.empty());
  REQUIRE(queue.numNodes() == 1);
  REQUIRE(queue.getBestLowerBound() == 7);
  REQUIRE(queue.performBounding(6) == 0);
  REQUIRE(queue.numNodes() == 0);
  REQUIRE(queue.getBestLowerBound() == kHighsInf);
}

TEST_CASE("node-queue-bounding-matches-independent-frontier",
          "[highs_node_queue][highs_node_lp_effort]") {
  for (int target = -3; target <= 7; ++target) {
    for (int cutoff = -4; cutoff <= 12; ++cutoff) {
      HighsNodeQueue queue;
      queue.setNumCol(1);
      queue.setOptimalityLimit(target);
      std::vector<int> kept;
      int active = 0;
      double expected_weight = 0;
      // Distinct domain changes also exercise removal from per-column indexes.
      for (int i = -4; i <= 12; ++i) {
        std::vector<HighsDomainChange> changes{
            {static_cast<double>(i), 0, HighsBoundType::kLower}};
        queue.emplaceNode(std::move(changes), {}, i, i + 1, 4);
        if (i <= target &&
            (i >= cutoff || (target < cutoff && i >= target)))
          expected_weight += 0.125;
        if (i < cutoff) {
          kept.push_back(i);
          // performBounding parks equality too when target < cutoff.
          active += target < cutoff ? i < target : i <= target;
        }
      }
      REQUIRE(queue.performBounding(cutoff) == expected_weight);
      REQUIRE(queue.numNodes() == static_cast<int64_t>(kept.size()));
      REQUIRE(queue.numActiveNodes() == active);
      REQUIRE(queue.getBestLowerBound() ==
              (kept.empty() ? kHighsInf : kept.front()));
      REQUIRE(queue.numNodesUp(0) == static_cast<int64_t>(kept.size()));
      for (const auto& entry : queue.getUpNodes(0)) {
        REQUIRE(entry.first < cutoff);
      }
      REQUIRE(queue.performBounding(cutoff) == 0);
      REQUIRE(queue.numNodes() == static_cast<int64_t>(kept.size()));
      queue.performBounding(-5);
      REQUIRE(queue.numNodes() == 0);
      REQUIRE(queue.getUpNodes(0).empty());
      // Cleared slots must remain usable with valid per-column links.
      queue.emplaceNode({{0, 0, HighsBoundType::kUpper}}, {}, -6, -5, 2);
      REQUIRE(queue.numNodesDown(0) == 1);
      REQUIRE(queue.popBestBoundNode().lower_bound == -6);
      REQUIRE(queue.numNodesDown(0) == 0);
    }
  }
}

TEST_CASE("node-queue-selectors-match-independent-finite-key-oracle",
          "[highs_node_queue][highs_node_lp_effort]") {
  struct Entry {
    int lower;
    int estimate;
    int changes;
    int slot;
    bool live;
  };
  for (bool hybrid : {false, true}) {
    for (unsigned seed = 1; seed <= 32; ++seed) {
      HighsNodeQueue queue;
      queue.setNumCol(4);
      std::vector<Entry> entries;
      uint32_t random = seed;
      auto draw = [&]() {
        random = random * 1664525u + 1013904223u;
        return random;
      };
      auto before = [&](const Entry& a, const Entry& b) {
        if (hybrid) {
          // Small integers make the weighted floating-point key exact.
          const int a_score = a.lower + a.estimate;
          const int b_score = b.lower + b.estimate;
          if (a_score != b_score) return a_score < b_score;
          if (a.changes != b.changes) return a.changes > b.changes;
        } else {
          if (a.lower != b.lower) return a.lower < b.lower;
          if (a.changes != b.changes) return a.changes < b.changes;
          if (a.estimate != b.estimate) return a.estimate < b.estimate;
        }
        return a.slot < b.slot;
      };
      for (int step = 0; step < 128; ++step) {
        const int lower = static_cast<int>(draw() % 17) - 8;
        const int estimate = static_cast<int>(draw() % 17) - 8;
        const int change_count = draw() % 5;
        int slot = 0;
        while (slot < static_cast<int>(entries.size()) && entries[slot].live)
          ++slot;
        Entry entry{lower, estimate, change_count, slot, true};
        if (slot == static_cast<int>(entries.size()))
          entries.push_back(entry);
        else
          entries[slot] = entry;
        std::vector<HighsDomainChange> changes;
        for (int col = 0; col < change_count; ++col)
          changes.push_back({static_cast<double>(step), col,
                             HighsBoundType::kLower});
        queue.emplaceNode(std::move(changes), {}, lower, estimate, slot + 1);
        if (step % 3 != 2) continue;
        int best = -1;
        for (const auto& candidate : entries)
          if (candidate.live && (best < 0 || before(candidate, entries[best])))
            best = candidate.slot;
        HighsNodeQueue::OpenNode node = std::move(
            hybrid ? queue.popBestNode() : queue.popBestBoundNode());
        REQUIRE(node.depth == best + 1);
        REQUIRE(node.lower_bound == entries[best].lower);
        REQUIRE(node.estimate == entries[best].estimate);
        REQUIRE(node.domchgstack.size() ==
                static_cast<size_t>(entries[best].changes));
        entries[best].live = false;
      }
      while (!queue.empty()) {
        int best = -1;
        for (const auto& candidate : entries)
          if (candidate.live && (best < 0 || before(candidate, entries[best])))
            best = candidate.slot;
        HighsNodeQueue::OpenNode node = std::move(
            hybrid ? queue.popBestNode() : queue.popBestBoundNode());
        REQUIRE(node.depth == best + 1);
        entries[best].live = false;
      }
      REQUIRE(queue.numNodes() == 0);
      REQUIRE(queue.getBestLowerBound() == kHighsInf);
    }
  }
}
