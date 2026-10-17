#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

#include "catch.hpp"
#include "mip/HighsSearchWork.h"

TEST_CASE("optional-probe-observation-cannot-charge-or-certify-parent-work",
          "[highs_node_lp_effort][highs_lp_work]") {
  HighsSearchWork work;
  work.observeStrongProbe(0, 1, 2, 3, 4, false, 100, 0.5, 1, false);
  REQUIRE(work.probeCalls() == 0);
  work.begin(10, true);
  for (int i = 0; i < 200; ++i)
    work.observeStrongProbe(i, 1, 2, 3, 4, false, 100, i, 1, false);
  REQUIRE(work.strongProbes().size() == 128);
  REQUIRE(work.probeDropped() == 72);
  REQUIRE(work.probeCertified() == 0);
  REQUIRE(work.remaining() == 10);
  REQUIRE(work.summary().local_bound_gain == 0);
  REQUIRE(work.summary().lp_calls[1].calls == 0);
  work.observeStrongProbe(0, 1, 2, 3, 4, true, -1, 1, 1, false);
  REQUIRE(work.probeUnavailable() == 1);
  REQUIRE(work.strongProbes().size() == 128);
  work.finish();
  REQUIRE(work.summary().max_lease_work == 0);
}

TEST_CASE("finite-managed-repair-does-not-renew-exhausted-tree-lease",
          "[highs_node_lp_effort][highs_lp_work]") {
  for (int64_t tree_work : {int64_t{5}, int64_t{15}}) {
    HighsSearchWork work;
    work.begin(10, true);
    work.chargeLp(tree_work);
    {
      HighsSearchWorkScope heuristic(work, HighsSearchWorkRole::kHeuristic);
      HighsRepairReservationScope repair(work, 25000);
      REQUIRE(repair.granted());
      REQUIRE(work.remaining() == 25000);
      REQUIRE_FALSE(work.skipOptional());
      HighsRepairReservationScope nested(work, 1);
      REQUIRE_FALSE(nested.granted());
      work.chargeLp(15000);  // Preparation can exceed the old per-LP ceiling.
      REQUIRE(work.remaining() == 10000);
      work.chargeSubMip(10000);
      REQUIRE(work.exhausted());
      REQUIRE(work.remaining() == 0);
    }
    REQUIRE(work.exhausted());
    REQUIRE(work.remaining() == 0);
    work.finish();
    REQUIRE(work.summary().leases == 1);
    REQUIRE(work.summary().repair_reservations == 1);
    REQUIRE(work.summary().repair_iterations == 25000);
    REQUIRE(work.summary().direct_iterations[0] == tree_work);
    REQUIRE(work.summary().direct_iterations[2] == 15000);
    REQUIRE(work.summary().submip_reported_iterations == 10000);
    REQUIRE(work.summary().max_lease_work == tree_work + 25000);
  }
}

TEST_CASE("managed-repair-reservation-honors-checkpoint-and-unwinding",
          "[highs_node_lp_effort][highs_lp_work]") {
  HighsSearchWork work;
  REQUIRE_FALSE(work.beginRepairReservation(10));
  work.begin(10, true);
  REQUIRE_FALSE(work.beginRepairReservation(0));
  REQUIRE_FALSE(work.beginRepairReservation(-1));
  REQUIRE_FALSE(work.beginRepairReservation(std::numeric_limits<int64_t>::max()));
  try {
    HighsRepairReservationScope repair(work, 20);
    REQUIRE(repair.granted());
    work.chargeLp(5);
    throw 1;
  } catch (int) {}
  REQUIRE(work.remaining() == 5);
  HighsCandidateCheckpoint checkpoint;
  work.attachCheckpoint(&checkpoint, checkpoint.sequence());
  checkpoint.request();
  REQUIRE_FALSE(work.beginRepairReservation(20));
  REQUIRE(work.skipOptional());
  work.finish();
}

TEST_CASE("lp-call-observation-does-not-charge-or-change-task-control",
          "[highs_node_lp_effort][highs_lp_work]") {
  HighsSearchWork work;
  work.recordLpCall(100, 5, 1.0);
  REQUIRE(work.summary().lp_calls[0].calls == 0);
  work.begin(10, true);
  work.setRole(HighsSearchWorkRole::kBranch);
  work.recordLpCall(900, 4, 0.25);
  REQUIRE(work.remaining() == 10);
  REQUIRE_FALSE(work.exhausted());
  REQUIRE_FALSE(work.skipOptional());
  REQUIRE(work.summary().direct_iterations[3] == 0);
  {
    HighsLpWorkScope probe(&work, HighsLpWorkKind::kStrongBranch);
    work.recordLpCall(7, 0, 0.5);
    REQUIRE(work.role() == HighsSearchWorkRole::kBranch);
    REQUIRE(work.phase() == HighsTaskPhase::kBranch);
    {
      HighsLpWorkScope restoration(&work, HighsLpWorkKind::kCutRestoration);
      work.recordLpCall(9, 2, 0.125);
    }
    REQUIRE(work.lpWorkKind() == HighsLpWorkKind::kStrongBranch);
  }
  REQUIRE(work.lpWorkKind() == HighsLpWorkKind::kRelaxation);
  work.chargeLp(10);
  REQUIRE(work.remaining() == 0);
  REQUIRE(work.exhausted());
  work.finish();
  REQUIRE_FALSE(work.observingLpWork());
  work.recordLpCall(999, 1, 5.0);
  REQUIRE(work.summary().lp_calls[0].calls == 1);
  REQUIRE(work.summary().lp_calls[1].simplex_iterations == 7);
  REQUIRE(work.summary().lp_calls[2].ipm_iterations == 2);
  REQUIRE(work.summary().lp_calls[3].calls == 0);
  REQUIRE(work.summary().max_lease_work == 10);
  work.setLpWorkKind(HighsLpWorkKind::kIterateRestoration);
  work.begin(10, false);
  REQUIRE(work.lpWorkKind() == HighsLpWorkKind::kRelaxation);
  work.finish();
}

TEST_CASE("lp-call-scopes-restore-on-return-and-unwinding",
          "[highs_node_lp_effort][highs_lp_work]") {
  HighsSearchWork work;
  work.begin(0, false);
  auto restore = [&]() {
    HighsLpWorkScope scope(&work, HighsLpWorkKind::kIterateRestoration);
    work.recordLpCall(3, 0, 0.25);
    return work.lpWorkKind();
  };
  REQUIRE(restore() == HighsLpWorkKind::kIterateRestoration);
  REQUIRE(work.lpWorkKind() == HighsLpWorkKind::kRelaxation);
  try {
    HighsLpWorkScope scope(&work, HighsLpWorkKind::kStrongBranch);
    throw 1;
  } catch (int) {
  }
  REQUIRE(work.lpWorkKind() == HighsLpWorkKind::kRelaxation);
  HighsLpWorkScope absent(nullptr, HighsLpWorkKind::kStrongBranch);
  REQUIRE(work.lpWorkKind() == HighsLpWorkKind::kRelaxation);
  work.finish();
}

TEST_CASE("lp-call-missing-measurements-and-saturation-remain-explicit",
          "[highs_node_lp_effort][highs_lp_work]") {
  HighsLpWorkMeasurement work;
  work.record(-1, 2, 0.25);
  work.record(3, -1, -1.0);
  work.record(0, 0, std::numeric_limits<double>::infinity());
  work.record(0, 0, std::numeric_limits<double>::quiet_NaN());
  REQUIRE(work.calls == 4);
  REQUIRE(work.unavailable_iteration_calls == 2);
  REQUIRE(work.unavailable_seconds_calls == 3);
  REQUIRE(work.simplex_iterations == 3);
  REQUIRE(work.ipm_iterations == 2);
  REQUIRE(work.seconds == 0.25);
  const int64_t max_count = (std::numeric_limits<int64_t>::max)();
  const double max_seconds = (std::numeric_limits<double>::max)();
  HighsLpWorkMeasurement large;
  large.record(max_count, max_count, max_seconds);
  work.merge(large);
  work.merge(large);
  REQUIRE(work.calls == 6);
  REQUIRE(work.simplex_iterations == max_count);
  REQUIRE(work.ipm_iterations == max_count);
  REQUIRE(std::isfinite(work.seconds));
  REQUIRE(work.seconds == max_seconds);
  work.calls = max_count;
  work.record(1, 1, 1.0);
  REQUIRE(work.calls == max_count);
}

TEST_CASE("lp-call-independent-operation-oracle-and-merge-properties",
          "[highs_node_lp_effort][highs_lp_work]") {
  uint64_t state = 0x9e3779b97f4a7c15ULL;
  for (int trial = 0; trial < 256; ++trial) {
    HighsSearchWork observed, control;
    std::array<int64_t, 4> calls{}, simplex{}, ipm{};
    std::array<double, 4> seconds{};
    observed.begin(1000, trial % 2 == 0);
    control.begin(1000, trial % 2 == 0);
    for (int step = 0; step < 128; ++step) {
      state = state * 6364136223846793005ULL + 1;
      const size_t kind = (state >> 32) & 3;
      const int64_t s = (state >> 36) & 31;
      const int64_t i = (state >> 41) & 7;
      const double elapsed = double((state >> 44) & 15) / 16;
      observed.setLpWorkKind(static_cast<HighsLpWorkKind>(kind));
      observed.recordLpCall(s, i, elapsed);
      ++calls[kind];
      simplex[kind] += s;
      ipm[kind] += i;
      seconds[kind] += elapsed;
      observed.chargeLp(s);
      control.chargeLp(s);
      REQUIRE(observed.remaining() == control.remaining());
      REQUIRE(observed.exhausted() == control.exhausted());
    }
    observed.finish();
    control.finish();
    REQUIRE(observed.summary().max_lease_work == control.summary().max_lease_work);
    REQUIRE(observed.summary().direct_iterations == control.summary().direct_iterations);
    for (size_t kind = 0; kind < 4; ++kind) {
      const auto& result = observed.summary().lp_calls[kind];
      REQUIRE(result.calls == calls[kind]);
      REQUIRE(result.simplex_iterations == simplex[kind]);
      REQUIRE(result.ipm_iterations == ipm[kind]);
      REQUIRE(result.seconds == seconds[kind]);
      REQUIRE(result.unavailable_iteration_calls == 0);
      REQUIRE(result.unavailable_seconds_calls == 0);
      HighsLpWorkMeasurement a, b;
      a.merge(result);
      b.merge(result);
      a.merge(b);
      REQUIRE(a.calls == 2 * calls[kind]);
      REQUIRE(a.simplex_iterations == 2 * simplex[kind]);
      REQUIRE(a.seconds == 2 * seconds[kind]);
    }
  }
}
