#ifndef HIGHS_SEARCH_WORK_H_
#define HIGHS_SEARCH_WORK_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <chrono>
#include <vector>
#include <cstdint>
#include <limits>
#include "mip/HighsCandidateCheckpoint.h"

enum class HighsSearchWorkRole { kEvaluation, kSeparation, kHeuristic, kBranch, kCount };

enum class HighsTaskPhase { kEvaluation, kSeparation, kHeuristic, kBranch,
                            kCutRestoration, kHeuristicPreparation, kChildMip };

enum class HighsLpWorkKind { kRelaxation, kStrongBranch, kCutRestoration,
                             kIterateRestoration, kCount };

inline const char* highsLpWorkKindName(HighsLpWorkKind kind) {
  switch (kind) {
    case HighsLpWorkKind::kRelaxation: return "relaxation";
    case HighsLpWorkKind::kStrongBranch: return "strong_branch";
    case HighsLpWorkKind::kCutRestoration: return "cut_restoration";
    case HighsLpWorkKind::kIterateRestoration: return "iterate_restoration";
    case HighsLpWorkKind::kCount: break;
  }
  return "unknown";
}

// Exclusive optimizeLp calls, independent of the cooperative scheduling proxy.
// Seconds sum worker-local elapsed intervals, not process CPU or wall time.
struct HighsLpWorkMeasurement {
  int64_t calls = 0;
  int64_t simplex_iterations = 0;
  int64_t ipm_iterations = 0;
  int64_t unavailable_iteration_calls = 0;
  int64_t unavailable_seconds_calls = 0;
  double seconds = 0;

  void record(int64_t simplex, int64_t ipm, double elapsed) {
    addCount(calls, 1);
    if (simplex < 0 || ipm < 0) addCount(unavailable_iteration_calls, 1);
    if (simplex >= 0) addCount(simplex_iterations, simplex);
    if (ipm >= 0) addCount(ipm_iterations, ipm);
    if (std::isfinite(elapsed) && elapsed >= 0)
      addSeconds(elapsed);
    else
      addCount(unavailable_seconds_calls, 1);
  }
  void merge(const HighsLpWorkMeasurement& other) {
    addCount(calls, other.calls);
    addCount(simplex_iterations, other.simplex_iterations);
    addCount(ipm_iterations, other.ipm_iterations);
    addCount(unavailable_iteration_calls, other.unavailable_iteration_calls);
    addCount(unavailable_seconds_calls, other.unavailable_seconds_calls);
    addSeconds(other.seconds);
  }

 private:
  static void addCount(int64_t& total, int64_t amount) {
    total += std::min(amount, std::numeric_limits<int64_t>::max() - total);
  }
  void addSeconds(double amount) {
    const double max = (std::numeric_limits<double>::max)();
    seconds += std::min(amount, max - seconds);
  }
};

inline const char* highsTaskPhaseName(HighsTaskPhase phase) {
  switch (phase) {
    case HighsTaskPhase::kEvaluation: return "evaluation";
    case HighsTaskPhase::kSeparation: return "separation";
    case HighsTaskPhase::kHeuristic: return "heuristic";
    case HighsTaskPhase::kBranch: return "branch";
    case HighsTaskPhase::kCutRestoration: return "cut_restoration";
    case HighsTaskPhase::kHeuristicPreparation: return "heuristic_preparation";
    case HighsTaskPhase::kChildMip: return "child_mip";
  }
  return "unknown";
}

struct HighsSearchWorkSummary {
  int64_t leases = 0;
  int64_t exhausted_leases = 0;
  int64_t yields = 0;
  int64_t skipped_optional = 0;
  int64_t submip_reported_iterations = 0;
  int64_t max_lease_work = 0;
  std::array<int64_t, 4> direct_iterations{};
  int64_t restoration_calls = 0;
  int64_t deferred_restorations = 0;
  double restoration_seconds = 0;
  int64_t local_bound_observations = 0;
  double local_bound_gain = 0;
  int64_t completion_exemptions = 0;
  int64_t candidate_yields = 0;
  int64_t task_records_dropped = 0;
  int64_t repair_reservations = 0;
  int64_t repair_iterations = 0;
  std::array<HighsLpWorkMeasurement, 4> lp_calls{};
};

struct HighsTaskMeasurement {
  int64_t work;
  double started_seconds;
  double seconds;
  double restoration_seconds;
  double local_bound_gain;
  int64_t completion_exemptions;
};

struct HighsStrongProbeMeasurement {
  int64_t lease, node, depth, column, candidates, reliable_directions;
  bool up;
  int64_t iterations;
  double seconds;
  int status;
  bool certified;
};

// Cooperative task budget. A depleted reservation is never a proof status.
// Direct LP iterations and the child's native (possibly adjusted) work are
// separate measurements; their sum is a scheduling proxy, not physical FLOPs.
class HighsSearchWork {
 public:
  void observeStrongProbe(int64_t node, int64_t depth, int64_t column,
                          int64_t candidates, int64_t reliable_directions,
                          bool up, int64_t iterations, double seconds,
                          int status, bool certified) {
    if (!active_) return;
    ++probe_calls_;
    probe_certified_ += certified;
    if (iterations < 0 || !std::isfinite(seconds) || seconds < 0) {
      ++probe_unavailable_;
      return;
    }
    HighsStrongProbeMeasurement entry{summary_.leases, node, depth, column,
        candidates, reliable_directions, up, iterations, seconds, status, certified};
    if (probes_.size() < 128) probes_.push_back(entry);
    else {
      auto shortest = std::min_element(probes_.begin(), probes_.end(),
          [](const HighsStrongProbeMeasurement& a, const HighsStrongProbeMeasurement& b) {
            return a.seconds < b.seconds;
          });
      if (entry.seconds > shortest->seconds) *shortest = entry;
      ++probe_dropped_;
    }
  }
  const std::vector<HighsStrongProbeMeasurement>& strongProbes() const { return probes_; }
  int64_t probeCalls() const { return probe_calls_; }
  int64_t probeCertified() const { return probe_certified_; }
  int64_t probeUnavailable() const { return probe_unavailable_; }
  int64_t probeDropped() const { return probe_dropped_; }
  void attachCheckpoint(HighsCandidateCheckpoint* checkpoint, uint64_t sequence) {
    checkpoint_ = checkpoint;
    batch_sequence_ = sequence;
  }
  bool checkpointEnabled() const { return checkpoint_ != nullptr; }
  bool checkpointRequested() const {
    const bool pending = checkpoint_ && checkpoint_->requestedSince(batch_sequence_);
    if (pending && first_checkpoint_seconds_ < 0)
    {
      first_checkpoint_seconds_ = started_seconds_ + std::chrono::duration<double>(
          std::chrono::steady_clock::now() - started_).count();
      checkpoint_phase_ = phase_;
    }
    return pending;
  }
  void requestCheckpoint() {
    if (checkpoint_) checkpoint_->request();
  }
  const HighsCandidateCheckpoint* checkpoint() const { return checkpoint_; }
  uint64_t batchSequence() const { return batch_sequence_; }
  double firstCheckpointSeconds() const { return first_checkpoint_seconds_; }
  double taskReturnSeconds() const { return task_return_seconds_; }
  HighsTaskPhase checkpointPhase() const { return checkpoint_phase_; }
  HighsTaskPhase phase() const { return phase_; }
  void setPhase(HighsTaskPhase phase) { phase_ = phase; }
  void recordCandidateYield() { ++summary_.candidate_yields; }
  void begin(int64_t limit, bool enforce, double started_seconds = 0) {
    // Observation is independent of enforcement, including unlimited tasks.
    active_ = true;
    repair_active_ = false;
    enforce_ = enforce && limit > 0;
    limit_ = limit;
    work_ = 0;
    role_ = HighsSearchWorkRole::kEvaluation;
    phase_ = HighsTaskPhase::kEvaluation;
    lp_kind_ = HighsLpWorkKind::kRelaxation;
    started_ = std::chrono::steady_clock::now();
    started_seconds_ = started_seconds;
    first_checkpoint_seconds_ = -1;
    task_return_seconds_ = -1;
    if (active_) {
      ++summary_.leases;
      restoration_before_ = summary_.restoration_seconds;
      gain_before_ = summary_.local_bound_gain;
      exemptions_before_ = summary_.completion_exemptions;
    }
  }
  void finish() {
    task_return_seconds_ = started_seconds_ + std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started_).count();
    if (active_) {
      summary_.exhausted_leases += limit_ > 0 && work_ >= limit_;
      summary_.max_lease_work = std::max(summary_.max_lease_work, work_);
      const HighsTaskMeasurement task{work_, started_seconds_, std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started_).count(),
            summary_.restoration_seconds - restoration_before_,
            summary_.local_bound_gain - gain_before_,
            summary_.completion_exemptions - exemptions_before_};
      if (tasks_.size() < 128) {
        tasks_.push_back(task);
      } else {
        // Keep the slowest bounded sample, not just the earliest tasks. These
        // observations never drive scheduling or proof decisions.
        auto shortest = std::min_element(tasks_.begin(), tasks_.end(),
            [](const HighsTaskMeasurement& a, const HighsTaskMeasurement& b) {
              return a.seconds < b.seconds;
            });
        if (task.seconds > shortest->seconds) *shortest = task;
        ++summary_.task_records_dropped;
      }
    }
    active_ = false;
    repair_active_ = false;
    checkpoint_ = nullptr;
  }
  HighsSearchWorkRole role() const { return role_; }
  bool observingLpWork() const { return active_; }
  HighsLpWorkKind lpWorkKind() const { return lp_kind_; }
  void setLpWorkKind(HighsLpWorkKind kind) { lp_kind_ = kind; }
  void recordLpCall(int64_t simplex, int64_t ipm, double seconds) {
    if (!active_ || lp_kind_ == HighsLpWorkKind::kCount) return;
    summary_.lp_calls[static_cast<std::size_t>(lp_kind_)].record(simplex, ipm, seconds);
  }
  void setRole(HighsSearchWorkRole role) {
    role_ = role;
    phase_ = static_cast<HighsTaskPhase>(role);
  }
  void chargeLp(int64_t iterations) {
    if (!active_ || iterations <= 0) return;
    add(summary_.direct_iterations[static_cast<std::size_t>(role_)], iterations);
    add(work_, iterations);
    if (repair_active_) {
      add(repair_work_, iterations);
      add(summary_.repair_iterations, iterations);
    }
  }
  void chargeSubMip(int64_t iterations) {
    if (!active_ || iterations <= 0) return;
    add(summary_.submip_reported_iterations, iterations);
    add(work_, iterations);
    if (repair_active_) {
      add(repair_work_, iterations);
      add(summary_.repair_iterations, iterations);
    }
  }
  // The manager owns this additional finite allocation. Its work still charges
  // the enclosing task, so ending repair cannot renew the tree lease.
  bool beginRepairReservation(int64_t limit) {
    if (!active_ || repair_active_ || limit <= 0 ||
        limit == std::numeric_limits<int64_t>::max() || checkpointRequested())
      return false;
    repair_active_ = true;
    repair_limit_ = limit;
    repair_work_ = 0;
    ++summary_.repair_reservations;
    return true;
  }
  void endRepairReservation() { repair_active_ = false; }
  bool exhausted() const {
    if (repair_active_) return repair_work_ >= repair_limit_;
    return active_ && enforce_ && work_ >= limit_;
  }
  int64_t remaining() const {
    if (repair_active_) return std::max<int64_t>(0, repair_limit_ - repair_work_);
    return active_ && enforce_ ? std::max<int64_t>(0, limit_ - work_)
                              : std::numeric_limits<int64_t>::max();
  }
  void recordRestoration(double seconds) {
    if (!active_) return;
    ++summary_.restoration_calls;
    summary_.restoration_seconds += std::max(0.0, seconds);
  }
  void recordDeferredRestoration() {
    if (active_) ++summary_.deferred_restorations;
  }
  void recordLocalBound(double before, double after) {
    if (!active_ || !std::isfinite(before) || !std::isfinite(after)) return;
    ++summary_.local_bound_observations;
    summary_.local_bound_gain += std::max(0.0, after - before);
  }
  void recordCompletionExemption() {
    if (active_ && enforce_) ++summary_.completion_exemptions;
  }
  bool skipOptional() {
    if (!exhausted() && !checkpointRequested()) return false;
    ++summary_.skipped_optional;
    return true;
  }
  void recordYield() { if (active_) ++summary_.yields; }
  const HighsSearchWorkSummary& summary() const { return summary_; }
  const std::vector<HighsTaskMeasurement>& tasks() const { return tasks_; }

 private:
  std::vector<HighsStrongProbeMeasurement> probes_;
  int64_t probe_calls_ = 0, probe_certified_ = 0;
  int64_t probe_unavailable_ = 0, probe_dropped_ = 0;
  static void add(int64_t& total, int64_t amount) {
    total += std::min(amount, std::numeric_limits<int64_t>::max() - total);
  }
  bool active_ = false;
  bool repair_active_ = false;
  int64_t repair_limit_ = 0;
  int64_t repair_work_ = 0;
  HighsCandidateCheckpoint* checkpoint_ = nullptr;
  uint64_t batch_sequence_ = 0;
  bool enforce_ = false;
  int64_t limit_ = 0;
  int64_t work_ = 0;
  HighsSearchWorkRole role_ = HighsSearchWorkRole::kEvaluation;
  HighsTaskPhase phase_ = HighsTaskPhase::kEvaluation;
  HighsLpWorkKind lp_kind_ = HighsLpWorkKind::kRelaxation;
  mutable HighsTaskPhase checkpoint_phase_ = HighsTaskPhase::kEvaluation;
  HighsSearchWorkSummary summary_;
  std::chrono::steady_clock::time_point started_;
  double started_seconds_ = 0;
  mutable double first_checkpoint_seconds_ = -1;
  double task_return_seconds_ = -1;
  double restoration_before_ = 0;
  double gain_before_ = 0;
  int64_t exemptions_before_ = 0;
  std::vector<HighsTaskMeasurement> tasks_;
};

class HighsRepairReservationScope {
 public:
  HighsRepairReservationScope(HighsSearchWork& work, int64_t limit,
                             bool own_inactive_lease = false, double started = 0)
      : work_(work), owned_lease_(own_inactive_lease && !work.observingLpWork() &&
                                limit > 0 && limit < std::numeric_limits<int64_t>::max()) {
    // Root construction has no enclosing tree task. Its manager allocation
    // owns this finite observation lease; existing task credit is never reset.
    if (owned_lease_) work_.begin(limit, true, started);
    granted_ = work_.beginRepairReservation(limit);
  }
  ~HighsRepairReservationScope() {
    if (granted_) work_.endRepairReservation();
    if (owned_lease_) work_.finish();
  }
  bool granted() const { return granted_; }
  HighsRepairReservationScope(const HighsRepairReservationScope&) = delete;
  HighsRepairReservationScope& operator=(const HighsRepairReservationScope&) = delete;
 private:
  HighsSearchWork& work_;
  bool owned_lease_;
  bool granted_;
};

class HighsSearchWorkScope {
 public:
  HighsSearchWorkScope(HighsSearchWork& work, HighsSearchWorkRole role)
      : work_(work), previous_(work.role()), phase_(work.phase()) { work_.setRole(role); }
  ~HighsSearchWorkScope() { work_.setRole(previous_); work_.setPhase(phase_); }
  HighsSearchWorkScope(const HighsSearchWorkScope&) = delete;
  HighsSearchWorkScope& operator=(const HighsSearchWorkScope&) = delete;
 private:
  HighsSearchWork& work_;
  HighsSearchWorkRole previous_;
  HighsTaskPhase phase_;
};

class HighsTaskPhaseScope {
 public:
  HighsTaskPhaseScope(HighsSearchWork& work, HighsTaskPhase phase)
      : HighsTaskPhaseScope(&work, phase) {}
  HighsTaskPhaseScope(HighsSearchWork* work, HighsTaskPhase phase)
      : work_(work), previous_(work ? work->phase() : phase) {
    if (work_) work_->setPhase(phase);
  }
  ~HighsTaskPhaseScope() { if (work_) work_->setPhase(previous_); }
  HighsTaskPhaseScope(const HighsTaskPhaseScope&) = delete;
  HighsTaskPhaseScope& operator=(const HighsTaskPhaseScope&) = delete;
 private:
  HighsSearchWork* work_;
  HighsTaskPhase previous_;
};

class HighsLpWorkScope {
 public:
  HighsLpWorkScope(HighsSearchWork* work, HighsLpWorkKind kind)
      : work_(work), previous_(work ? work->lpWorkKind() : kind) {
    if (work_) work_->setLpWorkKind(kind);
  }
  ~HighsLpWorkScope() { if (work_) work_->setLpWorkKind(previous_); }
  HighsLpWorkScope(const HighsLpWorkScope&) = delete;
  HighsLpWorkScope& operator=(const HighsLpWorkScope&) = delete;
 private:
  HighsSearchWork* work_;
  HighsLpWorkKind previous_;
};
#endif
