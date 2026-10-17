#ifndef HIGHS_PRIMAL_SNAPSHOT_H_
#define HIGHS_PRIMAL_SNAPSHOT_H_

#include <memory>
#include <mutex>
#include <chrono>
#include <atomic>

#include "mip/HighsCandidateVerifier.h"
#include "mip/HighsIncumbentMailbox.h"
#include "mip/HighsPrimalHeuristicManager.h"

struct HighsPrimalSnapshot {
  HighsWorkerCandidate candidate;
  std::vector<double> original_columns;
  double upper_bound = kHighsInf;
  int64_t sequence = 0;
  double admission_time = 0;
};

enum class HighsPrimalPublication { kImproved, kNoImprovement, kStale, kInfeasible };

// One immutable, checked witness per batch. Model/mapping owners must outlive
// the joined batch and must not mutate either model during publication. This
// service has no callbacks, node queues, postsolve stack or global proof state.
class HighsPrimalSnapshotStore {
 public:
  HighsPrimalSnapshotStore(const HighsLp& original, const HighsLp& reduced,
                          double tolerance, int64_t generation, int64_t epoch,
                          double clock_offset = 0)
      : original_(original), reduced_(reduced), tolerance_(tolerance),
        generation_(generation), epoch_(epoch), clock_offset_(clock_offset) {}

  bool publish(const HighsWorkerCandidate& candidate,
               std::vector<double> original_columns) {
    return publishChecked(candidate, std::move(original_columns)) ==
           HighsPrimalPublication::kImproved;
  }

  HighsPrimalPublication publishChecked(const HighsWorkerCandidate& candidate,
                                        std::vector<double> original_columns,
                                        HighsPrimalAdmissionReceipt* receipt = nullptr) {
    if (receipt) *receipt = {};
    if (candidate.metadata.admission_generation != generation_ ||
        candidate.metadata.work_epoch != epoch_)
      return HighsPrimalPublication::kStale;
    for (const HighsLp* lp : {&original_, &reduced_})
      for (HighsVarType type : lp->integrality_)
        if (type != HighsVarType::kContinuous && type != HighsVarType::kInteger &&
            type != HighsVarType::kImplicitInteger)
          return HighsPrimalPublication::kInfeasible;
    // Presolve's implied integers are native-supported. Keep the verifier's
    // explicit-integer contract: dual-implied columns can be fractional at a
    // nonoptimal feasible point. Semi-variable domains remain unsupported here.
    // Expensive checks and vector copies are outside the publication lock.
    const auto reduced_check = highsVerifyCandidate(
        reduced_, candidate.solution, nullptr, tolerance_);
    const auto original_check = highsVerifyCandidate(
        original_, original_columns, nullptr, tolerance_);
    if (!reduced_check.feasible || !original_check.feasible)
      return HighsPrimalPublication::kInfeasible;
    auto next = std::make_shared<HighsPrimalSnapshot>();
    next->candidate = candidate;
    next->original_columns = std::move(original_columns);
    next->upper_bound = static_cast<double>(
        static_cast<HighsInt>(original_.sense_) * original_check.objective -
        reduced_.offset_);
    // Be conservative if postsolve rounding changes the objective slightly.
    next->upper_bound = std::max(next->upper_bound, static_cast<double>(
        reduced_check.objective - reduced_.offset_));
    if (!std::isfinite(next->upper_bound)) return HighsPrimalPublication::kInfeasible;
    next->candidate.native_objective = next->upper_bound;
    if (receipt) receipt->checked_seed = candidate.solution;
    std::shared_ptr<const HighsPrimalSnapshot> retired;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (receipt) {
        receipt->measured = true;
        receipt->before = latest_ ? latest_->upper_bound : kHighsInf;
        receipt->after = receipt->before;
        receipt->candidate = next->upper_bound;
      }
      if (latest_ && next->upper_bound >= latest_->upper_bound)
        return HighsPrimalPublication::kNoImprovement;
      if (receipt) {
        receipt->improved = true;
        receipt->after = next->upper_bound;
      }
      next->sequence = ++publications_;
      next->admission_time = now();
      retired = std::move(latest_);
      latest_ = std::move(next);
      published_sequence_.store(publications_, std::memory_order_release);
    } // Potentially large retired vectors are freed outside the lock.
    return HighsPrimalPublication::kImproved;
  }

  std::shared_ptr<const HighsPrimalSnapshot> latest() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return latest_;
  }

  // The common LP-boundary poll has no new witness. Avoid a mutex and shared
  // pointer reference-count write in that case; publication still owns both.
  std::shared_ptr<const HighsPrimalSnapshot> latestAfter(int64_t sequence) const {
    if (published_sequence_.load(std::memory_order_acquire) <= sequence) return {};
    return latest();
  }

  double now() const {
    return clock_offset_ + std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started_).count();
  }

 private:
  const HighsLp& original_;
  const HighsLp& reduced_;
  const double tolerance_;
  const int64_t generation_;
  const int64_t epoch_;
  const double clock_offset_;
  const std::chrono::steady_clock::time_point started_ =
      std::chrono::steady_clock::now();
  mutable std::mutex mutex_;
  int64_t publications_ = 0;
  std::atomic<int64_t> published_sequence_{0};
  std::shared_ptr<const HighsPrimalSnapshot> latest_;
};

#endif
