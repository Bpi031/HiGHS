#ifndef HIGHS_ROOT_PRIMAL_HELPER_H_
#define HIGHS_ROOT_PRIMAL_HELPER_H_

#include <atomic>
#include <chrono>
#include <memory>

#include "Highs.h"
#include "mip/HighsCandidateVerifier.h"
#include "mip/HighsSolveCapacity.h"
#include "parallel/HighsParallel.h"

// Owns one candidate-only numerical search. It never reads the parent's
// mutable domain, callbacks, mappings, incumbent or proof statistics.
class HighsRootPrimalHelper {
 public:
  HighsRootPrimalHelper() = default;
  ~HighsRootPrimalHelper() { join(); }
  HighsRootPrimalHelper(const HighsRootPrimalHelper&) = delete;
  HighsRootPrimalHelper& operator=(const HighsRootPrimalHelper&) = delete;
  bool start(const HighsLp&, const std::vector<double>& seed,
             const HighsOptions&, HighsSolveCapacity&, double seconds);
  void join();
  const std::vector<double>& candidate() const { return candidate_; }
  int64_t work() const { return lp_work_; }
  double elapsed() const { return elapsed_; }
  bool started() const { return started_; }
  bool seeded() const { return seeded_; }
  bool feasibilityMode() const { return !seeded_; }
  bool finished() const { return finished_.load(); }
  bool weightedAttempted() const { return weighted_attempted_; }
  bool weightedImproved() const { return weighted_improved_; }
  double feasibilitySeconds() const { return feasibility_seconds_; }
  double completionSeconds() const { return completion_seconds_; }
  double weightedSeconds() const { return weighted_seconds_; }
  double weightedAllowance() const { return weighted_allowance_; }
  int64_t weightedWork() const { return weighted_work_; }

 private:
  using Clock = std::chrono::steady_clock;
  HighsSolveCapacity::Lease lease_;
  highs::parallel::TaskGroup tasks_;
  std::atomic<bool> stopping_{false};
  std::atomic<bool> finished_{false};
  bool started_ = false;
  bool joined_ = false;
  bool seeded_ = false;
  bool weighted_attempted_ = false;
  bool weighted_improved_ = false;
  int64_t lp_work_ = 0;
  double elapsed_ = 0;
  double feasibility_seconds_ = 0;
  double completion_seconds_ = 0;
  double weighted_seconds_ = 0;
  double weighted_allowance_ = 0;
  int64_t weighted_work_ = 0;
  std::vector<double> candidate_;
  std::vector<double> original_cost_;
  ObjSense original_sense_ = ObjSense::kMinimize;
  double original_offset_ = 0;
  std::unique_ptr<Highs> owner_;
};
#endif
