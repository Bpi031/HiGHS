#ifndef HIGHS_REFINEMENT_BUDGET_H_
#define HIGHS_REFINEMENT_BUDGET_H_

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>

#include "lp_data/HConst.h"

// Owned by one serial callback refinement, borrowed by all its nested MIPs
// and LPs. Only cancellation is written by another thread. No parent proof
// state or shared solver options are changed by this budget.
struct HighsRefinementBudget {
  using Clock = std::chrono::steady_clock;
  using Deadline = std::chrono::time_point<Clock, std::chrono::duration<double>>;
  int64_t limit;
  Deadline deadline;
  const std::atomic<bool>* cancel;
  int64_t simplex_iterations = 0;
  int64_t ipm_iterations = 0;
  int64_t lp_runs = 0;

  HighsRefinementBudget(int64_t limit, Deadline deadline,
                        const std::atomic<bool>* cancel)
      : limit(limit), deadline(deadline), cancel(cancel) {}

  int64_t remaining() const {
    const int64_t after_simplex = limit - std::min(limit, simplex_iterations);
    return after_simplex - std::min(after_simplex, ipm_iterations);
  }
  double secondsLeft() const {
    return std::max(0.0, std::chrono::duration<double>(deadline - Clock::now()).count());
  }
  HighsModelStatus stopStatus() const {
    if (cancel && cancel->load(std::memory_order_relaxed))
      return HighsModelStatus::kInterrupt;
    if (Clock::now() >= deadline) return HighsModelStatus::kTimeLimit;
    if (remaining() == 0) return HighsModelStatus::kIterationLimit;
    return HighsModelStatus::kNotset;
  }
};

// One optimizeLp invocation. Callbacks and final info overlap; only positive
// deltas are charged. Simplex/IPM counts are recorded separately: the combined
// cap bounds work counts, not equivalent seconds or computational cost.
struct HighsRefinementLpWork {
  HighsRefinementBudget* budget;
  int64_t simplex = 0;
  int64_t ipm = 0;
  explicit HighsRefinementLpWork(HighsRefinementBudget* budget) : budget(budget) {}

  void begin() {
    simplex = ipm = 0;
    if (budget) ++budget->lp_runs;
  }
  void observe(int64_t simplex_count, int64_t ipm_count) {
    if (!budget) return;
    auto charge = [](int64_t count, int64_t& seen, int64_t& total) {
      if (count <= seen) return;
      total += std::min(count - seen, std::numeric_limits<int64_t>::max() - total);
      seen = count;
    };
    charge(simplex_count, simplex, budget->simplex_iterations);
    charge(ipm_count, ipm, budget->ipm_iterations);
  }
};

#endif
