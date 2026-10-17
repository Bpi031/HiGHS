/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#ifndef HIGHS_SCHEDULER_CONSTANTS_H_
#define HIGHS_SCHEDULER_CONSTANTS_H_

#include <chrono>

struct HighsSchedulerConstants {
  using Clock = std::chrono::steady_clock;
  static_assert(Clock::is_steady, "scheduler intervals require a steady clock");
  enum Constants {
    kNumTryFac = 16,
    kMaxSpinTries = 4096,
    kMicroSecsBeforeSleep = 5000,
    kMicroSecsBeforeGlobalSync = 1000,
  };

  static int initialSpinTries(int numWorkers) {
    if (numWorkers <= 1) return kNumTryFac;
    const int others = numWorkers - 1;
    if (others >= kMaxSpinTries / kNumTryFac) return kMaxSpinTries;
    return kNumTryFac * others;
  }

  static int nextSpinTries(int tries) {
    // Bound the work between clock checks as well as avoiding signed overflow.
    if (tries <= 0) return 1;
    if (tries >= kMaxSpinTries / 2) return kMaxSpinTries;
    return 2 * tries;
  }
};

#endif
