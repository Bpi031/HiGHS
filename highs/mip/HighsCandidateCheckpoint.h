#ifndef HIGHS_CANDIDATE_CHECKPOINT_H_
#define HIGHS_CANDIDATE_CHECKPOINT_H_

#include <atomic>
#include <cstdint>
#include <limits>
#include <algorithm>
#include "lp_data/HConst.h"

// A basis-only restart must eventually get a completion opportunity. Repeated
// fixed-size chunks can spend the entire solve restoring the same expensive LP.
inline HighsInt highsCandidateLpIterationLimit(HighsInt requested,
                                               HighsInt yield_count) {
  if (yield_count >= 3) return kHighsIInf;
  const HighsInt chunk = 16384 * (HighsInt(1) << std::max<HighsInt>(0, yield_count));
  return std::min(requested, chunk);
}

// Advisory scheduling only. Candidate payloads remain worker-owned until join.
class HighsCandidateCheckpoint {
 public:
  explicit HighsCandidateCheckpoint(uint64_t initial_sequence = 0)
      : sequence_(initial_sequence) {}
  uint64_t sequence() const { return sequence_.load(std::memory_order_relaxed); }
  bool requestedSince(uint64_t batch_sequence) const {
    return sequence() != batch_sequence;
  }
  bool request() {
    uint64_t before = sequence();
    do {
      if (before == std::numeric_limits<uint64_t>::max()) {
        overflow_.store(true, std::memory_order_relaxed);
        return false;
      }
    } while (!sequence_.compare_exchange_weak(before, before + 1,
                                             std::memory_order_relaxed));
    return true;
  }
  bool overflowed() const { return overflow_.load(std::memory_order_relaxed); }

 private:
  std::atomic<uint64_t> sequence_;
  std::atomic<bool> overflow_{false};
};

// An optional LP reads only the advisory sequence, never another worker's
// payload or mutable domain. An observed request cancels work, not a proof.
class HighsCandidateLpInterrupt {
 public:
  void begin(const HighsCandidateCheckpoint* checkpoint, uint64_t sequence) {
    checkpoint_ = checkpoint;
    sequence_ = sequence;
    interrupted_.store(false, std::memory_order_relaxed);
  }
  bool poll() {
    if (checkpoint_ && checkpoint_->requestedSince(sequence_))
      interrupted_.store(true, std::memory_order_relaxed);
    return interrupted();
  }
  bool interrupted() const { return interrupted_.load(std::memory_order_relaxed); }

 private:
  const HighsCandidateCheckpoint* checkpoint_ = nullptr;
  uint64_t sequence_ = 0;
  std::atomic<bool> interrupted_{false};
};
#endif
