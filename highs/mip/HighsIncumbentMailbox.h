/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#ifndef HIGHS_INCUMBENT_MAILBOX_H_
#define HIGHS_INCUMBENT_MAILBOX_H_

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "lp_data/HConst.h"

struct HighsWorkerCandidateMetadata {
  HighsInt worker_id = -1;
  int64_t discovery_node = -1;
  int64_t discovery_lp_iterations = -1;
  int64_t work_epoch = -1;
  int64_t publication_sequence = -1;
  double discovery_time = -1.0;
  double publication_time = -1.0;
  int64_t admission_generation = -1;
};

struct HighsWorkerCandidate {
  std::vector<double> solution;
  double native_objective = kHighsInf;
  int solution_source = -1;
  HighsWorkerCandidateMetadata metadata;
};

// Deterministic exchange drains this worker-owned queue only after its task
// epoch has joined. Overflow is an integrity failure, never a reason to drop a
// candidate silently.
class HighsIncumbentMailbox {
 public:
  enum : size_t { kCapacity = 128 };

  bool publish(HighsWorkerCandidate candidate) {
    if (candidates_.size() == kCapacity) {
      overflowed_ = true;
      return false;
    }
    candidates_.push_back(std::move(candidate));
    return true;
  }

  std::vector<HighsWorkerCandidate> drain() {
    std::vector<HighsWorkerCandidate> candidates = std::move(candidates_);
    candidates_.clear();
    return candidates;
  }

  bool overflowed() const { return overflowed_; }
  size_t size() const { return candidates_.size(); }
  void clearOverflow() { overflowed_ = false; }

 private:
  std::vector<HighsWorkerCandidate> candidates_;
  bool overflowed_ = false;
};

#endif
