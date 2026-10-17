#ifndef HIGHS_CHILD_PROPOSAL_SINK_H_
#define HIGHS_CHILD_PROPOSAL_SINK_H_

#include <chrono>
#include <functional>
#include "mip/HighsCandidateVerifier.h"

// Owned by solveSubMip. A serial child may call this only with input-space
// values after postsolve; it never receives parent proof state or user callbacks.
class HighsChildProposalSink {
 public:
  using Consumer = std::function<bool(const std::vector<double>&)>;
  HighsChildProposalSink(const HighsLp& input, double tolerance,
                         Consumer consumer, size_t max_proposals = 8,
                         std::function<double()> parent_clock = {})
      : input_(input), tolerance_(tolerance), consumer_(std::move(consumer)),
        max_proposals_(max_proposals), parent_clock_(std::move(parent_clock)) {}

  bool propose(const std::vector<double>& values) {
    const auto started = std::chrono::steady_clock::now();
    const double discovery = parent_clock_ ? parent_clock_() : -1;
    current_discovery_s = discovery;
    const bool result = publish(values);
    check_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    if (result && first_discovery_s < 0) first_discovery_s = discovery;
    return result;
  }

  size_t events = 0, proposed = 0, published = 0, rejected = 0;
  size_t duplicates = 0, non_improving = 0, capacity_denied = 0;
  double first_discovery_s = -1, current_discovery_s = -1, check_seconds = 0;

 private:
  bool publish(const std::vector<double>& values) {
    ++events;
    if (published && values == last_values_) { ++duplicates; return false; }
    // Once exhausted, skip further full sparse-row checks as well as transport.
    if (proposed >= max_proposals_) { ++capacity_denied; return false; }
    const auto check = highsVerifyCandidate(input_, values, nullptr, tolerance_);
    if (!check.feasible) { ++rejected; return false; }
    const double objective = static_cast<HighsInt>(input_.sense_) *
                             static_cast<double>(check.objective);
    if (!std::isfinite(objective) || objective >= best_objective_) {
      ++non_improving;
      return false;
    }
    ++proposed;
    // One complete vector, never a growing solution pool. Rejected publication
    // does not suppress a later retry or the child's final readback fallback.
    if (!consumer_(values)) { ++rejected; return false; }
    ++published;
    best_objective_ = objective;
    last_values_ = values;
    return true;
  }

  const HighsLp& input_;
  double tolerance_;
  Consumer consumer_;
  size_t max_proposals_;
  std::function<double()> parent_clock_;
  double best_objective_ = kHighsInf;
  std::vector<double> last_values_;
};

#endif
