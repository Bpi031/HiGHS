/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef HIGHS_PRIMAL_HEURISTIC_MANAGER_H_
#define HIGHS_PRIMAL_HEURISTIC_MANAGER_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

enum class HighsPrimalHeuristicMethod : uint8_t {
  kRootReducedCost,
  kRens,
  kRins,
  kCount,
};

enum class HighsPrimalHeuristicSchedule : uint8_t {
  kOff,
  kCompatibility,
  kManagedRins,
};

enum class HighsPrimalHeuristicDecisionReason : uint8_t {
  kAllowed,
  kManagerOff,
  kNoIncumbent,
  kNoLpSolution,
  kRecursiveSubMip,
  kCallLimit,
  kCooldown,
  kDuplicateNeighbourhood,
  kLpWorkLimit,
  kProofReserve,
  kCount,
};

struct HighsSubMipCallBudget {
  int64_t max_leaves = -1;
  int64_t max_nodes = -1;
  int64_t max_stall_nodes = -1;
  int64_t max_lp_iterations = -1;
  double max_time = std::numeric_limits<double>::infinity();
  double min_fixing_rate = 0.0;
};

struct HighsPrimalHeuristicManagerConfig {
  HighsPrimalHeuristicSchedule schedule =
      HighsPrimalHeuristicSchedule::kOff;
  HighsSubMipCallBudget rins_budget;
  int64_t rins_max_calls = -1;
  int64_t rins_cooldown_nodes = 0;
  double proof_work_reserve = 0.0;
};

struct HighsPrimalHeuristicContext {
  HighsPrimalHeuristicMethod method = HighsPrimalHeuristicMethod::kRins;
  int64_t node = 0;
  int64_t main_lp_iterations = 0;
  int64_t heuristic_lp_iterations = 0;
  int64_t incumbent_sequence = 0;
  int64_t recursion_depth = 0;
  uint64_t neighbourhood_signature = 0;
  bool has_incumbent = false;
  bool has_lp_solution = false;
};

struct HighsPrimalHeuristicDecision {
  bool run = true;
  HighsPrimalHeuristicDecisionReason reason =
      HighsPrimalHeuristicDecisionReason::kAllowed;
  HighsSubMipCallBudget budget;
};

struct HighsPrimalHeuristicOutcome {
  bool started = false;
  bool completed = false;
  bool candidate_proposed = false;
  bool candidate_accepted = false;
  bool candidate_improved = false;
  double fixing_rate = 0.0;
  double elapsed = 0.0;
  int64_t nodes = 0;
  int64_t leaves = 0;
  int64_t lp_iterations = 0;
};

struct HighsPrimalHeuristicAccount {
  int64_t considered = 0;
  int64_t scheduled = 0;
  int64_t started = 0;
  int64_t completed = 0;
  int64_t proposed = 0;
  int64_t accepted = 0;
  int64_t improved = 0;
  int64_t nodes = 0;
  int64_t leaves = 0;
  int64_t lp_iterations = 0;
  double elapsed = 0.0;
  std::array<int64_t,
             static_cast<size_t>(HighsPrimalHeuristicDecisionReason::kCount)>
      decisions{};
};

constexpr size_t kHighsPrimalHeuristicMethodCount =
    static_cast<size_t>(HighsPrimalHeuristicMethod::kCount);

class HighsPrimalHeuristicManager {
 public:
  void initialise(const HighsPrimalHeuristicManagerConfig& config);

  HighsPrimalHeuristicDecision before(
      const HighsPrimalHeuristicContext& context);
  void record(const HighsPrimalHeuristicContext& context,
              const HighsPrimalHeuristicOutcome& outcome);

  bool enabled() const {
    return config_.schedule != HighsPrimalHeuristicSchedule::kOff;
  }
  const HighsPrimalHeuristicManagerConfig& config() const { return config_; }
  const HighsPrimalHeuristicAccount& account(
      HighsPrimalHeuristicMethod method) const;

  static const char* methodName(HighsPrimalHeuristicMethod method);
  static const char* scheduleName(HighsPrimalHeuristicSchedule schedule);
  static const char* reasonName(HighsPrimalHeuristicDecisionReason reason);

 private:
  static size_t methodIndex(HighsPrimalHeuristicMethod method);
  bool proofReserveReached(const HighsPrimalHeuristicContext& context) const;

  HighsPrimalHeuristicManagerConfig config_;
  std::array<HighsPrimalHeuristicAccount, kHighsPrimalHeuristicMethodCount>
      accounts_{};
  int64_t last_rins_node_ = -1;
  int64_t last_rins_incumbent_sequence_ = -1;
  uint64_t last_rins_signature_ = 0;
};

#endif
