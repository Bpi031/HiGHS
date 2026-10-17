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
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <vector>
#include <utility>

enum class HighsPrimalHeuristicMethod : uint8_t {
  kRootReducedCost,
  kRens,
  kRins,
  kLocalBranching,
  kCount,
};

enum class HighsPrimalHeuristicSchedule : uint8_t {
  kOff,
  kCompatibility,
  kManagedRins,
  kPortfolio,
  kPortfolioBurst,
  kPortfolioContinuation,
  kPortfolioDeferredContinuation,
  kPortfolioCostAware,
  kPortfolioUcbSuccess,
  kPortfolioUcbCost,
  kPortfolioAssignmentAdaptive,
  kPortfolioAssignmentComposed,
  kPortfolioAssignmentOpportunities,
  kPortfolioAssignmentProgressive,
};

enum class HighsPrimalNeighbourhoodKind : uint8_t {
  kUnspecified, kRens, kRins, kStaffPartition, kCoupledSlots,
  kAssignmentLocalBranching, kReducedLocalBranching,
};
constexpr size_t kHighsPrimalPartitionVariantCount = 5;

enum class HighsPrimalPortfolioChoice : uint8_t {
  kUnspecified,
  kFirstSweep,
  kForced,
  kLeastWork,
  kOwnGain,
  kExplore,
  kUcb,
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
  kPortfolioBusy,
  kInvalidRestriction,
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
  int64_t rins_lp_iterations_per_call = 25000;
  double proof_work_reserve = 0.0;
  bool trace_decisions = false;
  // Independent diagnostic values, not recovered proprietary work units.
  int64_t portfolio_max_calls = 6;
  int64_t portfolio_lp_cap = 400000;
  int64_t portfolio_bootstrap_lp = 200000;
  int64_t portfolio_lp_per_call = 100000;
  int64_t portfolio_proof_divisor = 5;
  double portfolio_time_per_call = 20.0;
  int64_t portfolio_burst_calls = 3;
  int64_t portfolio_burst_lp = 100000;
  double portfolio_burst_time = 20.0;
  int64_t portfolio_extra_continuations = 0;
  int64_t portfolio_solve_calls = 16;
  int64_t portfolio_cooldown_lp = 25000;
  int64_t portfolio_construction_lp_per_call = 100000;
  int64_t portfolio_late_reserve_calls = 0;
  int64_t portfolio_late_proof_progress = 0;
};

struct HighsPrimalHeuristicContext {
  HighsPrimalHeuristicMethod method = HighsPrimalHeuristicMethod::kRins;
  int64_t node = 0;
  int64_t main_lp_iterations = 0;
  int64_t heuristic_lp_iterations = 0;
  int64_t incumbent_sequence = 0;
  int64_t recursion_depth = 0;
  int64_t worker_id = 0;
  uint64_t neighbourhood_signature = 0;
  uint64_t burst_id = 0;
  double seed_upper_bound = std::numeric_limits<double>::infinity();
  bool has_incumbent = false;
  bool has_lp_solution = false;
  HighsPrimalPortfolioChoice portfolio_choice =
      HighsPrimalPortfolioChoice::kUnspecified;
  int64_t restart = 0;
  double time = 0.0;
  // Exact equality follows the signature prefilter; never hash-only denial.
  std::vector<double> restriction_identity;
  std::vector<double> checked_seed;
  int partition_variant = -1;
  int64_t partition_groups = 0;
  bool at_root = false;
  HighsPrimalNeighbourhoodKind neighbourhood = HighsPrimalNeighbourhoodKind::kUnspecified;
};

struct HighsPrimalHeuristicDecision {
  bool run = true;
  HighsPrimalHeuristicDecisionReason reason =
      HighsPrimalHeuristicDecisionReason::kAllowed;
  HighsSubMipCallBudget budget;
  uint64_t observation_id = 0;
  uint64_t reservation_id = 0;
  bool productive_continuation = false;
  bool deferred_continuation = false;
};

struct HighsPrimalAdmissionReceipt {
  bool measured = false;
  bool improved = false;
  double before = std::numeric_limits<double>::infinity();
  double after = std::numeric_limits<double>::infinity();
  double candidate = std::numeric_limits<double>::infinity();
  std::vector<double> checked_seed;
};

struct HighsPrimalChildReceipt {
  uint64_t invocation = 0;
  int status = 0;
  int64_t input_rows = 0;
  int64_t input_columns = 0;
  int64_t input_nonzeros = 0;
  int64_t working_rows = 0;
  int64_t working_columns = 0;
  int64_t working_nonzeros = 0;
  int64_t lp_iterations = 0;
  int64_t nodes = 0;
  double fixing_rate = 0;
  double started_time = 0;
  double returned_time = 0;
  double setup_seconds = 0;
  double first_lp_seconds = -1;
  int64_t first_lp_iterations = 0;
  bool proposed = false;
  bool accepted = false;
  bool admission_measured = false;
  bool own_gain = false;
};

struct HighsPrimalChildReceipts {
  static constexpr size_t kCapacity = 4;
  std::array<HighsPrimalChildReceipt, kCapacity> items{};
  size_t retained = 0;
  uint64_t considered = 0;
  void append(const HighsPrimalChildReceipt& receipt) {
    ++considered;
    if (retained < items.size()) items[retained++] = receipt;
  }
  void append(const HighsPrimalChildReceipts& other) {
    // RINS merges an earlier call with a later child, not with itself.
    if (this == &other) return;
    for (size_t i = 0; i < other.retained; ++i) append(other.items[i]);
    considered += other.considered - other.retained;
  }
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
  bool phase_measured = false;
  int64_t preparation_lp_iterations = 0;
  int64_t child_lp_iterations = 0;
  int64_t child_calls = 0;
  int64_t preparation_exhaustions = 0;
  double preparation_seconds = 0.0;
  double child_seconds = 0.0;
  double finished_time = -1.0;
  bool fixing_rate_measured = false;
  bool reservation_denied = false;
  // Observed conditions can coincide; unfinished LP is not an infeasibility proof.
  uint8_t preparation_stop_mask = 0;
  HighsPrimalAdmissionReceipt admission;
  double parent_lower_bound = -std::numeric_limits<double>::infinity();
  int64_t completion_lp_iterations = 0;
  HighsPrimalChildReceipts children;
};

constexpr uint8_t kHighsHeuristicStopDeadline = 1;
constexpr uint8_t kHighsHeuristicStopLpWork = 2;
constexpr uint8_t kHighsHeuristicStopNodes = 4;
constexpr uint8_t kHighsHeuristicStopCheckpoint = 8;
constexpr uint8_t kHighsHeuristicStopUnfinishedLp = 16;

struct HighsPrimalHeuristicObservation {
  uint64_t id = 0;
  HighsPrimalHeuristicContext context;
  HighsPrimalHeuristicDecision decision;
  HighsPrimalHeuristicOutcome outcome;
  bool outcome_recorded = false;
};

constexpr size_t kHighsPrimalHeuristicTraceCapacity = 128;

struct HighsPrimalHeuristicTrace {
  std::array<HighsPrimalHeuristicObservation,
             kHighsPrimalHeuristicTraceCapacity> observations{};
  size_t retained = 0;
  uint64_t considered = 0;
  uint64_t dropped = 0;
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
  int64_t phase_measured_calls = 0;
  int64_t preparation_lp_iterations = 0;
  int64_t child_lp_iterations = 0;
  int64_t child_calls = 0;
  int64_t preparation_exhaustions = 0;
  double preparation_seconds = 0.0;
  double child_seconds = 0.0;
  int64_t attributed_improvements = 0;
  int64_t reward_observations = 0;
  double reward = 0.0;
  int64_t selection_samples = 0;
  double selection_reward = 0.0;
  int64_t selection_reuses = 0;
  int64_t selection_explorations = 0;
  std::array<int64_t,
             static_cast<size_t>(HighsPrimalHeuristicDecisionReason::kCount)>
      decisions{};
};

constexpr size_t kHighsPrimalHeuristicMethodCount =
    static_cast<size_t>(HighsPrimalHeuristicMethod::kCount);

struct HighsPrimalPortfolioBurstAccount {
  int64_t started = 0;
  int64_t calls = 0;
  int64_t continuations = 0;
  int64_t no_gain = 0;
  int64_t stale_seed = 0;
  int64_t budget_exhausted = 0;
  int64_t extra_reserved = 0;
  int64_t extra_started = 0;
  int64_t extra_completed = 0;
  int64_t extra_limit_denials = 0;
  int64_t extra_lp_iterations = 0;
  int64_t deferred_issued = 0;
  int64_t deferred_claimed = 0;
  int64_t deferred_reserved = 0;
  int64_t deferred_started = 0;
  int64_t deferred_completed = 0;
  int64_t deferred_invalidated = 0;
  int64_t deferred_lp_iterations = 0;
};

struct HighsPrimalOpportunityAccount {
  int64_t considered = 0;
  int64_t allowed = 0;
  int64_t reserved_calls = 0;
  int64_t charged_lp = 0;
  double last_allowed_time = -1;
  double longest_sampled_denial = 0;
  std::array<int64_t, static_cast<size_t>(HighsPrimalHeuristicDecisionReason::kCount)>
      reasons{};
};

struct HighsPrimalOpportunityObservation {
  int64_t id = 0;
  int64_t restart = 0;
  int64_t sequence = -1;
  int64_t proof_lp = 0;
  int64_t charged_lp = 0;
  int64_t remaining_calls = 0;
  int64_t remaining_lp = 0;
  int64_t remaining_burst_calls = -1;
  int64_t remaining_burst_lp = -1;
  size_t outstanding = 0;
  double time = -1;
  double remaining_burst_seconds = -1;
  bool solve_scoped_calls = false;
  int64_t late_calls_held = 0;
  int64_t late_proof_needed = 0;
  HighsPrimalHeuristicDecisionReason reason = HighsPrimalHeuristicDecisionReason::kAllowed;
};

struct HighsPrimalOpportunityTrace {
  static constexpr size_t kCapacity = 128;
  std::array<HighsPrimalOpportunityObservation, kCapacity> observations{};
  size_t retained = 0;
  int64_t considered = 0;
};

struct HighsPrimalPartitionAccount {
  int64_t scheduled = 0;
  int64_t completed = 0;
  int64_t improved = 0;
  int64_t lp_iterations = 0;
  int64_t free_groups = 4;
  double elapsed = 0;
  double reward = 0;
  int64_t radius = 2;
  int64_t last_exercised_call = 0;
};

class HighsPrimalHeuristicManager {
 public:
  void initialise(const HighsPrimalHeuristicManagerConfig& config);

  HighsPrimalHeuristicDecision before(
      const HighsPrimalHeuristicContext& context);
  void record(const HighsPrimalHeuristicContext& context,
              const HighsPrimalHeuristicOutcome& outcome,
              uint64_t observation_id = 0, uint64_t reservation_id = 0);
  HighsPrimalHeuristicMethod selectPortfolio(
      const std::array<bool, kHighsPrimalHeuristicMethodCount>& eligible,
      HighsPrimalPortfolioChoice* choice = nullptr) const;
  bool portfolioOpportunityAvailable(int64_t proof_lp, int64_t restart,
      double time = -1, int64_t sequence = -1);
  HighsPrimalOpportunityAccount opportunityAccount() const;
  HighsPrimalOpportunityTrace opportunityTrace() const;
  std::array<HighsPrimalPartitionAccount, kHighsPrimalPartitionVariantCount> partitionAccounts() const;
  size_t partitionVariantCount() const { return composedEnabled() ? 5 : 3; }
  int selectPartitionVariant() const;
  uint64_t beginPortfolioBurst(int64_t worker, int64_t restart, double time);
  bool continuePortfolioBurst(uint64_t id, int64_t worker, int64_t restart,
                             int64_t sequence, double upper, double time,
                             const std::vector<double>& seed = {});
  bool deferPortfolioContinuation(uint64_t id, int64_t worker, int64_t restart,
                                 int64_t sequence, double upper, double time,
                                 const std::vector<double>& seed, bool checkpoint_boundary = false);
  uint64_t beginDeferredPortfolioBurst(int64_t worker, int64_t restart,
                                      double time, double upper,
                                      const std::vector<double>& seed);
  void endPortfolioBurst(uint64_t id);
  HighsPrimalPortfolioBurstAccount burstAccount() const;
  bool portfolioEnabled() const {
    return config_.schedule == HighsPrimalHeuristicSchedule::kPortfolio ||
           burstEnabled() || adaptiveEnabled();
  }
  bool adaptiveEnabled() const {
    return config_.schedule == HighsPrimalHeuristicSchedule::kPortfolioAssignmentAdaptive || composedEnabled();
  }
  bool composedEnabled() const {
    return config_.schedule == HighsPrimalHeuristicSchedule::kPortfolioAssignmentComposed ||
           config_.schedule == HighsPrimalHeuristicSchedule::kPortfolioAssignmentProgressive ||
           config_.schedule == HighsPrimalHeuristicSchedule::kPortfolioAssignmentOpportunities;
  }
  bool progressiveEnabled() const {
    return config_.schedule == HighsPrimalHeuristicSchedule::kPortfolioAssignmentProgressive;
  }
  bool opportunityCoverageEnabled() const {
    return config_.schedule == HighsPrimalHeuristicSchedule::kPortfolioAssignmentOpportunities;
  }
  bool burstEnabled() const {
    return config_.schedule == HighsPrimalHeuristicSchedule::kPortfolioBurst ||
           config_.schedule == HighsPrimalHeuristicSchedule::kPortfolioContinuation ||
           deferredEnabled();
  }
  bool deferredEnabled() const {
    return config_.schedule == HighsPrimalHeuristicSchedule::kPortfolioDeferredContinuation ||
           costSelectionEnabled() || ucbSelectionEnabled() || composedEnabled();
  }
  bool costSelectionEnabled() const {
    return config_.schedule == HighsPrimalHeuristicSchedule::kPortfolioCostAware;
  }
  bool ucbSelectionEnabled() const {
    return config_.schedule == HighsPrimalHeuristicSchedule::kPortfolioUcbSuccess ||
           config_.schedule == HighsPrimalHeuristicSchedule::kPortfolioUcbCost;
  }
  bool selectionEnabled() const {
    return costSelectionEnabled() || ucbSelectionEnabled() || adaptiveEnabled();
  }
  uint64_t integrityErrors() const;
  static double attributedReward(const HighsPrimalHeuristicOutcome& outcome);
  static const char* neighbourhoodName(HighsPrimalNeighbourhoodKind kind);

  bool enabled() const {
    return config_.schedule != HighsPrimalHeuristicSchedule::kOff;
  }
  const HighsPrimalHeuristicManagerConfig& config() const { return config_; }
  HighsPrimalHeuristicAccount account(
      HighsPrimalHeuristicMethod method) const;
  // Selection progress is not a child reservation or renewable work credit.
  int64_t nextSlotSelectionOrdinal();
  HighsPrimalHeuristicTrace decisionTrace() const;

  static const char* methodName(HighsPrimalHeuristicMethod method);
  static const char* scheduleName(HighsPrimalHeuristicSchedule schedule);
  static const char* reasonName(HighsPrimalHeuristicDecisionReason reason);

 private:
  static size_t methodIndex(HighsPrimalHeuristicMethod method);
  bool proofReserveReached(const HighsPrimalHeuristicContext& context) const;
  void observeDecision(const HighsPrimalHeuristicContext& context,
                       HighsPrimalHeuristicDecision& decision);
  void beforePortfolio(const HighsPrimalHeuristicContext& context,
                       HighsPrimalHeuristicDecision& decision);
  bool lateOpportunityReady(int64_t proof_lp);

  HighsPrimalHeuristicManagerConfig config_;
  std::array<HighsPrimalHeuristicAccount, kHighsPrimalHeuristicMethodCount>
      accounts_{};
  int64_t last_rins_node_ = -1;
  int64_t last_rins_incumbent_sequence_ = -1;
  uint64_t last_rins_signature_ = 0;
  mutable std::mutex mutex_;
  std::map<int64_t, int64_t> rins_reservations_;
  int64_t reserved_lp_iterations_ = 0;
  std::unique_ptr<HighsPrimalHeuristicTrace> trace_;
  struct Reservation {
    HighsPrimalHeuristicContext context;
    int64_t lp = 0;
    bool productive_continuation = false;
    bool deferred_continuation = false;
    Reservation(const HighsPrimalHeuristicContext& input, int64_t work, bool continuation,
                bool deferred = false)
        : context(input), lp(work), productive_continuation(continuation),
          deferred_continuation(deferred) {}
  };
  std::map<uint64_t, Reservation> portfolio_reservations_;
  std::vector<HighsPrimalHeuristicContext> portfolio_restrictions_;
  std::map<int64_t, int64_t> portfolio_calls_;
  uint64_t next_reservation_ = 0;
  uint64_t integrity_errors_ = 0;
  int64_t portfolio_charged_lp_ = 0;
  int64_t slot_selection_ordinal_ = 0;
  HighsPrimalOpportunityAccount opportunity_account_;
  std::unique_ptr<HighsPrimalOpportunityTrace> opportunity_trace_;
  std::array<HighsPrimalPartitionAccount, kHighsPrimalPartitionVariantCount> partition_accounts_{};
  int64_t last_portfolio_proof_ = -1;
  int64_t last_portfolio_sequence_ = -1;
  int64_t late_proof_baseline_ = -1;
  bool late_released_ = false;
  double denial_started_ = -1;
  struct Burst {
    uint64_t id = 0;
    int64_t worker = 0;
    int64_t restart = 0;
    int64_t calls = 0;
    int64_t lp = 0;
    double deadline = 0;
    int64_t sequence = 0;
    HighsPrimalHeuristicOutcome last;
    bool completed_call = false;
    bool continuation_authorized = false;
    int64_t authorized_sequence = 0;
    std::vector<double> authorized_seed;
    bool deferred_start = false;
    bool deferred_issued = false;
    double deferred_upper = std::numeric_limits<double>::infinity();
    std::vector<double> deferred_seed;
  } burst_;
  struct DeferredContinuation {
    uint64_t source_burst = 0;
    int64_t worker = 0;
    int64_t restart = 0;
    double upper = std::numeric_limits<double>::infinity();
    std::vector<double> seed;
  } deferred_;
  uint64_t next_burst_ = 0;
  HighsPrimalPortfolioBurstAccount burst_account_;
};

class HighsPrimalPortfolioBurstScope {
 public:
  HighsPrimalPortfolioBurstScope(HighsPrimalHeuristicManager& manager,
                               int64_t worker, int64_t restart, double time,
                               uint64_t claimed_id = 0)
      : manager_(manager), id_(claimed_id ? claimed_id :
            manager.beginPortfolioBurst(worker, restart, time)) {}
  ~HighsPrimalPortfolioBurstScope() { if (id_) manager_.endPortfolioBurst(id_); }
  HighsPrimalPortfolioBurstScope(const HighsPrimalPortfolioBurstScope&) = delete;
  HighsPrimalPortfolioBurstScope& operator=(const HighsPrimalPortfolioBurstScope&) = delete;
  uint64_t id() const { return id_; }
 private:
  HighsPrimalHeuristicManager& manager_;
  uint64_t id_;
};

// An abandoned call conservatively pays its reserved LP allowance. Its token
// cannot release a newer worker's reservation, including after a restart.
class HighsPrimalHeuristicReservation {
 public:
  HighsPrimalHeuristicReservation(HighsPrimalHeuristicManager& manager,
      HighsPrimalHeuristicContext context, HighsPrimalHeuristicDecision decision)
      : manager_(manager), context_(std::move(context)), decision_(decision) {}
  ~HighsPrimalHeuristicReservation() {
    if (!settled_) {
      HighsPrimalHeuristicOutcome cancelled;
      cancelled.lp_iterations = std::max<int64_t>(0, decision_.budget.max_lp_iterations);
      cancelled.elapsed = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - started_).count();
      cancelled.finished_time = context_.time + cancelled.elapsed;
      finish(cancelled);
    }
  }
  HighsPrimalHeuristicReservation(const HighsPrimalHeuristicReservation&) = delete;
  HighsPrimalHeuristicReservation& operator=(const HighsPrimalHeuristicReservation&) = delete;
  void finish(const HighsPrimalHeuristicOutcome& outcome) {
    if (settled_) return;
    manager_.record(context_, outcome, decision_.observation_id, decision_.reservation_id);
    settled_ = true;
  }
 private:
  HighsPrimalHeuristicManager& manager_;
  HighsPrimalHeuristicContext context_;
  HighsPrimalHeuristicDecision decision_;
  std::chrono::steady_clock::time_point started_ = std::chrono::steady_clock::now();
  bool settled_ = false;
};

#endif
