#ifndef HIGHS_PRIMAL_SERVICE_H_
#define HIGHS_PRIMAL_SERVICE_H_

#include <atomic>
#include <condition_variable>
#include <thread>
#include <stdexcept>

#include "Highs.h"
#include "mip/HighsPrimalSnapshot.h"
#include "mip/HighsMipSolverData.h"
#include "presolve/HighsPostsolveStack.h"

// Candidate-only authority and private repair owner. Neither the background
// owner nor its callbacks may inspect/mutate the running proof/search objects.
// The mapping is copied because reductionValues has a mutable decoding cursor.
class HighsPrimalService {
 public:
  HighsPrimalService(std::shared_ptr<HighsPrimalSnapshotStore> store,
                    const HighsLp& original, const HighsLp& reduced,
                    const HighsOptions& options,
                    const presolve::HighsPostsolveStack& mapping,
                    const HighsCallbackInput& input, HighsCCallbackType callback,
                    void* data, double lower_bound, int64_t epoch,
                    bool threaded)
      : store_(std::move(store)), mapping_(mapping), input_(input),
        callback_(callback), data_(data), sense_(static_cast<int>(original.sense_)),
        offset_(reduced.offset_), lower_bound_(lower_bound), epoch_(epoch),
        time_limit_(options.time_limit), base_sequence_(input.incumbent_sequence),
        threaded_(threaded) {
    HighsOptions private_options = options;
    // orig_model is already in the parent's native scaled coordinates.
    private_options.user_objective_scale = 0;
    private_options.user_bound_scale = 0;
    if (private_model_.passOptions(private_options) == HighsStatus::kError ||
        private_model_.passModel(original) == HighsStatus::kError)
      throw std::runtime_error("cannot freeze primal service model");
    private_model_.setOptionValue("output_flag", false);
    private_model_.setOptionValue("parallel", "off");
    // A background owner has its own thread-local scheduler. Inline one-thread
    // service keeps the existing scheduler's setting to avoid reinitialization.
    if (threaded_) private_model_.setOptionValue("threads", 1);
    private_model_.setOptionValue("mip_incumbent_exchange", "off");
    input_.highs = &private_model_;
    input_.refinement_cancel = &stopping_;
    input_.user_has_solution = false;
    input_.user_solution.assign(original.num_col_, kHighsUndefined);
    // A copied incumbent is not a new admission in this generation.
    if (const auto seed = store_->latest()) base_sequence_ -= seed->sequence;
  }

  ~HighsPrimalService() { stop(); }
  HighsPrimalService(const HighsPrimalService&) = delete;
  HighsPrimalService& operator=(const HighsPrimalService&) = delete;

  void start() {
    stopping_.store(false);
    if (threaded_ && callback_) thread_ = std::thread([this] {
      try {
        while (!stopping_.load()) {
          poll();
          std::unique_lock<std::mutex> lock(wait_mutex_);
          wake_.wait_for(lock, std::chrono::milliseconds(50),
                         [this] { return stopping_.load(); });
        }
      } catch (...) { failed_.store(true); }
    });
  }

  // Reuse the private matrix/mapping within one restart generation. Called
  // after joining the previous service; no large model copy per worker batch.
  void nextBatch(std::shared_ptr<HighsPrimalSnapshotStore> store,
                 const HighsCallbackInput& input, double lower_bound,
                 int64_t epoch) {
    store_ = std::move(store);
    lower_bound_ = lower_bound;
    epoch_ = epoch;
    input_.admission_generation = input.admission_generation;
    input_.incumbent_sequence = input.incumbent_sequence;
    input_.last_candidate_id = input.last_candidate_id;
    input_.admission = input.admission;
    base_sequence_ = input.incumbent_sequence;
    if (const auto seed = store_->latest()) base_sequence_ -= seed->sequence;
    notified_ = 0;
    next_query_ = 0;
  }

  // Inline mode runs at owner-safe simplex boundaries, preserving the entire
  // numerical call stack. Threaded mode never invokes callbacks on a worker.
  void atResidentBoundary() {
    if (!threaded_ && !stopping_.load()) {
      try { poll(); } catch (...) { failed_.store(true); }
    }
  }

  void stop() {
    const bool already_stopped = stopping_.exchange(true);
    wake_.notify_all();
    if (thread_.joinable()) thread_.join();
    // A short batch can finish before the service is scheduled. Flush only
    // notification, never launch a new repair after cancellation/join.
    if (!already_stopped) {
      try { poll(false); } catch (...) { failed_.store(true); }
    }
  }
  bool failed() const { return failed_.load(); }
  const HighsCallbackInput& joinedInput() const { return input_; }
  uint64_t queries() const { return queries_; }
  uint64_t notifications() const { return notifications_; }

 private:
  void invoke(int type, HighsCallbackOutput& output) {
    auto out = static_cast<HighsCallbackDataOut>(output);
    out.cutpool_num_col = -1;
    out.cutpool_num_cut = -1;
    out.cutpool_num_nz = -1;
    auto in = static_cast<HighsCallbackDataIn>(input_);
    callback_(type, "Primal service (frozen proof bound)", &out, &in, data_);
    input_ = in;
  }

  void poll(bool allow_repair = true) {
    if (!callback_ || (allow_repair && store_->now() >= time_limit_) || failed()) return;
    auto snapshot = store_->latest();
    if (!snapshot) return;
    const double now = store_->now();
    if (snapshot->sequence == notified_ && now < next_query_) return;
    HighsCallbackOutput output = {};
    output.highs = &private_model_;
    output.simplex_iteration_count = -1;
    output.ipm_iteration_count = -1;
    output.pdlp_iteration_count = -1;
    output.cutpool_num_col = -1;
    output.cutpool_num_cut = -1;
    output.running_time = now;
    output.mip_solution = snapshot->original_columns;
    output.mip_primal_bound = sense_ * (snapshot->upper_bound + offset_);
    // Frozen at batch start, never inferred from unjoined worker statistics.
    output.mip_dual_bound = sense_ * (lower_bound_ + offset_);
    output.mip_gap = std::isfinite(lower_bound_)
        ? std::abs(snapshot->upper_bound - lower_bound_) /
              std::max(std::abs(snapshot->upper_bound + offset_), 1e-16)
        : kHighsInf;
    output.objective_function_value = output.mip_primal_bound;
    const auto& meta = snapshot->candidate.metadata;
    output.mip_node_count = meta.discovery_node;
    output.mip_total_lp_iterations = meta.discovery_lp_iterations;
    output.mip_solution_source = snapshot->candidate.solution_source;
    output.mip_solution_sequence = base_sequence_ + snapshot->sequence;
    output.mip_solution_worker_id = meta.worker_id;
    output.mip_solution_discovery_node = meta.discovery_node;
    output.mip_solution_discovery_lp_iterations = meta.discovery_lp_iterations;
    output.mip_solution_work_epoch = epoch_;
    output.mip_solution_worker_sequence = meta.publication_sequence;
    output.mip_solution_discovery_time = meta.discovery_time;
    output.mip_solution_publication_time = meta.publication_time;
    output.mip_solution_admission_time = snapshot->admission_time;
    input_.incumbent_sequence = output.mip_solution_sequence;
    if (snapshot->sequence != notified_) {
      notified_ = snapshot->sequence;
      if (snapshot->upper_bound < notified_upper_bound_) {
        invoke(kCallbackMipImprovingSolution, output);
        notified_upper_bound_ = snapshot->upper_bound;
        ++notifications_;
      }
    }
    if (!allow_repair) return;
    input_.user_has_solution = false;
    input_.user_interrupt = false;
    input_.candidate_id = 0;
    input_.candidate_generation = -1;
    std::fill(input_.user_solution.begin(), input_.user_solution.end(), kHighsUndefined);
    private_model_.setOptionValue("time_limit", std::max(0.0, time_limit_ - now));
    ++queries_;
    invoke(kCallbackMipUserSolution, output);
    next_query_ = store_->now() + 1.0;
    if (!input_.user_has_solution) return;

    HighsCallbackSolutionOutcome receipt = {};
    receipt.candidate_id = input_.candidate_id;
    receipt.generation = input_.admission_generation;
    receipt.sequence_before = input_.incumbent_sequence;
    receipt.objective_before = output.mip_primal_bound;
    receipt.submitted_objective = kHighsInf;
    receipt.reduced_objective = kHighsInf;
    int status = kHighsAdmissionInfeasible;
    if (input_.candidate_generation != input_.admission_generation) {
      status = kHighsAdmissionStale;
    } else if (input_.candidate_id <= input_.last_candidate_id) {
      status = kHighsAdmissionDuplicate;
    } else {
      const auto checked = highsVerifyCandidate(private_model_.getLp(),
          input_.user_solution, nullptr,
          private_model_.getOptions().mip_feasibility_tolerance);
      if (checked.feasible) {
        HighsWorkerCandidate candidate;
        candidate.solution = mapping_.getReducedPrimalSolution(input_.user_solution);
        candidate.native_objective = static_cast<double>(sense_ * checked.objective - offset_);
        candidate.solution_source = kSolutionSourceUserSolution;
        candidate.metadata.worker_id = -1;
        candidate.metadata.admission_generation = input_.admission_generation;
        candidate.metadata.work_epoch = epoch_;
        candidate.metadata.publication_sequence = input_.candidate_id;
        candidate.metadata.discovery_time = store_->now();
        candidate.metadata.publication_time = candidate.metadata.discovery_time;
        receipt.submitted_objective = static_cast<double>(checked.objective);
        receipt.reduced_objective = candidate.native_objective;
        switch (store_->publishChecked(candidate, input_.user_solution)) {
          case HighsPrimalPublication::kImproved:
            status = kHighsAdmissionImproved;
            break;
          case HighsPrimalPublication::kNoImprovement:
            status = kHighsAdmissionNoImprovement;
            break;
          case HighsPrimalPublication::kStale:
            status = kHighsAdmissionStale;
            break;
          case HighsPrimalPublication::kInfeasible:
            status = kHighsAdmissionInfeasible;
            break;
        }
      }
    }
    const auto latest = store_->latest();
    input_.incumbent_sequence = base_sequence_ + latest->sequence;
    receipt.status = status;
    receipt.sequence_after = input_.incumbent_sequence;
    receipt.objective_after = sense_ * (latest->upper_bound + offset_);
    receipt.admission_time = store_->now();
    input_.admission = receipt;
    // IDs belong to a generation. A delayed old-generation submission must
    // not poison duplicate detection for the current private owner.
    if (status != kHighsAdmissionStale)
      input_.last_candidate_id = std::max(input_.last_candidate_id, input_.candidate_id);
    // Acknowledge promptly even when the caller's repair budget is exhausted.
    next_query_ = 0;
  }

  std::shared_ptr<HighsPrimalSnapshotStore> store_;
  Highs private_model_;
  presolve::HighsPostsolveStack mapping_;
  HighsCallbackInput input_;
  HighsCCallbackType callback_;
  void* data_;
  int sense_;
  double offset_, lower_bound_;
  int64_t epoch_;
  double time_limit_;
  int64_t base_sequence_;
  bool threaded_;
  std::atomic<bool> stopping_{false}, failed_{false};
  std::thread thread_;
  std::condition_variable wake_;
  std::mutex wait_mutex_;
  int64_t notified_ = 0;
  double notified_upper_bound_ = kHighsInf;
  double next_query_ = 0;
  uint64_t queries_ = 0, notifications_ = 0;
};

#endif
