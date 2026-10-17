#include "mip/HighsRootPrimalHelper.h"

#include <memory>

#include "mip/HighsRootLpPortfolio.h"

bool HighsRootPrimalHelper::start(const HighsLp& lp,
                                  const std::vector<double>& seed,
                                  const HighsOptions& parent,
                                  HighsSolveCapacity& capacity,
                                  double seconds) {
  const auto begin = Clock::now();
  const auto* deque = HighsTaskExecutor::getThisWorkerDeque();
  size_t bytes;
  if (started_ || !deque || deque->getNumWorkers() < 2 || !(seconds > 0) ||
      !std::isfinite(seconds) || !HighsRootLpPortfolio::copyBytes(lp, bytes) ||
      !capacity.reserve(2, bytes, lease_))
    return false;
  try {
    owner_.reset(new Highs());
    auto* owner = owner_.get();
    HighsOptions o = parent;
    o.log_file.clear();
    o.output_flag = o.log_to_console = false;
    o.write_solution_to_file = o.write_model_to_file = false;
    o.highs_analysis_level = 0;
    o.user_objective_scale = o.user_bound_scale = 0;
    o.parallel = kHighsOffString;
    o.threads = deque->getNumWorkers();
    o.mip_root_lp_portfolio = false;
    o.mip_root_primal_helper = false;
    o.mip_root_cut_exchange = false;
    o.mip_root_helper_private = true;
    o.mip_flexible_node_dispatch = false;
    o.mip_incumbent_exchange = "off";
    o.mip_candidate_checkpoint = false;
    o.mip_detect_symmetry = false;
    o.mip_allow_restart = false;
    // This helper already owns one finite candidate-only allowance. Do not
    // turn its weighted follow-up into a recursive heuristic portfolio.
    o.mip_heuristic_run_rens = false;
    o.mip_heuristic_run_rins = false;
    o.mip_heuristic_run_root_reduced_cost = false;
    o.mip_max_nodes = 50;
    o.mip_root_max_separation_rounds = 2;
    o.time_limit = seconds;
    seeded_ = highsVerifyCandidate(lp, seed, nullptr,
                                   parent.mip_feasibility_tolerance)
                  .feasible;
    HighsLp private_lp = lp;
    if (!seeded_) {
      // A cold helper must find a seed, not duplicate the parent's proof
      // objective. Only its checked vector returns; its bounds never do.
      std::fill(private_lp.col_cost_.begin(), private_lp.col_cost_.end(), 0.0);
      private_lp.offset_ = 0.0;
      private_lp.sense_ = ObjSense::kMinimize;
      o.objective_bound = kHighsInf;
      o.objective_target = -kHighsInf;
      o.mip_max_improving_sols = 1;
    }
    if (owner->passOptions(o) != HighsStatus::kOk ||
        owner->passModel(private_lp) != HighsStatus::kOk) {
      lease_.release();
      return false;
    }
    if (seeded_) {
      HighsSolution s;
      s.col_value = seed;
      s.value_valid = true;
      if (owner->setSolution(s) != HighsStatus::kOk) {
        lease_.release();
        return false;
      }
      candidate_ = seed;
    }
    owner->setCallback(
        [this](int, const std::string&, const HighsCallbackOutput*,
               HighsCallbackInput* in,
               void*) { in->user_interrupt = stopping_.load(); },
        nullptr);
    owner->startCallback(kCallbackMipInterrupt);
    owner->startCallback(kCallbackSimplexInterrupt);
    owner->startCallback(kCallbackIpmInterrupt);
    const double prepared =
        std::chrono::duration<double>(Clock::now() - begin).count();
    if (prepared >= seconds) {
      lease_.release();
      return false;
    }
    owner->setOptionValue("time_limit", seconds - prepared);
    original_cost_ = lp.col_cost_;
    original_sense_ = lp.sense_;
    original_offset_ = lp.offset_;
    tasks_.spawn([this, owner, begin, seconds]() {
      try {
        const double initial_allowance = seconds -
            std::chrono::duration<double>(Clock::now() - begin).count();
        if (!stopping_.load() && initial_allowance > 0 &&
            owner->setOptionValue("time_limit", initial_allowance) == HighsStatus::kOk) {
          const double seed_objective = seeded_
              ? static_cast<double>(highsVerifyCandidate(
                    owner->getLp(), candidate_, nullptr,
                    owner->getOptions().mip_feasibility_tolerance).objective)
              : 0.0;
          if (seeded_) {
            weighted_attempted_ = true;
            weighted_allowance_ = initial_allowance;
          }
          const auto initial_started = Clock::now();
          owner->run();
          const double initial_seconds =
              std::chrono::duration<double>(Clock::now() - initial_started).count();
          if (seeded_) weighted_seconds_ = initial_seconds;
          else feasibility_seconds_ = initial_seconds;
          lp_work_ =
              std::max<int64_t>(0, owner->getInfo().simplex_iteration_count);
          if (seeded_) weighted_work_ = lp_work_;
          const auto& values = owner->getSolution().col_value;
          if (owner->getSolution().value_valid && highsVerifyCandidate(
                  owner->getLp(), values, nullptr,
                  owner->getOptions().mip_feasibility_tolerance)
                  .feasible) {
            candidate_ = values;
            if (seeded_) {
              const auto after = highsVerifyCandidate(
                  owner->getLp(), candidate_, nullptr,
                  owner->getOptions().mip_feasibility_tolerance);
              weighted_improved_ = static_cast<int>(original_sense_) *
                  (static_cast<double>(after.objective) - seed_objective) < 0;
            }
          }

          // A zero-objective seed may leave continuous or implied-integer
          // auxiliaries at expensive bounds. Complete only those values, in
          // the same time allowance, without changing explicit integer choices.
          if (!seeded_ && !candidate_.empty() && !stopping_.load()) {
            HighsLp original = owner->getLp();
            original.col_cost_ = original_cost_;
            original.sense_ = original_sense_;
            original.offset_ = original_offset_;
            const double tolerance = owner->getOptions().mip_feasibility_tolerance;
            const auto before = highsVerifyCandidate(original, candidate_, nullptr, tolerance);
            HighsLp completion = original;
            bool supported = before.feasible;
            for (HighsInt col = 0; col < completion.num_col_; ++col) {
              const auto type = completion.integrality_.empty()
                                    ? HighsVarType::kContinuous
                                    : completion.integrality_[col];
              if (type == HighsVarType::kInteger) {
                completion.col_lower_[col] = completion.col_upper_[col] =
                    std::round(candidate_[col]);
              } else if (type != HighsVarType::kContinuous &&
                         type != HighsVarType::kImplicitInteger) {
                supported = false;
              }
            }
            if (supported && owner->passModel(completion) == HighsStatus::kOk &&
                owner->setOptionValue("solve_relaxation", true) == HighsStatus::kOk &&
                owner->setOptionValue("solver", "simplex") == HighsStatus::kOk) {
              const double remaining = seconds -
                  std::chrono::duration<double>(Clock::now() - begin).count();
              if (remaining > 0 && !stopping_.load() &&
                  owner->setOptionValue("time_limit", remaining) == HighsStatus::kOk) {
                const auto completion_started = Clock::now();
                owner->run();
                completion_seconds_ += std::chrono::duration<double>(
                    Clock::now() - completion_started).count();
                lp_work_ += std::max<int64_t>(0, owner->getInfo().simplex_iteration_count);
                const auto& completed = owner->getSolution();
                const auto after = highsVerifyCandidate(original, completed.col_value,
                                                        nullptr, tolerance);
                bool integral = after.feasible;
                if (integral)
                  for (HighsInt col = 0; col < original.num_col_; ++col)
                    if (!original.integrality_.empty() &&
                        original.integrality_[col] == HighsVarType::kImplicitInteger &&
                        std::abs(completed.col_value[col] - std::round(completed.col_value[col])) > tolerance)
                      integral = false;
                if (completed.value_valid && integral &&
                    static_cast<int>(original_sense_) * (after.objective - before.objective) < 0)
                  candidate_ = completed.col_value;
              }
            }

            // Feasibility and fixed-choice completion are not roster
            // improvement. Spend only the helper's remaining deadline on the
            // original weighted MIP, retaining the checked seed on failure.
            const double remaining = seconds -
                std::chrono::duration<double>(Clock::now() - begin).count();
            const auto seed_check = highsVerifyCandidate(original, candidate_, nullptr, tolerance);
            if (remaining > 0 && !stopping_.load() && seed_check.feasible &&
                owner->passModel(original) == HighsStatus::kOk &&
                owner->setOptionValue("solve_relaxation", false) == HighsStatus::kOk &&
                owner->setOptionValue("solver", "choose") == HighsStatus::kOk &&
                owner->setOptionValue("mip_max_improving_sols", kHighsIInf) == HighsStatus::kOk) {
              HighsSolution seed;
              seed.col_value = candidate_;
              seed.value_valid = true;
              if (owner->setSolution(seed) == HighsStatus::kOk) {
                const double allowance = seconds -
                    std::chrono::duration<double>(Clock::now() - begin).count();
                if (allowance > 0 && !stopping_.load() &&
                    owner->setOptionValue("time_limit", allowance) == HighsStatus::kOk) {
                  weighted_attempted_ = true;
                  weighted_allowance_ = allowance;
                  const auto weighted_started = Clock::now();
                  owner->run();
                  weighted_seconds_ += std::chrono::duration<double>(
                      Clock::now() - weighted_started).count();
                  weighted_work_ += std::max<int64_t>(0, owner->getInfo().simplex_iteration_count);
                  lp_work_ += weighted_work_;
                  const auto& improved = owner->getSolution();
                  const auto check = highsVerifyCandidate(original, improved.col_value, nullptr, tolerance);
                  if (improved.value_valid && check.feasible &&
                      static_cast<int>(original_sense_) *
                          (check.objective - seed_check.objective) < 0) {
                    candidate_ = improved.col_value;
                    weighted_improved_ = true;
                  }
                }
              }
            }
          }
        }
      } catch (...) {
        // A failed optional search must not discard its checked starting seed.
      }
      elapsed_ = std::chrono::duration<double>(Clock::now() - begin).count();
      finished_.store(true);
    });
    started_ = true;
    return true;
  } catch (...) {
    lease_.release();
    return false;
  }
}

void HighsRootPrimalHelper::join() {
  if (!started_ || joined_) return;
  stopping_.store(true);
  tasks_.taskWait();
  joined_ = true;
  lease_.release();
}
