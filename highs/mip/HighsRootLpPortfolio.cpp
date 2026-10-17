#include "mip/HighsRootLpPortfolio.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>

#include "parallel/HighsParallel.h"

namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
bool addBytes(size_t count, size_t width, size_t& bytes) {
  if (count > (std::numeric_limits<size_t>::max() - bytes) / width)
    return false;
  bytes += count * width;
  return true;
}
bool within(long double x, double lo, double up, double tol) {
  return std::isfinite(x) && x >= (long double)lo - tol &&
         x <= (long double)up + tol;
}
}  // namespace

bool HighsRootLpPortfolio::copyBytes(const HighsLp& lp, size_t& bytes) {
  bytes = sizeof(HighsLp);
  if (lp.num_col_ < 0 || lp.num_row_ < 0) return false;
  if (!addBytes(lp.num_col_, 3 * sizeof(double), bytes) ||
      !addBytes(lp.num_row_, 2 * sizeof(double), bytes) ||
      !addBytes(lp.a_matrix_.start_.size(), sizeof(HighsInt), bytes) ||
      !addBytes(lp.a_matrix_.index_.size(), sizeof(HighsInt), bytes) ||
      !addBytes(lp.a_matrix_.value_.size(), sizeof(double), bytes) ||
      !addBytes(lp.integrality_.size(), sizeof(HighsVarType), bytes))
    return false;
  for (const auto& name : lp.col_names_)
    if (!addBytes(name.size(), 1, bytes) ||
        !addBytes(1, sizeof(std::string), bytes))
      return false;
  for (const auto& name : lp.row_names_)
    if (!addBytes(name.size(), 1, bytes) ||
        !addBytes(1, sizeof(std::string), bytes))
      return false;
  // Two private models plus temporary passModel copies. This is deliberately
  // separate from factorization/solver workspace, which requires RSS evidence.
  if (bytes > std::numeric_limits<size_t>::max() / 4) return false;
  bytes *= 4;
  return true;
}

bool HighsRootLpPortfolio::checkedOptimalBasis(const HighsLp& lp,
                                               const HighsSolution& s,
                                               const HighsBasis& b,
                                               const HighsOptions& o) {
  if (lp.num_col_ < 0 || lp.num_row_ < 0 ||
      lp.col_cost_.size() != size_t(lp.num_col_) ||
      lp.col_lower_.size() != size_t(lp.num_col_) ||
      lp.col_upper_.size() != size_t(lp.num_col_) ||
      lp.row_lower_.size() != size_t(lp.num_row_) ||
      lp.row_upper_.size() != size_t(lp.num_row_) ||
      lp.a_matrix_.start_.size() != size_t(lp.num_col_) + 1 ||
      lp.a_matrix_.start_[0] != 0 ||
      lp.a_matrix_.index_.size() != lp.a_matrix_.value_.size() ||
      lp.a_matrix_.start_.back() < 0 ||
      size_t(lp.a_matrix_.start_.back()) != lp.a_matrix_.value_.size() ||
      lp.isMip() || !lp.a_matrix_.isColwise() || !b.valid || !s.value_valid ||
      !s.dual_valid || !std::isfinite(lp.offset_) ||
      s.col_value.size() != size_t(lp.num_col_) ||
      s.col_dual.size() != size_t(lp.num_col_) ||
      s.row_value.size() != size_t(lp.num_row_) ||
      s.row_dual.size() != size_t(lp.num_row_) ||
      b.col_status.size() != size_t(lp.num_col_) ||
      b.row_status.size() != size_t(lp.num_row_))
    return false;
  HighsInt basic = 0;
  auto checkStatus = [&](HighsBasisStatus status) {
    if (status == HighsBasisStatus::kBasic) ++basic;
    return status == HighsBasisStatus::kBasic ||
           status == HighsBasisStatus::kLower ||
           status == HighsBasisStatus::kUpper ||
           status == HighsBasisStatus::kZero ||
           status == HighsBasisStatus::kNonbasic;
  };
  for (auto status : b.col_status)
    if (!checkStatus(status)) return false;
  for (auto status : b.row_status)
    if (!checkStatus(status)) return false;
  if (basic != lp.num_row_) return false;
  const double pt = o.primal_feasibility_tolerance;
  const double dt = o.dual_feasibility_tolerance;
  if (!(pt > 0) || !(dt > 0) || !std::isfinite(pt) || !std::isfinite(dt))
    return false;
  const long double sense = static_cast<int>(lp.sense_);
  std::vector<long double> rows(lp.num_row_, 0);
  long double primal = sense * lp.offset_, dual = primal, allowance = 0;
  auto dualTerm = [&](double multiplier, double lo, double up,
                      long double value) {
    if (!std::isfinite(multiplier)) return false;
    const long double d = sense * multiplier;
    if (d == 0) return true;
    const double bound = d > 0 ? lo : up;
    // A tiny residual on a free/non-active variable is allowed by native dual
    // tolerance. It contributes a residual allowance, never a proof bound.
    if (std::abs(d) <= dt) {
      allowance += std::abs(d * value);
      return true;
    }
    if (!std::isfinite(bound) || std::abs(bound) >= kHighsInf ||
        std::abs(value - bound) > pt)
      return false;
    dual += d * bound;
    allowance += std::abs(d) * pt;
    return true;
  };
  for (HighsInt j = 0; j < lp.num_col_; ++j) {
    if (lp.a_matrix_.start_[j] > lp.a_matrix_.start_[j + 1] ||
        lp.a_matrix_.start_[j + 1] < 0 ||
        size_t(lp.a_matrix_.start_[j + 1]) > lp.a_matrix_.value_.size() ||
        !std::isfinite(lp.col_cost_[j]) ||
        !within(s.col_value[j], lp.col_lower_[j], lp.col_upper_[j], pt) ||
        !dualTerm(s.col_dual[j], lp.col_lower_[j], lp.col_upper_[j],
                  s.col_value[j]))
      return false;
    long double stationarity = lp.col_cost_[j] - s.col_dual[j];
    primal += sense * lp.col_cost_[j] * s.col_value[j];
    for (HighsInt k = lp.a_matrix_.start_[j]; k < lp.a_matrix_.start_[j + 1];
         ++k) {
      const HighsInt i = lp.a_matrix_.index_[k];
      const double a = lp.a_matrix_.value_[k];
      if (i < 0 || i >= lp.num_row_ || !std::isfinite(a) ||
          !std::isfinite(s.row_dual[i]))
        return false;
      rows[i] += (long double)a * s.col_value[j];
      stationarity -= (long double)a * s.row_dual[i];
    }
    if (std::abs(stationarity) > dt) return false;
    allowance += dt * std::abs(s.col_value[j]);
  }
  for (HighsInt i = 0; i < lp.num_row_; ++i) {
    if (!within(rows[i], lp.row_lower_[i], lp.row_upper_[i], pt) ||
        !std::isfinite(s.row_value[i]) ||
        std::abs(rows[i] - s.row_value[i]) > pt ||
        !dualTerm(s.row_dual[i], lp.row_lower_[i], lp.row_upper_[i], rows[i]))
      return false;
  }
  const long double roundoff = 64 * std::numeric_limits<double>::epsilon() *
                               (1 + std::abs(primal) + std::abs(dual));
  return std::isfinite(primal) && std::isfinite(dual) &&
         std::abs(primal - dual) <= allowance + roundoff;
}

HighsRootLpPortfolioResult HighsRootLpPortfolio::offerBasis(
    Highs& owner, HighsSolveCapacity& capacity, double seconds,
    const std::function<bool()>& owner_poll) {
  HighsRootLpPortfolioResult result;
  const auto start = Clock::now();
  const HighsLp& lp = owner.getLp();
  size_t bytes;
  auto* deque = HighsTaskExecutor::getThisWorkerDeque();
  if (!deque || deque->getNumWorkers() < 2 || !(seconds > 0) ||
      !std::isfinite(seconds) || owner.getBasis().valid ||
      owner.getModel().hessian_.dim_ != 0 || lp.isMip() ||
      !lp.a_matrix_.isColwise() || !copyBytes(lp, bytes))
    return result;
  HighsSolveCapacity::Lease lease;
  if (!capacity.reserve(2, bytes, lease)) return result;
  if (owner_poll && owner_poll()) {
    result.cancelled = true;
    return result;
  }
  result.attempted = true;
  std::atomic<int> winner{-1};
  std::atomic<bool> cancelled{false};
  std::exception_ptr owner_exception;
  std::unique_ptr<Highs> arms[2];
  int64_t iterations[2] = {0, 0};
  auto run = [&](int id) {
    try {
      if (elapsed(start) >= seconds || winner.load() >= 0 || cancelled.load())
        return;
      arms[id].reset(new Highs());
      auto& arm = *arms[id];
      HighsOptions options = owner.getOptions();
      // Do not inherit files, callbacks, owner profiling, or an inner executor.
      options.log_file.clear();
      options.output_flag = false;
      options.log_to_console = false;
      options.write_solution_to_file = false;
      options.write_model_to_file = false;
      options.highs_analysis_level = 0;
      options.threads = deque->getNumWorkers();
      options.parallel = kHighsOffString;
      options.presolve = kHighsOffString;
      options.solver = kSimplexString;
      options.simplex_strategy = id == 0 ? 4 : 1;
      options.objective_bound = kHighsInf;
      options.objective_target = -kHighsInf;
      // Scaling was already applied by the MIP owner.
      options.user_objective_scale = 0;
      options.user_bound_scale = 0;
      if (arm.passOptions(options) != HighsStatus::kOk ||
          arm.passModel(lp) != HighsStatus::kOk)
        return;
      const double left = seconds - elapsed(start);
      if (!(left > 0) || winner.load() >= 0 || cancelled.load()) return;
      arm.setOptionValue("time_limit", left);
      arm.setCallback(
          [&](int, const std::string&, const HighsCallbackOutput*,
              HighsCallbackInput* input, void*) {
            // Arm zero runs synchronously on the MIP owner. The private dual
            // arm never calls external code or touches the owner's callback.
            if (id == 0 && owner_poll && owner_poll()) cancelled.store(true);
            input->user_interrupt = cancelled.load() || winner.load() >= 0 ||
                                    elapsed(start) >= seconds;
          },
          nullptr);
      arm.startCallback(kCallbackSimplexInterrupt);
      const auto status = arm.run();
      result.status[id] = arm.getModelStatus();
      iterations[id] =
          std::max<HighsInt>(0, arm.getInfo().simplex_iteration_count);
      if (status == HighsStatus::kOk &&
          arm.getModelStatus() == HighsModelStatus::kOptimal &&
          elapsed(start) < seconds && !cancelled.load() &&
          checkedOptimalBasis(lp, arm.getSolution(), arm.getBasis(), options)) {
        int expected = -1;
        winner.compare_exchange_strong(expected, id);
      }
    } catch (...) {
      if (id == 0) {
        owner_exception = std::current_exception();
        cancelled.store(true);
      }
      // A failed optional arm has no authority over the owner's status.
      result.status[id] = HighsModelStatus::kSolveError;
    }
  };
  {
    highs::parallel::TaskGroup tasks;
    tasks.spawn([&]() { run(1); });
    run(0);
    tasks.taskWait();
  }
  result.iterations = iterations[0] + iterations[1];
  if (owner_exception) std::rethrow_exception(owner_exception);
  if (owner_poll && owner_poll()) cancelled.store(true);
  result.cancelled = cancelled.load();
  result.winner = winner.load();
  if (result.winner >= 0 && !result.cancelled && elapsed(start) < seconds) {
    try {
      result.basis_installed =
          owner.setBasis(arms[result.winner]->getBasis(),
                         "checked root LP portfolio") == HighsStatus::kOk;
    } catch (...) {
      result.basis_installed = false;
    }
    if (!result.basis_installed) owner.setBasis();
  }
  result.elapsed = elapsed(start);
  return result;
}
