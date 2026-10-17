// This allocation interposer belongs only in a static, isolated test executable.
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <vector>

#include "simplex/HEkk.h"

namespace {
std::atomic<int> fail_after{-1};
std::atomic<int> allocation_calls{0};

void* allocate(std::size_t bytes) {
  ++allocation_calls;
  if (fail_after.load() >= 0 && fail_after.fetch_sub(1) == 0) {
    fail_after.store(-1);
    throw std::bad_alloc();
  }
  void* pointer = std::malloc(bytes ? bytes : 1);
  if (!pointer) throw std::bad_alloc();
  return pointer;
}

bool sameBasis(const SimplexBasis& a, const SimplexBasis& b) {
  return a.basicIndex_ == b.basicIndex_ &&
         a.nonbasicFlag_ == b.nonbasicFlag_ &&
         a.nonbasicMove_ == b.nonbasicMove_ && a.hash == b.hash &&
         a.debug_id == b.debug_id &&
         a.debug_update_count == b.debug_update_count &&
         a.debug_origin_name == b.debug_origin_name;
}

bool checkIdentityFactor(HEkk& ekk) {
  for (bool transpose : {false, true}) {
    HVector rhs;
    rhs.setup(2);
    rhs.array[0] = 3;
    rhs.array[1] = -4;
    rhs.index[0] = 0;
    rhs.index[1] = 1;
    rhs.count = 2;
    if (transpose)
      ekk.btran(rhs, 1);
    else
      ekk.ftran(rhs, 1);
    if (rhs.array[0] != 3 || rhs.array[1] != -4) return false;
  }
  return true;
}

int checkCapture(bool warm, bool edge_weights) {
  int injected = 0;
  for (int ordinal = 0; ordinal < 64; ++ordinal) {
    HighsOptions options;
    options.output_flag = false;
    HighsTimer timer;
    HEkk ekk;
    ekk.lp_.num_row_ = 2;
    ekk.lp_.num_col_ = 2;
    ekk.lp_.a_matrix_.num_row_ = 2;
    ekk.lp_.a_matrix_.num_col_ = 2;
    ekk.lp_.a_matrix_.format_ = MatrixFormat::kColwise;
    ekk.lp_.a_matrix_.start_ = {0, 2, 4};
    ekk.lp_.a_matrix_.index_ = {0, 1, 0, 1};
    ekk.lp_.a_matrix_.value_ = {2, -1, 1, 3};
    ekk.lp_.col_cost_ = {-1, 2};
    ekk.lp_.col_lower_ = {0, 0};
    ekk.lp_.col_upper_ = {10, 10};
    ekk.lp_.row_lower_ = {0, 0};
    ekk.lp_.row_upper_ = {0, 0};
    ekk.setPointers(nullptr, &options, &timer);
    ekk.initialiseEkk();
    ekk.info_.simplex_strategy = kSimplexStrategyDualPlain;
    if (ekk.initialiseSimplexLpBasisAndFactor() != HighsStatus::kOk)
      return 2;
    ekk.status_.has_dual_steepest_edge_weights = edge_weights;
    if (edge_weights) ekk.dual_edge_weight_ = {1, 4};
    if (warm) {
      ekk.status_.has_dual_steepest_edge_weights = false;
      ekk.putIterate();
      ekk.status_.has_dual_steepest_edge_weights = edge_weights;
    }
    // A long metadata value checks exact copying; libc++ string allocations may
    // not pass through this executable's vector-allocation interposer.
    ekk.basis_.debug_origin_name.assign(512, 'x');
    const SimplexBasis expected_basis = ekk.basis_;
    const std::vector<double> expected_weights = ekk.dual_edge_weight_;
    allocation_calls.store(0);
    fail_after.store(ordinal);
    bool threw = false;
    try {
      ekk.putIterate();
    } catch (const std::bad_alloc&) {
      threw = true;
      ++injected;
    }
    fail_after.store(-1);
    const int calls = allocation_calls.load();
    if (threw && ekk.getIterate() != HighsStatus::kError) {
      std::fprintf(stderr,
                   "incomplete iterate accepted: warm=%d weights=%d ordinal=%d\n",
                   int(warm), int(edge_weights), ordinal);
      return 1;
    }
    if (!sameBasis(ekk.basis_, expected_basis) ||
        ekk.dual_edge_weight_ != expected_weights || !checkIdentityFactor(ekk))
      return 3;
    // A failed snapshot must not damage the live factor or prevent a retry.
    if (threw) ekk.putIterate();
    if (ekk.getIterate() != HighsStatus::kOk ||
        !sameBasis(ekk.basis_, expected_basis) ||
        ekk.dual_edge_weight_ != expected_weights || !checkIdentityFactor(ekk))
      return 4;
    if (!threw) {
      if ((!warm || edge_weights) && injected == 0) return 5;
      if (ordinal != calls || injected != calls) return 5;
      std::printf("iterate-allocation-check: warm=%d weights=%d injections=%d "
                  "allocations=%d forward_transpose_checks=%d passed\n",
                  int(warm), int(edge_weights), injected, calls,
                  4 * (injected + 1));
      return 0;
    }
  }
  return 6;
}
}  // namespace

void* operator new(std::size_t bytes) { return allocate(bytes); }
void* operator new[](std::size_t bytes) { return allocate(bytes); }
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }
#if defined(__cpp_sized_deallocation)
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { std::free(pointer); }
#endif

int main() {
  for (bool warm : {false, true}) {
    for (bool weights : {false, true}) {
      const int status = checkCapture(warm, weights);
      if (status) return status;
    }
  }
  return 0;
}
