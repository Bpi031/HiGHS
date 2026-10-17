// Isolated allocation-failure testing: replacement new/delete must never be
// linked into the solver or the shared unit-test process.
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <vector>

#include "simplex/HEkkDual.h"
#include "util/HFactor.h"

namespace {
std::atomic<long> live_blocks{0};
std::atomic<int> fail_after{-1};
std::atomic<int> allocation_calls{0};
std::atomic<bool> track_allocations{false};
std::atomic<void*> tracked_blocks[128]{};

void* allocate(std::size_t bytes) {
  ++allocation_calls;
  const int remaining = fail_after.load();
  if (remaining >= 0 && fail_after.fetch_sub(1) == 0) {
    fail_after.store(-1);
    throw std::bad_alloc();
  }
  void* result = std::malloc(bytes ? bytes : 1);
  if (!result) throw std::bad_alloc();
  // Count only owned addresses allocated during the selected update. Runtime
  // and setup allocation paths need not interpose the same way on every host.
  if (track_allocations.load()) {
    bool registered = false;
    for (auto& block : tracked_blocks) {
      void* empty = nullptr;
      if (block.compare_exchange_strong(empty, result)) {
        registered = true;
        ++live_blocks;
        break;
      }
    }
    if (!registered) {
      std::free(result);
      throw std::bad_alloc();
    }
  }
  return result;
}

void release(void* pointer) noexcept {
  if (!pointer) return;
  for (auto& block : tracked_blocks) {
    void* expected = pointer;
    if (block.compare_exchange_strong(expected, nullptr)) {
      --live_blocks;
      break;
    }
  }
  std::free(pointer);
}

void packedVector(HVector& vector, double first, double second) {
  vector.setup(2);
  vector.array[0] = first;
  vector.array[1] = second;
  vector.count = 0;
  if (first) vector.index[vector.count++] = 0;
  if (second) vector.index[vector.count++] = 1;
  vector.packFlag = true;
  vector.pack();
}

bool correctSolve(double first, double second) {
  return std::isfinite(first) && std::isfinite(second) &&
         std::abs(first - 2) <= 1e-10 && std::abs(second + 1) <= 1e-10;
}
}  // namespace

void* operator new(std::size_t bytes) { return allocate(bytes); }
void* operator new[](std::size_t bytes) { return allocate(bytes); }
void* operator new(std::size_t bytes, const std::nothrow_t&) noexcept {
  try {
    return allocate(bytes);
  } catch (...) {
    return nullptr;
  }
}
void* operator new[](std::size_t bytes, const std::nothrow_t& tag) noexcept {
  return operator new(bytes, tag);
}
void operator delete(void* pointer) noexcept { release(pointer); }
void operator delete[](void* pointer) noexcept { release(pointer); }
void operator delete(void* pointer, const std::nothrow_t&) noexcept {
  release(pointer);
}
void operator delete[](void* pointer, const std::nothrow_t&) noexcept {
  release(pointer);
}
#if defined(__cpp_sized_deallocation)
void operator delete(void* pointer, std::size_t) noexcept { release(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { release(pointer); }
#endif

int checkAllocations(bool simplex_caller) {
  const char* path = simplex_caller ? "simplex" : "factor";
  int injected = 0;
  // Visit each allocation ordinal until the complete update no longer throws.
  // Failed factors are destroyed, not reused: this checks resource ownership,
  // not a strong exception guarantee for a partially changed factorization.
  for (int ordinal = 0; ordinal < 128; ++ordinal) {
    const long original_live = live_blocks.load();
    bool threw = false;
    int calls = 0;
    {
      HighsSparseMatrix matrix;
      matrix.num_row_ = 2;
      matrix.num_col_ = 2;
      matrix.format_ = MatrixFormat::kColwise;
      matrix.start_ = {0, 2, 4};
      matrix.index_ = {0, 1, 0, 1};
      matrix.value_ = {2, 1, 1, 2};
      std::vector<HighsInt> basis{2, 3};
      HFactor factor;
      factor.setup(matrix, basis);
      if (factor.build()) return 2;
      HighsOptions options;
      options.output_flag = false;
      HighsTimer timer;
      HEkk ekk;
      ekk.lp_.num_row_ = 2;
      ekk.lp_.num_col_ = 2;
      ekk.lp_.a_matrix_ = matrix;
      ekk.lp_.col_cost_ = {0, 0};
      ekk.lp_.col_lower_ = {0, 0};
      ekk.lp_.col_upper_ = {kHighsInf, kHighsInf};
      ekk.lp_.row_lower_ = {0, 0};
      ekk.lp_.row_upper_ = {0, 0};
      ekk.setPointers(nullptr, &options, &timer);
      ekk.initialiseEkk();
      ekk.info_.simplex_strategy = kSimplexStrategyDualPlain;
      if (ekk.initialiseSimplexLpBasisAndFactor() != HighsStatus::kOk) return 2;
      HEkkDual dual(ekk);
      HVector columns[2];
      HVector rows[2];
      // Partial transforms are unchanged for the starting identity basis.
      packedVector(columns[0], 2, 1);
      packedVector(columns[1], 1, 2);
      packedVector(rows[0], 1, 0);
      packedVector(rows[1], 0, 1);
      columns[0].next = &columns[1];
      rows[0].next = &rows[1];
      HighsInt leaving_rows[2] = {0, 1};
      HighsInt hint = 0;
      dual.multi_nFinish = 2;
      dual.rebuild_reason = 0;
      for (int row = 0; row < 2; ++row) {
        dual.multi_finish[row].col_aq = &columns[row];
        dual.multi_finish[row].row_ep = &rows[row];
        dual.multi_finish[row].row_out = row;
        // Real simplex pivots replace basic indices before updating INVERT.
        ekk.basis_.basicIndex_[row] = row;
      }
      allocation_calls.store(0);
      track_allocations.store(true);
      fail_after.store(ordinal);
      try {
        if (simplex_caller)
          dual.majorUpdateFactor();
        else
          factor.update(columns, rows, leaving_rows, &hint);
      } catch (const std::bad_alloc&) {
        threw = true;
        ++injected;
      }
      fail_after.store(-1);
      track_allocations.store(false);
      calls = allocation_calls.load();
      if (!threw) {
        basis[0] = 0;
        basis[1] = 1;
        std::vector<double> forward{3, 0};
        std::vector<double> transpose{3, 0};
        if (simplex_caller) {
          HVector forward_vector;
          HVector transpose_vector;
          packedVector(forward_vector, 3, 0);
          packedVector(transpose_vector, 3, 0);
          ekk.ftran(forward_vector, 1);
          ekk.btran(transpose_vector, 1);
          forward = forward_vector.array;
          transpose = transpose_vector.array;
        } else {
          factor.ftranCall(forward);
          factor.btranCall(transpose);
        }
        if (!correctSolve(forward[0], forward[1]) ||
            !correctSolve(transpose[0], transpose[1]))
          return 3;
      }
    }
    if (live_blocks.load() != original_live) {
      std::fprintf(stderr, "%s allocation ordinal %d leaked %ld blocks\n",
                   path, ordinal, live_blocks.load() - original_live);
      return 1;
    }
    if (!threw) {
      if (injected < 7 || ordinal < calls) return 4;
      std::printf("%s-allocation-check: passed injections=%d allocations=%d "
                  "forward_transpose_checks=4\n", path, injected, calls);
      return 0;
    }
  }
  std::fprintf(stderr, "%s exceeded the bounded allocation test\n", path);
  return 5;
}

int main() {
  // Calibrate address tracking and the numerical oracle before solver testing.
  track_allocations.store(true);
  void* allocation = operator new[](8);
  track_allocations.store(false);
  if (live_blocks.load() != 1) return 6;
  operator delete[](allocation);
  if (live_blocks.load() != 0 || !correctSolve(2, -1) ||
      correctSolve(std::numeric_limits<double>::quiet_NaN(), -1) ||
      correctSolve(2, std::numeric_limits<double>::infinity()))
    return 6;
  const int factor_status = checkAllocations(false);
  return factor_status ? factor_status : checkAllocations(true);
}
