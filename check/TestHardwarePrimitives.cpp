#include <cstdint>
#include <cmath>
#include <cstring>
#include <future>
#include <limits>
#include <mutex>
#include <random>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <vector>

#include "catch.hpp"
#include <random>
#include "Highs.h"
#include "interfaces/highs_c_api.h"
#include "parallel/HighsCacheAlign.h"
#include "parallel/HighsCombinable.h"
#include "parallel/HighsHardwareTopology.h"
#include "parallel/HighsParallel.h"
#include "parallel/HighsSpinMutex.h"
#include "qpsolver/qpvector.hpp"
#include "simplex/HEkkDualRow.h"
#include "util/FactorTimer.h"
#include "util/HFactor.h"
#include "util/HVector.h"
#include "util/HighsTimer.h"
#include "util/HighsSparseVectorSum.h"

namespace {
struct SchedulerScope {
  explicit SchedulerScope(int workers) {
    HighsTaskExecutor::shutdown(true);
    highs::parallel::initialize_scheduler(workers);
  }
  ~SchedulerScope() { HighsTaskExecutor::shutdown(true); }
};

struct alignas(256) OverAligned {
  unsigned char bytes[256];
};

struct TrackedValue {
  static int live;
  int value;
  explicit TrackedValue(int value = 0) : value(value) { ++live; }
  TrackedValue(const TrackedValue& other) : value(other.value) { ++live; }
  TrackedValue(TrackedValue&& other) noexcept : value(other.value) { ++live; }
  TrackedValue& operator=(TrackedValue&& other) noexcept {
    value = other.value;
    return *this;
  }
  ~TrackedValue() { --live; }
};
int TrackedValue::live = 0;

int failingWorker = 0;
std::atomic<int> startedWorkers{0};
std::atomic<int> exitedWorkers{0};
std::thread launchWithFailure(void (*run)(int, HighsTaskExecutor*), int id,
                              HighsTaskExecutor* executor) {
  if (id == failingWorker)
    throw std::system_error(
        std::make_error_code(std::errc::resource_unavailable_try_again));
  return std::thread([=]() {
    ++startedWorkers;
    run(id, executor);
    ++exitedWorkers;
  });
}
std::thread launchEmpty(void (*)(int, HighsTaskExecutor*), int,
                        HighsTaskExecutor*) {
  return std::thread();
}
std::thread launchDelayed(void (*run)(int, HighsTaskExecutor*), int id,
                          HighsTaskExecutor* executor) {
  return std::thread([=]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    run(id, executor);
  });
}
}  // namespace

namespace {
std::vector<long double> denseFactorOracle(
    const HighsSparseMatrix& matrix, const std::vector<HighsInt>& basis,
    const std::vector<double>& rhs, bool transpose) {
  const HighsInt n = matrix.num_row_;
  std::vector<long double> dense(n * n, 0);
  for (HighsInt col = 0; col < n; ++col) {
    const HighsInt variable = basis[col];
    if (variable >= matrix.num_col_) {
      const HighsInt row = variable - matrix.num_col_;
      dense[transpose ? col * n + row : row * n + col] = 1;
    } else {
      for (HighsInt k = matrix.start_[variable];
           k < matrix.start_[variable + 1]; ++k) {
        const HighsInt row = matrix.index_[k];
        dense[transpose ? col * n + row : row * n + col] += matrix.value_[k];
      }
    }
  }
  std::vector<long double> result(rhs.begin(), rhs.end());
  for (HighsInt col = 0; col < n; ++col) {
    HighsInt pivot = col;
    for (HighsInt row = col + 1; row < n; ++row)
      if (std::abs(dense[row * n + col]) >
          std::abs(dense[pivot * n + col])) pivot = row;
    REQUIRE(std::abs(dense[pivot * n + col]) > 1e-14L);
    for (HighsInt other = col; other < n; ++other)
      std::swap(dense[col * n + other], dense[pivot * n + other]);
    std::swap(result[col], result[pivot]);
    for (HighsInt row = col + 1; row < n; ++row) {
      const long double multiplier = dense[row * n + col] / dense[col * n + col];
      for (HighsInt other = col + 1; other < n; ++other)
        dense[row * n + other] -= multiplier * dense[col * n + other];
      result[row] -= multiplier * result[col];
    }
  }
  for (HighsInt row = n - 1; row >= 0; --row) {
    for (HighsInt col = row + 1; col < n; ++col)
      result[row] -= dense[row * n + col] * result[col];
    result[row] /= dense[row * n + row];
  }
  return result;
}

void checkIndexedFactorSolve(const HFactor& factor,
                            const HighsSparseMatrix& matrix,
                            const std::vector<HighsInt>& basis,
                            const std::vector<double>& input) {
  const HighsInt n = matrix.num_row_;
  const auto original_invert = factor.getInvert();
  for (bool transpose : {false, true}) {
    const auto reference = denseFactorOracle(matrix, basis, input, transpose);
    for (double expected_density : {0.0, 1.0}) {
      CAPTURE(transpose, expected_density);
      HVector rhs;
      rhs.setup(n);
      for (HighsInt row = 0; row < n; ++row)
        if (input[row] != 0) {
          rhs.index[rhs.count++] = row;
          rhs.array[row] = input[row];
        }
      if (transpose) factor.btranCall(rhs, expected_density);
      else factor.ftranCall(rhs, expected_density);
      REQUIRE(rhs.count >= 0);
      REQUIRE(rhs.count <= n);
      std::vector<bool> indexed(n, false);
      for (HighsInt k = 0; k < rhs.count; ++k) {
        REQUIRE(rhs.index[k] >= 0);
        REQUIRE(rhs.index[k] < n);
        REQUIRE_FALSE(indexed[rhs.index[k]]);
        indexed[rhs.index[k]] = true;
      }
      for (HighsInt row = 0; row < n; ++row) {
        REQUIRE(std::isfinite(rhs.array[row]));
        REQUIRE(std::abs(rhs.array[row] - reference[row]) <=
                1e-10L * (1 + std::abs(reference[row])));
        if (rhs.array[row] != 0) REQUIRE(indexed[row]);
      }
      // Each solve owns its scratch marks; a later RHS must not see old DFS state.
      for (char mark : rhs.cwork) REQUIRE(mark == 0);
    }
  }
  const auto final_invert = factor.getInvert();
  const auto integer_members = {
           &InvertibleRepresentation::l_pivot_index,
           &InvertibleRepresentation::l_pivot_lookup,
           &InvertibleRepresentation::l_start,
           &InvertibleRepresentation::l_index,
           &InvertibleRepresentation::lr_start,
           &InvertibleRepresentation::lr_index,
           &InvertibleRepresentation::u_pivot_lookup,
           &InvertibleRepresentation::u_pivot_index,
           &InvertibleRepresentation::u_start,
           &InvertibleRepresentation::u_last_p,
           &InvertibleRepresentation::u_index,
           &InvertibleRepresentation::ur_start,
           &InvertibleRepresentation::ur_lastp,
           &InvertibleRepresentation::ur_space,
           &InvertibleRepresentation::ur_index,
           &InvertibleRepresentation::pf_start,
           &InvertibleRepresentation::pf_index,
           &InvertibleRepresentation::pf_pivot_index};
  const auto real_members = {
           &InvertibleRepresentation::l_value,
           &InvertibleRepresentation::lr_value,
           &InvertibleRepresentation::u_pivot_value,
           &InvertibleRepresentation::u_value,
           &InvertibleRepresentation::ur_value,
           &InvertibleRepresentation::pf_value,
           &InvertibleRepresentation::pf_pivot_value};
  auto reused_invert = original_invert;
  for (auto member : integer_members) (reused_invert.*member).push_back(-1);
  for (auto member : real_members) (reused_invert.*member).push_back(-0.0);
  factor.getInvert(reused_invert);
  for (auto member : integer_members) {
    REQUIRE(original_invert.*member == final_invert.*member);
    REQUIRE(original_invert.*member == reused_invert.*member);
  }
  for (auto member : real_members) {
    const auto& original = original_invert.*member;
    const auto& final = final_invert.*member;
    REQUIRE(original.size() == final.size());
    if (!original.empty())
      REQUIRE(std::memcmp(original.data(), final.data(),
                          original.size() * sizeof(double)) == 0);
    const auto& reused = reused_invert.*member;
    REQUIRE(original.size() == reused.size());
    if (!original.empty())
      REQUIRE(std::memcmp(original.data(), reused.data(),
                          original.size() * sizeof(double)) == 0);
  }
}
}  // namespace

TEST_CASE("Sparse vector sum resize preserves only the retained prefix",
          "[highs_hardware][sparse_sum_resize_contract]") {
  HighsSparseVectorSum sum(128);
  sum.add(100, 3.0);
  sum.add(5, 1.25);
  sum.add(101, HighsCDouble(-4.0));
  sum.add(2, HighsCDouble(-2.5));
  sum.add(100, -3.0);
  sum.setDimension(8);
  REQUIRE(sum.values.size() == 8);
  REQUIRE(sum.getNonzeros() == std::vector<HighsInt>{5, 2});
  REQUIRE(sum.getValue(5) == 1.25);
  REQUIRE(sum.getValue(2) == -2.5);

  sum.setDimension(8);
  REQUIRE(sum.getNonzeros() == std::vector<HighsInt>{5, 2});
  sum.setDimension(128);
  REQUIRE(sum.getNonzeros() == std::vector<HighsInt>{5, 2});
  REQUIRE(sum.getValue(100) == 0.0);
  sum.add(100, 7.0);
  REQUIRE(sum.getNonzeros() == std::vector<HighsInt>{5, 2, 100});
  sum.setDimension(8);
  std::vector<HighsCDouble>(sum.values).swap(sum.values);
  sum.clear();
  REQUIRE(sum.getNonzeros().empty());
  for (HighsInt i = 0; i < 8; ++i) REQUIRE(sum.getValue(i) == 0.0);

  sum.add(7, 2.0);
  sum.setDimension(0);
  REQUIRE(sum.values.empty());
  REQUIRE(sum.getNonzeros().empty());
  sum.clear();
  sum.setDimension(8);
  for (HighsInt i = 0; i < 8; ++i) REQUIRE(sum.getValue(i) == 0.0);
}

TEST_CASE("Sparse vector sum lifecycle matches an independent dense oracle",
          "[highs_hardware][sparse_sum_lifecycle_contract]") {
  std::mt19937 random(30073);
  for (int trial = 0; trial < 64; ++trial) {
    HighsSparseVectorSum sum;
    std::vector<long double> reference;
    std::vector<bool> touched;
    for (int step = 0; step < 400; ++step) {
      const unsigned operation = random() % 7;
      if (operation == 0 || reference.empty()) {
        const HighsInt dimension = random() % 129;
        sum.setDimension(dimension);
        reference.resize(dimension, 0.0L);
        touched.resize(dimension, false);
      } else if (operation <= 3) {
        const HighsInt index = random() % reference.size();
        const double value = (static_cast<int>(random() % 17) - 8) * 0.25;
        if (step % 2) sum.add(index, value);
        else sum.add(index, HighsCDouble(value));
        reference[index] += value;
        touched[index] = true;
      } else if (operation == 4) {
        sum.cleanup([](HighsInt, double value) {
          return std::abs(value) <= 0.25;
        });
        for (size_t i = 0; i < reference.size(); ++i)
          if (std::abs(reference[i]) <= 0.25L) {
            reference[i] = 0.0L;
            touched[i] = false;
          }
      } else if (operation == 5) {
        const HighsInt even_count = sum.partition([](HighsInt index) {
          return index % 2 == 0;
        });
        HighsInt expected_count = 0;
        for (size_t i = 0; i < touched.size(); ++i)
          if (touched[i] && i % 2 == 0) ++expected_count;
        REQUIRE(even_count == expected_count);
        const auto& indices = sum.getNonzeros();
        for (HighsInt i = 0; i < static_cast<HighsInt>(indices.size()); ++i)
          REQUIRE((indices[i] % 2 == 0) == (i < even_count));
      } else {
        sum.clear();
        std::fill(reference.begin(), reference.end(), 0.0L);
        std::fill(touched.begin(), touched.end(), false);
      }

      REQUIRE(sum.values.size() == reference.size());
      std::vector<bool> seen(reference.size(), false);
      for (HighsInt index : sum.getNonzeros()) {
        REQUIRE(index >= 0);
        REQUIRE(index < static_cast<HighsInt>(reference.size()));
        REQUIRE_FALSE(seen[index]);
        seen[index] = true;
      }
      REQUIRE(seen == touched);
      for (size_t i = 0; i < reference.size(); ++i) {
        const double value = sum.getValue(static_cast<HighsInt>(i));
        if (reference[i] == 0.0L) {
          // Cancellation retains an index with the existing tiny sentinel.
          REQUIRE(std::abs(value) <= (std::numeric_limits<double>::min)());
        } else {
          REQUIRE(value == static_cast<double>(reference[i]));
        }
      }
      HighsSparseVectorSum copy = sum;
      copy.clear();
      REQUIRE(copy.getNonzeros().empty());
      REQUIRE(sum.getNonzeros().size() ==
              static_cast<size_t>(std::count(touched.begin(), touched.end(), true)));
    }
  }
}

TEST_CASE("Dual pivot-row packing preserves append order and source state",
          "[highs_hardware][dual_row_pack_contract]") {
  std::mt19937 random(7291);
  for (int trial = 0; trial < 160; ++trial) {
    const HighsInt size = trial < 8 ? trial : random() % 257;
    HVector left;
    HVector right;
    left.setup(size);
    right.setup(size);
    const HighsInt left_count = size ? random() % (size + 1) : 0;
    const HighsInt right_count = size ? random() % (size + 1) : 0;
    for (HighsInt i = 0; i < size; ++i) {
      left.index[i] = i;
      right.index[i] = i;
    }
    std::shuffle(left.index.begin(), left.index.end(), random);
    std::shuffle(right.index.begin(), right.index.end(), random);
    const double values[] = {0.0, -0.0, 1.0, -3.0, kHighsTiny / 2,
                             std::numeric_limits<double>::denorm_min(),
                             1e100};
    for (HighsInt i = 0; i < left_count; ++i)
      left.array[left.index[i]] = values[i % 7];
    for (HighsInt i = 0; i < right_count; ++i)
      right.array[right.index[i]] = values[(i + 3) % 7];
    left.count = left_count;
    right.count = right_count;
    left.synthetic_tick = 11;
    right.synthetic_tick = 19;
    left.packFlag = true;
    right.packFlag = true;
    const auto left_values = left.array;
    const auto right_values = right.array;
    const auto left_indices = left.index;
    const auto right_indices = right.index;

    HEkk simplex;
    HEkkDualRow packed(simplex);
    const HighsInt prefix = trial % 4;
    const HighsInt capacity = prefix + 2 * size;
    packed.packIndex.assign(capacity, -23);
    packed.packValue.assign(capacity, 17.0);
    packed.packCount = prefix;
    packed.workCount = 7;
    packed.workDelta = -2;
    auto expected_indices = packed.packIndex;
    auto expected_values = packed.packValue;
    HighsInt expected_count = prefix;
    for (const auto* input : {&left, &right}) {
      const HighsInt offset = input == &left ? 0 : size;
      for (HighsInt i = 0; i < input->count; ++i) {
        expected_indices[expected_count] = input->index[i] + offset;
        expected_values[expected_count++] = input->array[input->index[i]];
      }
      packed.chooseMakepack(input, offset);
      REQUIRE(packed.packCount == expected_count);
      REQUIRE(packed.packIndex == expected_indices);
      if (capacity)
        REQUIRE(std::memcmp(packed.packValue.data(), expected_values.data(),
                            capacity * sizeof(double)) == 0);
      REQUIRE(packed.workCount == 7);
      REQUIRE(packed.workDelta == -2);
    }
    REQUIRE(left.index == left_indices);
    REQUIRE(right.index == right_indices);
    REQUIRE(left.count == left_count);
    REQUIRE(right.count == right_count);
    REQUIRE(left.synthetic_tick == 11);
    REQUIRE(right.synthetic_tick == 19);
    REQUIRE(left.packFlag);
    REQUIRE(right.packFlag);
    if (size) {
      REQUIRE(std::memcmp(left.array.data(), left_values.data(),
                          size * sizeof(double)) == 0);
      REQUIRE(std::memcmp(right.array.data(), right_values.data(),
                          size * sizeof(double)) == 0);
    }
  }
}

TEST_CASE("Dual pivot-row packing is invariant under indexed row splitting",
          "[highs_hardware][dual_row_pack_contract]") {
  for (HighsInt size : {0, 1, 7, 32, 127}) {
    HVector whole;
    whole.setup(size);
    whole.count = size;
    for (HighsInt i = 0; i < size; ++i) {
      whole.index[i] = size - i - 1;
      whole.array[i] = i % 3 ? (i - 17) / 8.0 : -0.0;
    }
    for (HighsInt split = 0; split <= size; ++split) {
      HVector first(whole);
      HVector second(whole);
      first.count = split;
      second.count = size - split;
      std::copy(whole.index.begin() + split, whole.index.end(),
                second.index.begin());
      HEkk simplex;
      HEkkDualRow complete(simplex);
      HEkkDualRow pieces(simplex);
      complete.packIndex.resize(size);
      complete.packValue.resize(size);
      pieces.packIndex.resize(size);
      pieces.packValue.resize(size);
      complete.chooseMakepack(&whole, 13);
      pieces.chooseMakepack(&first, 13);
      pieces.chooseMakepack(&second, 13);
      REQUIRE(complete.packCount == size);
      REQUIRE(pieces.packCount == size);
      REQUIRE(complete.packIndex == pieces.packIndex);
      if (size)
        REQUIRE(std::memcmp(complete.packValue.data(), pieces.packValue.data(),
                            size * sizeof(double)) == 0);
    }
  }
}

TEST_CASE("Dual candidates obey sign, strict threshold and relaxed step bound",
          "[highs_hardware][dual_candidate_contract]") {
  std::mt19937 random(12073);
  for (HighsInt update_count : {0, 9, 10, 19, 20}) {
    const double threshold = update_count < 10 ? 1e-9
                             : update_count < 20 ? 3e-8 : 1e-6;
    for (double tolerance : {1e-7, 0.25}) {
      for (int direction : {-1, 1}) {
        for (int trial = 0; trial < 80; ++trial) {
          const HighsInt size = trial < 8 ? trial : random() % 129;
          const HighsInt count = size ? random() % (size + 1) : 0;
          std::vector<HighsInt> indices(size);
          std::vector<double> values(size);
          std::vector<int8_t> moves(size);
          std::vector<double> duals(size);
          const double boundary[] = {
              0.0, -0.0, threshold,
              std::nextafter(threshold, 0.0),
              std::nextafter(threshold, kHighsInf),
              -threshold, 0.125, -0.25, 2.0, -4.0, 1e100};
          for (HighsInt i = 0; i < size; ++i) {
            indices[i] = i;
            values[i] = boundary[(i + trial) % 11];
            moves[i] = static_cast<int8_t>(static_cast<int>(random() % 3) - 1);
            duals[i] = moves[i] * ((random() % 33) / 8.0);
          }
          std::shuffle(indices.begin(), indices.end(), random);
          const auto original_indices = indices;
          const auto original_values = values;
          const auto original_moves = moves;
          const auto original_duals = duals;
          std::vector<std::pair<HighsInt, double>> expected(
              size, std::make_pair(-41, 17.0));
          HighsInt expected_count = 0;
          long double limiting_step = std::numeric_limits<long double>::infinity();
          for (HighsInt k = 0; k < count; ++k) {
            const HighsInt col = indices[k];
            const long double aligned =
                static_cast<long double>(values[k]) * direction * moves[col];
            if (aligned <= static_cast<long double>(threshold)) continue;
            expected[expected_count++] = std::make_pair(col, static_cast<double>(aligned));
            const long double relaxed =
                static_cast<long double>(duals[col]) * moves[col] + tolerance;
            limiting_step = std::min(limiting_step, relaxed / aligned);
          }
          HighsOptions options;
          options.dual_feasibility_tolerance = tolerance;
          HEkk simplex;
          simplex.options_ = &options;
          simplex.info_.update_count = update_count;
          HEkkDualRow row(simplex);
          row.packIndex = indices;
          row.packValue = values;
          row.packCount = count;
          row.workMove = moves.data();
          row.workDual = duals.data();
          row.workData.assign(size, std::make_pair(-41, 17.0));
          row.workCount = size;
          row.workTheta = -23;
          row.workDelta = direction;
          row.choosePossible();
          REQUIRE(row.workCount == expected_count);
          for (HighsInt k = 0; k < size; ++k) {
            REQUIRE(row.workData[k].first == expected[k].first);
            REQUIRE(std::memcmp(&row.workData[k].second, &expected[k].second,
                                sizeof(double)) == 0);
          }
          if (expected_count) {
            REQUIRE(std::isfinite(row.workTheta));
            const long double difference =
                std::abs(static_cast<long double>(row.workTheta) - limiting_step);
            REQUIRE(difference <= 2e-14L * std::abs(limiting_step));
          } else {
            REQUIRE(row.workTheta == kHighsInf);
          }
          REQUIRE(row.packCount == count);
          REQUIRE(row.packIndex == original_indices);
          REQUIRE(moves == original_moves);
          REQUIRE(row.workDelta == direction);
          REQUIRE(simplex.info_.update_count == update_count);
          REQUIRE(options.dual_feasibility_tolerance == tolerance);
          if (size) {
            REQUIRE(std::memcmp(row.packValue.data(), original_values.data(),
                                size * sizeof(double)) == 0);
            REQUIRE(std::memcmp(duals.data(), original_duals.data(),
                                size * sizeof(double)) == 0);
          }
        }
      }
    }
  }
}

TEST_CASE("Dual candidate filtering preserves partitioned candidate order",
          "[highs_hardware][dual_candidate_contract]") {
  for (HighsInt size : {0, 1, 7, 64}) {
    HighsOptions options;
    options.dual_feasibility_tolerance = 0.25;
    HEkk simplex;
    simplex.options_ = &options;
    simplex.info_.update_count = 0;
    std::vector<HighsInt> indices(size);
    std::vector<double> values(size);
    std::vector<int8_t> moves(size);
    std::vector<double> duals(size);
    for (HighsInt k = 0; k < size; ++k) {
      indices[k] = size - k - 1;
      values[k] = k % 3 ? (k + 1) / 8.0 : -0.0;
      moves[k] = k % 5 ? 1 : -1;
      duals[k] = moves[k] * (k % 7);
    }
    auto filter = [&](HighsInt start, HighsInt end, double& step) {
      HEkkDualRow row(simplex);
      row.packIndex.assign(indices.begin() + start, indices.begin() + end);
      row.packValue.assign(values.begin() + start, values.begin() + end);
      row.packCount = end - start;
      row.workData.resize(end - start);
      row.workMove = moves.data();
      row.workDual = duals.data();
      row.workDelta = 1;
      row.choosePossible();
      step = row.workTheta;
      row.workData.resize(row.workCount);
      return row.workData;
    };
    double complete_step;
    const auto complete = filter(0, size, complete_step);
    for (HighsInt split = 0; split <= size; ++split) {
      double left_step;
      double right_step;
      auto combined = filter(0, split, left_step);
      const auto right = filter(split, size, right_step);
      combined.insert(combined.end(), right.begin(), right.end());
      REQUIRE(combined == complete);
      REQUIRE(std::min(left_step, right_step) == complete_step);
    }
  }
}

TEST_CASE("Dual candidate joins accept empty slices and preserve ordered state",
          "[highs_hardware][dual_join_contract]") {
  HEkk simplex;
  const std::pair<HighsInt, double> sentinel(-91, -17.0);
  for (HighsInt capacity : {0, 1, 7, 64}) {
    for (HighsInt used = 0; used <= capacity; ++used) {
      for (HighsInt appended = 0; appended <= capacity - used; ++appended) {
        for (double other_theta : {kHighsInf, 0.0, 0.25, 8.0}) {
          HEkkDualRow destination(simplex);
          HEkkDualRow source(simplex);
          destination.workData.assign(capacity, sentinel);
          destination.workCount = used;
          destination.workTheta = 0.5;
          for (HighsInt i = 0; i < used; ++i)
            destination.workData[i] = std::make_pair(i, (i + 1) / 8.0);
          source.workData.resize(appended);
          source.workCount = appended;
          source.workTheta = other_theta;
          for (HighsInt i = 0; i < appended; ++i)
            source.workData[i] = std::make_pair(used + i, -(i + 1) / 16.0);
          const auto source_before = source.workData;
          auto expected = destination.workData;
          for (HighsInt i = 0; i < appended; ++i)
            expected[used + i] = source_before[i];
          const auto* const storage = destination.workData.data();
          const auto allocation = destination.workData.capacity();
          destination.chooseJoinpack(&source);
          REQUIRE(destination.workCount == used + appended);
          REQUIRE(destination.workTheta == std::min(0.5, other_theta));
          REQUIRE(destination.workData == expected);
          REQUIRE(destination.workData.data() == storage);
          REQUIRE(destination.workData.capacity() == allocation);
          REQUIRE(source.workData == source_before);
          REQUIRE(source.workCount == appended);
          REQUIRE(source.workTheta == other_theta);
        }
      }
    }
  }
}

TEST_CASE("Dual candidate joins agree with every three-slice partition",
          "[highs_hardware][dual_join_contract]") {
  HEkk simplex;
  for (HighsInt size : {0, 1, 7, 31}) {
    std::vector<std::pair<HighsInt, double>> expected;
    for (HighsInt i = 0; i < size; ++i)
      expected.emplace_back(size - i - 1, (i % 5 - 2) / 8.0);
    for (HighsInt first = 0; first <= size; ++first) {
      for (HighsInt second = first; second <= size; ++second) {
        HEkkDualRow combined(simplex);
        combined.workData.resize(size);
        combined.workCount = 0;
        combined.workTheta = kHighsInf;
        const HighsInt offsets[] = {0, first, second, size};
        const double steps[] = {4.0, 2.0, 3.0};
        for (int slice = 0; slice < 3; ++slice) {
          HEkkDualRow part(simplex);
          part.workData.assign(expected.begin() + offsets[slice],
                               expected.begin() + offsets[slice + 1]);
          part.workCount = offsets[slice + 1] - offsets[slice];
          part.workTheta = steps[slice];
          combined.chooseJoinpack(&part);
          REQUIRE(combined.workCount == offsets[slice + 1]);
        }
        REQUIRE(combined.workData == expected);
        REQUIRE(combined.workTheta == 2.0);
      }
    }
  }
}

TEST_CASE("Indexed factor paths match dense independent solves across updates",
          "[highs_hardware][factor_contract]") {
  const HighsInt n = 64;
  for (int shape = 0; shape < 4; ++shape) {
    HighsSparseMatrix matrix;
    matrix.num_row_ = n;
    matrix.num_col_ = 2 * n;
    matrix.format_ = MatrixFormat::kColwise;
    matrix.start_.assign(1, 0);
    for (HighsInt col = 0; col < matrix.num_col_; ++col) {
      const HighsInt diagonal = col % n;
      const HighsInt block = shape == 1 ? 8 : n;
      for (HighsInt row = 0; row < n; ++row) {
        double value = 0;
        if (row == diagonal) value = col < n ? 4.0 : 5.0;
        else if (shape && row / block == diagonal / block &&
                 (row == (diagonal + 1) % n ||
                  (shape >= 2 && row == (diagonal + n - 1) % n) ||
                  (shape == 3 && row == (diagonal + 7) % n)))
          value = ((row + col) % 2 ? -1.0 : 1.0) * 0.25;
        if (value) {
          matrix.index_.push_back(row);
          matrix.value_.push_back(value);
        }
      }
      matrix.start_.push_back(matrix.index_.size());
    }
    std::vector<HighsInt> basis(n);
    for (HighsInt row = 0; row < n; ++row) basis[row] = row;
    HFactor factor;
    factor.setup(matrix, basis);
    REQUIRE(factor.build() == 0);
    for (int step = 0; step < 6; ++step) {
      CAPTURE(shape, step);
      for (int support = 0; support < 4; ++support) {
        std::vector<double> input(n, 0);
        if (support == 1) input[(step * 11) % n] = 1;
        if (support == 2) {
          input[(step * 11) % n] = -2;
          input[(step * 11 + 17) % n] = 3;
        }
        if (support == 3)
          for (HighsInt row = 0; row < n; ++row) input[row] = row % 7 - 3;
        checkIndexedFactorSolve(factor, matrix, basis, input);
      }
      const HighsInt target = step * 11;
      HighsInt pivot_row = 0;
      while (pivot_row < n && basis[pivot_row] != target) ++pivot_row;
      REQUIRE(pivot_row < n);
      const HighsInt entering = n + target;
      HVector aq, ep;
      aq.setup(n);
      ep.setup(n);
      aq.packFlag = ep.packFlag = true;
      for (HighsInt k = matrix.start_[entering];
           k < matrix.start_[entering + 1]; ++k) {
        aq.index[aq.count++] = matrix.index_[k];
        aq.array[matrix.index_[k]] = matrix.value_[k];
      }
      ep.index[0] = pivot_row;
      ep.array[pivot_row] = 1;
      ep.count = 1;
      factor.ftranCall(aq, 0);
      factor.btranCall(ep, 0);
      REQUIRE(std::abs(aq.array[pivot_row]) > 0.5);
      HighsInt hint = 0;
      factor.update(&aq, &ep, &pivot_row, &hint);
      REQUIRE(hint == 0);
      basis[pivot_row] = entering;
    }
    std::vector<double> final_rhs(n, 0);
    final_rhs[63] = -3;
    checkIndexedFactorSolve(factor, matrix, basis, final_rhs);
  }
}

TEST_CASE("Unit and mixed factor diagonals preserve exact division results",
          "[highs_hardware][factor_contract]") {
  const HighsInt n = 28;
  const double diagonals[] = {1.0, -1.0, 2.0, 3.0, 0.5,
                              std::nextafter(1.0, 2.0),
                              std::nextafter(1.0, 0.0)};
  const double values[] = {0.0, -0.0, kHighsTiny, -kHighsTiny,
                           8 * kHighsTiny, -8 * kHighsTiny, 1.0, -1.0,
                           std::nextafter(1.0, 2.0),
                           std::numeric_limits<double>::max() / 16,
                           -std::numeric_limits<double>::max() / 16};
  HighsSparseMatrix matrix;
  matrix.num_row_ = matrix.num_col_ = n;
  matrix.format_ = MatrixFormat::kColwise;
  matrix.start_.assign(1, 0);
  std::vector<HighsInt> basis(n);
  for (HighsInt row = 0; row < n; ++row) {
    basis[row] = row;
    matrix.index_.push_back(row);
    matrix.value_.push_back(diagonals[row % 7]);
    matrix.start_.push_back(row + 1);
  }
  HFactor factor;
  factor.setup(matrix, basis);
  REQUIRE(factor.build() == 0);
  for (int offset = 0; offset < 11; ++offset) {
    for (bool sparse : {false, true}) {
      std::vector<double> input(n, 0), expected(n, 0);
      for (HighsInt row = 0; row < n; ++row) {
        if (sparse && row != offset) continue;
        input[row] = values[(row + offset) % 11];
        if (std::abs(input[row]) > kHighsTiny)
          expected[row] = input[row] / matrix.value_[row];
      }
      for (double density : {0.0, 1.0}) {
        HVector rhs;
        rhs.setup(n);
        for (HighsInt row = 0; row < n; ++row)
          if (input[row] != 0) {
            rhs.index[rhs.count++] = row;
            rhs.array[row] = input[row];
          }
        factor.ftranCall(rhs, density);
        for (HighsInt row = 0; row < n; ++row) {
          CAPTURE(offset, sparse, density, row);
          REQUIRE(std::memcmp(&rhs.array[row], &expected[row], sizeof(double)) == 0);
        }
      }
      auto dense_rhs = input;
      factor.ftranCall(dense_rhs);
      REQUIRE(std::memcmp(dense_rhs.data(), expected.data(), n * sizeof(double)) == 0);
    }
  }
}

TEST_CASE("Dense factor solves own each profiling clock exactly once",
          "[highs_hardware][factor_contract]") {
  HighsSparseMatrix matrix;
  matrix.num_row_ = 3;
  matrix.num_col_ = 3;
  matrix.format_ = MatrixFormat::kColwise;
  matrix.start_ = {0, 2, 5, 7};
  matrix.index_ = {0, 1, 0, 1, 2, 1, 2};
  matrix.value_ = {4, 0.25, -0.125, 5, 0.25, -0.125, 6};
  for (int padding : {0, 16}) {
    std::vector<HighsInt> basis{0, 1, 2};
    HFactor factor;
    factor.setup(matrix, basis);
    REQUIRE(factor.build() == 0);
    HighsTimer timer;
    for (int i = 0; i < padding; ++i) timer.clock_def("Padding");
    HighsTimerClock clocks;
    clocks.timer_pointer_ = &timer;
    FactorTimer().initialiseFactorClocks(clocks);
    for (bool transpose : {false, true}) {
      const HighsInt clock = clocks.clock_[transpose ? FactorBtran : FactorFtran];
      for (const std::vector<double>& input :
           {std::vector<double>{0, 0, 0}, {1, 0, 0}, {3, -2, 1}}) {
        const auto expected = denseFactorOracle(matrix, basis, input, transpose);
        auto output = input;
        const HighsInt calls_before = timer.clock_num_call[clock];
        if (transpose) factor.btranCall(output, &clocks);
        else factor.ftranCall(output, &clocks);
        REQUIRE(timer.clock_num_call[clock] == calls_before + 1);
        REQUIRE(timer.clock_start[clock] > 0);
        REQUIRE(std::isfinite(timer.read(clock)));
        REQUIRE(timer.read(clock) >= 0);
        for (size_t row = 0; row < input.size(); ++row)
          REQUIRE(std::abs(output[row] - expected[row]) < 1e-12L);
        auto unprofiled = input;
        if (transpose) factor.btranCall(unprofiled);
        else factor.ftranCall(unprofiled);
        REQUIRE(output == unprofiled);
        REQUIRE(timer.clock_num_call[clock] == calls_before + 1);
      }
    }
  }
}

TEST_CASE("Factor snapshots preserve logical columns and FT updates",
          "[highs_hardware][factor_contract]") {
  const HighsInt n = 8;
  HighsSparseMatrix matrix;
  matrix.num_row_ = matrix.num_col_ = n;
  matrix.format_ = MatrixFormat::kColwise;
  matrix.start_.assign(1, 0);
  for (HighsInt col = 0; col < n; ++col) {
    matrix.index_.push_back(col);
    matrix.value_.push_back(4);
    if (col + 1 < n) {
      matrix.index_.push_back(col + 1);
      matrix.value_.push_back(-0.25);
    }
    matrix.start_.push_back(matrix.index_.size());
  }
  std::vector<HighsInt> basis(n);
  for (HighsInt row = 0; row < n; ++row) basis[row] = n + row;
  HFactor live;
  live.setup(matrix, basis);
  REQUIRE(live.build() == 0);
  for (HighsInt step = 0; step <= 4; ++step) {
    CAPTURE(step);
    auto replay_basis = basis;
    HFactor replay;
    replay.setup(matrix, replay_basis);
    replay.setInvert(live.getInvert());
    for (const auto& input : {std::vector<double>(n, 0),
                             std::vector<double>{1, 0, 0, 0, 0, 0, 0, -2},
                             std::vector<double>{1, -2, 3, -4, 5, -6, 7, -8}}) {
      checkIndexedFactorSolve(replay, matrix, replay_basis, input);
      for (double density : {0.0, 1.0}) {
        HVector expected, actual;
        expected.setup(n);
        actual.setup(n);
        for (HighsInt row = 0; row < n; ++row)
          if (input[row]) {
            expected.index[expected.count++] = row;
            actual.index[actual.count++] = row;
            expected.array[row] = actual.array[row] = input[row];
          }
        live.ftranCall(expected, density);
        replay.ftranCall(actual, density);
        REQUIRE(actual.array == expected.array);
        REQUIRE(actual.count == expected.count);
        REQUIRE(actual.synthetic_tick == expected.synthetic_tick);
        REQUIRE(std::equal(actual.index.begin(), actual.index.begin() + actual.count,
                           expected.index.begin()));
      }
    }
    REQUIRE(replay_basis == basis);
    if (step == 4) break;
    HighsInt pivot = 0;
    while (pivot < n && basis[pivot] != n + step) ++pivot;
    REQUIRE(pivot < n);
    HVector aq, ep;
    aq.setup(n);
    ep.setup(n);
    aq.packFlag = ep.packFlag = true;
    for (HighsInt k = matrix.start_[step]; k < matrix.start_[step + 1]; ++k) {
      aq.index[aq.count++] = matrix.index_[k];
      aq.array[matrix.index_[k]] = matrix.value_[k];
    }
    ep.index[0] = pivot;
    ep.array[pivot] = 1;
    ep.count = 1;
    live.ftranCall(aq, 0);
    live.btranCall(ep, 0);
    HighsInt hint = 0;
    live.update(&aq, &ep, &pivot, &hint);
    REQUIRE(hint == 0);
    basis[pivot] = step;
  }
}

TEST_CASE("Upper-factor scans preserve dependency order after FT updates",
          "[highs_hardware][factor_contract]") {
  const HighsInt n = 24;
  for (unsigned seed : {1u, 42u, 103u}) {
    std::minstd_rand random(seed);
    HighsSparseMatrix matrix;
    matrix.num_row_ = n;
    matrix.num_col_ = 2 * n;
    matrix.format_ = MatrixFormat::kColwise;
    matrix.start_.assign(1, 0);
    for (HighsInt col = 0; col < 2 * n; ++col) {
      for (HighsInt row = 0; row < n; ++row) {
        const double value = row == col % n ? (col < n ? 4.0 : 5.0)
            : (random() % 7 == 0 ? (random() % 2 ? 0.125 : -0.125) : 0.0);
        if (value) {
          matrix.index_.push_back(row);
          matrix.value_.push_back(value);
        }
      }
      matrix.start_.push_back(matrix.index_.size());
    }
    std::vector<HighsInt> basis(n);
    for (HighsInt row = 0; row < n; ++row) basis[row] = row;
    HFactor factor;
    factor.setup(matrix, basis);
    REQUIRE(factor.build() == 0);
    for (HighsInt step = 0; step <= n; ++step) {
      CAPTURE(seed, step);
      const auto invert = factor.getInvert();
      for (HighsInt ordinal = 0;
           ordinal < static_cast<HighsInt>(invert.u_pivot_index.size()); ++ordinal) {
        const HighsInt row = invert.u_pivot_index[ordinal];
        if (row < 0) continue;
        REQUIRE(invert.u_pivot_lookup[row] == ordinal);
        for (HighsInt k = invert.u_start[ordinal]; k < invert.u_last_p[ordinal]; ++k) {
          REQUIRE(invert.u_pivot_lookup[invert.u_index[k]] < ordinal);
          REQUIRE(invert.u_pivot_lookup[invert.u_index[k]] >= 0);
        }
      }
      for (HighsInt support : {HighsInt{0}, HighsInt{1}, HighsInt{2}, n}) {
        std::vector<double> input(n, 0);
        for (HighsInt k = 0; k < support; ++k)
          input[(step + 7 * k) % n] = k % 2 ? -2.0 : 1.0;
        checkIndexedFactorSolve(factor, matrix, basis, input);
      }
      if (step == n) break;
      const HighsInt target = step;
      HighsInt pivot_row = 0;
      while (pivot_row < n && basis[pivot_row] != target) ++pivot_row;
      REQUIRE(pivot_row < n);
      const HighsInt entering = n + target;
      HVector aq, ep;
      aq.setup(n);
      ep.setup(n);
      aq.packFlag = ep.packFlag = true;
      for (HighsInt k = matrix.start_[entering];
           k < matrix.start_[entering + 1]; ++k) {
        aq.index[aq.count++] = matrix.index_[k];
        aq.array[matrix.index_[k]] = matrix.value_[k];
      }
      ep.index[0] = pivot_row;
      ep.array[pivot_row] = 1;
      ep.count = 1;
      factor.ftranCall(aq, 1);
      factor.btranCall(ep, 1);
      REQUIRE(std::abs(aq.array[pivot_row]) > 0.5);
      HighsInt hint = 0;
      factor.update(&aq, &ep, &pivot_row, &hint);
      REQUIRE(hint == 0);
      basis[pivot_row] = entering;
    }
  }
}

TEST_CASE("Scheduler backoff has bounded monotonic batches", "[highs_hardware]") {
  using Constants = HighsSchedulerConstants;
  REQUIRE(Constants::Clock::is_steady);
  for (int workers : {std::numeric_limits<int>::min(), 0, 1, 2, 4, 128,
                      std::numeric_limits<int>::max()}) {
    int tries = Constants::initialSpinTries(workers);
    REQUIRE(tries > 0);
    REQUIRE(tries <= Constants::kMaxSpinTries);
    for (int round = 0; round < 64; ++round) {
      const int next = Constants::nextSpinTries(tries);
      REQUIRE(next >= tries);
      REQUIRE(next <= Constants::kMaxSpinTries);
      tries = next;
    }
    REQUIRE(tries == Constants::kMaxSpinTries);
  }
  REQUIRE(Constants::initialSpinTries(4) == 48);
  REQUIRE(Constants::nextSpinTries(10) == 20);
  REQUIRE(Constants::nextSpinTries(0) == 1);
  REQUIRE(Constants::nextSpinTries(std::numeric_limits<int>::min()) == 1);
  REQUIRE(Constants::nextSpinTries(std::numeric_limits<int>::max()) ==
          Constants::kMaxSpinTries);
}

TEST_CASE("Semaphore handoffs survive delayed releases", "[highs_hardware]") {
  // Distinct non-atomic payloads check publication through acquire/release.
  for (int delay : {0, 1, 8}) {
    HighsBinarySemaphore request;
    HighsBinarySemaphore response;
    int payload = 0;
    std::thread consumer([&]() {
      for (int round = 1; round <= 64; ++round) {
        request.acquire();
        payload += round;
        if (round == 1 && delay)
          std::this_thread::sleep_for(std::chrono::milliseconds(delay));
        response.release();
      }
    });
    bool matches = true;
    for (int round = 1; round <= 64; ++round) {
      payload = round;
      request.release();
      response.acquire();
      matches = matches && payload == 2 * round;
    }
    consumer.join();
    REQUIRE(matches);
  }
}

TEST_CASE("Scheduler errors restore the user-scaled model", "[highs_hardware]") {
  SchedulerScope scheduler(1);
  HighsLp lp;
  lp.num_col_ = 1;
  lp.num_row_ = 1;
  lp.col_cost_ = {3};
  lp.col_lower_ = {0};
  lp.col_upper_ = {10};
  lp.row_lower_ = {3};
  lp.row_upper_ = {kHighsInf};
  lp.a_matrix_.start_ = {0, 1};
  lp.a_matrix_.index_ = {0};
  lp.a_matrix_.value_ = {1};
  lp.offset_ = 1;
  for (bool previously_solved : {false, true}) {
    Highs highs;
    REQUIRE(highs.setOptionValue("output_flag", false) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("threads", 1) == HighsStatus::kOk);
    REQUIRE(highs.passModel(lp) == HighsStatus::kOk);
    if (previously_solved) {
      REQUIRE(highs.run() == HighsStatus::kOk);
      REQUIRE(highs.getInfo().objective_function_value == 10);
    }
    REQUIRE(highs.setOptionValue("user_objective_scale", -4) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("user_bound_scale", 3) == HighsStatus::kOk);
    REQUIRE(highs.setOptionValue("threads", 2) == HighsStatus::kOk);
    for (int attempt = 0; attempt < 3; ++attempt) {
      INFO("previously solved=" << previously_solved << " attempt=" << attempt);
      REQUIRE(Highs_run(&highs) == kHighsStatusError);
      const HighsLp& restored = highs.getLp();
      REQUIRE(restored.col_cost_ == lp.col_cost_);
      REQUIRE(restored.col_lower_ == lp.col_lower_);
      REQUIRE(restored.col_upper_ == lp.col_upper_);
      REQUIRE(restored.row_lower_ == lp.row_lower_);
      REQUIRE(restored.row_upper_ == lp.row_upper_);
      REQUIRE(restored.a_matrix_.value_ == lp.a_matrix_.value_);
      REQUIRE(restored.offset_ == lp.offset_);
      REQUIRE_FALSE(highs.getRunData().valid);
      REQUIRE_FALSE(highs.getSolution().value_valid);
      REQUIRE(highs.getModelStatus() == HighsModelStatus::kSolveError);
    }
    REQUIRE(highs.setOptionValue("threads", 1) == HighsStatus::kOk);
    REQUIRE(highs.run() == HighsStatus::kOk);
    REQUIRE(highs.getInfo().objective_function_value == 10);
    REQUIRE(highs.getSolution().col_value == std::vector<double>{3});
    REQUIRE(highs.getModelStatus() == HighsModelStatus::kOptimal);
    REQUIRE(highs.getLp().col_cost_ == lp.col_cost_);
    REQUIRE(highs.getLp().row_lower_ == lp.row_lower_);
  }
}

TEST_CASE("Cache allocation rejects overflowing extents", "[highs_hardware]") {
  using highs::cache_aligned;
  REQUIRE_THROWS_AS(cache_aligned::alloc(std::numeric_limits<std::size_t>::max()),
                    std::bad_alloc);
  REQUIRE_THROWS_AS(cache_aligned::make_unique_array<std::uint64_t>(
                        std::numeric_limits<std::size_t>::max() / 8 + 1),
                    std::bad_alloc);
  REQUIRE_THROWS_AS(cache_aligned::alloc(8, 129), std::bad_alloc);
  for (std::size_t size : {std::size_t{0}, std::size_t{1}, std::size_t{63},
                           std::size_t{128}, std::size_t{1025}}) {
    void* ptr = cache_aligned::alloc(size);
    REQUIRE(reinterpret_cast<std::uintptr_t>(ptr) % cache_aligned::alignment() == 0);
    if (size) {
      static_cast<unsigned char*>(ptr)[0] = 1;
      static_cast<unsigned char*>(ptr)[size - 1] = 2;
    }
    cache_aligned::free(ptr);
  }
  cache_aligned::free(nullptr);
}

TEST_CASE("Cache allocation honors over-aligned element storage", "[highs_hardware]") {
  for (std::size_t count = 1; count <= 16; ++count) {
    auto ptr = highs::cache_aligned::make_unique_array<OverAligned>(count);
    REQUIRE(reinterpret_cast<std::uintptr_t>(ptr.get()) % alignof(OverAligned) == 0);
  }
  auto object = highs::cache_aligned::make_unique<OverAligned>();
  REQUIRE(reinterpret_cast<std::uintptr_t>(object.get()) % alignof(OverAligned) == 0);
  auto shared = highs::cache_aligned::make_shared<OverAligned>();
  REQUIRE(reinterpret_cast<std::uintptr_t>(shared.get()) % alignof(OverAligned) == 0);
  SchedulerScope scheduler(1);
  HighsCombinable<OverAligned> local;
  REQUIRE(reinterpret_cast<std::uintptr_t>(&local.local()) % alignof(OverAligned) == 0);
}

TEST_CASE("Cache allocation releases failed object construction", "[highs_hardware]") {
  struct ThrowingValue {
    ThrowingValue() { throw std::runtime_error("injected constructor failure"); }
  };
  // ASan/LSan supplies the allocation-lifetime oracle in sanitizer builds.
  for (int i = 0; i < 16; ++i) {
    REQUIRE_THROWS_AS(highs::cache_aligned::make_unique<ThrowingValue>(),
                      std::runtime_error);
    REQUIRE_THROWS_AS(highs::cache_aligned::make_shared<ThrowingValue>(),
                      std::runtime_error);
  }
}

TEST_CASE("Combinable retries failed construction without publishing a value",
          "[highs_hardware]") {
  SchedulerScope scheduler(1);
  int calls = 0;
  HighsCombinable<int> values([&]() {
    if (++calls == 1) throw std::runtime_error("injected construction failure");
    return 7;
  });
  REQUIRE_THROWS_AS(values.local(), std::runtime_error);
  int visited = 0;
  values.combine_each([&](int&) { ++visited; });
  REQUIRE(visited == 0);
  REQUIRE(values.local() == 7);
  REQUIRE(calls == 2);
  const auto& view = values;
  REQUIRE(view.local() == 7);
  values.clear();
  REQUIRE(values.combine([](int a, int b) { return a + b; }) == 0);
}

TEST_CASE("Combinable combines every worker", "[highs_hardware]") {
  SchedulerScope scheduler(4);
  HighsCombinable<int> values([]() { return 0; });
  REQUIRE(highs::parallel::num_threads() == 4);
  highs::parallel::for_each(0, 65536, [&](HighsInt start, HighsInt end) {
    for (HighsInt i = start; i != end; ++i) ++values.local();
  });
  int expected = 0;
  int copies = 0;
  values.combine_each([&](int value) { expected += value; ++copies; });
  INFO("Actual initialized worker copies: " << copies);
  REQUIRE(copies >= 1);
  REQUIRE(expected == 65536);
  REQUIRE(values.combine([](int a, int b) { return a + b; }) == expected);
}

TEST_CASE("Combinable move assignment destroys replaced live copies",
          "[highs_hardware]") {
  SchedulerScope scheduler(1);
  REQUIRE(TrackedValue::live == 0);
  {
    HighsCombinable<TrackedValue> source([]() { return TrackedValue(7); });
    HighsCombinable<TrackedValue> target([]() { return TrackedValue(3); });
    REQUIRE(source.local().value == 7);
    REQUIRE(target.local().value == 3);
    REQUIRE(TrackedValue::live == 2);
    target = std::move(source);
    REQUIRE(TrackedValue::live == 1);
    REQUIRE(target.local().value == 7);
    target.clear();
    REQUIRE(TrackedValue::live == 0);
    REQUIRE(target.local().value == 7);
    HighsCombinable<TrackedValue> moved(std::move(target));
    REQUIRE(moved.local().value == 7);
    target.clear();
    source.clear();
  }
  REQUIRE(TrackedValue::live == 0);
}

TEST_CASE("Spin mutex protects publication on one and four threads",
          "[highs_hardware]") {
  for (int count : {1, 4}) {
    HighsSpinMutex mutex;
    int value = 0;
    std::vector<std::thread> workers;
    for (int worker = 0; worker < count; ++worker) {
      workers.emplace_back([&]() {
        for (int i = 0; i < 2000; ++i) {
          std::lock_guard<HighsSpinMutex> lock(mutex);
          ++value;
        }
      });
    }
    for (auto& worker : workers) worker.join();
    REQUIRE(value == count * 2000);
    REQUIRE(mutex.try_lock());
    REQUIRE_FALSE(mutex.try_lock());
    mutex.unlock();
  }
}

TEST_CASE("Scheduler shutdown clears thread-local publication",
          "[highs_hardware]") {
  HighsTaskExecutor::shutdown(true);
  for (int workers : {1, 4, 1, 4}) {
    HighsTaskExecutor::initialize(workers);
    REQUIRE(HighsTaskExecutor::getThisWorkerDeque() != nullptr);
    HighsTaskExecutor::shutdown(true);
    REQUIRE(HighsTaskExecutor::getThisWorkerDeque() == nullptr);
    HighsTaskExecutor::shutdown(true);
  }
}

TEST_CASE("Scheduler rolls back every partial startup before retry",
          "[highs_hardware]") {
  HighsTaskExecutor::shutdown(true);
  for (int failure = 1; failure <= 3; ++failure) {
    for (int attempt = 0; attempt < 8; ++attempt) {
      failingWorker = failure;
      startedWorkers = 0;
      exitedWorkers = 0;
      REQUIRE_THROWS_AS(HighsTaskExecutor::initialize(4, launchWithFailure),
                        std::system_error);
      REQUIRE(HighsTaskExecutor::getThisWorkerDeque() == nullptr);
      REQUIRE(startedWorkers == failure - 1);
      REQUIRE(exitedWorkers == failure - 1);
      HighsTaskExecutor::initialize(4);
      std::atomic<int> visited{0};
      highs::parallel::for_each(0, 1024, [&](HighsInt start, HighsInt end) {
        visited += static_cast<int>(end - start);
      });
      REQUIRE(visited == 1024);
      HighsTaskExecutor::shutdown(true);
      REQUIRE(HighsTaskExecutor::getThisWorkerDeque() == nullptr);
    }
  }
  REQUIRE_THROWS_AS(HighsTaskExecutor::initialize(4, launchEmpty),
                    std::system_error);
  REQUIRE(HighsTaskExecutor::getThisWorkerDeque() == nullptr);
  for (int workers : {0, -1}) {
    REQUIRE_THROWS_AS(HighsTaskExecutor::initialize(workers), std::invalid_argument);
    REQUIRE(HighsTaskExecutor::getThisWorkerDeque() == nullptr);
  }
  SchedulerScope scheduler(1);
  REQUIRE(highs::parallel::num_threads() == 1);
}

TEST_CASE("Scheduler handles repeated nonblocking shutdown",
          "[highs_hardware]") {
  HighsTaskExecutor::shutdown(true);
  for (int attempt = 0; attempt < 32; ++attempt) {
    HighsTaskExecutor::initialize(4);
    HighsTaskExecutor::shutdown(false);
    REQUIRE(HighsTaskExecutor::getThisWorkerDeque() == nullptr);
    HighsTaskExecutor::initialize(1);
    REQUIRE(highs::parallel::num_threads() == 1);
    HighsTaskExecutor::shutdown(true);
  }
}

TEST_CASE("Scheduler publishes its first task after delayed workers register",
          "[highs_hardware][scheduler_startup]") {
  for (int workers : {2, 4}) {
    for (int attempt = 0; attempt < 8; ++attempt) {
      HighsTaskExecutor::shutdown(true);
      HighsTaskExecutor::initialize(workers, launchDelayed);
      std::promise<int> started;
      auto ready = started.get_future();
      std::future_status state;
      int owner;
      {
        highs::parallel::TaskGroup group;
        group.spawn([&]() { started.set_value(highs::parallel::thread_num()); });
        // Do not let the owner execute the task through sync before observing
        // worker progress. A lost startup wakeup fails this bounded regression.
        state = ready.wait_for(std::chrono::seconds(2));
        group.taskWait();
        owner = ready.get();
      }
      HighsTaskExecutor::shutdown(true);
      REQUIRE(state == std::future_status::ready);
      REQUIRE(owner != 0);
    }
  }
}

TEST_CASE("Dense vector operations agree with direct arithmetic",
          "[highs_hardware]") {
  HVector source;
  HVector target;
  source.setup(4);
  target.setup(4);
  source.array = {1.0, -2.0, 0.0, 4.0};
  source.count = -1;
  SECTION("Dense norm") { REQUIRE(source.norm2() == 21.0); }
  SECTION("Dense copy") {
    target.copy(&source);
    REQUIRE(target.array == source.array);
    REQUIRE(target.norm2() == source.norm2());
  }
  SECTION("Dense self-copy") {
    source.copy(&source);
    REQUIRE(source.array == std::vector<double>{1.0, -2.0, 0.0, 4.0});
    REQUIRE(source.count == -1);
  }
}

TEST_CASE("Sparse vector self-copy preserves values and empty vectors are valid",
          "[highs_hardware]") {
  HVector source;
  source.setup(4);
  source.count = 2;
  source.index[0] = 1;
  source.index[1] = 3;
  source.array[1] = -2.0;
  source.array[3] = 4.0;
  source.copy(&source);
  REQUIRE(source.count == 2);
  REQUIRE(source.norm2() == 20.0);
  REQUIRE(source.array[1] == -2.0);
  REQUIRE(source.array[3] == 4.0);
  source.setup(0);
  HVector target;
  target.setup(0);
  target.copy(&source);
  REQUIRE(target.norm2() == 0.0);
  target.saxpy(1.0, &source);
  REQUIRE(target.count == 0);
}

TEST_CASE("Vector representations satisfy dense reference properties",
          "[highs_hardware]") {
  std::mt19937 random(42);
  for (int trial = 0; trial < 200; ++trial) {
    const int size = trial < 2 ? trial : random() % 258;
    HVector source;
    HVector target;
    HVectorQuad quad;
    source.setup(size);
    target.setup(size);
    quad.setup(size);
    double expected = 0;
    for (int i = 0; i < size; ++i) {
      if (random() % 100 >= static_cast<unsigned>(trial % 101)) continue;
      const double value = static_cast<int>(random() % 33) - 16;
      if (value == 0) continue;
      source.array[i] = value;
      source.index[source.count++] = i;
      expected += value * value;
    }
    if (trial % 2) source.count = -1;
    const auto reference = source.array;
    REQUIRE(source.norm2() == expected);
    quad.copy(&source);
    REQUIRE(static_cast<double>(quad.norm2()) == expected);
    target.copy(&quad);
    REQUIRE(target.array == reference);
    REQUIRE(target.norm2() == expected);
    source.copy(&source);
    REQUIRE(source.array == reference);
    target.clear();
    REQUIRE(target.norm2() == 0.0);
    REQUIRE(target.array == std::vector<double>(size, 0.0));
  }
}

TEST_CASE("Dense vector packing preserves values without an index list",
          "[highs_hardware][vector_contract]") {
  HVector vector;
  vector.setup(5);
  vector.array = {0.0, 2.0, -3.0, kHighsZero, -0.0};
  vector.count = -1;
  vector.packFlag = true;
  vector.pack();
  REQUIRE(vector.count == -1);
  REQUIRE(vector.packCount == 3);
  REQUIRE_FALSE(vector.packFlag);
  REQUIRE(vector.packIndex[0] == 1);
  REQUIRE(vector.packIndex[1] == 2);
  REQUIRE(vector.packIndex[2] == 3);
  REQUIRE(vector.packValue[0] == 2.0);
  REQUIRE(vector.packValue[1] == -3.0);
  REQUIRE(vector.packValue[2] == kHighsZero);
  vector.pack();
  REQUIRE(vector.packCount == 3);
}

namespace {
template <typename Vector>
void checkTightAndPack() {
  using Real = typename decltype(Vector().array)::value_type;
  const double controls[] = {0.0, -0.0, kHighsZero, -kHighsZero,
      kHighsTiny, -kHighsTiny, std::nextafter(kHighsTiny, 0.0),
      std::nextafter(kHighsTiny, 1.0), 1.0, -2.0,
      std::numeric_limits<double>::denorm_min(),
      std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::quiet_NaN()};
  std::mt19937 random(81);
  for (HighsInt size : {0, 1, 7, 64, 129}) {
    for (int trial = 0; trial < 32; ++trial) {
      for (bool dense : {false, true}) {
        for (bool packing : {false, true}) {
          Vector vector;
          vector.setup(size);
          for (HighsInt i = 0; i < size; ++i) vector.index[i] = i;
          std::shuffle(vector.index.begin(), vector.index.end(), random);
          vector.count = dense ? -1 : trial % (size + 1);
          const HighsInt entries = dense ? size : vector.count;
          for (HighsInt i = 0; i < entries; ++i) {
            const HighsInt row = dense ? i : vector.index[i];
            vector.array[row] = Real(controls[(i + trial) % 14]);
            if (i % 14 == 8) vector.array[row] += 1e-30;
          }
          vector.packFlag = packing;
          vector.packCount = size / 2;
          std::fill(vector.packIndex.begin(), vector.packIndex.end(), -3);
          std::fill(vector.packValue.begin(), vector.packValue.end(), Real(-7));
          vector.synthetic_tick = 7.5;
          vector.next = &vector;
          auto expected = vector;
          const auto original_array = vector.array;
          const auto original_index = vector.index;
          HighsInt survivors = 0;
          for (HighsInt i = 0; i < entries; ++i) {
            const HighsInt row = dense ? i : original_index[i];
            const double value = static_cast<double>(original_array[row]);
            const bool keep = dense ? !(std::abs(value) < kHighsTiny)
                                    : std::abs(value) >= kHighsTiny;
            if (!keep) expected.array[row] = Real(0);
            if (!dense && keep) expected.index[survivors++] = row;
          }
          if (!dense) expected.count = survivors;
          if (packing) {
            expected.packFlag = false;
            expected.packCount = 0;
            for (HighsInt i = 0; i < (dense ? size : survivors); ++i) {
              const HighsInt row = dense ? i : expected.index[i];
              if (dense && expected.array[row] == Real(0)) continue;
              expected.packIndex[expected.packCount] = row;
              expected.packValue[expected.packCount++] = expected.array[row];
            }
          }
          vector.tight();
          vector.pack();
          CAPTURE(size, trial, dense, packing);
          REQUIRE(vector.count == expected.count);
          REQUIRE(vector.index == expected.index);
          REQUIRE(vector.packCount == expected.packCount);
          REQUIRE(vector.packFlag == expected.packFlag);
          REQUIRE(vector.packIndex == expected.packIndex);
          REQUIRE(vector.synthetic_tick == 7.5);
          REQUIRE(vector.next == &vector);
          REQUIRE(vector.iwork == expected.iwork);
          REQUIRE(vector.cwork == expected.cwork);
          if (size) {
            REQUIRE(std::memcmp(vector.array.data(), expected.array.data(),
                                size * sizeof(Real)) == 0);
            REQUIRE(std::memcmp(vector.packValue.data(), expected.packValue.data(),
                                size * sizeof(Real)) == 0);
          }
        }
      }
    }
  }
}
}  // namespace

TEST_CASE("Tight and pack preserves double state against an independent oracle",
          "[highs_hardware][vector_contract][tight_pack_contract]") {
  checkTightAndPack<HVector>();
}

TEST_CASE("Tight and pack preserves compensated state against an independent oracle",
          "[highs_hardware][vector_contract][tight_pack_contract]") {
  checkTightAndPack<HVectorQuad>();
}

TEST_CASE("Pricing paths preserve independent matrix products and partitions",
          "[highs_hardware][pricing_contract]") {
  std::mt19937 random(415);
  for (HighsInt rows : {0, 1, 7, 31}) {
    for (HighsInt cols : {0, 1, 8, 37}) {
      for (int trial = 0; trial < 12; ++trial) {
        HighsSparseMatrix matrix;
        matrix.num_row_ = rows;
        matrix.num_col_ = cols;
        matrix.format_ = MatrixFormat::kColwise;
        matrix.start_.assign(1, 0);
        std::vector<long double> dense(rows * cols, 0);
        for (HighsInt col = 0; col < cols; ++col) {
          for (HighsInt row = 0; row < rows; ++row) {
            const double value = double(int(random() % 9) - 4) * 0.5;
            if (value && random() % 100 < unsigned(trial * 9)) {
              dense[row * cols + col] = value;
              matrix.index_.push_back(row);
              matrix.value_.push_back(value);
            }
          }
          matrix.start_.push_back(matrix.index_.size());
        }
        HVector input;
        input.setup(rows);
        std::vector<long double> expected(cols, 0);
        for (HighsInt row = 0; row < rows; ++row) {
          const double value = double(int(random() % 7) - 3);
          if (value) {
            input.array[row] = value;
            input.index[input.count++] = row;
          }
          for (HighsInt col = 0; col < cols; ++col)
            expected[col] += dense[row * cols + col] * value;
        }
        std::shuffle(input.index.begin(), input.index.begin() + input.count,
                     random);
        std::vector<int8_t> flags(cols, 1);
        for (HighsInt col = 0; col < cols; ++col)
          flags[col] = (col + trial) % 3 != 0;
        const auto original_input = input;
        const auto original_matrix = matrix;
        HVector result;
        result.setup(cols);
        result.synthetic_tick = 19;
        result.packFlag = true;
        result.packCount = cols / 2;
        for (bool quad : {false, true}) {
          for (int technique = 0; technique < 5; ++technique) {
            const bool partitioned = technique == 4;
            HighsSparseMatrix rowwise;
            if (partitioned)
              rowwise.createRowwisePartitioned(matrix, flags.data());
            else
              rowwise.createRowwise(matrix);
            for (int repetition = 0; repetition < 2; ++repetition) {
              result.clear();
              result.synthetic_tick = 19;
              result.packFlag = true;
              result.packCount = cols / 2;
              if (technique == 0)
                matrix.priceByColumn(quad, result, input);
              else if (technique == 1)
                rowwise.priceByRow(quad, result, input);
              else
                rowwise.priceByRowWithSwitch(quad, result, input,
                                             technique == 2 ? 1.0 : 0.0,
                                             0, kHyperPriceDensity);
              CAPTURE(rows, cols, trial, quad, technique, repetition);
              REQUIRE(result.count >= 0);
              REQUIRE(result.count <= cols);
              REQUIRE(result.index.size() == size_t(cols));
              std::vector<bool> seen(cols, false);
              for (HighsInt i = 0; i < result.count; ++i) {
                const HighsInt col = result.index[i];
                REQUIRE(col >= 0);
                REQUIRE(col < cols);
                REQUIRE_FALSE(seen[col]);
                seen[col] = true;
              }
              for (HighsInt col = 0; col < cols; ++col) {
                const long double value = partitioned && !flags[col]
                                              ? 0 : expected[col];
                REQUIRE(result.array[col] == value);
                REQUIRE(seen[col] == (value != 0));
              }
              REQUIRE(result.synthetic_tick == 19);
              REQUIRE(result.packFlag);
              REQUIRE(result.packCount == cols / 2);
              REQUIRE(input.array == original_input.array);
              REQUIRE(input.index == original_input.index);
              REQUIRE(input.count == original_input.count);
              REQUIRE(matrix == original_matrix);
            }
          }
        }
      }
    }
  }
}

TEST_CASE("Continued pricing retains an independently computed prefix",
          "[highs_hardware][pricing_contract][pricing_continuation]") {
  std::mt19937 random(416);
  for (HighsInt rows : {1, 7, 19}) {
    const HighsInt cols = rows * 2 + 1;
    HighsSparseMatrix matrix;
    matrix.num_row_ = rows;
    matrix.num_col_ = cols;
    matrix.format_ = MatrixFormat::kColwise;
    matrix.start_.assign(1, 0);
    std::vector<long double> dense(rows * cols, 0);
    for (HighsInt col = 0; col < cols; ++col) {
      for (HighsInt row = 0; row < rows; ++row) {
        const double value = double(int(random() % 7) - 3) * 0.5;
        if (value && (row + col) % 3) {
          dense[row * cols + col] = value;
          matrix.index_.push_back(row);
          matrix.value_.push_back(value);
        }
      }
      matrix.start_.push_back(matrix.index_.size());
    }
    HVector input;
    input.setup(rows);
    for (HighsInt row = 0; row < rows; ++row) {
      input.index[input.count++] = row;
      input.array[row] = row % 2 ? -2.0 : 1.0;
    }
    std::shuffle(input.index.begin(), input.index.begin() + input.count,
                 random);
    const auto original_input = input;
    const auto original_matrix = matrix;
    std::vector<int8_t> flags(cols, 1);
    for (HighsInt col = 0; col < cols; ++col) flags[col] = col % 3 != 0;
    for (bool partitioned : {false, true}) {
      HighsSparseMatrix rowwise;
      if (partitioned)
        rowwise.createRowwisePartitioned(matrix, flags.data());
      else
        rowwise.createRowwise(matrix);
      for (bool quad : {false, true}) {
        for (HighsInt split = 0; split <= input.count; ++split) {
          for (double density : {0.0, 1.0}) {
            HVector result;
            result.setup(cols);
            result.synthetic_tick = 29;
            result.packFlag = true;
            result.packCount = cols / 2;
            std::vector<long double> prefix(cols, 0), expected(cols, 0);
            for (HighsInt i = 0; i < input.count; ++i) {
              const HighsInt row = input.index[i];
              for (HighsInt col = 0; col < cols; ++col) {
                if (partitioned && !flags[col]) continue;
                const long double term = dense[row * cols + col] * input.array[row];
                expected[col] += term;
                if (i < split) prefix[col] += term;
              }
            }
            for (HighsInt col = 0; col < cols; ++col) {
              result.array[col] = double(prefix[col]);
              if (prefix[col]) result.index[result.count++] = col;
            }
            rowwise.priceByRowWithSwitch(quad, result, input, density,
                                         split, kHyperPriceDensity);
            CAPTURE(rows, partitioned, quad, split, density);
            REQUIRE(result.count >= 0);
            REQUIRE(result.count <= cols);
            REQUIRE(result.index.size() == size_t(cols));
            std::vector<bool> seen(cols, false);
            for (HighsInt i = 0; i < result.count; ++i) {
              const HighsInt col = result.index[i];
              REQUIRE(col >= 0);
              REQUIRE(col < cols);
              REQUIRE_FALSE(seen[col]);
              seen[col] = true;
            }
            for (HighsInt col = 0; col < cols; ++col) {
              REQUIRE(result.array[col] == expected[col]);
              REQUIRE(seen[col] == (expected[col] != 0));
            }
            REQUIRE(result.synthetic_tick == 29);
            REQUIRE(result.packFlag);
            REQUIRE(result.packCount == cols / 2);
            REQUIRE(input.array == original_input.array);
            REQUIRE(input.index == original_input.index);
            REQUIRE(input.count == original_input.count);
            REQUIRE(matrix == original_matrix);
          }
        }
      }
    }
  }
}

TEST_CASE("Compensated continuation clears cancelled prefix values",
          "[highs_hardware][pricing_contract][pricing_continuation]") {
  HighsSparseMatrix matrix;
  matrix.num_row_ = 2;
  matrix.num_col_ = 17;
  matrix.format_ = MatrixFormat::kColwise;
  matrix.start_.assign(1, 0);
  for (HighsInt col = 0; col < matrix.num_col_; ++col) {
    if (col == 3 || col == 11) {
      matrix.index_.push_back(0);
      matrix.value_.push_back(col == 3 ? 2.0 : 4.0);
      matrix.index_.push_back(1);
      matrix.value_.push_back(col == 3 ? -2.0 : -1.0);
    }
    matrix.start_.push_back(matrix.index_.size());
  }
  HVector input;
  input.setup(2);
  input.count = 2;
  input.index[0] = 0;
  input.index[1] = 1;
  input.array[0] = input.array[1] = 1.0;
  for (bool partitioned : {false, true}) {
    HighsSparseMatrix rowwise;
    std::vector<int8_t> flags(matrix.num_col_, 1);
    flags[11] = 0;
    if (partitioned)
      rowwise.createRowwisePartitioned(matrix, flags.data());
    else
      rowwise.createRowwise(matrix);
    HVector result;
    result.setup(matrix.num_col_);
    result.index[result.count++] = 3;
    result.array[3] = 2.0;
    if (!partitioned) {
      result.index[result.count++] = 11;
      result.array[11] = 4.0;
    }
    // Both sparse rows fit without switching; the prefix cancels at column 3.
    rowwise.priceByRowWithSwitch(true, result, input, -kHighsInf, 1,
                                kHighsInf);
    CAPTURE(partitioned);
    REQUIRE(result.count == (partitioned ? 0 : 1));
    if (!partitioned) REQUIRE(result.index[0] == 11);
    for (HighsInt col = 0; col < matrix.num_col_; ++col)
      REQUIRE(result.array[col] == (!partitioned && col == 11 ? 3.0 : 0.0));
    result.clear();
    for (double value : result.array) REQUIRE(value == 0.0);
    REQUIRE(input.array == std::vector<double>{1.0, 1.0});
    REQUIRE(input.count == 2);
  }
}

TEST_CASE("Dense pricing preserves its existing strict tiny threshold",
          "[highs_hardware][pricing_contract][pricing_dense_boundary]") {
  const std::vector<double> values = {
      0.0, -0.0, kHighsTiny, -kHighsTiny,
      std::nextafter(kHighsTiny, 0.0),
      -std::nextafter(kHighsTiny, 0.0),
      std::nextafter(kHighsTiny, kHighsInf),
      -std::nextafter(kHighsTiny, kHighsInf),
      std::numeric_limits<double>::denorm_min(),
      (std::numeric_limits<double>::min)(), 1e150, -1e150};
  HighsSparseMatrix matrix;
  matrix.num_row_ = 1;
  matrix.num_col_ = values.size();
  matrix.format_ = MatrixFormat::kColwise;
  matrix.start_.assign(1, 0);
  for (double value : values) {
    matrix.index_.push_back(0);
    matrix.value_.push_back(value);
    matrix.start_.push_back(matrix.index_.size());
  }
  HighsSparseMatrix rowwise;
  rowwise.createRowwise(matrix);
  HVector input;
  input.setup(1);
  input.count = 1;
  input.index[0] = 0;
  input.array[0] = 1;
  for (bool quad : {false, true}) {
    HVector result;
    result.setup(matrix.num_col_);
    rowwise.priceByRowWithSwitch(quad, result, input, 1.0, 0,
                                 kHyperPriceDensity);
    std::vector<HighsInt> expected_indices;
    for (HighsInt col = 0; col < matrix.num_col_; ++col) {
      const double expected = std::abs(values[col]) < kHighsTiny
                                  ? 0.0 : values[col];
      CAPTURE(quad, col, values[col]);
      REQUIRE(std::memcmp(&expected, &result.array[col], sizeof(double)) == 0);
      if (expected) expected_indices.push_back(col);
    }
    REQUIRE(result.count == HighsInt(expected_indices.size()));
    REQUIRE(std::equal(expected_indices.begin(), expected_indices.end(),
                       result.index.begin()));
    REQUIRE(input.array[0] == 1);
    REQUIRE(input.count == 1);
    REQUIRE(input.index[0] == 0);
  }
}

TEST_CASE("Timer reset restores all clock vectors and discards old times",
          "[highs_hardware][timer_contract]") {
  HighsTimer timer;
  for (int repetition = 0; repetition < 64; ++repetition) {
    const HighsInt extra = timer.clock_def("Temporary");
    timer.add(timer.solve_clock, 3, 123.0);
    timer.add(extra, 2, 17.0);
    timer.setPrintfFlag(false, false);
    timer.resetHighsTimer();
    REQUIRE(timer.num_clock == 4);
    REQUIRE(timer.clock_num_call.size() == 4);
    REQUIRE(timer.clock_start.size() == 4);
    REQUIRE(timer.clock_time.size() == 4);
    REQUIRE(timer.clock_names == std::vector<std::string>{
                                    "Run HiGHS", "Presolve", "Solve", "Postsolve"});
    REQUIRE(timer.printf_flag);
    REQUIRE(timer.presolve_clock == 1);
    REQUIRE(timer.solve_clock == 2);
    REQUIRE(timer.postsolve_clock == 3);
    for (HighsInt clock = 0; clock < timer.num_clock; ++clock) {
      REQUIRE(timer.read(clock) == 0.0);
      REQUIRE(timer.numCall(clock) == 0);
      REQUIRE_FALSE(timer.running(clock));
    }
    timer.add(timer.solve_clock, 1, 5.0);
    REQUIRE(timer.read(timer.solve_clock) == 5.0);
    timer.zeroAllClocks();
    REQUIRE(timer.num_clock == 4);
    REQUIRE(timer.read(timer.solve_clock) == 0.0);
    REQUIRE(timer.numCall(timer.solve_clock) == 0);
  }
}

namespace {
template <typename Vector>
void checkReindex(const std::vector<double>& values) {
  Vector vector;
  vector.setup(static_cast<HighsInt>(values.size()));
  vector.array.assign(values.begin(), values.end());
  const auto original = vector.array;
  vector.count = -1;
  vector.synthetic_tick = 7.5;
  vector.packFlag = true;
  std::vector<HighsInt> expected;
  for (HighsInt i = 0; i < vector.size; ++i)
    if (static_cast<double>(vector.array[i]) != 0.0) expected.push_back(i);
  vector.reIndex();
  REQUIRE(vector.count == static_cast<HighsInt>(expected.size()));
  REQUIRE(std::equal(expected.begin(), expected.end(), vector.index.begin()));
  REQUIRE(vector.synthetic_tick == 7.5);
  REQUIRE(vector.packFlag);
  if (!values.empty())
    REQUIRE(std::memcmp(vector.array.data(), original.data(),
                        values.size() * sizeof(vector.array[0])) == 0);
  vector.pack();
  REQUIRE(vector.packCount == vector.count);
  for (HighsInt i = 0; i < vector.count; ++i) {
    REQUIRE(vector.packIndex[i] == expected[i]);
    const auto& value = vector.array[expected[i]];
    REQUIRE(std::memcmp(&vector.packValue[i], &value, sizeof(value)) == 0);
  }
  vector.clear();
  for (const auto& value : vector.array)
    REQUIRE(static_cast<double>(value) == 0.0);
}
}  // namespace

TEST_CASE("Vector reindex preserves exact sorted support and packing",
          "[highs_hardware][vector_contract][reindex_contract]") {
  for (const auto& values : std::vector<std::vector<double>>{
           {}, {0.0}, {1.0}, {0.0, -0.0, kHighsZero, kHighsTiny, -kHighsTiny},
           {std::numeric_limits<double>::denorm_min(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity()}}) {
    checkReindex<HVector>(values);
    checkReindex<HVectorQuad>(values);
  }
  std::mt19937 random(431);
  for (int trial = 0; trial < 200; ++trial) {
    std::vector<double> values(random() % 513);
    for (auto& value : values)
      value = random() % 100 < unsigned(trial % 101)
                  ? double(int(random() % 100) - 50)
                  : 0.0;
    checkReindex<HVector>(values);
    checkReindex<HVectorQuad>(values);
  }
  HVector sparse;
  sparse.setup(20);
  sparse.array[15] = 2.0;
  sparse.array[3] = -4.0;
  sparse.index[0] = 15;
  sparse.index[1] = 3;
  sparse.count = 2;
  const auto original_index = sparse.index;
  sparse.reIndex();
  REQUIRE(sparse.count == 2);
  REQUIRE(sparse.index == original_index);
  sparse.count = 3;
  sparse.reIndex();
  REQUIRE(sparse.count == 2);
  REQUIRE(sparse.index[0] == 3);
  REQUIRE(sparse.index[1] == 15);
}

TEST_CASE("Dense vector saxpy uses values rather than stale indices",
          "[highs_hardware][vector_contract]") {
  HVector source;
  HVector target;
  source.setup(3);
  target.setup(3);
  source.array = {2.0, 0.0, -3.0};
  source.count = -1;
  target.array = {1.0, 5.0, 7.0};
  target.index = {0, 1, 2};
  target.count = 3;
  target.saxpy(2.0, &source);
  REQUIRE(target.array == std::vector<double>{5.0, 5.0, 1.0});
  REQUIRE(target.count < 0);
  source.count = 2;
  source.index = {0, 2, 0};
  target.array = {0.0, 5.0, 0.0};
  target.count = -1;
  target.saxpy(2.0, &source);
  REQUIRE(target.array == std::vector<double>{4.0, 5.0, -6.0});
  REQUIRE(target.count < 0);
  target.clear();
  REQUIRE(target.array == std::vector<double>(3, 0.0));
}

TEST_CASE("Vector saxpy agrees across sparse and dense representations",
          "[highs_hardware][vector_contract]") {
  std::mt19937 random(187);
  for (int trial = 0; trial < 200; ++trial) {
    const int size = trial < 2 ? trial : random() % 129;
    std::vector<double> left(size);
    std::vector<double> right(size);
    for (int i = 0; i < size; ++i) {
      left[i] = static_cast<int>(random() % 17) - 8;
      right[i] = static_cast<int>(random() % 17) - 8;
    }
    const double multiplier = static_cast<int>(random() % 5) - 2;
    std::vector<double> expected = left;
    for (int i = 0; i < size; ++i) {
      if (right[i] == 0) continue;
      const double value = left[i] + multiplier * right[i];
      expected[i] = std::abs(value) < kHighsTiny ? kHighsZero : value;
    }
    for (bool dense_left : {false, true}) {
      for (bool dense_right : {false, true}) {
        HVector target;
        HVectorQuad source;
        target.setup(size);
        source.setup(size);
        target.array = left;
        for (int i = 0; i < size; ++i) {
          source.array[i] = right[i];
          if (left[i] != 0) target.index[target.count++] = i;
          if (right[i] != 0) source.index[source.count++] = i;
        }
        if (dense_left) target.count = -1;
        if (dense_right) source.count = -1;
        target.saxpy(HighsCDouble(multiplier), &source);
        REQUIRE(target.array == expected);
        target.packFlag = true;
        target.pack();
        std::vector<double> unpacked(size, 0.0);
        for (HighsInt i = 0; i < target.packCount; ++i) {
          REQUIRE(target.packIndex[i] >= 0);
          REQUIRE(target.packIndex[i] < size);
          REQUIRE(unpacked[target.packIndex[i]] == 0.0);
          unpacked[target.packIndex[i]] = target.packValue[i];
        }
        REQUIRE(unpacked == expected);
        target.clear();
        REQUIRE(target.array == std::vector<double>(size, 0.0));
      }
    }
  }
}

TEST_CASE("Dense vector copies replace all values and reset transient scalars",
          "[highs_hardware][vector_contract]") {
  HVector source;
  HVectorQuad target;
  source.setup(5);
  target.setup(5);
  source.array = {0.0, 2.0, -3.0, kHighsZero, -0.0};
  source.count = -1;
  source.synthetic_tick = 17;
  for (int attempt = 0; attempt < 8; ++attempt) {
    target.array.assign(5, HighsCDouble(99.0));
    target.count = attempt % 2 ? -1 : 0;
    target.packFlag = true;
    target.next = &target;
    target.copy(&source);
    REQUIRE(target.count == -1);
    REQUIRE(target.synthetic_tick == 17);
    REQUIRE_FALSE(target.packFlag);
    REQUIRE(target.next == nullptr);
    for (int i = 0; i < 5; ++i)
      REQUIRE(static_cast<double>(target.array[i]) == source.array[i]);
  }
}

TEST_CASE("Vector saxpy preserves alias and tiny-value contracts",
          "[highs_hardware][vector_contract]") {
  for (bool dense : {false, true}) {
    HVector vector;
    vector.setup(3);
    vector.array = {2.0, -3.0, 0.0};
    vector.index = {0, 1, 0};
    vector.count = dense ? -1 : 2;
    vector.saxpy(-1.0, &vector);
    REQUIRE(vector.array == std::vector<double>{kHighsZero, kHighsZero, 0.0});
    vector.clear();
    REQUIRE(vector.array == std::vector<double>(3, 0.0));
  }
  HVectorQuad target;
  HVector source;
  target.setup(3);
  source.setup(3);
  target.array = {HighsCDouble(kHighsTiny / 2), HighsCDouble(2), HighsCDouble(4)};
  target.count = -1;
  source.array = {0.0, 1.0, -2.0};
  source.count = -1;
  target.saxpy(2.0, &source);
  REQUIRE(target.array[0] == HighsCDouble(kHighsTiny / 2));
  REQUIRE(target.array[1] == HighsCDouble(4));
  REQUIRE(target.array[2] == HighsCDouble(kHighsZero));
  REQUIRE(target.count == -1);
}

TEST_CASE("Vector setup rejects overflowing work extents before mutation",
          "[highs_hardware][vector_contract]") {
  HVector vector;
  vector.setup(3);
  vector.array = {2.0, 0.0, 0.0};
  vector.index[0] = 0;
  vector.count = 1;
  for (HighsInt size : {HighsInt{-1}, std::numeric_limits<HighsInt>::max(),
                        std::numeric_limits<HighsInt>::max() / 4 + 1}) {
    REQUIRE_THROWS_AS(vector.setup(size), std::length_error);
    REQUIRE(vector.size == 3);
    REQUIRE(vector.count == 1);
    REQUIRE(vector.array == std::vector<double>{2.0, 0.0, 0.0});
  }
}

TEST_CASE("QP vector repopulation preserves its own source",
          "[highs_hardware][vector_contract]") {
  QpVector vector(3);
  vector.value = {2.0, 0.0, -3.0};
  vector.resparsify();
  const QpVector reference(vector);
  vector.repopulate(vector);
  REQUIRE(vector.value == reference.value);
  REQUIRE(vector.index == reference.index);
  REQUIRE(vector.num_nz == reference.num_nz);
}

TEST_CASE("QP vector affine saxpy agrees with a frozen aliased source",
          "[highs_hardware][vector_contract]") {
  for (double a : {0.0, -1.0, 2.0}) {
    for (double b : {0.0, -1.0, 3.0}) {
      QpVector target(5);
      target.value = {2.0, 0.0, -3.0, 1e-16, -0.0};
      target.resparsify();
      const QpVector original(target);
      QpVector reference(target);
      reference.saxpy(a, b, original);
      target.saxpy(a, b, target);
      REQUIRE(target.value == reference.value);
      REQUIRE(target.num_nz == reference.num_nz);
      REQUIRE(target.index == reference.index);
    }
  }
}

TEST_CASE("Factor setup rejects invalid dimensions before replacing bindings",
          "[highs_hardware][factor_contract]") {
  const std::vector<HighsInt> starts{0};
  std::vector<HighsInt> basis{0, 1};
  HFactor factor;
  factor.setup(0, 2, starts.data(), nullptr, nullptr, basis.data());
  SECTION("Negative row count") {
    REQUIRE_THROWS_AS(factor.setup(0, -1, nullptr, nullptr, nullptr, nullptr),
                      std::length_error);
  }
  SECTION("Negative column count") {
    REQUIRE_THROWS_AS(factor.setup(-1, 2, nullptr, nullptr, nullptr, nullptr),
                      std::length_error);
  }
  SECTION("Negative basic count") {
    REQUIRE_THROWS_AS(factor.setupGeneral(0, 2, -1, nullptr, nullptr, nullptr,
                                         nullptr), std::length_error);
  }
  SECTION("Logical column index overflow") {
    REQUIRE_THROWS_AS(factor.setup(std::numeric_limits<HighsInt>::max(), 2,
                                   nullptr, nullptr, nullptr, nullptr),
                      std::length_error);
  }
  SECTION("Row workspace overflow") {
    REQUIRE_THROWS_AS(factor.setup(0, std::numeric_limits<HighsInt>::max(),
                                   nullptr, nullptr, nullptr, nullptr),
                      std::length_error);
  }
  SECTION("Basic workspace overflow") {
    REQUIRE_THROWS_AS(factor.setupGeneral(
                         0, 2, std::numeric_limits<HighsInt>::max(),
                         nullptr, nullptr, nullptr, nullptr),
                      std::length_error);
  }
  REQUIRE(factor.num_row == 2);
  REQUIRE(factor.num_col == 0);
  REQUIRE(factor.num_basic == 2);
  REQUIRE(factor.getBaseIndex() == basis.data());
  REQUIRE(factor.getAstart() == starts.data());
  REQUIRE(factor.build() == 0);
}

TEST_CASE("Factor workspace supports wide matrices and changing bases",
          "[highs_hardware][factor_contract]") {
  for (HighsInt rows : {2, 4, 8}) {
    for (HighsInt width : {1, 32}) {
      HighsSparseMatrix matrix;
      matrix.num_row_ = rows;
      matrix.num_col_ = rows * width;
      matrix.format_ = MatrixFormat::kColwise;
      matrix.start_.assign(1, 0);
      for (HighsInt col = 0; col < matrix.num_col_; ++col) {
        for (HighsInt row = 0; row < rows; ++row) {
          matrix.index_.push_back(row);
          matrix.value_.push_back(row == col % rows ? 2.0 : 1.0);
        }
        matrix.start_.push_back(matrix.index_.size());
      }
      REQUIRE(matrix.start_.size() ==
              static_cast<size_t>(matrix.num_col_ + 1));
      std::vector<HighsInt> basis(rows);
      for (HighsInt row = 0; row < rows; ++row)
        basis[row] = matrix.num_col_ + row;
      HFactor factor;
      factor.setup(matrix, basis);
      for (int attempt = 0; attempt < 4; ++attempt) {
        for (HighsInt row = 0; row < rows; ++row)
          basis[row] = attempt % 2 || row < rows / 2
              ? (attempt % width) * rows + row : matrix.num_col_ + row;
        CAPTURE(rows, width, attempt, basis);
        REQUIRE(factor.build() == 0);
        std::vector<double> rhs(rows);
        for (HighsInt row = 0; row < rows; ++row) rhs[row] = row - 2.0;
        auto forward = rhs;
        factor.ftranCall(forward);
        std::vector<double> residual(rows, 0.0);
        for (HighsInt col = 0; col < rows; ++col) {
          const HighsInt variable = basis[col];
          if (variable >= matrix.num_col_)
            residual[variable - matrix.num_col_] += forward[col];
          else
            for (HighsInt entry = matrix.start_[variable];
                 entry < matrix.start_[variable + 1]; ++entry)
              residual[matrix.index_[entry]] +=
                  matrix.value_[entry] * forward[col];
        }
        for (HighsInt row = 0; row < rows; ++row)
          REQUIRE(std::abs(residual[row] - rhs[row]) < 1e-9);
        auto transpose = rhs;
        factor.btranCall(transpose);
        for (HighsInt col = 0; col < rows; ++col) {
          const HighsInt variable = basis[col];
          double value = 0;
          if (variable >= matrix.num_col_)
            value = transpose[variable - matrix.num_col_];
          else
            for (HighsInt entry = matrix.start_[variable];
                 entry < matrix.start_[variable + 1]; ++entry)
              value += matrix.value_[entry] * transpose[matrix.index_[entry]];
          REQUIRE(std::abs(value - rhs[col]) < 1e-9);
        }
      }
    }
  }
}

TEST_CASE("Factor basis storage supports repeated singular structural columns",
          "[highs_hardware][factor_contract]") {
  HighsSparseMatrix matrix;
  matrix.num_row_ = 3;
  matrix.num_col_ = 1;
  matrix.format_ = MatrixFormat::kColwise;
  matrix.start_ = {0, 3};
  matrix.index_ = {0, 1, 2};
  matrix.value_ = {1.0, 2.0, 3.0};
  for (HighsInt variable : {0, 1, 2, 3}) {
    std::vector<HighsInt> basis(3, variable);
    HFactor factor;
    factor.setup(matrix, basis);
    REQUIRE(factor.build() == 2);
  }
}

TEST_CASE("Factor mixed-length bases agree with an independent rank oracle",
          "[highs_hardware][factor_contract]") {
  HighsSparseMatrix matrix;
  matrix.num_row_ = 3;
  matrix.num_col_ = 5;
  matrix.format_ = MatrixFormat::kColwise;
  matrix.start_ = {0, 2, 4, 6, 9, 9};
  matrix.index_ = {0, 1, 1, 2, 0, 2, 0, 1, 2};
  matrix.value_ = {2, 1, 2, 1, 1, 2, 1, 1, 1};
  const long double columns[8][3] = {
      {2, 1, 0}, {0, 2, 1}, {1, 0, 2}, {1, 1, 1},
      {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  // Exhaust every unique full/partial basis. Tiny integer matrices have
  // unambiguous rank; dense elimination is independent of sparse HFactor.
  for (unsigned mask = 1; mask < 256; ++mask) {
    std::vector<HighsInt> basis;
    for (HighsInt variable = 0; variable < 8; ++variable)
      if (mask & (1u << variable)) basis.push_back(variable);
    if (basis.size() > 3) continue;
    long double dense[3][3] = {};
    for (size_t col = 0; col < basis.size(); ++col)
      for (int row = 0; row < 3; ++row)
        dense[row][col] = columns[basis[col]][row];
    HighsInt rank = 0;
    for (size_t col = 0; col < basis.size(); ++col) {
      HighsInt pivot = rank;
      while (pivot < 3 && std::abs(dense[pivot][col]) < 1e-12L) ++pivot;
      if (pivot == 3) continue;
      for (size_t other = col; other < basis.size(); ++other)
        std::swap(dense[rank][other], dense[pivot][other]);
      for (HighsInt row = rank + 1; row < 3; ++row) {
        const long double multiplier = dense[row][col] / dense[rank][col];
        for (size_t other = col; other < basis.size(); ++other)
          dense[row][other] -= multiplier * dense[rank][other];
      }
      ++rank;
    }
    CAPTURE(mask, basis, rank);
    HFactor factor;
    factor.setup(matrix, basis);
    const HighsInt deficiency = static_cast<HighsInt>(basis.size()) - rank;
    REQUIRE(factor.build() == deficiency);
    if (basis.size() != 3 || deficiency) continue;
    const std::vector<double> rhs{1, -2, 3};
    auto forward = rhs;
    factor.ftranCall(forward);
    auto transpose = rhs;
    factor.btranCall(transpose);
    for (int row = 0; row < 3; ++row) {
      long double value = 0;
      for (int col = 0; col < 3; ++col)
        value += columns[basis[col]][row] * forward[col];
      REQUIRE(std::abs(value - rhs[row]) < 1e-9L);
    }
    for (int col = 0; col < 3; ++col) {
      long double value = 0;
      for (int row = 0; row < 3; ++row)
        value += columns[basis[col]][row] * transpose[row];
      REQUIRE(std::abs(value - rhs[col]) < 1e-9L);
    }
  }
}

TEST_CASE("Cache domain selection uses level and sharing rather than index3",
          "[highs_hardware]") {
  HighsCacheTopologyEntry l1;
  l1.level = 1;
  l1.type = "Data";
  l1.shared_cpus = "0-1";
  HighsCacheTopologyEntry l2 = l1;
  l2.level = 2;
  l2.shared_cpus = "0-7";
  HighsCacheTopologyEntry l3 = l2;
  l3.level = 3;
  l3.type = "Unified";
  l3.shared_cpus = "0-31";
  auto cache = highsLastLevelDataCache({l3, l1, l2});
  REQUIRE(cache.first == 3);
  REQUIRE(cache.second == "cpus:0-31");
  cache = highsLastLevelDataCache({l1, l2});
  REQUIRE(cache.first == 2);
  l3.shared_cpus.clear();
  REQUIRE(highsLastLevelDataCache({l1, l3}).first == 0);
  l3.id = 17;
  REQUIRE(highsLastLevelDataCache({l1, l3}).second == "Unified:id:17");
  l3.type = "Instruction";
  REQUIRE(highsLastLevelDataCache({l1, l3}).first == 1);
  REQUIRE(highsLastLevelDataCache({}).first == 0);
  HighsInt node = -1;
  REQUIRE(highsTopologySuffix("node0", "node", node));
  REQUIRE(node == 0);
  REQUIRE(highsTopologySuffix("node17", "node", node));
  REQUIRE(node == 17);
  for (const std::string name : {"node", "node-1", "node2x", "other2",
                                 "node999999999999999999999999999999"}) {
    REQUIRE_FALSE(highsTopologySuffix(name, "node", node));
    REQUIRE(node == 17);
  }
  const HighsHardwareTopology topology = highsDetectHardwareTopology();
#if defined(__APPLE__)
  REQUIRE(topology.numa_nodes == 0);
  REQUIRE(topology.last_level_cache_domains == 0);
  REQUIRE_FALSE(topology.complete);
#endif
  REQUIRE(highsTopologyAwareWorkerCount(4, topology) >= 1);
  REQUIRE(highsTopologyAwareWorkerCount(4, topology) <= 4);
}

TEST_CASE("Configured worker capacity does not depend on cache metadata",
          "[highs_hardware][highs_worker_capacity]") {
  HighsHardwareTopology topology;
  topology.logical_cpus = 10;
  topology.physical_cores = 10;
  topology.performance_cores = 4;
  topology.heterogeneous_core_classes = 2;
  REQUIRE(highsTopologyAwareWorkerCount(8, topology) == 8);
  REQUIRE(highsTopologyAwareWorkerCount(10, topology) == 10);
  REQUIRE(highsTopologyAwareWorkerCount(16, topology) == 10);

  topology = HighsHardwareTopology{};
  topology.logical_cpus = 32;
  REQUIRE_FALSE(topology.complete);
  REQUIRE(highsTopologyAwareWorkerCount(8, topology) == 8);
  REQUIRE(highsTopologyAwareWorkerCount(64, topology) == 32);
  topology.logical_cpus = 0;
  REQUIRE(highsTopologyAwareWorkerCount(8, topology) == 8);
  REQUIRE(highsTopologyAwareWorkerCount(1, topology) == 1);

  for (HighsInt visible = 0; visible <= 64; ++visible) {
    topology.logical_cpus = visible;
    for (HighsInt requested = 1; requested <= 128; ++requested) {
      const HighsInt expected = visible == 0 || requested <= visible
                                    ? requested : visible;
      for (bool complete : {false, true}) {
        topology.complete = complete;
        topology.physical_cores = 1;
        topology.performance_cores = 1;
        topology.heterogeneous_core_classes = 2;
        REQUIRE(highsTopologyAwareWorkerCount(requested, topology) == expected);
      }
    }
  }
  topology.logical_cpus = (std::numeric_limits<HighsInt>::max)();
  REQUIRE(highsTopologyAwareWorkerCount(topology.logical_cpus, topology) ==
          topology.logical_cpus);
}
