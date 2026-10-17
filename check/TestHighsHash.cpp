#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>

#include "HCheckConfig.h"
#include "catch.hpp"
#include "util/HighsHash.h"
#include "util/HighsHashTree.h"
#include "util/HighsRandom.h"

TEST_CASE("Highs_random_real_finite_interval_boundaries",
          "[util][highs_hardware][random_real_contract]") {
  const double largest = (std::numeric_limits<double>::max)();
  const double smallest = std::numeric_limits<double>::denorm_min();
  const std::vector<std::pair<double, double>> intervals = {
      {-largest, largest}, {-largest, largest / 2},
      {-largest / 2, largest}, {-largest, smallest},
      {-smallest, largest}, {-largest, -largest / 2},
      {largest / 2, largest}, {-smallest, smallest},
      {0, smallest}, {-smallest, 0}, {largest, largest},
      {-largest, -largest}, {0, 0}};
  for (HighsUInt seed : {HighsUInt{0}, HighsUInt{1}, HighsUInt{42},
                        (std::numeric_limits<HighsUInt>::max)()}) {
    for (const auto& interval : intervals) {
      for (bool reverse : {false, true}) {
        const double a = reverse ? interval.second : interval.first;
        const double b = reverse ? interval.first : interval.second;
        HighsRandom random(seed);
        HighsRandom reference(seed);
        for (int sample = 0; sample < 256; ++sample) {
          const double fraction = reference.closedFraction();
          const double value = random.real(a, b);
          REQUIRE(std::isfinite(value));
          REQUIRE(value >= (std::min)(a, b));
          REQUIRE(value <= (std::max)(a, b));
          if (a == -largest && b == largest) {
            // Normalized oracle avoids both the overflowing difference and
            // dependence on a platform's long-double width.
            REQUIRE(std::fabs(value / largest - (2 * fraction - 1)) <=
                    8 * std::numeric_limits<double>::epsilon());
          }
          REQUIRE(random.integer() == reference.integer());
        }
      }
    }
  }
}

TEST_CASE("Highs_random_real_keeps_ordinary_draws_and_state",
          "[util][highs_hardware][random_real_contract]") {
  const std::vector<std::pair<double, double>> intervals = {
      {0, 1}, {0.1, 0.9}, {0.4, 0.6}, {0.6, 0.6},
      {-1e-4, 1e-4}, {-10000, 10001}, {1e300, 1.1e300},
      {-1e300, 1e300}, {-1e300, -1.1e300}};
  for (HighsUInt seed : {HighsUInt{0}, HighsUInt{1}, HighsUInt{42},
                        (std::numeric_limits<HighsUInt>::max)()}) {
    for (const auto& interval : intervals) {
      HighsRandom random(seed);
      HighsRandom reference(seed);
      for (int sample = 0; sample < 1024; ++sample) {
        const double expected = interval.first +
                                (interval.second - interval.first) *
                                    reference.closedFraction();
        const double value = random.real(interval.first, interval.second);
        REQUIRE(std::memcmp(&value, &expected, sizeof(value)) == 0);
        REQUIRE(random.integer() == reference.integer());
        REQUIRE(random.fraction() == reference.fraction());
        REQUIRE(random.bit() == reference.bit());
      }
    }
  }
}

TEST_CASE("Highs_random_real_generated_finite_intervals",
          "[util][highs_hardware][random_real_contract]") {
  std::mt19937_64 inputs(63041);
  HighsRandom random(42);
  for (int trial = 0; trial < 4096; ++trial) {
    // Generate finite binary64 endpoints, not NaN/inf or undefined inputs.
    uint64_t bits_a = inputs();
    uint64_t bits_b = inputs();
    bits_a &= ~(uint64_t{0x7ff} << 52);
    bits_b &= ~(uint64_t{0x7ff} << 52);
    bits_a |= uint64_t(inputs() % 2047) << 52;
    bits_b |= uint64_t(inputs() % 2047) << 52;
    double a, b;
    static_assert(sizeof(double) == sizeof(uint64_t), "binary64 fixture");
    std::memcpy(&a, &bits_a, sizeof(a));
    std::memcpy(&b, &bits_b, sizeof(b));
    INFO("trial=" << trial << " bits_a=" << bits_a << " bits_b=" << bits_b);
    REQUIRE(std::isfinite(a));
    REQUIRE(std::isfinite(b));
    for (int sample = 0; sample < 8; ++sample) {
      const double value = random.real(a, b);
      REQUIRE(std::isfinite(value));
      REQUIRE(value >= (std::min)(a, b));
      REQUIRE(value <= (std::max)(a, b));
    }
  }
}

TEST_CASE("Highs_random_signed_range_boundaries",
          "[util][highs_hardware][random_range_contract]") {
  const HighsInt low = (std::numeric_limits<HighsInt>::min)();
  const HighsInt high = (std::numeric_limits<HighsInt>::max)();
  std::vector<std::pair<HighsInt, HighsInt>> intervals = {
      {low, high}, {low, 1}, {low, 0}, {-1, high},
      {-high, high}, {low, -1}, {low, low + 2}, {high - 2, high},
      {low + 1, high - 1}, {-128, 128}, {0, high}};
#ifdef HIGHSINT64
  const HighsInt power32 = HighsInt{1} << 32;
  intervals.emplace_back(0, power32 - 1);
  intervals.emplace_back(0, power32);
  intervals.emplace_back(0, power32 + 1);
  intervals.emplace_back(-(HighsInt{1} << 31), HighsInt{1} << 31);
  intervals.emplace_back(-1, power32 - 1);
#endif
  for (HighsUInt seed : {HighsUInt{0}, HighsUInt{1}, HighsUInt{42},
                        (std::numeric_limits<HighsUInt>::max)()}) {
    for (const auto& interval : intervals) {
      const HighsInt lower = interval.first;
      const HighsInt upper = interval.second;
      const HighsUInt width = HighsUInt(upper) - HighsUInt(lower);
      HighsRandom random(seed);
      bool below_midpoint = false;
      bool above_midpoint = false;
      for (int sample = 0; sample < 512; ++sample) {
        const HighsInt value = random.integer(lower, upper);
        REQUIRE(value >= lower);
        REQUIRE(value < upper);
        const HighsUInt distance = HighsUInt(value) - HighsUInt(lower);
        REQUIRE(distance < width);
        below_midpoint |= distance < width / 2;
        above_midpoint |= distance >= width / 2;
      }
      // A deterministic degeneracy guard, not a statistical uniformity proof.
      REQUIRE(below_midpoint);
      REQUIRE(above_midpoint);
    }
  }
}

TEST_CASE("Highs_random_translated_legacy_sequence",
          "[util][highs_hardware][random_range_contract]") {
  for (HighsUInt seed : {HighsUInt{0}, HighsUInt{1}, HighsUInt{42},
                        (std::numeric_limits<HighsUInt>::max)()}) {
    for (HighsInt lower : {-10000, -513, -2, 0, 7, 1024}) {
      for (HighsInt width : {1, 2, 3, 7, 16, 65537}) {
        HighsRandom shifted(seed);
        HighsRandom reference(seed);
        for (int sample = 0; sample < 200; ++sample) {
          REQUIRE(shifted.integer(lower, lower + width) ==
                  lower + reference.integer(width));
        }
        REQUIRE(shifted.integer() == reference.integer());
        REQUIRE(shifted.fraction() == reference.fraction());
        REQUIRE(shifted.bit() == reference.bit());
      }
    }
  }
}

TEST_CASE("Highs_random_one_argument_and_degenerate_ranges",
          "[util][highs_hardware][random_range_contract]") {
  const HighsInt low = (std::numeric_limits<HighsInt>::min)();
  const HighsInt high = (std::numeric_limits<HighsInt>::max)();
  std::vector<HighsInt> limits = {1, 2, 3, 7, 16, 65537, high};
#ifdef HIGHSINT64
  const HighsInt power32 = HighsInt{1} << 32;
  limits.push_back(power32 - 1);
  limits.push_back(power32);
  limits.push_back(power32 + 1);
#endif
  for (HighsInt upper : limits) {
    HighsRandom random(42);
    for (int sample = 0; sample < 512; ++sample) {
      const HighsInt value = random.integer(upper);
      REQUIRE(value >= 0);
      REQUIRE(value < upper);
    }
  }
  HighsRandom random(42);
  HighsRandom untouched(42);
  REQUIRE(random.integer(low, low + 1) == low);
  REQUIRE(random.integer(high - 1, high) == high - 1);
  REQUIRE(random.integer(low, low) == low);
  REQUIRE(random.integer(high, high) == high);
  REQUIRE(random.integer(0) == 0);
  REQUIRE(random.integer(1) == 0);
  REQUIRE(random.integer() == untouched.integer());
}

TEST_CASE("Highs_random_frozen_legacy_trace",
          "[util][highs_hardware][random_range_contract]") {
  uint64_t checksum = 1469598103934665603ULL;
  uint64_t samples = 0;
  auto mix = [&](uint64_t value) {
    checksum ^= value;
    checksum *= 1099511628211ULL;
    ++samples;
  };
  for (HighsUInt seed : {HighsUInt{0}, HighsUInt{1}, HighsUInt{42},
                        (std::numeric_limits<HighsUInt>::max)()}) {
    for (HighsInt upper : {1, 2, 3, 7, 16, 65537, 2147483647}) {
      HighsRandom random(seed);
      for (int sample = 0; sample < 512; ++sample) {
        mix(static_cast<uint64_t>(random.integer(upper)));
        mix(static_cast<uint64_t>(random.integer(-10000, 10001)));
        mix(static_cast<uint64_t>(random.integer()));
        const double fraction = random.fraction();
        uint64_t bits;
        std::memcpy(&bits, &fraction, sizeof(bits));
        mix(bits);
        mix(random.bit());
      }
    }
  }
  REQUIRE(samples == 71680);
  // Frozen from the pre-fix header, not generated by the tested implementation.
#ifdef HIGHSINT64
  REQUIRE(checksum == 4757867706049468292ULL);
#else
  REQUIRE(checksum == 13157778048245073ULL);
#endif
}

TEST_CASE("Highs_log2i", "[util]") {
  // test 32 bit and 64 bit values whoes log2 value should be floored
  uint32_t x = 12345;
  uint64_t y = 1234567891011;
  REQUIRE(HighsHashHelpers::log2i(x) == 13);
  REQUIRE(HighsHashHelpers::log2i(y) == 40);

  // test 2^13 and 2^40 which should yield the same result
  x = 8192;
  y = 1099511627776;
  REQUIRE(HighsHashHelpers::log2i(x) == 13);
  REQUIRE(HighsHashHelpers::log2i(y) == 40);
}

TEST_CASE("Highs_HashTree", "[util]") {
  // test 32 bit and 64 bit values whoes log2 value should be floored
  HighsHashTree<int> htree;

  htree.insert(1);
  htree.insert(2);
  htree.insert(3);

  REQUIRE(!htree.contains(4));
  REQUIRE(htree.contains(1));
  REQUIRE(htree.contains(2));
  REQUIRE(htree.contains(3));

  htree.erase(2);
  REQUIRE(htree.contains(1));
  REQUIRE(!htree.contains(2));
  REQUIRE(htree.contains(3));

  REQUIRE(!htree.insert(1));
  REQUIRE(!htree.insert(3));

  constexpr int NUM_CHECK = 10000;
  for (int i = 0; i < NUM_CHECK; ++i) {
    htree.insert(i);
  }

  std::vector<int> test;
  std::vector<int> testReference;
  testReference.resize(NUM_CHECK);
  std::iota(testReference.begin(), testReference.end(), 0);

  htree.for_each([&](const HighsHashTableEntry<int>& entry) {
    test.push_back(entry.key());
    return false;
  });

  REQUIRE(test.size() == NUM_CHECK);
  std::sort(test.begin(), test.end());
  REQUIRE(std::equal(test.begin(), test.end(), testReference.begin()));

  for (int i = 0; i < NUM_CHECK; ++i) {
    REQUIRE(htree.contains(i));
  }

  testReference.clear();

  for (int i = 0; i < NUM_CHECK; ++i) {
    if ((i % 7) == 0) {
      REQUIRE(htree.contains(i));
      htree.erase(i);
      REQUIRE(!htree.contains(i));
    } else {
      testReference.push_back(i);
    }
  }

  test.clear();
  htree.for_each([&](const HighsHashTableEntry<int>& entry) {
    test.push_back(entry.key());
    return false;
  });

  REQUIRE(test.size() == testReference.size());
  std::sort(test.begin(), test.end());
  REQUIRE(std::equal(test.begin(), test.end(), testReference.begin()));

  HighsHashTree<int> htree2;
  for (int i = 0; i < NUM_CHECK; ++i) {
    if (htree.contains(i)) {
      REQUIRE((i % 7) != 0);
    } else {
      REQUIRE((i % 7) == 0);
      htree2.insert(i);
    }
  }

  const auto* commonElement = htree.find_common(htree2);
  REQUIRE(commonElement == nullptr);

  for (int i = 0; i < NUM_CHECK; ++i) {
    if (i % 7 != 0) {
      htree2.insert(i);
      commonElement = htree.find_common(htree2);
      REQUIRE(commonElement != nullptr);
      REQUIRE(commonElement->key() == i);
      htree.erase(i);

      commonElement = htree.find_common(htree2);
      REQUIRE(commonElement == nullptr);
    }
  }

  HighsHashTree<int> htree3 = htree2;

  for (int i = 0; i < NUM_CHECK; ++i) {
    const int* a = htree2.find(i);
    const int* b = htree3.find(i);

    if (a == b) {
      REQUIRE(a == nullptr);
    } else {
      REQUIRE(*a == *b);
    }
  }
}
