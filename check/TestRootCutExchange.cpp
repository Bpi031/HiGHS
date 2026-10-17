#include "catch.hpp"
#include "mip/HighsRootCutExchange.h"

TEST_CASE("oversized root cut does not hide later fitting packets",
          "[root_cut_exchange]") {
  HighsLp lp;
  lp.num_col_ = 2;
  lp.num_row_ = 2;
  lp.integrality_.assign(2, HighsVarType::kInteger);
  lp.row_lower_.assign(2, -kHighsInf);
  lp.row_upper_ = {3, 3};
  lp.a_matrix_.format_ = MatrixFormat::kRowwise;
  lp.a_matrix_.start_ = {0, 2, 3};
  lp.a_matrix_.index_ = {0, 1, 0};
  lp.a_matrix_.value_ = {2, 2, 2};
  HighsRootCutExchange exchange(lp, {"m", "p", 0, 0, 0});
  const auto cuts = exchange.discover(2, 1);
  REQUIRE(cuts.size() == 1);
  REQUIRE(cuts[0].source_row == 1);
  REQUIRE(cuts[0].sequence == 1);
  REQUIRE(exchange.check(cuts[0]));
  REQUIRE(exchange.discover(0, 10).empty());
  REQUIRE(exchange.discover(2, 0).empty());
}

TEST_CASE("root cut packets have independent integral-point certificates",
          "[root_cut_exchange]") {
  for (int signs = 0; signs != 8; ++signs)
    for (int scale = 1; scale <= 8; scale *= 2) {
      HighsLp lp;
      lp.num_col_ = 3;
      lp.num_row_ = 1;
      lp.col_cost_ = {1, 2, -3};
      lp.col_lower_.assign(3, -2);
      lp.col_upper_.assign(3, 2);
      lp.integrality_.assign(3, HighsVarType::kInteger);
      lp.row_lower_ = {-1.5 * scale};
      lp.row_upper_ = {2.5 * scale};
      lp.a_matrix_.format_ = MatrixFormat::kRowwise;
      lp.a_matrix_.start_ = {0, 3};
      lp.a_matrix_.index_ = {0, 1, 2};
      for (int j = 0; j != 3; ++j)
        lp.a_matrix_.value_.push_back((signs & (1 << j) ? -1 : 1) * scale);
      HighsRootCutExchange exchange(lp, {"model", "mapping", 1, 2, 3});
      for (int corruption = 0; corruption != 4; ++corruption) {
        auto bad = lp;
        if (corruption == 0) bad.a_matrix_.start_.clear();
        if (corruption == 1) bad.a_matrix_.index_[0] = bad.num_col_;
        if (corruption == 2) bad.row_lower_.clear();
        if (corruption == 3) bad.a_matrix_.start_[1] = 100;
        HighsRootCutExchange rejected(bad, {"model", "mapping", 1, 2, 3});
        REQUIRE(rejected.discover(4, 12).empty());
      }
      auto cuts = exchange.discover(4, 12);
      REQUIRE(cuts.size() == 2);
      for (const auto& cut : cuts) {
        REQUIRE(exchange.check(cut));
        for (int x = -2; x <= 2; ++x)
          for (int y = -2; y <= 2; ++y)
            for (int z = -2; z <= 2; ++z) {
              const int values[3] = {x, y, z};
              double activity = 0, lhs = 0;
              for (int j = 0; j != 3; ++j)
                activity += lp.a_matrix_.value_[j] * values[j];
              if (activity < lp.row_lower_[0] || activity > lp.row_upper_[0])
                continue;
              for (size_t j = 0; j < cut.indices.size(); ++j)
                lhs += cut.coefficients[j] * values[cut.indices[j]];
              REQUIRE(lhs <= cut.rhs);
            }
        for (int mutation = 0; mutation != 9; ++mutation) {
          auto bad = cut;
          if (mutation == 0) ++bad.generation.restart;
          if (mutation == 1) bad.unconditional = false;
          if (mutation == 2) bad.domain_generation = 1;
          if (mutation == 3) bad.dependencies.push_back(1);
          if (mutation == 4) bad.coefficients[0] *= -1;
          if (mutation == 5) bad.indices[0] = 3;
          if (mutation == 6) bad.rhs -= 1;
          if (mutation == 7) bad.divisor *= 2;
          if (mutation == 8) bad.sequence = 0;
          REQUIRE_FALSE(exchange.check(bad));
          std::vector<HighsRootCutPacket> received{cut};
          REQUIRE_FALSE(exchange.stage({bad}, 4, 12, received));
          REQUIRE(received.size() == 1);
          REQUIRE(received[0].rhs == cut.rhs);
        }
      }
      std::vector<HighsRootCutPacket> batch;
      REQUIRE(exchange.stage({cuts[0], cuts[0], cuts[1]}, 2, 6, batch));
      REQUIRE(batch.size() == 2);
      REQUIRE_FALSE(exchange.stage(cuts, 1, 6, batch));
      REQUIRE(batch.size() == 2);
      auto reused_sequence = cuts;
      reused_sequence[1].sequence = reused_sequence[0].sequence;
      REQUIRE_FALSE(exchange.stage(reused_sequence, 2, 6, batch));
      REQUIRE(batch.size() == 2);
    }
}
