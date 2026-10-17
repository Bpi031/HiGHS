#ifndef HIGHS_ROOT_LP_PORTFOLIO_H_
#define HIGHS_ROOT_LP_PORTFOLIO_H_

#include <cstddef>
#include <cstdint>
#include <functional>

#include "Highs.h"
#include "mip/HighsSolveCapacity.h"

struct HighsRootLpPortfolioResult {
  bool attempted = false;
  bool basis_installed = false;
  bool cancelled = false;
  int winner = -1;  // 0: primal, 1: dual. Opportunistic, not deterministic.
  int64_t iterations = 0;
  double elapsed = 0;
  HighsModelStatus status[2] = {HighsModelStatus::kNotset,
                                HighsModelStatus::kNotset};
};

// Pure LP replay only. No candidate, cut, node, or global bound is imported.
// The caller must reoptimize the owner after a successful basis installation.
class HighsRootLpPortfolio {
 public:
  static HighsRootLpPortfolioResult offerBasis(
      Highs& owner, HighsSolveCapacity& capacity, double seconds,
      const std::function<bool()>& owner_poll = {});
  static bool checkedOptimalBasis(const HighsLp& lp,
                                  const HighsSolution& solution,
                                  const HighsBasis& basis,
                                  const HighsOptions& options);
  static bool copyBytes(const HighsLp& lp, size_t& bytes);
};
#endif
