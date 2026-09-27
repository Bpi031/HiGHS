/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#ifndef HIGHS_PRIMAL_HEURISTICS_H_
#define HIGHS_PRIMAL_HEURISTICS_H_

#include <atomic>
#include <cstdint>
#include <vector>

#include "lp_data/HStruct.h"
#include "lp_data/HighsLp.h"
#include "mip/HighsPrimalHeuristicManager.h"
#include "util/HighsRandom.h"

class HighsMipSolver;
class HighsMipWorker;
class HighsLpRelaxation;

enum class HighsSubMipOrigin {
  kRootReducedCost,
  kRootRens,
  kTreeRens,
  kTreeRins,
};

class HighsPrimalHeuristics {
 private:
  const HighsMipSolver& mipsolver;
  std::vector<HighsInt> intcols;
  double successObservations;
  HighsInt numSuccessObservations;
  double infeasObservations;
  HighsInt numInfeasObservations;

  HighsRandom randgen;
  std::atomic<int64_t> nextSubMipInvocation{0};

 public:
  HighsPrimalHeuristics(HighsMipSolver& mipsolver);

  void setupIntCols();

  bool solveSubMip(HighsMipWorker& worker, const HighsLp& lp,
                   const HighsBasis& basis, double fixingRate,
                   std::vector<double> colLower, std::vector<double> colUpper,
                   HighsInt maxleaves, HighsInt maxnodes, HighsInt stallnodes,
                   HighsSubMipOrigin origin, double max_time = kHighsInf,
                   const HighsSubMipCallBudget* call_budget = nullptr,
                   HighsPrimalHeuristicOutcome* outcome = nullptr);

  static const char* subMipMethodName(HighsSubMipOrigin origin);

  static const char* subMipParentPhaseName(HighsSubMipOrigin origin);

  double determineTargetFixingRate(HighsMipWorker& worker);

  void rootReducedCost(HighsMipWorker& worker,
                       double max_submip_time = kHighsInf);

  void RENS(HighsMipWorker& worker, const std::vector<double>& relaxationsol,
            double max_submip_time = kHighsInf,
            HighsSubMipOrigin origin = HighsSubMipOrigin::kTreeRens);

  void RINS(HighsMipWorker& worker, const std::vector<double>& relaxationsol,
            const HighsSubMipCallBudget* call_budget = nullptr,
            HighsPrimalHeuristicOutcome* outcome = nullptr);

  void feasibilityPump(HighsMipWorker& worker);

  void centralRounding(HighsMipWorker& worker);

  void flushStatistics(HighsMipSolver& mipsolver, HighsMipWorker& worker);

  bool tryRoundedPoint(HighsMipWorker& worker, const std::vector<double>& point,
                       const int solution_source);

  bool linesearchRounding(HighsMipWorker& worker,
                          const std::vector<double>& point1,
                          const std::vector<double>& point2,
                          const int solution_source);

  void randomizedRounding(HighsMipWorker& worker,
                          const std::vector<double>& relaxationsol);

  void shifting(HighsMipWorker& worker,
                const std::vector<double>& relaxationsol);

  void ziRound(HighsMipWorker& worker,
               const std::vector<double>& relaxationsol);

  bool addIncumbent(const std::vector<double>& sol, double solobj,
                    const int solution_source, HighsMipWorker& worker);

  bool trySolution(const std::vector<double>& solution,
                   const int solution_source, HighsMipWorker& worker);

  HighsInt getNumSuccessObservations(HighsMipWorker& worker) const;

  HighsInt getNumInfeasObservations(HighsMipWorker& worker) const;

  double getSuccessObservations(HighsMipWorker& worker) const;

  double getInfeasObservations(HighsMipWorker& worker) const;

  HighsInt getHeuristicRandom(const HighsInt sup) {
    return randgen.integer(sup);
  }
};

#endif
