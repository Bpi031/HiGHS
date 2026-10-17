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
#include <istream>
#include <vector>

#include "lp_data/HStruct.h"
#include "lp_data/HighsLp.h"
#include "mip/HighsPrimalHeuristicManager.h"
#include "util/HighsRandom.h"

class HighsMipSolver;
class HighsMipWorker;
class HighsLpRelaxation;
class HighsOptions;
struct HighsReplayContext;

enum class HighsSubMipOrigin {
  kRootReducedCost,
  kRootRens,
  kTreeRens,
  kTreeRins,
  kTreeLocalBranching,
};

struct HighsLocalBranchingRow {
  std::vector<HighsInt> index;
  std::vector<double> value;
  double upper = 0;
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
  bool portfolioCall(HighsMipWorker& worker, const std::vector<double>& relaxation,
                     uint64_t burst_id, bool followup, bool at_root);

 public:
  HighsPrimalHeuristics(HighsMipSolver& mipsolver);

  void setupIntCols();

  bool solveSubMip(HighsMipWorker& worker, const HighsLp& lp,
                   const HighsBasis& basis, double fixingRate,
                   std::vector<double> colLower, std::vector<double> colUpper,
                   HighsInt maxleaves, HighsInt maxnodes, HighsInt stallnodes,
                   HighsSubMipOrigin origin, double max_time = kHighsInf,
                   const HighsSubMipCallBudget* call_budget = nullptr,
                   HighsPrimalHeuristicOutcome* outcome = nullptr,
                   const std::vector<double>* starting_solution = nullptr,
                   const HighsLocalBranchingRow* local_branching = nullptr,
                   const HighsReplayContext* replay_context = nullptr,
                   bool original_space = false);

  // Candidate-only restrictions. Failure leaves the supplied bounds intact.
  static bool buildRinsNeighbourhood(
      const HighsLp& model, const std::vector<double>& relaxation,
      const std::vector<double>& incumbent, double tolerance,
      double min_fixing_rate, std::vector<double>& lower,
      std::vector<double>& upper, double& fixing_rate);
  static bool buildRensNeighbourhood(const HighsLp& model,
      const std::vector<double>& relaxation, double tolerance,
      std::vector<double>& lower, std::vector<double>& upper, double& fixing_rate);
  static bool buildLocalBranchingRow(const HighsLp& model,
      const std::vector<double>& incumbent, double tolerance, HighsInt radius,
      HighsLocalBranchingRow& row);
  static bool buildAssignmentLocalBranchingRow(const HighsLp& model,
      const std::vector<double>& incumbent, const std::vector<HighsInt>& groups,
      double tolerance, HighsInt radius, HighsLocalBranchingRow& row);
  // Explicit caller-owned binary partition; non-support auxiliaries stay free.
  // Bounds only: the caller must independently check seed rows and candidates.
  static bool buildBinaryPartitionNeighbourhood(
      const HighsLp& model, const std::vector<double>& incumbent,
      const std::vector<HighsInt>& support,
      const std::vector<HighsInt>& free_support, double tolerance,
      std::vector<double>& lower, std::vector<double>& upper,
      double& fixing_rate);
  static bool readBinaryPartitionContract(std::istream& input,
      const HighsLp& model, std::vector<HighsInt>& groups, HighsInt& free_groups);
  static bool selectBinaryPartition(const HighsLp& model,
      const std::vector<double>& incumbent, const std::vector<HighsInt>& groups,
      HighsInt free_groups, int64_t ordinal, std::vector<HighsInt>& support,
      std::vector<HighsInt>& free_support, int variant = 0);
  void portfolio(HighsMipWorker& worker, const std::vector<double>& relaxation,
                 bool at_root = false);

  static const char* subMipMethodName(HighsSubMipOrigin origin);

  static const char* subMipParentPhaseName(HighsSubMipOrigin origin);

  // Every helper is a terminal neighbourhood search. Allowing it to launch
  // RENS/RINS/reduced-cost sub-MIPs recursively defeats the parent's work
  // budget and can indefinitely delay the proof search.
  static void configureSubMipOptions(HighsOptions& options,
                                     HighsSubMipOrigin origin);

  static bool subMipLaunchAllowed(bool parent_is_submip,
                                  HighsInt parent_submip_level) {
    return !parent_is_submip && parent_submip_level == 0;
  }

  // Return the remaining wall-clock allowance for one complete heuristic
  // call. The same deadline is shared by preprocessing, retries, and the
  // terminal sub-MIP rather than being renewed for every retry.
  static double remainingHeuristicTime(double started, double now,
                                       double allowance);

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
                   const int solution_source, HighsMipWorker& worker,
                   HighsPrimalAdmissionReceipt* receipt = nullptr);

  HighsInt getNumSuccessObservations(HighsMipWorker& worker) const;

  HighsInt getNumInfeasObservations(HighsMipWorker& worker) const;

  double getSuccessObservations(HighsMipWorker& worker) const;

  double getInfeasObservations(HighsMipWorker& worker) const;

  HighsInt getHeuristicRandom(const HighsInt sup) {
    return randgen.integer(sup);
  }
};

#endif
