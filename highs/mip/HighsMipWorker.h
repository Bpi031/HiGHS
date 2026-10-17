/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available Hias open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#ifndef HIGHS_MIP_WORKER_H_
#define HIGHS_MIP_WORKER_H_

#include <cassert>

#include "mip/HighsConflictPool.h"
#include "mip/HighsCutPool.h"
#include "mip/HighsImplications.h"
#include "mip/HighsIncumbentMailbox.h"
#include "mip/HighsLpRelaxation.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsNodeLpEffort.h"
#include "mip/HighsNodeQueue.h"
#include "mip/HighsPrimalHeuristics.h"
#include "mip/HighsPrimalSnapshot.h"
#include "mip/HighsPseudocost.h"
#include "mip/HighsSeparation.h"
#include "mip/HighsSearchWork.h"

class HighsSearch;
class HighsPrimalService;

class HighsMipWorker {
 private:
  struct SepaStatistics {
    SepaStatistics() : numNeighbourhoodQueries(0), sepa_lp_iterations(0) {}

    int64_t numNeighbourhoodQueries;
    int64_t sepa_lp_iterations;
  };
  struct HeurStatistics {
    HeurStatistics()
        : total_repair_lp(0),
          total_repair_lp_feasible(0),
          total_repair_lp_iterations(0),
          lp_iterations(0) {
      successObservations = 0;
      numSuccessObservations = 0;
      infeasObservations = 0;
      numInfeasObservations = 0;
      max_submip_level = 0;
      termination_status_ = HighsModelStatus::kNotset;
    }

    int64_t total_repair_lp;
    int64_t total_repair_lp_feasible;
    int64_t total_repair_lp_iterations;
    int64_t lp_iterations;

    double successObservations;
    HighsInt numSuccessObservations;
    double infeasObservations;
    HighsInt numInfeasObservations;
    HighsInt max_submip_level;
    HighsModelStatus termination_status_;
  };
  const HighsMipSolver& mipsolver_;
  const HighsMipSolverData& mipdata_;

  HighsLpRelaxation* lp_;
  HighsDomain* globaldom_;
  HighsCutPool* cutpool_;
  HighsConflictPool* conflictpool_;
  HighsPseudocost* pseudocost_;

  bool heuristics_allowed;
  HeurStatistics heur_stats;
  SepaStatistics sepa_stats;
  HighsInt worker_id_;
  int64_t work_epoch_ = -1;
  int64_t publication_sequence_ = 0;
  HighsNodeLpEffortController node_lp_effort_;
  // Passive resident-work accounting must not train bounded-LP budgets.
  HighsNodeLpEffortController resident_lp_observer_;
  std::shared_ptr<HighsPrimalSnapshotStore> primal_store_;
  HighsPrimalService* primal_service_ = nullptr;
  uint64_t resident_boundaries_ = 0;
  std::shared_ptr<const HighsPrimalSnapshot> repair_snapshot_;
  int64_t peer_imports_ = 0;
  int64_t repair_sequence_ = 0;
  double peer_wait_max_ = 0;

 public:
  HighsSearchWork search_work;
  std::unique_ptr<HighsSearch> search_ptr_;
  std::unique_ptr<HighsSeparation> sepa_ptr_;
  HighsNodeQueue nodequeue;

  double upper_bound;
  double upper_limit;
  double optimality_limit;

  HighsIncumbentMailbox incumbent_mailbox_;

  HighsRandom randgen;

  const HighsMipSolver& getMipSolver() const;

  HighsMipWorker(const HighsMipSolver& mipsolver, HighsLpRelaxation* lp,
                 HighsDomain* domain, HighsCutPool* cutpool,
                 HighsConflictPool* conflictpool, HighsPseudocost* pseudocost,
                 HighsInt worker_id);

  ~HighsMipWorker() {
    search_ptr_.reset();
    sepa_ptr_.reset();
  }

  void resetSearch();

  void resetSepa();

  HighsDomain& getGlobalDomain() const { return *globaldom_; }

  void setGlobalDomain(HighsDomain* globaldom) { globaldom_ = globaldom; }

  HighsPseudocost& getPseudocost() const { return *pseudocost_; }

  void setPseudocost(HighsPseudocost* pseudocost) { pseudocost_ = pseudocost; }

  HighsConflictPool& getConflictPool() const { return *conflictpool_; }

  void setConflictPool(HighsConflictPool* conflictpool) {
    conflictpool_ = conflictpool;
  }

  HighsCutPool& getCutPool() const { return *cutpool_; }

  void setCutPool(HighsCutPool* cutpool) { cutpool_ = cutpool; }

  HighsLpRelaxation& getLpRelaxation() const { return *lp_; }

  void setLpRelaxation(HighsLpRelaxation* lp) { lp_ = lp; }

  bool addIncumbent(const std::vector<double>& sol, double solobj,
                    int solution_source, HighsPrimalAdmissionReceipt* receipt = nullptr);

  // Serial/joined master only; publishes a witness, never global proof state.
  void publishMasterIncumbent(std::vector<double> reduced_columns,
                              int solution_source);

  void attachPrimalStore(std::shared_ptr<HighsPrimalSnapshotStore> store) {
    // Called only by the joined owner. No stale mapping survives a new batch.
    repair_snapshot_.reset();
    primal_store_ = std::move(store);
  }
  void attachPrimalService(HighsPrimalService* service) { primal_service_ = service; }
  bool residentLpEnabled() const { return primal_store_ != nullptr; }
  void residentLpBoundary();
  uint64_t residentBoundaries() const { return resident_boundaries_; }
  void importPrimalSnapshot();
  const std::vector<double>& repairIncumbent() const;
  int64_t repairIncumbentSequence() const;
  std::shared_ptr<const HighsPrimalSnapshot> pendingPrimalSnapshot() const {
    return primal_store_ ? primal_store_->latest() : nullptr;
  }
  int64_t peerImports() const { return peer_imports_; }
  double peerWaitMax() const { return peer_wait_max_; }

  HighsInt workerId() const { return worker_id_; }
  int64_t workEpoch() const { return work_epoch_; }
  int64_t nextPublicationSequence() const { return publication_sequence_; }
  void setWorkEpoch(int64_t work_epoch) { work_epoch_ = work_epoch; }

  bool nodeLpEffortEnabled() const { return node_lp_effort_.enabled(); }
  HighsInt nodeLpIterationLimit(HighsInt depth,
                                HighsInt yield_count) const;
  void recordNodeLpAttempt(bool resident = false) {
    (resident ? resident_lp_observer_ : node_lp_effort_).recordAttempt();
  }
  void recordNodeLpCompleted(const HighsNodeLpEffortObservation& observation,
                            bool resident = false) {
    (resident ? resident_lp_observer_ : node_lp_effort_).recordCompleted(observation);
  }
  void recordNodeLpYield(HighsInt iterations, bool resident = false) {
    (resident ? resident_lp_observer_ : node_lp_effort_).recordYield(iterations);
  }
  void recordNodeLpSerialCompletion() {
    node_lp_effort_.recordSerialCompletion();
  }
  void recordTaskLease(double seconds) {
    node_lp_effort_.recordTaskLease(seconds);
  }
  const HighsNodeLpEffortSummary& nodeLpEffortSummary() const {
    return node_lp_effort_.summary();
  }
  const HighsNodeLpEffortSummary& residentLpSummary() const {
    return resident_lp_observer_.summary();
  }

  std::pair<bool, double> transformNewIntegerFeasibleSolution(
      const std::vector<double>& sol,
      std::vector<double>* original_columns = nullptr) const;

  bool trySolution(const std::vector<double>& solution, int solution_source,
                   HighsPrimalAdmissionReceipt* receipt = nullptr);

  void setAllowHeuristics(const bool allowed) { heuristics_allowed = allowed; }

  bool getAllowHeuristics() const { return heuristics_allowed; }

  int64_t& getNumNeighbourhoodQueries() {
    return sepa_stats.numNeighbourhoodQueries;
  }

  int64_t& getSepaLpIterations() { return sepa_stats.sepa_lp_iterations; }

  void resetSepaStats();

  void updateHeurStatsLpIters(int64_t lp_iters, int64_t total_repair_lp,
                              int64_t total_repair_lp_feasible,
                              int64_t total_repair_lp_iters);

  void updateHeurStatsInfeasObservations(double fixingRate);

  void updateHeurStatsSuccessObservations(double fixingRate);

  int64_t& getHeurLpIterations() { return heur_stats.lp_iterations; }

  HighsInt& getHeurNumSuccessObservations() {
    return heur_stats.numSuccessObservations;
  }

  HighsInt& getHeurNumInfeasObservations() {
    return heur_stats.numInfeasObservations;
  }

  double& getHeurSuccessObservations() {
    return heur_stats.successObservations;
  }

  double& getHeurInfeasObservations() { return heur_stats.infeasObservations; }

  void setHeurTerminationStatus(HighsModelStatus status) {
    heur_stats.termination_status_ = status;
  }

  HighsModelStatus getHeurTerminationStatus() const {
    return heur_stats.termination_status_;
  }

  bool terminatorTerminated() const {
    return heur_stats.termination_status_ != HighsModelStatus::kNotset;
  }

  void updateHeurStatsMaxSubMipLevel(HighsInt new_level) {
    heur_stats.max_submip_level =
        std::max(new_level, heur_stats.max_submip_level);
  }

  void getHeurStatsValues(
      int64_t& total_repair_lp, int64_t& total_repair_lp_feasible,
      int64_t& total_repair_lp_iterations, int64_t& lp_iterations,
      double& successObservations, HighsInt& numSuccessObservations,
      double& infeasObservations, HighsInt& numInfeasObservations,
      HighsInt& max_submip_level, HighsModelStatus& termination_status) const;

  void resetHeurStats();
};

#endif
