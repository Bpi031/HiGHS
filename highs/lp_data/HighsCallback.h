/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file lp_data/HighsCallback.h
 * @brief
 */
#ifndef LP_DATA_HIGHSCALLBACK_H_
#define LP_DATA_HIGHSCALLBACK_H_

#include <functional>
#include <atomic>

// forward declaration to avoid circular dependency
enum class HighsStatus;
class Highs;
struct HighsRefinementBudget;

#include "lp_data/HStruct.h"
#include "lp_data/HighsCallbackStruct.h"

enum ExternalMipSolutionQueryOrigin {
  kExternalMipSolutionQueryOriginAfterSetup = 0,
  kExternalMipSolutionQueryOriginBeforeDive,
  kExternalMipSolutionQueryOriginEvaluateRootNode0,
  kExternalMipSolutionQueryOriginEvaluateRootNode1,
  kExternalMipSolutionQueryOriginEvaluateRootNode2,
  kExternalMipSolutionQueryOriginEvaluateRootNode3,
  kExternalMipSolutionQueryOriginEvaluateRootNode4
};

/**
 * Struct to handle callback output data
 */
struct HighsCallbackOutput {
  Highs* highs = nullptr;
  HighsLogType log_type;
  double running_time;
  HighsInt simplex_iteration_count;
  HighsInt ipm_iteration_count;
  HighsInt pdlp_iteration_count;
  double objective_function_value;
  int64_t mip_node_count;
  int64_t mip_total_lp_iterations;
  double mip_primal_bound;
  double mip_dual_bound;
  double mip_gap;
  std::vector<double> mip_solution;
  HighsInt mip_solution_source;
  int64_t mip_solution_sequence;
  HighsInt mip_solution_worker_id;
  int64_t mip_solution_discovery_node;
  int64_t mip_solution_discovery_lp_iterations;
  int64_t mip_solution_work_epoch;
  int64_t mip_solution_worker_sequence;
  double mip_solution_discovery_time;
  double mip_solution_publication_time;
  double mip_solution_admission_time;
  HighsInt cutpool_num_col;
  HighsInt cutpool_num_cut;
  std::vector<HighsInt> cutpool_start;
  std::vector<HighsInt> cutpool_index;
  std::vector<double> cutpool_value;
  std::vector<double> cutpool_lower;
  std::vector<double> cutpool_upper;
  ExternalMipSolutionQueryOrigin external_solution_query_origin;

  operator HighsCallbackDataOut() const;
};

struct HighsCallbackInput {
  Highs* highs = nullptr;
  bool user_interrupt = false;
  bool user_has_solution = false;
  std::vector<double> user_solution;
  int64_t admission_generation = 0;
  int64_t incumbent_sequence = 0;
  int64_t last_candidate_id = 0;
  int64_t candidate_id = 0;
  int64_t candidate_generation = -1;
  HighsCallbackSolutionOutcome admission = {};
  HighsInt refinement_status = 0;
  int64_t refinement_simplex_iterations = 0;
  int64_t refinement_ipm_iterations = 0;
  int64_t refinement_lp_runs = 0;
  const std::atomic<bool>* refinement_cancel = nullptr;

  HighsStatus setSolution(HighsInt num_entries, const double* value);

  HighsStatus setSolution(HighsInt num_entries, const HighsInt* index,
                          const double* value);

  HighsStatus repairSolution();

  // Rost helpers return candidates only, never a bound for the parent MIP.
  HighsStatus refineSolution(double seconds, HighsInt nodes,
                             HighsInt lp_iterations, bool require_optimal);

  operator HighsCallbackDataIn() const;
  HighsCallbackInput& operator=(const HighsCallbackDataIn& data_in);
};

using HighsCallbackFunctionType =
    std::function<void(int, const std::string&, const HighsCallbackOutput*,
                       HighsCallbackInput*, void*)>;

struct HighsCallback {
  HighsCallback(Highs* highs) : highs(highs) {
    data_out.highs = highs;
    data_in.highs = highs;
    clear();
  }

  // Function pointers cannot be used for Pybind11, so use std::function
  HighsCallbackFunctionType user_callback = nullptr;
  HighsCCallbackType c_callback = nullptr;
  HighsRefinementBudget* refinement_budget = nullptr;
  // Separate, explicitly opted-in owner. Never uses the legacy callback input.
  HighsCCallbackType primal_service_callback = nullptr;
  void* primal_service_data = nullptr;
  void* user_callback_data = nullptr;
  Highs* highs = nullptr;
  std::vector<bool> active;
  HighsCallbackOutput data_out;
  HighsCallbackInput data_in;
  bool callbackActive(const int callback_type);
  bool callbackAction(const int callback_type, std::string message = "");
  void clearHighsCallbackOutput();
  void clearHighsCallbackInput();
  void clear();
};
#endif /* LP_DATA_HIGHSCALLBACK_H_ */
