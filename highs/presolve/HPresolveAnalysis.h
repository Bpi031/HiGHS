/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file presolve/HPresolveAnalysis.h
 * @brief
 */
#ifndef PRESOLVE_HIGHS_PRESOLVE_ANALYSIS_H_
#define PRESOLVE_HIGHS_PRESOLVE_ANALYSIS_H_

class HPresolveAnalysis {
  const HighsLp* model;
  const HighsOptions* options;
  const bool* allow_rule;
  const HighsInt* numDeletedRows;
  const HighsInt* numDeletedCols;

  // store original problem sizes for reference
  HighsInt original_num_col_;
  HighsInt original_num_row_;

 public:
  std::vector<bool> allow_rule_;

  bool allow_logging_;
  bool logging_on_;

  int log_rule_type_;
  HighsInt num_deleted_rows0_;
  HighsInt num_deleted_cols0_;
  HighsPresolveLog presolve_log_;

  // for LP presolve
  //
  // Transform options->presolve_rule_off into logical settings in
  // allow_rule_[*], commenting on the rules switched off
  void setup(const HighsLp* model_, const HighsOptions* options_,
             const HighsInt& numDeletedRows_, const HighsInt& numDeletedCols_,
             const bool silent);
  void startPresolveSummary(const HighsInt num_row, const HighsInt num_col,
                            const HighsInt num_nz);
  void recordPresolvePhase(const PresolvePhaseType phase,
                           const HighsInt before_num_row,
                           const HighsInt before_num_col,
                           const HighsInt before_num_nz,
                           const double before_time,
                           const HighsInt after_num_row,
                           const HighsInt after_num_col,
                           const HighsInt after_num_nz,
                           const double after_time);
  void finishPresolveSummary(const HighsInt num_row, const HighsInt num_col,
                             const HighsInt num_nz,
                             const HighsInt postsolve_reductions);
  void resetNumDeleted();

  std::string presolveReductionTypeToString(const HighsInt reduction_type);
  void startPresolveRuleLog(const HighsInt rule_type);
  void stopPresolveRuleLog(const HighsInt rule_type);
  bool analysePresolveRuleLog(const bool report = false);
  friend class HPresolve;
};

#endif
