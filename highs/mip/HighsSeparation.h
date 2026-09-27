/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#ifndef HIGHS_SEPARATION_H_
#define HIGHS_SEPARATION_H_

#include <cstdint>
#include <vector>

#include "mip/HighsCutPool.h"
#include "mip/HighsLpRelaxation.h"
#include "mip/HighsNodeSeparation.h"
#include "mip/HighsSeparator.h"

class HighsMipSolver;
class HighsMipWorker;
class HighsImplications;
class HighsCliqueTable;

class HighsSeparation {
 public:
  struct RoundLimits {
    HighsInt max_selected_rows = -1;
    int64_t max_selected_nonzeros = -1;
    bool base_model_rows_only = false;
    std::array<HighsInt, kHighsCutOriginCount> max_selected_by_origin;

    RoundLimits() { max_selected_by_origin.fill(-1); }
  };

  HighsInt separationRound(HighsDomain& propdomain,
                           HighsLpRelaxation::Status& status);

  HighsNodeSeparationRoundResult separationRoundDetailed(
      HighsDomain& propdomain, HighsLpRelaxation::Status& status,
      const RoundLimits& limits);

  void separate(HighsDomain& propdomain);

  HighsNodeSeparationSummary separate(
      HighsDomain& propdomain, const HighsNodeSeparationConfig& config,
      const HighsNodeSeparationContext& context);

  int64_t nextNodeSeparationOpportunity() {
    return node_separation_opportunities_++;
  }

  void setLpRelaxation(HighsLpRelaxation* lp) { this->lp = lp; }

  HighsSeparation(HighsMipWorker& mipworker);

 private:
  HighsMipWorker& mipworker_;
  HighsInt implBoundClock;
  HighsInt cliqueClock;
  std::vector<std::unique_ptr<HighsSeparator>> separators;
  HighsCutSet cutset;
  HighsLpRelaxation* lp;
  int64_t node_separation_opportunities_ = 0;
};

#endif
