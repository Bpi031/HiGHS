/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "mip/HighsSeparation.h"

#include <algorithm>
#include <cassert>
#include <queue>

#include "io/HighsIO.h"
#include "mip/HighsCliqueTable.h"
#include "mip/HighsDomain.h"
#include "mip/HighsImplications.h"
#include "mip/HighsLpAggregator.h"
#include "mip/HighsLpRelaxation.h"
#include "mip/HighsMipSolverData.h"
#include "mip/HighsModkSeparator.h"
#include "mip/HighsPathSeparator.h"
#include "mip/HighsTableauSeparator.h"
#include "mip/HighsTransformedLp.h"

HighsSeparation::HighsSeparation(HighsMipWorker& mipworker)
    : mipworker_(mipworker) {
  /*
  if (mipworker.mipsolver_.profiling_->mip_) {
    implBoundClock =
        mipworker.mipsolver_.profiling_->getSepaClockIndex(kImplboundSepaString);
    cliqueClock =
        mipworker.mipsolver_.profiling_->getSepaClockIndex(kCliqueSepaString);
  }
  */
  implBoundClock = 990;
  cliqueClock = 991;
  const HighsMipSolver& mipsolver = mipworker.getMipSolver();
  separators.emplace_back(new HighsTableauSeparator(mipsolver));
  separators.emplace_back(new HighsPathSeparator(mipsolver));
  separators.emplace_back(new HighsModkSeparator(mipsolver));
}

HighsNodeSeparationRoundResult HighsSeparation::separationRoundDetailed(
    HighsDomain& propdomain, HighsLpRelaxation::Status& status,
    const RoundLimits& limits) {
  HighsNodeSeparationRoundResult result;
  const double start_time = lp->getMipSolver().timer_.read();
  const int64_t start_lp_iterations = lp->getNumLpIterations();
  result.objective_before = lp->getObjective();
  result.fractional_before = lp->getFractionalIntegers().size();
  result.active_rows_before = lp->numRows();
  result.active_nonzeros_before = lp->numNonzeros();

  const HighsSolution& sol = lp->getLpSolver().getSolution();

  HighsMipSolverData& mipdata = *lp->getMipSolver().mipdata_;
  HighsLpAggregator lpAggregator(*lp, limits.base_model_rows_only);

  auto poolCuts = [&]() {
    HighsInt cuts = mipworker_.getCutPool().getNumCuts();
    if (&mipworker_.getCutPool() != &mipdata.getCutPool())
      cuts += mipdata.getCutPool().getNumCuts();
    return cuts;
  };
  const HighsInt start_pool_cuts = poolCuts();
  auto acceptedByOrigin = [&]() {
    std::array<int64_t, kHighsCutOriginCount> counts{};
    const auto& local = mipworker_.getCutPool().getAcceptedOriginCounts();
    const auto& global = mipdata.getCutPool().getAcceptedOriginCounts();
    for (size_t i = 0; i != counts.size(); ++i) {
      counts[i] = local[i];
      if (&mipworker_.getCutPool() != &mipdata.getCutPool())
        counts[i] += global[i];
    }
    return counts;
  };
  const auto start_origin_counts = acceptedByOrigin();

  auto finish = [&]() {
    result.candidates_generated =
        std::max<HighsInt>(0, poolCuts() - start_pool_cuts);
    const auto end_origin_counts = acceptedByOrigin();
    for (size_t i = 0; i != result.generated_by_origin.size(); ++i)
      result.generated_by_origin[i] =
          std::max<int64_t>(0, end_origin_counts[i] - start_origin_counts[i]);
    result.generator_model_row_uses = lpAggregator.getModelRowUses();
    result.generator_cut_pool_row_uses = lpAggregator.getCutPoolRowUses();
    result.rejected_cut_pool_row_uses =
        lpAggregator.getRejectedCutPoolRowUses();
    result.lp_iterations =
        std::max<int64_t>(0, lp->getNumLpIterations() - start_lp_iterations);
    result.elapsed_seconds =
        std::max(0.0, lp->getMipSolver().timer_.read() - start_time);
    result.objective_after = lp->getObjective();
    result.fractional_after = lp->getFractionalIntegers().size();
    result.active_rows_after = lp->numRows();
    result.active_nonzeros_after = lp->numNonzeros();
    result.lp_optimal = lp->scaledOptimal(status);
    result.node_infeasible =
        status == HighsLpRelaxation::Status::kInfeasible ||
        propdomain.infeasible() || mipworker_.getGlobalDomain().infeasible();
    return result;
  };

  auto propagateAndResolve = [&]() {
    if (propdomain.infeasible() || mipworker_.getGlobalDomain().infeasible()) {
      status = HighsLpRelaxation::Status::kInfeasible;
      propdomain.clearChangedCols();
      return -1;
    }

    propdomain.propagate();
    if (propdomain.infeasible()) {
      status = HighsLpRelaxation::Status::kInfeasible;
      propdomain.clearChangedCols();
      return -1;
    }

    // only modify cliquetable for master worker.
    if (&propdomain == &mipdata.getDomain())
      mipdata.cliquetable.cleanupFixed(mipdata.getDomain());

    if (mipworker_.getGlobalDomain().infeasible()) {
      status = HighsLpRelaxation::Status::kInfeasible;
      propdomain.clearChangedCols();
      return -1;
    }

    int numBoundChgs = (int)propdomain.getChangedCols().size();

    while (!propdomain.getChangedCols().empty()) {
      lp->setObjectiveLimit(mipworker_.upper_limit);
      status = lp->resolveLp(&propdomain);
      if (!lp->scaledOptimal(status)) return -1;

      if (&propdomain == &mipdata.getDomain() &&
          lp->unscaledDualFeasible(status)) {
        mipdata.redcostfixing.addRootRedcost(
            mipdata.mipsolver, lp->getSolution().col_dual, lp->getObjective());
        if (mipworker_.upper_limit != kHighsInf)
          mipdata.redcostfixing.propagateRootRedcost(mipdata.mipsolver);
      }
    }

    return numBoundChgs;
  };

  if (!mipdata.parallelLockActive())
    lp->getMipSolver().profiling_->start(implBoundClock);
  mipdata.implications.separateImpliedBounds(
      *lp, lp->getSolution().col_value, mipworker_.getCutPool(),
      mipdata.feastol, mipworker_.getGlobalDomain(),
      mipdata.parallelLockActive());
  if (!mipdata.parallelLockActive())
    lp->getMipSolver().profiling_->stop(implBoundClock);

  HighsInt ncuts = 0;
  HighsInt numboundchgs = propagateAndResolve();
  if (numboundchgs == -1)
    return finish();
  else
    ncuts += numboundchgs;

  if (!mipdata.parallelLockActive())
    lp->getMipSolver().profiling_->start(cliqueClock);
  mipdata.cliquetable.separateCliques(
      lp->getMipSolver(), sol.col_value, mipworker_.getCutPool(),
      mipdata.feastol,
      mipdata.parallelLockActive() ? mipworker_.randgen
                                   : mipdata.cliquetable.getRandgen(),
      mipdata.parallelLockActive()
          ? mipworker_.getNumNeighbourhoodQueries()
          : mipdata.cliquetable.getNumNeighbourhoodQueries());
  if (!mipdata.parallelLockActive())
    lp->getMipSolver().profiling_->stop(cliqueClock);

  numboundchgs = propagateAndResolve();
  if (numboundchgs == -1)
    return finish();
  else
    ncuts += numboundchgs;

  if (&propdomain != &mipworker_.getGlobalDomain())
    lp->computeBasicDegenerateDuals(
        mipdata.feastol, propdomain, mipworker_.getGlobalDomain(),
        mipworker_.getConflictPool(), mipworker_.getPseudocost(), true);

  HighsTransformedLp transLp(*lp, mipdata.implications,
                             mipworker_.getGlobalDomain());
  if (mipworker_.getGlobalDomain().infeasible()) {
    status = HighsLpRelaxation::Status::kInfeasible;
    return finish();
  }
  for (const std::unique_ptr<HighsSeparator>& separator : separators) {
    separator->run(*lp, lpAggregator, transLp, mipworker_.getCutPool());
    if (mipworker_.getGlobalDomain().infeasible()) {
      status = HighsLpRelaxation::Status::kInfeasible;
      return finish();
    }
  }

  numboundchgs = propagateAndResolve();
  if (numboundchgs == -1)
    return finish();
  else
    ncuts += numboundchgs;

  cutset.clear();
  mipworker_.getCutPool().separate(sol.col_value, propdomain, cutset,
                                   mipdata.feastol, mipdata.cutpools, false,
                                   limits.max_selected_rows,
                                   limits.max_selected_nonzeros,
                                   &limits.max_selected_by_origin);
  // Also separate the global cut pool
  if (&mipworker_.getCutPool() != &mipdata.getCutPool()) {
    HighsInt remaining_rows = limits.max_selected_rows;
    if (remaining_rows >= 0) remaining_rows -= cutset.numCuts();
    int64_t remaining_nonzeros = limits.max_selected_nonzeros;
    if (remaining_nonzeros >= 0)
      remaining_nonzeros -= static_cast<int64_t>(cutset.ARindex_.size());
    mipdata.getCutPool().separate(sol.col_value, propdomain, cutset,
                                  mipdata.feastol, mipdata.cutpools, true,
                                  remaining_rows < 0
                                      ? -1
                                      : std::max<HighsInt>(0, remaining_rows),
                                  remaining_nonzeros < 0
                                      ? -1
                                      : std::max<int64_t>(0,
                                                          remaining_nonzeros),
                                  &limits.max_selected_by_origin);
  }

  if (cutset.numCuts() > 0) {
    result.cuts_selected = cutset.numCuts();
    result.cuts_added = cutset.numCuts();
    result.selected_nonzeros = cutset.ARindex_.size();
    for (HighsCutOrigin origin : cutset.origins)
      ++result.selected_by_origin[static_cast<size_t>(origin)];
    ncuts += cutset.numCuts();
    lp->addCuts(cutset);
    status = lp->resolveLp(&propdomain);
    lp->performAging(true);

    // only for the master domain.
    if (&propdomain == &mipdata.getDomain() &&
        lp->unscaledDualFeasible(status)) {
      mipdata.redcostfixing.addRootRedcost(
          mipdata.mipsolver, lp->getSolution().col_dual, lp->getObjective());
      if (mipdata.upper_limit != kHighsInf)
        mipdata.redcostfixing.propagateRootRedcost(mipdata.mipsolver);
    }
  }

  result.bound_changes = ncuts - result.cuts_added;
  return finish();
}

HighsInt HighsSeparation::separationRound(
    HighsDomain& propdomain, HighsLpRelaxation::Status& status) {
  const HighsNodeSeparationRoundResult result =
      separationRoundDetailed(propdomain, status, RoundLimits{});
  return result.bound_changes + result.cuts_added;
}

void HighsSeparation::separate(HighsDomain& propdomain) {
  HighsLpRelaxation::Status status = lp->getStatus();
  const HighsMipSolver& mipsolver = lp->getMipSolver();

  if (lp->scaledOptimal(status) && !lp->getFractionalIntegers().empty()) {
    // double firstobj = lp->getObjective();
    double firstobj = mipsolver.mipdata_->rootlpsolobj;

    while (lp->getObjective() < mipworker_.optimality_limit) {
      double lastobj = lp->getObjective();

      int64_t nlpiters = -lp->getNumLpIterations();
      HighsInt ncuts = separationRound(propdomain, status);
      nlpiters += lp->getNumLpIterations();

      if (mipsolver.mipdata_->parallelLockActive()) {
        mipworker_.getSepaLpIterations() += nlpiters;
      } else {
        mipsolver.mipdata_->sepa_lp_iterations += nlpiters;
        mipsolver.mipdata_->total_lp_iterations += nlpiters;
      }

      // printf("separated %" HIGHSINT_FORMAT " cuts\n", ncuts);

      // printf(
      //     "separation round %" HIGHSINT_FORMAT " at node %" HIGHSINT_FORMAT "
      //     added %" HIGHSINT_FORMAT " cuts objective changed " "from %g to %g,
      //     first obj is %g\n", nrounds, (HighsInt)nnodes, ncuts, lastobj,
      //     lp->getObjective(), firstobj);
      if (ncuts == 0 || !lp->scaledOptimal(status) ||
          lp->getFractionalIntegers().empty())
        break;

      // if the objective improved considerably we continue
      if ((lp->getObjective() - firstobj) <=
          std::max((lastobj - firstobj), mipsolver.mipdata_->feastol) * 1.01)
        break;
    }

    // printf("done separating\n");
  } else {
    // printf("no separation, just aging. status: %" HIGHSINT_FORMAT "\n",
    //        (HighsInt)status);
    lp->performAging(true);

    mipworker_.getCutPool().performAging();
  }
}

HighsNodeSeparationSummary HighsSeparation::separate(
    HighsDomain& propdomain, const HighsNodeSeparationConfig& config,
    const HighsNodeSeparationContext& context) {
  if (!config.enabled) {
    separate(propdomain);
    HighsNodeSeparationSummary summary;
    summary.stop_reason =
        HighsNodeSeparationReason::kControllerDisabled;
    summary.opportunity_sequence = context.opportunity_sequence;
    summary.tree_nodes_processed = context.tree_nodes_processed;
    summary.depth = context.depth;
    return summary;
  }

  HighsNodeSeparationController controller;
  controller.beginNode(config, context);
  HighsLpRelaxation::Status status = lp->getStatus();
  const HighsMipSolver& mipsolver = lp->getMipSolver();

  auto capture = [&]() {
    HighsNodeSeparationContext current = context;
    current.active_rows = lp->numRows();
    current.active_nonzeros = lp->numNonzeros();
    current.lp_iterations = lp->getNumLpIterations();
    current.time = mipsolver.timer_.read();
    current.objective = lp->getObjective();
    current.fractional_integers = lp->getFractionalIntegers().size();
    return current;
  };

  if (!controller.eligible()) {
    lp->performAging(true);
    mipworker_.getCutPool().performAging();
  } else if (!lp->scaledOptimal(status)) {
    controller.stop(HighsNodeSeparationReason::kLpNotOptimal);
    lp->performAging(true);
    mipworker_.getCutPool().performAging();
  } else if (lp->getFractionalIntegers().empty()) {
    controller.stop(HighsNodeSeparationReason::kIntegral);
    lp->performAging(true);
    mipworker_.getCutPool().performAging();
  } else {
    const double legacy_first_objective = mipsolver.mipdata_->rootlpsolobj;
    while (lp->getObjective() < mipworker_.optimality_limit) {
      const HighsNodeSeparationContext current = capture();
      const HighsNodeSeparationDecision decision =
          controller.beforeRound(current);
      if (!decision.continue_separation) {
        controller.stop(decision.reason);
        break;
      }

      RoundLimits limits;
      limits.max_selected_rows = controller.remainingRows();
      limits.max_selected_nonzeros = controller.remainingNonzeros();
      limits.max_selected_by_origin = controller.remainingCutsPerOrigin();
      limits.base_model_rows_only =
          controller.config().base_model_rows_only;
      const double last_objective = lp->getObjective();
      const HighsNodeSeparationRoundResult round =
          separationRoundDetailed(propdomain, status, limits);

      if (mipsolver.mipdata_->parallelLockActive()) {
        mipworker_.getSepaLpIterations() += round.lp_iterations;
      } else {
        mipsolver.mipdata_->sepa_lp_iterations += round.lp_iterations;
        mipsolver.mipdata_->total_lp_iterations += round.lp_iterations;
      }
      controller.recordRound(round);

      if (round.node_infeasible || !round.lp_optimal) {
        controller.stop(HighsNodeSeparationReason::kLpNotOptimal);
        break;
      }
      if (round.bound_changes + round.cuts_added == 0) {
        controller.stop(HighsNodeSeparationReason::kNoCuts);
        break;
      }
      if (lp->getFractionalIntegers().empty()) {
        controller.stop(HighsNodeSeparationReason::kIntegral);
        break;
      }

      if (controller.summary().effective_mode ==
              HighsNodeCutMode::kUnlimitedLegacy &&
          (lp->getObjective() - legacy_first_objective) <=
              std::max(last_objective - legacy_first_objective,
                       mipsolver.mipdata_->feastol) *
                  1.01) {
        controller.stop(HighsNodeSeparationReason::kLowMarginalValue);
        break;
      }
    }
    if (controller.summary().stop_reason ==
            HighsNodeSeparationReason::kNone &&
        lp->getObjective() >= mipworker_.optimality_limit)
      controller.stop(HighsNodeSeparationReason::kObjectiveLimit);
  }

  const HighsNodeSeparationSummary& summary = controller.summary();
  highsLogUser(
      mipsolver.options_mip_->log_options, HighsLogType::kInfo,
      "MIP-NodeSeparation: schema=rost/highs-node-separation/v1 "
      "event=node_exit opportunity=%lld tree_nodes=%lld depth=%lld "
      "requested_mode=%s "
      "effective_mode=%s eligible=%d reason=%s rounds=%lld "
      "phase_time=%.17g phase_lp_iterations=%lld "
      "candidates_generated=%lld cuts_selected=%lld cuts_added=%lld "
      "selected_nonzeros=%lld dual_gain=%.17g "
      "fractional_reduction=%lld low_value_rounds=%lld "
      "base_model_rows_only=%d generator_model_rows=%lld "
      "generator_cut_pool_rows=%lld rejected_cut_pool_rows=%lld\n",
      static_cast<long long>(summary.opportunity_sequence),
      static_cast<long long>(summary.tree_nodes_processed),
      static_cast<long long>(summary.depth),
      HighsNodeSeparationController::modeName(summary.requested_mode),
      HighsNodeSeparationController::modeName(summary.effective_mode),
      summary.eligible ? 1 : 0,
      HighsNodeSeparationController::reasonName(summary.stop_reason),
      static_cast<long long>(summary.rounds), summary.elapsed_seconds,
      static_cast<long long>(summary.lp_iterations),
      static_cast<long long>(summary.candidates_generated),
      static_cast<long long>(summary.cuts_selected),
      static_cast<long long>(summary.cuts_added),
      static_cast<long long>(summary.selected_nonzeros), summary.dual_gain,
      static_cast<long long>(summary.fractional_reduction),
      static_cast<long long>(summary.consecutive_low_value_rounds),
      controller.config().base_model_rows_only ? 1 : 0,
      static_cast<long long>(summary.generator_model_row_uses),
      static_cast<long long>(summary.generator_cut_pool_row_uses),
      static_cast<long long>(summary.rejected_cut_pool_row_uses));
  for (size_t i = 0; i != summary.generated_by_origin.size(); ++i) {
    if (summary.generated_by_origin[i] == 0) continue;
    highsLogUser(
        mipsolver.options_mip_->log_options, HighsLogType::kInfo,
        "MIP-NodeSeparation: schema=rost/highs-node-separation/v1 "
        "event=family opportunity=%lld tree_nodes=%lld depth=%lld origin=%s "
        "generated=%lld "
        "selected=%lld\n",
        static_cast<long long>(summary.opportunity_sequence),
        static_cast<long long>(summary.tree_nodes_processed),
        static_cast<long long>(summary.depth),
        HighsCutPool::originName(static_cast<HighsCutOrigin>(i)),
        static_cast<long long>(summary.generated_by_origin[i]),
        static_cast<long long>(summary.selected_by_origin[i]));
  }
  return summary;
}
