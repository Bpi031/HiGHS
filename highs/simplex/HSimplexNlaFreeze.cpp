/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file simplex/HSimplexNlaFreeze.cpp
 *
 * @brief Interface to HFactor allowing non-HFactor updates, NLA-only
 * scaling and shifting of NLA analysis below simplex level.
 */
#include <stdio.h>

#include "simplex/HSimplexNla.h"

void SimplexIterate::clear() {
  this->valid_ = false;
  this->basis_.clear();
  this->invert_.clear();
  this->dual_edge_weight_.clear();
}

void HSimplexNla::putInvert() {
  // HEkk publishes validity only after the basis and weights are copied too.
  simplex_iterate_.valid_ = false;
  factor_.getInvert(simplex_iterate_.invert_);
}
void HSimplexNla::getInvert() { factor_.setInvert(simplex_iterate_.invert_); }
