#ifndef HIGHS_ROOT_CUT_EXCHANGE_H_
#define HIGHS_ROOT_CUT_EXCHANGE_H_

#include <vector>

#include "lp_data/HighsLp.h"
#include "mip/HighsProofFrontier.h"

struct HighsRootCutPacket {
  HighsProofFrontier::Generation generation;
  uint64_t producer = 0;
  uint64_t sequence = 0;
  uint64_t domain_generation = 0;
  bool unconditional = false;
  HighsInt source_row = -1;
  bool lower_side = false;
  double divisor = 1;
  std::vector<uint64_t> dependencies;
  std::vector<HighsInt> indices;
  std::vector<double> coefficients;
  double rhs = 0;
};

// First audited cut family: exact integral row activities lie on a lattice.
// Only power-of-two divisors are used, so coefficient/side division is exact
// in binary arithmetic. No bound, symmetry or incumbent-cutoff premise is
// used. Other native cut families need separate dependency certificates.
class HighsRootCutExchange {
 public:
  HighsRootCutExchange(const HighsLp&, const HighsProofFrontier::Generation&);
  std::vector<HighsRootCutPacket> discover(size_t max_rows,
                                           size_t max_nonzeros) const;
  bool check(const HighsRootCutPacket&) const;
  bool stage(const std::vector<HighsRootCutPacket>&, size_t max_rows,
             size_t max_nonzeros, std::vector<HighsRootCutPacket>&) const;

 private:
  bool derive(HighsInt row, bool lower, HighsRootCutPacket&) const;
  HighsLp source_;
  HighsProofFrontier::Generation generation_;
  bool source_valid_ = false;
};
#endif
