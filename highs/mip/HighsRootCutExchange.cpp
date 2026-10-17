#include "mip/HighsRootCutExchange.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

HighsRootCutExchange::HighsRootCutExchange(
    const HighsLp& source, const HighsProofFrontier::Generation& generation)
    : source_(source), generation_(generation) {
  const auto& a = source_.a_matrix_;
  if (source_.num_col_ < 0 || source_.num_row_ < 0 ||
      source_.row_lower_.size() != size_t(source_.num_row_) ||
      source_.row_upper_.size() != size_t(source_.num_row_) ||
      (!a.isColwise() && !a.isRowwise()) || a.index_.size() != a.value_.size())
    return;
  const HighsInt major = a.isColwise() ? source_.num_col_ : source_.num_row_;
  const HighsInt minor = a.isColwise() ? source_.num_row_ : source_.num_col_;
  if (a.start_.size() != size_t(major) + 1 || a.start_.front() != 0 ||
      a.start_.back() < 0 || size_t(a.start_.back()) != a.value_.size())
    return;
  for (HighsInt i = 0; i < major; ++i)
    if (a.start_[i] > a.start_[i + 1] || a.start_[i] < 0) return;
  for (size_t i = 0; i < a.index_.size(); ++i)
    if (a.index_[i] < 0 || a.index_[i] >= minor || !std::isfinite(a.value_[i]))
      return;
  source_.a_matrix_.ensureRowwise();
  source_valid_ = true;
}

bool HighsRootCutExchange::derive(HighsInt row, bool lower,
                                  HighsRootCutPacket& packet) const {
  if (!source_valid_ || row < 0 || row >= source_.num_row_ ||
      source_.integrality_.size() != size_t(source_.num_col_))
    return false;
  const double side =
      lower ? -source_.row_lower_[row] : source_.row_upper_[row];
  if (!std::isfinite(side) || std::abs(side) >= 1099511627776.0) return false;
  const auto& a = source_.a_matrix_;
  const auto start = a.start_[row], end = a.start_[row + 1];
  if (end <= start || end - start > 128) return false;
  uint64_t gcd = 0;
  for (HighsInt p = start; p != end; ++p) {
    if (source_.integrality_[a.index_[p]] != HighsVarType::kInteger ||
        !std::isfinite(a.value_[p]) || a.value_[p] == 0 ||
        std::abs(a.value_[p]) > 1048576.0 ||
        a.value_[p] != std::round(a.value_[p]))
      return false;
    uint64_t value = static_cast<uint64_t>(std::abs(a.value_[p]));
    while (value) {
      const uint64_t next = gcd % value;
      gcd = value;
      value = next;
    }
  }
  if (!gcd || (gcd & (gcd - 1))) return false;
  const double divisor = static_cast<double>(gcd);
  const double rhs = std::floor(side / divisor);
  if (rhs == side / divisor) return false;
  HighsRootCutPacket out;
  out.generation = generation_;
  out.unconditional = true;
  out.source_row = row;
  out.lower_side = lower;
  out.divisor = divisor;
  out.rhs = rhs;
  std::vector<std::pair<HighsInt, double>> entries;
  for (HighsInt p = start; p != end; ++p)
    entries.emplace_back(a.index_[p], (lower ? -1 : 1) * a.value_[p] / divisor);
  std::sort(entries.begin(), entries.end());
  for (const auto& e : entries) {
    if (!out.indices.empty() && e.first == out.indices.back()) return false;
    out.indices.push_back(e.first);
    out.coefficients.push_back(e.second);
  }
  packet = std::move(out);
  return true;
}

std::vector<HighsRootCutPacket> HighsRootCutExchange::discover(
    size_t max_rows, size_t max_nonzeros) const {
  std::vector<HighsRootCutPacket> packets;
  if (!source_valid_) return packets;
  size_t nnz = 0;
  for (HighsInt i = 0; i < source_.num_row_; ++i)
    for (bool lower : {false, true}) {
      HighsRootCutPacket packet;
      if (!derive(i, lower, packet)) continue;
      if (packets.size() == max_rows) return packets;
      // A wide row must not hide later valid cuts that fit the same budget.
      if (packet.indices.size() > max_nonzeros - nnz) continue;
      packet.sequence = packets.size() + 1;
      nnz += packet.indices.size();
      packets.push_back(std::move(packet));
    }
  return packets;
}

bool HighsRootCutExchange::check(const HighsRootCutPacket& packet) const {
  if (!(packet.generation == generation_) || !packet.unconditional ||
      packet.domain_generation || !packet.dependencies.empty() ||
      packet.sequence == 0)
    return false;
  HighsRootCutPacket expected;
  return derive(packet.source_row, packet.lower_side, expected) &&
         packet.divisor == expected.divisor && packet.rhs == expected.rhs &&
         packet.indices == expected.indices &&
         packet.coefficients == expected.coefficients;
}

bool HighsRootCutExchange::stage(
    const std::vector<HighsRootCutPacket>& proposals, size_t max_rows,
    size_t max_nonzeros, std::vector<HighsRootCutPacket>& received) const {
  std::vector<HighsRootCutPacket> pending;
  size_t nnz = 0;
  for (const auto& p : proposals) {
    if (!check(p)) return false;
    bool duplicate = false;
    for (const auto& q : pending) {
      const bool same_cut = p.indices == q.indices &&
                            p.coefficients == q.coefficients && p.rhs == q.rhs;
      if (p.producer == q.producer && p.sequence == q.sequence && !same_cut)
        return false;
      duplicate |= same_cut;
    }
    if (duplicate) continue;
    if (pending.size() == max_rows || p.indices.size() > max_nonzeros - nnz)
      return false;
    nnz += p.indices.size();
    pending.push_back(p);
  }
  received.swap(pending);  // malformed/over-budget batch changes nothing
  return true;
}
