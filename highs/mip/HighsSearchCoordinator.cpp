#include "mip/HighsSearchCoordinator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

HighsNodeQueue::OpenNode HighsDetachedNode::node() const {
  auto c = changes;
  auto b = branchings;
  return HighsNodeQueue::OpenNode(std::move(c), std::move(b), bound, estimate,
                                  depth, basis, yields);
}

bool HighsDetachedNode::valid(HighsInt columns) const {
  if (columns < 0 || depth < 1 || yields < 0 || std::isnan(bound) ||
      bound >= kHighsInf || std::isnan(estimate) || estimate >= kHighsInf)
    return false;
  for (const auto& c : changes)
    if (c.column < 0 || c.column >= columns || !std::isfinite(c.boundval) ||
        (c.boundtype != HighsBoundType::kLower &&
         c.boundtype != HighsBoundType::kUpper))
      return false;
  HighsInt previous = -1;
  for (HighsInt p : branchings) {
    if (p <= previous || p < 0 || size_t(p) >= changes.size()) return false;
    previous = p;
  }
  if (basis && basis->col_status.size() != size_t(columns)) return false;
  return true;
}

size_t HighsDetachedNode::bytes() const {
  size_t bytes = sizeof(HighsDetachedNode);
  auto add = [&](size_t n, size_t w) {
    if (n > (std::numeric_limits<size_t>::max() - bytes) / w) return false;
    bytes += n * w;
    return true;
  };
  if (!add(changes.size(), sizeof(HighsDomainChange)) ||
      !add(branchings.size(), sizeof(HighsInt)))
    return std::numeric_limits<size_t>::max();
  // Basis references share immutable storage; charge their full size anyway.
  if (basis && (!add(basis->col_status.size(), sizeof(HighsBasisStatus)) ||
                !add(basis->row_status.size(), sizeof(HighsBasisStatus))))
    return std::numeric_limits<size_t>::max();
  return bytes;
}

HighsSearchCoordinator::HighsSearchCoordinator(const Generation& generation,
                                               HighsInt columns, size_t workers,
                                               size_t byte_limit,
                                               double inherited_bound)
    : generation_(generation),
      columns_(columns),
      byte_limit_(byte_limit),
      waiting_(workers, false),
      ledger_(generation) {
  if (!workers || !ledger_.initialize(0, inherited_bound))
    throw std::invalid_argument("invalid search epoch");
}

bool HighsSearchCoordinator::offer(const Generation& generation,
                                   const HighsDetachedNode& node) {
  if (!(generation == generation_) || !node.valid(columns_)) return false;
  const size_t bytes = node.bytes();
  std::lock_guard<std::mutex> lock(mutex_);
  if (failed_ || bytes > byte_limit_ - bytes_ ||
      next_id_ == std::numeric_limits<uint64_t>::max())
    return false;
  // Allocate the complete packet before transferring the donor's subtree.
  auto copy = std::make_shared<const HighsDetachedNode>(node);
  Pending packet;
  packet.receipt.id = next_id_;
  packet.receipt.generation = generation_;
  packet.receipt.node = std::move(copy);
  pending_.push_back(std::move(packet));
  ++next_id_;
  ++offered_;
  bytes_ += bytes;
  ready_.notify_all();
  return true;
}

bool HighsSearchCoordinator::take(uint64_t worker, Receipt& out, bool wait) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (worker >= waiting_.size() || waiting_[worker]) return false;
  waiting_[worker] = true;
  ready_.notify_all();
  auto available = [&]() {
    return std::find_if(pending_.begin(), pending_.end(),
                        [](const Pending& p) { return !p.leased; });
  };
  auto all_waiting = [&]() {
    return std::all_of(waiting_.begin(), waiting_.end(),
                       [](bool x) { return x; });
  };
  if (wait)
    ready_.wait(lock, [&]() {
      return failed_ || available() != pending_.end() || all_waiting();
    });
  auto it = available();
  if (failed_ || it == pending_.end()) {
    // A retired worker cannot produce new work. Keep its waiting bit set so
    // the last idle worker observes termination instead of waiting forever.
    ready_.notify_all();
    return false;
  }
  Receipt copy = it->receipt;
  copy.worker = worker;
  out = std::move(copy);
  it->receipt.worker = worker;
  it->leased = true;
  waiting_[worker] = false;
  return true;
}

bool HighsSearchCoordinator::installed(const Receipt& receipt) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!(receipt.generation == generation_)) return false;
  auto it = std::find_if(
      pending_.begin(), pending_.end(),
      [&](const Pending& p) { return p.receipt.id == receipt.id; });
  if (it == pending_.end() || !it->leased ||
      it->receipt.worker != receipt.worker || it->receipt.node != receipt.node)
    return false;
  bytes_ -= it->receipt.node->bytes();
  pending_.erase(it);
  ++installed_;
  return true;
}

void HighsSearchCoordinator::fail() {
  std::lock_guard<std::mutex> lock(mutex_);
  failed_ = true;
  ready_.notify_all();
}
void HighsSearchCoordinator::retire(uint64_t worker) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (worker < waiting_.size()) waiting_[worker] = true;
  ready_.notify_all();
}
bool HighsSearchCoordinator::failed() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return failed_;
}
std::vector<HighsDetachedNode> HighsSearchCoordinator::drain() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<HighsDetachedNode> nodes;
  nodes.reserve(pending_.size());
  for (const auto& p : pending_) nodes.push_back(*p.receipt.node);
  pending_.clear();
  bytes_ = 0;
  return nodes;
}
double HighsSearchCoordinator::conservativeBound() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return ledger_.lowerBound();
}
size_t HighsSearchCoordinator::offered() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return offered_;
}
size_t HighsSearchCoordinator::installed() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return installed_;
}
