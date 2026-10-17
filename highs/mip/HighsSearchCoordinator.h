#ifndef HIGHS_SEARCH_COORDINATOR_H_
#define HIGHS_SEARCH_COORDINATOR_H_

#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>

#include "mip/HighsNodeQueue.h"
#include "mip/HighsProofFrontier.h"

// Only mathematical node data crosses a worker boundary. In particular no
// queue allocator links, domain registrations or numerical workspace is copied.
struct HighsDetachedNode {
  std::vector<HighsDomainChange> changes;
  std::vector<HighsInt> branchings;
  std::shared_ptr<const HighsBasis> basis;
  double bound = -kHighsInf;
  double estimate = -kHighsInf;
  HighsInt depth = 0;
  HighsInt yields = 0;
  HighsNodeQueue::OpenNode node() const;
  bool valid(HighsInt columns) const;
  size_t bytes() const;
};

// Epoch-local owner for immutable donations. The native global bound stays
// frozen until all workers join. A covering obligation remains in the ledger
// for the whole epoch; local estimates never supply a new proof bound.
class HighsSearchCoordinator {
 public:
  using Generation = HighsProofFrontier::Generation;
  struct Receipt {
    uint64_t id = 0;
    uint64_t worker = 0;
    Generation generation;
    std::shared_ptr<const HighsDetachedNode> node;
  };
  HighsSearchCoordinator(const Generation&, HighsInt columns, size_t workers,
                         size_t byte_limit, double inherited_bound);
  bool offer(const Generation&, const HighsDetachedNode&);
  bool take(uint64_t worker, Receipt&, bool wait);
  bool installed(const Receipt&);
  void retire(uint64_t worker);
  void fail();
  bool failed() const;
  // May be called only after every producer/receiver has joined.
  std::vector<HighsDetachedNode> drain();
  double conservativeBound() const;
  size_t offered() const;
  size_t installed() const;

 private:
  struct Pending {
    Receipt receipt;
    bool leased = false;
  };
  const Generation generation_;
  const HighsInt columns_;
  const size_t byte_limit_;
  mutable std::mutex mutex_;
  std::condition_variable ready_;
  std::vector<Pending> pending_;
  std::vector<bool> waiting_;
  size_t bytes_ = 0;
  size_t offered_ = 0;
  size_t installed_ = 0;
  uint64_t next_id_ = 1;
  bool failed_ = false;
  HighsProofFrontier ledger_;
};
#endif
