#ifndef HIGHS_SOLVE_CAPACITY_H_
#define HIGHS_SOLVE_CAPACITY_H_

#include <cstddef>
#include <mutex>

// Reservations account for executor slots and model-copy bytes, not peak RSS.
// The owner must outlive every lease. Numerical factor fill is not bounded
// here.
class HighsSolveCapacity {
 public:
  class Lease {
   public:
    Lease() = default;
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    ~Lease() { release(); }
    bool active() const { return owner_ != nullptr; }
    void release() {
      if (!owner_) return;
      std::lock_guard<std::mutex> lock(owner_->mutex_);
      owner_->slots_used_ -= slots_;
      owner_->bytes_used_ -= bytes_;
      owner_ = nullptr;
    }

   private:
    friend class HighsSolveCapacity;
    HighsSolveCapacity* owner_ = nullptr;
    size_t slots_ = 0;
    size_t bytes_ = 0;
  };

  HighsSolveCapacity(size_t slots, size_t bytes)
      : slots_(slots), bytes_(bytes) {}
  bool reserve(size_t slots, size_t bytes, Lease& lease) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (lease.active() || !slots || slots > slots_ - slots_used_ ||
        bytes > bytes_ - bytes_used_)
      return false;
    slots_used_ += slots;
    bytes_used_ += bytes;
    lease.owner_ = this;
    lease.slots_ = slots;
    lease.bytes_ = bytes;
    return true;
  }

 private:
  std::mutex mutex_;
  const size_t slots_;
  const size_t bytes_;
  size_t slots_used_ = 0;
  size_t bytes_used_ = 0;
};
#endif
