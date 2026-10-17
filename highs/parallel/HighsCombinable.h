/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef HIGHS_COMBINABLE_H_
#define HIGHS_COMBINABLE_H_

#include <functional>
#include <type_traits>

#include "parallel/HighsCacheAlign.h"
#include "parallel/HighsTaskExecutor.h"

template <typename T, typename FConstruct_ = std::function<T(void)>>
class HighsCombinable {
  FConstruct_ construct_;
  struct PaddedData {
    alignas(highs::cache_aligned::alignment()) bool initialized_;
    typename std::aligned_storage<sizeof(T), alignof(T)>::type storage_;
    PaddedData() : initialized_(false) {}
    T* value() { return reinterpret_cast<T*>(&storage_); }
  };
  int numThreads;
  highs::cache_aligned::unique_ptr<PaddedData[]> threadCopies_;

 public:
  HighsCombinable()
      : construct_([]() { return T(); }),
        numThreads(HighsTaskExecutor::getNumWorkerThreads()),
        threadCopies_(
            highs::cache_aligned::make_unique_array<PaddedData>(numThreads)) {
    for (int i = 0; i < numThreads; ++i) new (&threadCopies_[i]) PaddedData();
  }

  template <typename FConstruct>
  HighsCombinable(FConstruct&& fConstruct)
      : construct_(std::forward<FConstruct>(fConstruct)),
        numThreads(HighsTaskExecutor::getNumWorkerThreads()),
        threadCopies_(
            highs::cache_aligned::make_unique_array<PaddedData>(numThreads)) {
    for (int i = 0; i < numThreads; ++i) new (&threadCopies_[i]) PaddedData();
  }

  HighsCombinable<T, FConstruct_>& operator=(
      HighsCombinable<T, FConstruct_>&& other) {
    if (this != &other) {
      clear();
      construct_ = std::move(other.construct_);
      threadCopies_ = std::move(other.threadCopies_);
      numThreads = other.numThreads;
      other.numThreads = 0;
    }
    return *this;
  }
  HighsCombinable(HighsCombinable<T, FConstruct_>&& other)
      : construct_(std::move(other.construct_)),
        numThreads(other.numThreads),
        threadCopies_(std::move(other.threadCopies_)) {
    other.numThreads = 0;
  }

  void clear() {
    for (int i = 0; i < numThreads; ++i) {
      if (threadCopies_[i].initialized_) {
        threadCopies_[i].initialized_ = false;
        threadCopies_[i].value()->~T();
      }
    }
  }

  T& local() {
    int threadId = HighsTaskExecutor::getThisWorkerDeque()->getOwnerId();
    if (!threadCopies_[threadId].initialized_) {
      new (&threadCopies_[threadId].storage_) T(construct_());
      threadCopies_[threadId].initialized_ = true;
    }

    return *threadCopies_[threadId].value();
  }

  const T& local() const {
    int threadId = HighsTaskExecutor::getThisWorkerDeque()->getOwnerId();
    if (!threadCopies_[threadId].initialized_) {
      new (&threadCopies_[threadId].storage_) T(construct_());
      threadCopies_[threadId].initialized_ = true;
    }

    return *threadCopies_[threadId].value();
  }

  template <typename FCombine>
  void combine_each(FCombine&& combine) {
    for (int i = 0; i < numThreads; ++i)
      if (threadCopies_[i].initialized_) combine(*threadCopies_[i].value());
  }

  template <typename FCombine>
  T combine(FCombine&& combine) {
    T combined{};
    int i;
    for (i = 0; i < numThreads; ++i) {
      if (threadCopies_[i].initialized_) {
        combined = std::move(*threadCopies_[i].value());
        break;
      }
    }

    for (++i; i < numThreads; ++i) {
      if (threadCopies_[i].initialized_) {
        combined =
            combine(std::move(combined), std::move(*threadCopies_[i].value()));
      }
    }

    return combined;
  }

  ~HighsCombinable() {
    if (threadCopies_) clear();
  }
};

template <typename U, typename FConstruct>
HighsCombinable<U, FConstruct> makeHighsCombinable(FConstruct&& fconstruct) {
  return HighsCombinable<U, FConstruct>(std::forward<FConstruct>(fconstruct));
}

#endif
