/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file util/HighsDataStack.h
 * @brief A stack of unstructured data stored as bytes
 */

#ifndef UTIL_HIGHS_DATA_STACK_H_
#define UTIL_HIGHS_DATA_STACK_H_

#include <cassert>
#include <cstring>
#include <type_traits>
#include <vector>

#include "util/HighsInt.h"

#if __GNUG__ && __GNUC__ < 5 && !defined(__clang__)
#define IS_TRIVIALLY_COPYABLE(T) __has_trivial_copy(T)
#else
#define IS_TRIVIALLY_COPYABLE(T) std::is_trivially_copyable<T>::value
#endif

class HighsDataStack {
  std::vector<char> data;
  std::size_t position = 0;

  template <typename T,
            typename std::enable_if<IS_TRIVIALLY_COPYABLE(T), int>::type = 0>
  void popAt(T& r, std::size_t& cursor) const {
    assert(cursor <= data.size() && cursor >= sizeof(T));
    cursor -= sizeof(T);
    std::memcpy(&r, data.data() + cursor, sizeof(T));
  }

  template <typename T>
  void popAt(std::vector<T>& r, std::size_t& cursor) const {
    std::size_t numData;
    popAt(numData, cursor);
    assert(numData <= cursor / sizeof(T));
    cursor -= numData * sizeof(T);
    r.resize(numData);
    if (numData) std::memcpy(r.data(), data.data() + cursor, numData * sizeof(T));
  }

 public:
  // Borrow stable bytes, but never share a decoding cursor. The stack must
  // outlive its readers and must not be mutated/moved while they are in use.
  class Reader {
   public:
    explicit Reader(const HighsDataStack& stack)
        : stack_(stack), position_(stack.data.size()) {}

    template <typename T>
    void pop(T& r) { stack_.popAt(r, position_); }

   private:
    const HighsDataStack& stack_;
    std::size_t position_;
  };

  Reader reader() const { return Reader(*this); }

  void resetPosition() { position = data.size(); }

  template <typename T,
            typename std::enable_if<IS_TRIVIALLY_COPYABLE(T), int>::type = 0>
  void push(const T& r) {
    std::size_t dataSize = data.size();
    data.resize(dataSize + sizeof(T));
    std::memcpy(data.data() + dataSize, &r, sizeof(T));
  }

  template <typename T,
            typename std::enable_if<IS_TRIVIALLY_COPYABLE(T), int>::type = 0>
  void pop(T& r) {
    popAt(r, position);
  }

  template <typename T>
  void push(const std::vector<T>& r) {
    std::size_t offset = data.size();
    std::size_t numData = r.size();
    // store the data
    data.resize(offset + numData * sizeof(T) + sizeof(std::size_t));
    if (!r.empty())
      std::memcpy(data.data() + offset, r.data(), numData * sizeof(T));
    // store the vector size
    offset += numData * sizeof(T);
    std::memcpy(data.data() + offset, &numData, sizeof(std::size_t));
  }

  template <typename T>
  void pop(std::vector<T>& r) {
    popAt(r, position);
  }

  void setPosition(size_t position_) { this->position = position_; }

  size_t getCurrentDataSize() const { return data.size(); }
};

#endif
