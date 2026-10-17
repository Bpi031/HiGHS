/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#ifndef HIGHS_CACHE_ALIGN_H_
#define HIGHS_CACHE_ALIGN_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <utility>

namespace highs {

struct cache_aligned {
  static constexpr std::size_t alignment() {
#if defined(__APPLE__) && defined(__aarch64__)
    return 128;
#else
    return 64;
#endif
  }

  static void* alloc(std::size_t size,
                     std::size_t requested_alignment = alignment()) {
    if (requested_alignment < alignment()) requested_alignment = alignment();
    if ((requested_alignment & (requested_alignment - 1)) != 0 ||
        size > std::numeric_limits<std::size_t>::max() - requested_alignment)
      throw std::bad_alloc();
    void* base = ::operator new(size + requested_alignment);
    void* ptr = static_cast<char*>(base) + sizeof(void*);
    std::size_t space = size + requested_alignment - sizeof(void*);
    void* aligned_ptr = std::align(requested_alignment, size, ptr, space);
    if (!aligned_ptr) {
      ::operator delete(base);
      throw std::bad_alloc();
    }
    std::memcpy(static_cast<char*>(aligned_ptr) - sizeof(void*), &base,
                sizeof(void*));
    return aligned_ptr;
  }

  static void free(void* aligned_ptr) {
    if (aligned_ptr) {
      void* freeptr;
      std::memcpy(&freeptr,
                  static_cast<char*>(aligned_ptr) - sizeof(void*),
                  sizeof(void*));
      ::operator delete(freeptr);
    }
  }

  template <typename T>
  struct Deleter {
    void operator()(T* ptr) const {
      ptr->~T();
      free(ptr);
    }
  };

  template <typename T>
  struct Deleter<T[]> {
    void operator()(T* ptr) const { free(ptr); }
  };

  template <typename T>
  using unique_ptr = std::unique_ptr<T, Deleter<T>>;

  template <typename T>
  using shared_ptr = std::shared_ptr<T>;

  template <typename T, typename... Args>
  static shared_ptr<T> make_shared(Args&&... args) {
    return shared_ptr<T>(construct<T>(std::forward<Args>(args)...), Deleter<T>());
  }

  template <typename T, typename... Args>
  static unique_ptr<T> make_unique(Args&&... args) {
    return unique_ptr<T>(construct<T>(std::forward<Args>(args)...));
  }

  // Raw array storage: its owner must construct and destroy nontrivial elements.
  template <typename T, typename... Args>
  static unique_ptr<T[]> make_unique_array(std::size_t N) {
    if (N > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::bad_alloc();
    return unique_ptr<T[]>(static_cast<T*>(alloc(sizeof(T) * N, alignof(T))));
  }

 private:
  template <typename T, typename... Args>
  static T* construct(Args&&... args) {
    void* ptr = alloc(sizeof(T), alignof(T));
    try {
      return new (ptr) T(std::forward<Args>(args)...);
    } catch (...) {
      free(ptr);
      throw;
    }
  }
};

}  // namespace highs

#endif
