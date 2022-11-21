#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <cassert>
#include <functional>
#include "ndarray_backend_cpu_impl.hpp"

namespace needle::cpu {

/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
 * here by default.
 */

AlignedArray::AlignedArray(const size_t size_) : size(size_) {
  if (int ret = posix_memalign((void **) &ptr, ALIGNMENT, size * ELEM_SIZE);ret != 0) throw std::bad_alloc();
}

[[nodiscard]] size_t AlignedArray::ptr_as_int() const { return (size_t) ptr; }

void Fill(AlignedArray *out, scalar_t val) {
  /**
   * Fill the values of an aligned array with val
   */
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}

std::vector<uint32_t> get_indices(std::vector<uint32_t> const &shape,
                                  std::vector<uint32_t> const &strides) {

  std::vector<uint32_t> res{};

  auto index_size = static_cast<int>(shape.size());
  std::vector<uint32_t> indices(index_size, 0);

  auto next = [&](std::vector<uint32_t> &indices) {
    for (int i = index_size - 1; i >= 0; --i) {
      if (++indices[i] == shape[i]) indices[i] = 0;
      else return true;
    }
    return false;
  };

  auto add_index = [&](std::vector<uint32_t> &indices) {
    uint32_t index = 0;
    for (int i = 0; i < index_size; ++i) {
      index += indices[i] * strides[i];
    }
    res.push_back(index);
  };

  do {
    add_index(indices);
  } while (next(indices));

  return res;
}

void Compact(const AlignedArray &a, AlignedArray *out, const std::vector<uint32_t> &shape,
             const std::vector<uint32_t> &strides, size_t offset) {
  /**
   * Compact an array in memory
   *
   * Args:
   *   a: non-compact representation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   *
   * Returns:
   *  void (you need to modify out directly, rather than returning anything; this is true for all the
   *  function will implement here, so we won't repeat this note.)
   */
  /// BEGIN YOUR SOLUTION
  auto const indices = get_indices(shape, strides);
  int counter = 0;
  for (auto &index : indices) {
    out->ptr[counter] = a.ptr[offset + index];
    ++counter;
  }
  /// END YOUR SOLUTION
}

void EwiseSetitem(const AlignedArray &a, AlignedArray *out, const std::vector<uint32_t> &shape,
                  const std::vector<uint32_t> &strides, size_t offset) {
  /**
   * Set items in a (non-compact) array
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN YOUR SOLUTION

  auto const indices = get_indices(shape, strides);

  int counter = 0;
  for (auto &index : indices) {
    out->ptr[offset + index] = a.ptr[counter];
    ++counter;
  }
  /// END YOUR SOLUTION
}

void ScalarSetitem(const size_t size, scalar_t val, AlignedArray *out, const std::vector<uint32_t> &shape,
                   const std::vector<uint32_t> &strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but convenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */

  /// BEGIN YOUR SOLUTION
  auto const indices = get_indices(shape, strides);
  for (auto &index : indices) {
    out->ptr[offset + index] = val;
  }
  /// END YOUR SOLUTION
}

/**
 * In the code the follows, use the above template to create analogous element-wise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

/// BEGIN YOUR SOLUTION
EWISE_BINARY_OP(Add, std::plus{})
SCALAR_BINARY_OP(Add, std::plus{})
EWISE_BINARY_OP(Mul, std::multiplies{})
SCALAR_BINARY_OP(Mul, std::multiplies{})
EWISE_BINARY_OP(Div, std::divides{})
SCALAR_BINARY_OP(Div, std::divides{})
SCALAR_BINARY_OP(Power, std::pow)
EWISE_BINARY_OP(Maximum, std::max)
SCALAR_BINARY_OP(Maximum, std::max)
EWISE_BINARY_OP(Eq, [](scalar_t lhs, scalar_t rhs) { return static_cast<scalar_t >(lhs == rhs); })
SCALAR_BINARY_OP(Eq, [](scalar_t lhs, scalar_t rhs) { return static_cast<scalar_t >(lhs == rhs); })
EWISE_BINARY_OP(Ge, [](scalar_t lhs, scalar_t rhs) { return static_cast<scalar_t >(lhs >= rhs); })
SCALAR_BINARY_OP(Ge, [](scalar_t lhs, scalar_t rhs) { return static_cast<scalar_t >(lhs >= rhs); })
EWISE_UNARY_OP(Log, std::log)
EWISE_UNARY_OP(Exp, std::exp)
EWISE_UNARY_OP(Tanh, std::tanh)
/// END YOUR SOLUTION

void Matmul(const AlignedArray &a, const AlignedArray &b, AlignedArray *out, uint32_t m, uint32_t n,
            uint32_t p) {
  /**
   * Multiply two (compact) matrices into an output (also compact) matrix.  For this implementation
   * you can use the "naive" three-loop algorithm.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: compact 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   */

  /// BEGIN YOUR SOLUTION

  for (uint32_t i = 0; i < m; ++i) {
    for (uint32_t j = 0; j < p; ++j) {
      scalar_t sum = 0;
      for (uint32_t k = 0; k < n; ++k) {
        sum += a.ptr[i * n + k] * b.ptr[k * p + j];
      }
      out->ptr[i * p + j] = sum;
    }
  }

  /// END YOUR SOLUTION
}

inline void AlignedDot(const float *__restrict__ a,
                       const float *__restrict__ b,
                       float *__restrict__ out) {

  /**
   * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
   * the result to the existing out, which you should not set to zero beforehand).  We are including
   * the compiler flags here that enable the compile to properly use vector operators to implement
   * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and
   * out don't have any overlapping memory (which is necessary in order for vector operations to be
   * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b,
   * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the
   * compiler that the input array will be aligned to the appropriate blocks in memory, which also
   * helps the compiler vectorize the code.
   *
   * Args:
   *   a: compact 2D array of size TILE x TILE
   *   b: compact 2D array of size TILE x TILE
   *   out: compact 2D array of size TILE x TILE to write to
   */

  a = (const float *) __builtin_assume_aligned(a, TILE * ELEM_SIZE);
  b = (const float *) __builtin_assume_aligned(b, TILE * ELEM_SIZE);
  out = (float *) __builtin_assume_aligned(out, TILE * ELEM_SIZE);

  /// BEGIN YOUR SOLUTION

  for (uint32_t i = 0; i < TILE; ++i) {
    for (uint32_t j = 0; j < TILE; ++j) {
      for (uint32_t k = 0; k < TILE; ++k) {
        out[i * TILE + j] += a[i * TILE + k] * b[k * TILE + j];
      }
    }
  }
  /// END YOUR SOLUTION
}

void MatmulTiled(const AlignedArray &a, const AlignedArray &b, AlignedArray *out, uint32_t m,
                 uint32_t n, uint32_t p) {
  /**
   * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
   * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
   *   a[m/TILE][n/TILE][TILE][TILE]
   * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
   * function should call `AlignedDot()` implemented above).
   *
   * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
   * assume that this division happens without any remainder.
   *
   * Args:
   *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
   *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
   *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   *
   */
  /// BEGIN YOUR SOLUTION

  // init out to zero
  for (int i = 0; i < m * p; ++i) {
    out->ptr[i] = 0;
  }

  for (uint32_t i = 0; i < m / TILE; ++i) {
    for (uint32_t j = 0; j < p / TILE; ++j) {
      for (uint32_t k = 0; k < n / TILE; ++k) {
        AlignedDot(a.ptr + (i * n / TILE + k) * TILE * TILE,
                   b.ptr + (k * p / TILE + j) * TILE * TILE,
                   out->ptr + (i * p / TILE + j) * TILE * TILE);
      }
    }
  }

  /// END YOUR SOLUTION
}

void ReduceMax(const AlignedArray &a, AlignedArray *out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN YOUR SOLUTION
  std::vector<scalar_t> tmp(reduce_size);

  for (size_t i = 0; i < out->size; i++) {
    for (size_t j = 0; j < reduce_size; j++) {
      tmp[j] = a.ptr[i * reduce_size + j];
    }

    out->ptr[i] = *std::max_element(tmp.begin(), tmp.end());
  }


  /// END YOUR SOLUTION
}

void ReduceSum(const AlignedArray &a, AlignedArray *out, size_t reduce_size) {
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN YOUR SOLUTION

  for (size_t i = 0; i < out->size; i++) {
    float sum = 0;
    for (size_t j = 0; j < reduce_size; j++) {
      sum += a.ptr[i * reduce_size + j];
    }
    out->ptr[i] = sum;
  }

  /// END YOUR SOLUTION
}

}
