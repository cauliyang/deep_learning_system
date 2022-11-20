//
// Created by Yangyang Li on 11/20/22.
//

#ifndef NEEDLE_SRC_NDARRAY_BACKEND_CPU_H_
#define NEEDLE_SRC_NDARRAY_BACKEND_CPU_H_
#include <vector>
namespace needle::cpu {

constexpr int TILE = 8;
constexpr int ALIGNMENT = 256;

using scalar_t = float;
constexpr size_t ELEM_SIZE = sizeof(scalar_t);

struct AlignedArray {

  explicit AlignedArray(size_t size_);
  AlignedArray(const AlignedArray &other) = delete;
  AlignedArray &operator=(const AlignedArray &other) = delete;
  ~AlignedArray() { free(ptr); }

  [[nodiscard]] size_t ptr_as_int() const;

  scalar_t *ptr{nullptr};
  size_t size{};
};

/**
 * Fill the values of an aligned array with val
 */
void Fill(AlignedArray *out, scalar_t val);

std::vector<uint32_t> get_indices(std::vector<uint32_t> const &shape,
                                  std::vector<uint32_t> const &strides);
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
void Compact(const AlignedArray &a, AlignedArray *out, const std::vector<uint32_t> &shape,
             const std::vector<uint32_t> &strides, size_t offset);
/**
 *
 * Set items in a (non-compact) array
 *
 * Args:
 *   a: _compact_ array whose items will be written to out
 *   out: non-compact array whose items are to be written
 *   shape: shapes of each dimension for a and out
 *   strides: strides of the *out* array (not a, which has compact strides)
 *   offset: offset of the *out* array (not a, which has zero offset, being compact)
 */
void EwiseSetitem(const AlignedArray &a, AlignedArray *out, const std::vector<uint32_t> &shape,
                  const std::vector<uint32_t> &strides, size_t offset);

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
void ScalarSetitem(const size_t size, scalar_t val, AlignedArray *out, const std::vector<uint32_t> &shape,
                   const std::vector<uint32_t> &strides, size_t offset);
/**
* Set entries in out to be the sum of corresponding entires in a and b.
*/
void EwiseAdd(const AlignedArray &a, const AlignedArray &b, AlignedArray *out);

/**
 * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
 */
void ScalarAdd(const AlignedArray &a, scalar_t val, AlignedArray *out);

//*   - EwiseMul, ScalarMul
//*   - EwiseDiv, ScalarDiv
//*   - ScalarPower
//*   - EwiseMaximum, ScalarMaximum
//*   - EwiseEq, ScalarEq
//*   - EwiseGe, ScalarGe
//*   - EwiseLog
//*   - EwiseExp
//*   - EwiseTanh



#define EWISE_BINARY_OP(name, op) \
  void Ewise##name(const AlignedArray &a, const AlignedArray &b, AlignedArray *out){ \
  for (size_t i = 0; i < a.size; i++) { \
      out->ptr[i] = op(a.ptr[i], b.ptr[i]); \
    } \
  }

#define EWISE_UNARY_OP(name, op) \
  void Ewise##name(const AlignedArray &a,AlignedArray *out){ \
  for (size_t i = 0; i < a.size; i++) { \
      out->ptr[i] = op(a.ptr[i]); \
    } \
  }

#define SCALAR_BINARY_OP(name, op) \
  void Scalar##name(const AlignedArray &a, scalar_t val, AlignedArray *out){ \
  for (size_t i = 0; i < a.size; i++) { \
      out->ptr[i] = op(a.ptr[i],val); \
    } \
  }

void EwiseMul(const AlignedArray &a, const AlignedArray &b, AlignedArray *out);
void ScalarMul(const AlignedArray &a, scalar_t val, AlignedArray *out);
void EwiseDiv(const AlignedArray &a, const AlignedArray &b, AlignedArray *out);
void ScalarDiv(const AlignedArray &a, scalar_t val, AlignedArray *out);
void ScalarPower(const AlignedArray &a, scalar_t val, AlignedArray *out);
void EwiseMaximum(const AlignedArray &a, const AlignedArray &b, AlignedArray *out);
void ScalarMaximum(const AlignedArray &a, scalar_t val, AlignedArray *out);
void EwiseEq(const AlignedArray &a, const AlignedArray &b, AlignedArray *out);
void ScalarEq(const AlignedArray &a, scalar_t val, AlignedArray *out);
void EwiseGe(const AlignedArray &a, const AlignedArray &b, AlignedArray *out);
void ScalarGe(const AlignedArray &a, scalar_t val, AlignedArray *out);
void EwiseLog(const AlignedArray &a, AlignedArray *out);
void EwiseExp(const AlignedArray &a, AlignedArray *out);
void EwiseTanh(const AlignedArray &a, AlignedArray *out);

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
void Matmul(const AlignedArray &a, const AlignedArray &b, AlignedArray *out, uint32_t m, uint32_t n,
            uint32_t p);

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
inline void AlignedDot(const float *__restrict__ a,
                       const float *__restrict__ b,
                       float *__restrict__ out);

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
void MatmulTiled(const AlignedArray &a, const AlignedArray &b, AlignedArray *out, uint32_t m,
                 uint32_t n, uint32_t p);

/**
 * Reduce by taking maximum over `reduce_size` contiguous blocks.
 *
 * Args:
 *   a: compact array of size a.size = out.size * reduce_size to reduce over
 *   out: compact array to write into
 *   reduce_size: size of the dimension to reduce over
 */

void ReduceMax(const AlignedArray &a, AlignedArray *out, size_t reduce_size);

void ReduceSum(const AlignedArray &a, AlignedArray *out, size_t reduce_size);
}
#endif //NEEDLE_SRC_NDARRAY_BACKEND_CPU_H_
