#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

constexpr int TILE = 8;
constexpr int BASE_THREAD_NUM = 256;

using scalar_t = float;
constexpr size_t
    ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() {
    return (size_t)
        ptr;
  }

  scalar_t *ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t
                        size) {
  /**
 * Utility function to get cuda dimensions for 1D call
 */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  uint32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<uint32_t> &x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimensions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t *out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray *out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Utility function to convert contiguous index i to memory location from strides

//  map from an index in the compact array to one in the strided array.
__device__ size_t get_index(size_t i, CudaVec const &shape, const CudaVec &strides, size_t offset) {

  size_t output = offset;

  for (int j = 0; j < shape.size; j++) {
    size_t temp = 1;
    for (int k = j + 1; k < shape.size; k++) {
      temp *= shape.data[k];
    }

    output += (i / temp) * strides.data[j];
    i = i % temp;
  }

  return output;
}

__global__ void CompactKernel(const scalar_t *a, scalar_t *out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact operation.  This should effectively map a single entry in the
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   *
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN YOUR SOLUTION
  if (gid < size) {
    out[gid] = a[get_index(gid, shape, strides, offset)];
  }
  /// END YOUR SOLUTION
}

void Compact(const CudaArray &a, CudaArray *out, std::vector<uint32_t> shape,
             std::vector<uint32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give
   * you the code for this function, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to
   * execute the underlying function.
   *
   * Args:
   *   a: non-compact representation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}

__global__ void EwiseSetitemKernel(scalar_t *out, scalar_t const *a, size_t size, CudaVec shape,
                                   CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[get_index(gid, shape, strides, offset)] = a[gid];
  }
}

void EwiseSetitem(const CudaArray &a, CudaArray *out, std::vector<uint32_t> shape,
                  std::vector<uint32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  You will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN YOUR SOLUTION
  CudaDims dim = CudaOneDim(a.size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(out->ptr, a.ptr, a.size, VecToCuda(shape),
                                              VecToCuda(strides), offset);
  /// END YOUR SOLUTION
}

__global__ void ScalarSetitemKernel(scalar_t *out, scalar_t val, size_t size, CudaVec shape,
                                    CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[get_index(gid, shape, strides, offset)] = val;
  }
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray *out, std::vector<uint32_t> shape,
                   std::vector<uint32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN YOUR SOLUTION
  CudaDims dim = CudaOneDim(size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(out->ptr, val, size, VecToCuda(shape),
                                               VecToCuda(strides), offset);
  /// END YOUR SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

// __global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
//   size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
//   if (gid < size) out[gid] = a[gid] + b[gid];
// }

// void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
//   /**
//    * Add together two CUDA array
//    */
//   CudaDims dim = CudaOneDim(out->size);
//   EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
// }

// __global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
//   size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
//   if (gid < size) out[gid] = a[gid] + val;
// }

// void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
//   /**
//    * Add together a CUDA array and a scalar value.
//    */
//   CudaDims dim = CudaOneDim(out->size);
//   ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
// }

#define EWISE_BINARY_OP_KERNEL(name, op)                                                                  \
  __global__ void Ewise##name##Kernel(const scalar_t *a, const scalar_t *b, scalar_t *out, size_t size) { \
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;                                                   \
    if (gid < size) out[gid] = op(a[gid], b[gid]);                                                        \
  }

#define EWISE_BINARY_OP(name)                                                        \
  void Ewise##name(const CudaArray &a, const CudaArray &b, CudaArray *out) {         \
    CudaDims dim = CudaOneDim(out->size);                                            \
    Ewise##name##Kernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size); \
  }

#define SCALAR_BINARY_OP_KERNEL(name, op)                                                             \
  __global__ void Scalar##name##Kernel(const scalar_t *a, scalar_t val, scalar_t *out, size_t size) { \
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;                                               \
    if (gid < size) out[gid] = op(a[gid], val);                                                       \
  }

#define SCALAR_BINARY_OP(name)                                                      \
  void Scalar##name(const CudaArray &a, scalar_t val, CudaArray *out) {             \
    CudaDims dim = CudaOneDim(out->size);                                           \
    Scalar##name##Kernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size); \
  }

#define EWISE_UNARY_OP_KERNEL(name, op)                                                \
  __global__ void Ewise##name##Kernel(const scalar_t *a, scalar_t *out, size_t size) { \
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;                                \
    if (gid < size) out[gid] = op(a[gid]);                                             \
  }

#define EWISE_UNARY_OP(name)                                                  \
  void Ewise##name(const CudaArray &a, CudaArray *out) {                      \
    CudaDims dim = CudaOneDim(out->size);                                     \
    Ewise##name##Kernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size); \
  }
/**
 * In the code the follows, use the above template to create analogous elementise
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
EWISE_BINARY_OP_KERNEL(Add, [](scalar_t lhs, scalar_t rhs) { return lhs + rhs; })
EWISE_BINARY_OP(Add)

SCALAR_BINARY_OP_KERNEL(Add, [](scalar_t lhs, scalar_t rhs) { return lhs + rhs; })
SCALAR_BINARY_OP(Add)

EWISE_BINARY_OP_KERNEL(Mul, [](scalar_t lhs, scalar_t rhs) { return lhs * rhs; })
EWISE_BINARY_OP(Mul)

SCALAR_BINARY_OP_KERNEL(Mul, [](scalar_t lhs, scalar_t rhs) { return lhs * rhs; })
SCALAR_BINARY_OP(Mul)

EWISE_BINARY_OP_KERNEL(Div, [](scalar_t lhs, scalar_t rhs) { return lhs / rhs; })
EWISE_BINARY_OP(Div)

SCALAR_BINARY_OP_KERNEL(Div, [](scalar_t lhs, scalar_t rhs) { return lhs / rhs; })
SCALAR_BINARY_OP(Div)

SCALAR_BINARY_OP_KERNEL(Power, std::pow)
SCALAR_BINARY_OP(Power)

EWISE_BINARY_OP_KERNEL(Maximum, [](scalar_t lhs, scalar_t rhs) { return lhs > rhs ? lhs : rhs; })
EWISE_BINARY_OP(Maximum)

SCALAR_BINARY_OP_KERNEL(Maximum, [](scalar_t lhs, scalar_t rhs) { return lhs > rhs ? lhs : rhs; })
SCALAR_BINARY_OP(Maximum)

EWISE_BINARY_OP_KERNEL(Eq, [](scalar_t lhs, scalar_t rhs) { return static_cast<scalar_t>(lhs == rhs); })
EWISE_BINARY_OP(Eq)

SCALAR_BINARY_OP_KERNEL(Eq, [](scalar_t lhs, scalar_t rhs) { return static_cast<scalar_t>(lhs == rhs); })
SCALAR_BINARY_OP(Eq)

EWISE_BINARY_OP_KERNEL(Ge, [](scalar_t lhs, scalar_t rhs) { return static_cast<scalar_t>(lhs >= rhs); })
EWISE_BINARY_OP(Ge)

SCALAR_BINARY_OP_KERNEL(Ge, [](scalar_t lhs, scalar_t rhs) { return static_cast<scalar_t>(lhs >= rhs); })
SCALAR_BINARY_OP(Ge)

EWISE_UNARY_OP_KERNEL(Log, std::log)
EWISE_UNARY_OP(Log)

EWISE_UNARY_OP_KERNEL(Exp, std::exp)
EWISE_UNARY_OP(Exp)

EWISE_UNARY_OP_KERNEL(Tanh, std::tanh)
EWISE_UNARY_OP(Tanh)
/**
 * In the code the follows, use the above template to create analogous elementise
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

/// END YOUR SOLUTION

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

constexpr int THREAD_PER_BLOCK = 32;

__global__ void _MatmulKernel(const scalar_t *a, const scalar_t *b, scalar_t *c,
                              size_t M, size_t N, size_t P, size_t size) {
  size_t tx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t ty = blockDim.y * blockIdx.y + threadIdx.y;
  size_t gid = tx * P + ty;

  if (gid < size) {
    if (tx < M && ty < P) {
      scalar_t val = 0.0;
      for (size_t i = 0; i < N; ++i) {
        val += a[tx * N + i] * b[ty + i * P];
      }
      c[gid] = val;
    }
  }
}

void Matmul(const CudaArray &a, const CudaArray &b, CudaArray *out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling,
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   *
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN YOUR SOLUTION
  size_t grid_x = (M + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  size_t grid_y = (P + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  dim3 grids = dim3(grid_x, grid_y);
  dim3 blocks = dim3(THREAD_PER_BLOCK, THREAD_PER_BLOCK);
  _MatmulKernel<<<grids, blocks>>>(a.ptr, b.ptr, out->ptr, M, N, P, out->size);
  /// END YOUR SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

__global__ void ReduceMaxKernel(const scalar_t *a, scalar_t *out, size_t size, size_t reduce_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  // reduce max over all elements in a
  // use thread to reduce max over all elements in a
  scalar_t max_value = a[gid * reduce_size];

  for (size_t i = 0; i < reduce_size; ++i) {
    max_value = std::fmaxf(max_value, a[gid * reduce_size + i]);
  }

  if (gid < size) out[gid] = max_value;
}

void ReduceMax(const CudaArray &a, CudaArray *out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */
  /// BEGIN YOUR SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
  /// END YOUR SOLUTION
}

__global__ void ReduceSumKernel(const scalar_t *a, scalar_t *out, size_t size, size_t reduce_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  scalar_t sum = 0;
  for (size_t i = 0; i < reduce_size; ++i) {
    sum += a[gid * reduce_size + i];
  }
  if (gid < size) out[gid] = sum;
}

void ReduceSum(const CudaArray &a, CudaArray *out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you
   * can perform each reduction in a single CUDA thread.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN YOUR SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
  /// END YOUR SOLUTION
}

}// namespace cuda
}// namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") =
      TILE;

  py::class_<CudaArray>(m,
                        "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray &a, std::vector<size_t> shape, std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides
                       .begin(),
                   numpy_strides
                       .end(),
                   numpy_strides
                       .begin(),
                   [](
                       size_t &c) {
                     return c * ELEM_SIZE;
                   });

    // copy memory to host
    scalar_t *host_ptr = (scalar_t *) std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void *p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray *out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
