//
// Created by Yangyang Li <yangyang.li@northwestern.edu> on 9/17/22
//

#ifndef DEEP_LEARNING_SYSTEM_HW0_SRC_SIMPLE_ML_EXT_H_
#define DEEP_LEARNING_SYSTEM_HW0_SRC_SIMPLE_ML_EXT_H_
#include <cassert>

template<typename T>
struct Matrix {

  size_t rows{}; // number of rows 0-indexed
  size_t columns{}; // number of columns 0-indexed
  size_t size{}; // may overflow
  T *begin{nullptr};
  bool is_destroy{true}; // control whether to delete the memory

  //----------------Define the constructor----------------//
  Matrix(size_t rows_, size_t columns_, T init_value = 0)
      : rows{rows_}, columns{columns_}, size{rows * columns},
        begin{allocate(size)} {
    init(init_value);
  }

  // if is_destroy is false do not have ownership
  Matrix(size_t rows_, size_t columns_, T *begin_, bool is_destroy_)
      : rows{rows_}, columns{columns_}, size{rows * columns}, begin{begin_}, is_destroy{is_destroy_} {
  }

  //----------------Define the destructor----------------//
  ~Matrix() {
    if (begin != nullptr && is_destroy)
      free(begin);
  }

  //----------------Define the copy constructor----------------//
  Matrix(Matrix const &other)
      : rows{other.rows}, columns{other.columns}, size{other.size}, begin{allocate(other.size)} {
    init(other);
  }

  Matrix &operator=(Matrix const &other) {
    if (this != &other) {
      if (begin != nullptr && is_destroy)
        free(begin);
      rows = other.rows;
      columns = other.columns;
      size = other.size;
      begin = allocate(size);
      init(other);
    }
  }

  //----------------Define the move constructor----------------//
  Matrix(Matrix &&other)
  noexcept
      : rows{other.rows}, columns{other.columns}, size{other.size}, begin{
      std::move(other.begin)} {
    other.begin = nullptr;
  }

  Matrix &operator=(Matrix &&other)
  noexcept {
    rows = other.rows;
    columns = other.columns;
    size = other.size;
    begin = std::move(other.begin);
    other.begin = nullptr;
  }

  //----------------Define transpose----------------//
  Matrix transpose() const {
    auto res = Matrix{columns, rows};

    for (size_t index_row = 0; index_row < columns; ++index_row) {
      for (size_t index_column = 0; index_column < rows; ++index_column) {
        res.set(index_row, index_column, at(index_column, index_row));
      }
    }

    return res;
  }

  //----------------Define at to get the value----------------//
  T at(size_t i) const {
    assert(i < size);
    return *(begin + i);
  }

  T at(size_t i, size_t j) const {
    assert(i < rows && j < columns);
    return at(i * columns + j);
  }

  void set(size_t i, T value) {
    assert(i < size);
    *(begin + i) = value;
  }

  void set(size_t i, size_t j, T value) {
    assert(i < rows && j < columns);
    set(i * columns + j, value);
  }

  template<typename Func>
  void for_each(Func func) {

    for (size_t index = 0; index < size; ++index) {
      set(index, func(at(index)));
    }
  }

  template<typename Func>
  void apply_ith_row(size_t row, Func func) {
    assert(row < rows);

    T *new_begin = begin + row * columns;

    for (size_t index = 0; index < columns; ++index) {
      // maybe problematic
      *(new_begin + index) = func(*(new_begin + index));
    }
  }

  // [)
  // Return shallow copy
  Matrix shallow_copy_by_row(size_t start, size_t end) const {
    assert(end > start);

    // permit end > rows
    if (end > rows)
      end = rows;

    size_t num_of_rows = end - start;

    T *new_begin = begin + start * columns;
    // return Matrix doesn't have ownership of underlying memory
    // use it carefully
    return {num_of_rows, columns, new_begin, false};
  }

  bool same_shape(Matrix const &other) const {
    return rows == other.rows && columns == other.columns;
  }


  //----------------Define the operator +----------------//

  friend bool operator==(Matrix const &lhs, Matrix const &rhs) {
    if (!lhs.same_shape(rhs))
      return false;

    for (size_t index = 0; index < lhs.size; ++index) {
      if (lhs.at(index) != rhs.at(index))
        return false;
    }

    return true;
  }

  Matrix operator-() const {
    auto res = Matrix{*this};
    res.for_each([](T x) { return -x; });
    return res;
  }

  // element-wise subtraction
  Matrix &operator-=(Matrix const &other) {
    assert(same_shape(other));

    for (size_t index = 0; index < rows * columns; ++index) {

      set(index, at(index) - other.at(index));
    }

    return *this;
  }

  // element-wise addition
  Matrix &operator+=(Matrix const &other) {
    assert(same_shape(other));

    for (size_t index = 0; index < rows * columns; ++index) {
      set(index, at(index) + other.at(index));
    }

    return *this;
  }

  // element-wise addition
  friend Matrix operator+(Matrix const &lhs, Matrix const &rhs) {
    assert(lhs.same_shape(rhs));
    auto res = Matrix{lhs};
    res += rhs;
    return res;
  }

  // element-wise subtraction
  friend Matrix operator-(Matrix const &lhs, Matrix const &rhs) {
    assert(lhs.same_shape(rhs));

    auto res = Matrix{lhs};
    res -= rhs;
    return res;
  }

  /**
 * @brief  matrix and scalar deletion
 * @return lhs - scalar
 */
  friend Matrix operator-(Matrix const &lhs, T value) {
    auto res = Matrix{lhs};
    res.for_each([=](T item) { return item - value; });
    return res;
  }

  /**
* @brief  matrix and scalar deletion
* @return lhs + scalar
*/
  friend Matrix operator+(Matrix const &lhs, T value) {
    auto res = Matrix{lhs};
    res.for_each([=](T item) { return item + value; });
    return res;
  }

  /**
   * @brief vectors from two matrix dot product
   * @param lhs left hand side matrix
   * @param rhs right hand side matrix
   * @param lhs_row row index of left hand side matrix
   * @param rhs_column column index of right hand side matrix
   * @return dot product of two vectors on lhs_row of lhs and rhs_column of rhs
   */
  friend T vector_product(Matrix const &lhs, Matrix const &rhs, size_t lhs_row,
                          size_t rhs_column) {
    assert(lhs_row < lhs.rows && rhs_column < rhs.columns);
    T res = 0;
    for (size_t index = 0; index < lhs.columns; ++index) {
      res += lhs.at(lhs_row, index) * rhs.at(index, rhs_column);
    }
    return res;
  }

  /**
   * @brief matrix multiplication
   * @return  lhs @ rhs
   */
  friend Matrix operator*(Matrix const &lhs, Matrix const &rhs) {
    Matrix res = Matrix{lhs.rows, rhs.columns};
    for (size_t row = 0; row < lhs.rows; ++row) {
      for (size_t column = 0; column < rhs.columns; ++column) {
        res.set(row, column, vector_product(lhs, rhs, row, column));
      }
    }
    return res;
  }

  /**
   * @brief matrix scalar multiplication by broadcasting
   * @return  a new matrix
   */
  friend Matrix operator*(Matrix const &lhs, T scale) {
    Matrix res = Matrix{lhs};
    res.for_each([scale](T item) { return item * scale; });
    return res;
  }

  friend Matrix operator*(T scale, Matrix const &lhs) { return lhs * scale; }

  // element-wise multiplication no-broadcasting
  // TODO: implement broadcasting
  friend Matrix elementwise_mul(Matrix const &lhs, Matrix const &rhs) {
    assert(lhs.same_shape(rhs));
    Matrix res = Matrix{lhs.rows, lhs.columns};

    for (size_t index = 0; index < res.rows * res.columns; ++index) {
      res.set(index, lhs.at(index) * rhs.at(index));
    }

    return res;
  }

  /**
   * @brief element-wise matrix division by broadcasting
   * @return a new matrix
   */
  friend Matrix operator/(Matrix const &lhs, Matrix const &rhs) {
    // not check condition broadcast

    if (lhs.same_shape(rhs)) {

      Matrix res = Matrix{lhs.rows, lhs.columns};
      for (size_t index = 0; index < lhs.rows * lhs.columns; ++index) {
        // NOTE: maybe problematic due to does not check zero division
        res.set(index, lhs.at(index) / rhs.at(index));
      }
      return res;

    } else if (lhs.rows == rhs.rows && rhs.columns == 1) {
      // broadcast rhs to lhs
      Matrix res{lhs};

      for (size_t index = 0; index < res.rows; ++index) {
        res.apply_ith_row(index,
                          [&](T item) { return item / rhs.at(index, 0); });
      }
      return res;
    }

    // TODO add rhs.row == 1
    assert(false);
  }

  // matrix scalar division
  friend Matrix operator/(Matrix const &lhs, T scale) {
    Matrix res = Matrix{lhs};
    res.for_each([scale](T item) { return item / scale; });
    return res;
  }

  // matrix scalar division
  friend Matrix operator/(T scale, Matrix const &lhs) {
    Matrix res = Matrix{lhs};
    res.for_each([scale](T item) { return scale / item; });
    return res;
  }

  //-------------Define debug function----------------//
  friend std::ostream &operator<<(std::ostream &os, Matrix const &matrix) {
    std::cout << "rows: " << matrix.rows << " columns: " << matrix.columns
              << " size: " << matrix.size << "\n";
    for (size_t index_row = 0; index_row < matrix.rows; ++index_row) {
      for (size_t index_column = 0; index_column < matrix.columns; ++index_column) {
        os << matrix.at(index_row, index_column) << " ";
      }
      os << "\n";
    }
    return os;
  }

 private:
  T *allocate(size_t size_) const
  noexcept {
    return (T *) malloc(size_ * sizeof(T));
  }

  void init(T init_value)
  noexcept {
    for (size_t index = 0; index < rows * columns; ++index) {
      set(index, init_value);
    }
  }

  void init(Matrix const &other)
  noexcept {
    assert(same_shape(other));
    for (size_t index = 0; index < size; ++index) {
      set(index, other.at(index));
    }
  }
};

/**
 * @brief create identity matrix
 */
template<typename T>
Matrix<T> eye(size_t row) {
  auto res = Matrix<T>{row, row};
  for (size_t index = 0; index < row; ++index) {
    res.set(index, index, 1);
  }
  return res;
}

/**
 * @brief relu activation function
 */
template<typename T>
Matrix<T> relu(Matrix<T> const &other) {

  auto res = Matrix<T>{other};

  res.for_each([](T item) {
    return item > 0 ? item : 0;
  });
  return res;
}

/**
 * @brief sum of all elements in matrix by axis
 * @param axis 0 for row, 1 for column
 */
template<typename T>
Matrix<T> sum(Matrix<T> const &matrix, size_t axis) {

  if (axis == 1) {
    auto tmp = Matrix<T>{matrix.columns, 1, 1};
    return matrix * tmp;
  } else if (axis == 0) {
    auto tmp = Matrix<T>{1, matrix.rows, 1};
    return tmp * matrix;
  }
  // cannot reach here
  assert(false);
}

//--------Define Helper Functions----------------//
template<typename T>
Matrix<T> exp_normalize(Matrix<T> const &lhs, Matrix<T> const &rhs) {
  auto res = lhs * rhs;
  res.for_each([](T item) { return std::exp(item); });
  auto row_sum = sum(res, 1);
  return res / row_sum;
}

template<typename T>
Matrix<T> one_hots(const unsigned char *begin, size_t size, size_t k) {

  auto res = Matrix<T>{size, k};

  for (size_t index_row = 0; index_row < size; ++index_row) {
    res.set(index_row, *(begin + index_row), 1);
  }
  return res;
}

#endif //DEEP_LEARNING_SYSTEM_HW0_SRC_SIMPLE_ML_EXT_H_
