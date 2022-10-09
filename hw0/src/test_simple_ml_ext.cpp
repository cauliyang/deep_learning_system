#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "simple_ml_ext.hpp"

DOCTEST_MAKE_STD_HEADERS_CLEAN_FROM_WARNINGS_ON_WALL_BEGIN
#include <iostream>
#include <initializer_list>
#include <array>
DOCTEST_MAKE_STD_HEADERS_CLEAN_FROM_WARNINGS_ON_WALL_END

template<typename T>
Matrix<T> creat_matrix(size_t row, size_t column, std::initializer_list<T> list) {
  Matrix<T> matrix(row, column);
  assert(list.size() == row * column);

  for (size_t i = 0; i < matrix.size; ++i) {
    matrix.set(i, *(list.begin() + i));
  }
  return matrix;
}

TEST_CASE("Test matrix constructor") {
  Matrix<float> matrix(2, 3, 1.0f);

  CHECK(matrix.rows == 2);
  CHECK(matrix.columns == 3);
  CHECK(matrix.size == 6);
  CHECK(matrix.at(0) == 1.0f);

  Matrix<float> matrix2(2, 3, matrix.begin, false);
  CHECK(matrix2.rows == 2);
  CHECK(matrix2.columns == 3);
  CHECK(matrix2.size == 6);
  CHECK(matrix2.is_destroy == false);

  matrix2.set(3, 2.0f);
  CHECK(matrix2.at(3) == 2.0f);
  CHECK(matrix.at(3) == 2.0f);

  Matrix<double> matrix3(10000, 10000, 1.0);
  CHECK(matrix3.rows == 10000);
  CHECK(matrix3.columns == 10000);
}

TEST_CASE("Test at and set") {
  Matrix<float> matrix(2, 3, 1.0f);
  matrix.set(0, 2.0f);
  matrix.set(1, 3.0f);
  matrix.set(2, 4.0f);

  matrix.set(1, 0, 5.0f);
  matrix.set(1, 1, 6.0f);
  matrix.set(1, 2, 7.0f);

  CHECK(matrix.at(0) == 2.0f);
  CHECK(matrix.at(1) == 3.0f);
  CHECK(matrix.at(2) == 4.0f);
  CHECK(matrix.at(3) == 5.0f);
  CHECK(matrix.at(4) == 6.0f);
  CHECK(matrix.at(5) == 7.0f);

  CHECK(matrix.at(0, 0) == 2.0f);
  CHECK(matrix.at(0, 1) == 3.0f);
  CHECK(matrix.at(0, 2) == 4.0f);
  CHECK(matrix.at(1, 0) == 5.0f);
  CHECK(matrix.at(1, 1) == 6.0f);
  CHECK(matrix.at(1, 2) == 7.0f);

}

TEST_CASE("Test matrix copy and move constructor") {
  Matrix<float> matrix(2, 3, 1.0f);
  Matrix<float> matrix2(matrix);
  CHECK(matrix2.rows == 2);
  CHECK(matrix2.columns == 3);
  CHECK(matrix2.size == 6);

  matrix2.set(0, 2.0f);
  CHECK(matrix.at(0) == 1.0f);

  Matrix<float> matrix3(std::move(matrix));
  CHECK(matrix3.rows == 2);
  CHECK(matrix3.columns == 3);
  CHECK(matrix3.size == 6);
  CHECK(matrix3.at(0) == 1.0f);
  CHECK(matrix.begin == nullptr);
}

TEST_CASE("Test matrix transpose") {
  Matrix<float> matrix = creat_matrix(2, 3, {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f});

  Matrix<float> matrix2 = matrix.transpose();
  CHECK(matrix2.rows == 3);
  CHECK(matrix2.columns == 2);

  CHECK(matrix.at(0, 1) == matrix2.at(1, 0));
  CHECK(matrix.at(1, 2) == matrix2.at(2, 1));
}

TEST_CASE("Test for each") {
  Matrix<double> matrix(10, 10, 42.0);
  matrix.for_each([](double value) { return value * 2; });
  matrix.for_each([](double value) {
    INFO("value is ", value);
    return value;
  });

}

TEST_CASE("Test apply_ith_row") {
  Matrix<double> matrix(10, 10, 42.0);
  matrix.apply_ith_row(0, [](double value) { return value * 2; });
  CHECK(matrix.at(0, 0) == 84.0);
  CHECK(matrix.at(0, 1) == 84.0);
  CHECK(matrix.at(0, 9) == 84.0);
}

TEST_CASE("Test split row") {
  // Create shallow copy of matrix
  Matrix<double> matrix(5, 10, 42.0);

  Matrix<double> matrix2 = matrix.shallow_copy_by_row(2, 4);
  CHECK(matrix2.rows == 2);
  CHECK(matrix2.columns == 10);
  CHECK(matrix2.is_destroy == false);
  CHECK(matrix2.at(0, 0) == 42.0);

  matrix.set(3, 5, 0.0);

  CHECK(matrix2.at(1, 5) == 0.0);

  matrix2.set(0, 0, -100.9);

  CHECK(matrix.at(2, 0) == -100.9);
}

TEST_CASE("Test same shape") {

  Matrix<double> matrix(5, 10, 42.0);
  Matrix<double> matrix2(5, 10, 42.0);
  Matrix<double> matrix3(5, 11, 42.0);
  Matrix<double> matrix4(6, 10, 42.0);
  Matrix<double> matrix5(6, 11, 42.0);

  CHECK(matrix.same_shape(matrix2));
  CHECK(!matrix.same_shape(matrix3));
  CHECK(!matrix.same_shape(matrix4));
  CHECK(!matrix.same_shape(matrix5));

}

TEST_CASE("Test matrix operator ") {
  Matrix<double> matrix = creat_matrix(2, 3, {2.0, 3.0, 4.0, 5.0, 6.0, 7.0});
  Matrix<double> matrix2 = creat_matrix(2, 3, {2.0, 3.0, 4.0, 5.0, 6.0, 7.0});
  CHECK(matrix == matrix2);

  SUBCASE("-matrix") {
    Matrix<double> matrix3 = -matrix;
    CHECK(matrix3.at(0, 0) == -2.0);
    CHECK(matrix3.at(0, 1) == -3.0);
    CHECK(matrix3.at(0, 2) == -4.0);
    CHECK(matrix3.at(1, 0) == -5.0);
    CHECK(matrix3.at(1, 1) == -6.0);
    CHECK(matrix3.at(1, 2) == -7.0);
  }

  SUBCASE("matrix + matrix") {
    Matrix<double> matrix3 = matrix + matrix2;
    CHECK(matrix3.at(0, 0) == 4.0);
    CHECK(matrix3.at(0, 1) == 6.0);
    CHECK(matrix3.at(1, 1) == 12.0);

    matrix3 += matrix2;
    CHECK(matrix3.at(0, 0) == 6.0);
    CHECK(matrix3.at(0, 1) == 9.0);
  }

  // matrix - matrix
  SUBCASE("matrix - matrix") {

    Matrix<double> matrix4 = matrix - matrix2;
    CHECK(matrix4.at(0, 0) == 0.0);
    CHECK(matrix4.at(0, 1) == 0.0);
    CHECK(matrix4.at(1, 1) == 0.0);

    matrix4 -= matrix2;
    CHECK(matrix4.at(0, 0) == -2.0);
    CHECK(matrix4.at(0, 1) == -3.0);

  }

  // matrix + scalar
  SUBCASE("matrix + scalar") {
    Matrix<double> matrix5 = matrix + 2.0;
    CHECK(matrix5.at(0, 0) == 4.0);
    CHECK(matrix5.at(0, 1) == 5.0);
    CHECK(matrix5.at(1, 1) == 8.0);
  }

  // matrix - scalar
  SUBCASE("matrix - scalar") {
    Matrix<double> matrix6 = matrix - 2.0;
    CHECK(matrix6.at(0, 0) == 0.0);
    CHECK(matrix6.at(0, 1) == 1.0);
    CHECK(matrix6.at(1, 1) == 4.0);
  }

  // matrix * scalar
  SUBCASE("matrix * scalar") {
    Matrix<double> matrix7 = matrix * 2.0;
    Matrix<double> matrix8 = 2 * matrix;
    CHECK(matrix7.at(0, 0) == 4.0);
    CHECK(matrix7.at(0, 1) == 6.0);
    CHECK(matrix7.at(1, 1) == 12.0);
    CHECK(matrix8.at(0, 0) == 4.0);

  }

  // matrix @ matrix
  SUBCASE("matrix @ matrix") {
    Matrix<double> matrix9 = matrix.transpose();
    Matrix<double> matrix10 = matrix9 * matrix;

    CHECK(matrix10.rows == 3);
    CHECK(matrix10.columns == 3);

    CHECK(matrix10.at(0, 0) == 29.0);
    CHECK(matrix10.at(0, 1) == 36.0);
    CHECK(matrix10.at(2, 2) == 65.0);
  }

  SUBCASE("element-wise matrix * matrix") {

    Matrix<double> matrix11 = elementwise_mul(matrix, matrix);
    CHECK(matrix11.at(0, 0) == 4.0);
    CHECK(matrix11.at(0, 1) == 9.0);
    CHECK(matrix11.at(1, 1) == 36.0);
  }

  SUBCASE("element-wise matrix / matrix") {

    Matrix<double> matrix12 = matrix / matrix;
    CHECK(matrix12.at(0, 0) == 1.0);
    CHECK(matrix12.at(0, 1) == 1.0);
    CHECK(matrix12.at(1, 1) == 1.0);


    // test broadcast

    Matrix<double> matrix13 = creat_matrix(2, 1, {2.0, 3.0});
    Matrix<double> matrix14 = matrix / matrix13;

    CHECK(matrix14.at(0, 0) == 1.0);
    CHECK(matrix14.at(0, 2) == 2.0);
    CHECK(matrix14.at(1, 1) == 2.0);

  }

  SUBCASE("matrix / scalar and scalar / matrix") {
    Matrix<double> matrix15 = matrix / 2.0;
    CHECK(matrix15.at(0, 0) == 1.0);
    CHECK(matrix15.at(0, 1) == 1.5);
    CHECK(matrix15.at(1, 1) == 3);
    CHECK(matrix15.at(1, 2) == 3.5);

    Matrix<double> matrix16 = 2.0 / matrix;
    CHECK(matrix16.at(0, 0) == 1.0);
    CHECK(matrix16.at(0, 1) == 2.0 / 3.0);
    CHECK(matrix16.at(1, 1) == 2.0 / 6.0);
  }
}

TEST_CASE("Test operator <<") {
  Matrix<double> matrix = creat_matrix(2, 2, {2.0, 3.0, 4.0, 5.0});
  std::cout << matrix << '\n';
}

TEST_CASE("Test eye") {
  Matrix<double> matrix = eye<double>(3);
  CHECK(matrix.at(0, 0) == 1.0);
  CHECK(matrix.at(1, 1) == 1.0);
  CHECK(matrix.at(2, 2) == 1.0);
  std::cout << matrix << '\n';
}

TEST_CASE("Test relu") {
  Matrix<double> matrix = creat_matrix(2, 2, {-2.0, 3.0, 4.0, -5.0});
  Matrix<double> matrix2 = relu(matrix);
  CHECK(matrix2.at(0, 0) == 0.0);
  CHECK(matrix2.at(0, 1) == 3.0);
  CHECK(matrix2.at(1, 1) == 0.0);
  std::cout << matrix2 << '\n';
}

TEST_CASE("Test sum by axis") {
  Matrix<double> matrix = creat_matrix(3, 2, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  Matrix<double> matrix2 = sum(matrix, 0);
  Matrix<double> matrix3 = sum(matrix, 1);

  CHECK(matrix2.rows == 1);
  CHECK(matrix2.columns == 2);

  CHECK(matrix3.rows == 3);
  CHECK(matrix3.columns == 1);

  CHECK(matrix2.at(0, 0) == 9.0);
  CHECK(matrix2.at(0, 1) == 12.0);
  CHECK(matrix3.at(0, 0) == 3.0);
  CHECK(matrix3.at(1, 0) == 7.0);
  CHECK(matrix3.at(2, 0) == 11.0);
}

TEST_CASE("Test exp normalize") {
  Matrix<double> matrix = creat_matrix(2, 2, {1.0, 0.0, 0.0, 1.0});
  Matrix<double> matrix2 = exp_normalize(matrix, matrix);

  CHECK(matrix2.at(0, 0) == std::exp(1) / (std::exp(1) + 1));
}

TEST_CASE("Test one hots") {
  constexpr std::array<unsigned char, 4> labels = {0, 1, 5, 2};

  Matrix<double> matrix = one_hots<double>(labels.data(), 4, 10);
  CHECK_EQ(matrix.rows, 4);
  CHECK_EQ(matrix.columns, 10);
  CHECK_EQ(matrix.at(0, 0), 1.0);
  CHECK_EQ(matrix.at(1, 1), 1.0);
  CHECK_EQ(matrix.at(2, 5), 1.0);
  CHECK_EQ(matrix.at(3, 2), 1.0);
  std::cout << matrix << '\n';

}
