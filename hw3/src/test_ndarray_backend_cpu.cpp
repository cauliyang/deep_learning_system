#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "ndarray_backend_cpu_impl.hpp"

DOCTEST_MAKE_STD_HEADERS_CLEAN_FROM_WARNINGS_ON_WALL_BEGIN
#include <iostream>
#include <initializer_list>
#include <array>
DOCTEST_MAKE_STD_HEADERS_CLEAN_FROM_WARNINGS_ON_WALL_END

TEST_CASE("test_fill") {
  needle::cpu::AlignedArray a(10);
  needle::cpu::Fill(&a, 1.0);
  for (int i = 0; i < a.size; i++) {
    CHECK(1.0 == a.ptr[i]);
  }
}

TEST_CASE("test_get_indices_1d") {
  using namespace needle::cpu;
  std::vector<uint32_t> shape{2, 3};
  std::vector<uint32_t> strides{3, 1};
  auto res = get_indices(shape, strides);
  std::vector<uint32_t> expected{0, 1, 2, 3, 4, 5};
  CHECK(res == expected);

}

TEST_CASE("test_get_indices_3d") {
  using namespace needle::cpu;
  std::vector<uint32_t> shape{2, 3, 2};
  std::vector<uint32_t> strides{6, 2, 1};
  auto res = get_indices(shape, strides);
  std::vector<uint32_t> expected{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  CHECK(res == expected);
}

TEST_CASE("temp") {
  using namespace needle::cpu;
  std::vector<uint32_t> shape{2, 2};
  std::vector<uint32_t> strides{4, 1};
  auto res = get_indices(shape, strides);
  for (auto i : res) {
    std::cout << i << std::endl;
  }
}
