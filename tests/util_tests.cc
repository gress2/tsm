#include "util.hpp"
#include "gtest/gtest.h"

TEST(sample_gaussian_test, returns_correct_vector_size) {
  std::vector<double> samples = sample_gaussian(0, 10, 5);
  ASSERT_EQ(samples.size(), 5);
}

TEST(sum_test, returns_correct_value_lval) {
  std::vector<int> nums = {1, 2, 3, 4};
  ASSERT_EQ(sum(nums), 10);
}

TEST(sum_test, returns_correct_value_rval) {
  ASSERT_EQ(sum(std::vector<int>{1, 2, 3, 4}), 10);
}

TEST(mean_test, returns_correct_value_lval) {
  std::vector<int> nums = {1, 2, 3, 4};
  ASSERT_EQ(mean(nums), 2.5);
}

TEST(mean_test, returns_correct_value_rval) {
  ASSERT_EQ(mean(std::vector<int>{1, 2, 3, 4}), 2.5);
}

TEST(multiply_test, returns_correct_value_lval) {
  std::vector<int> nums = {1, 2, 3, 4};
  int factor = 2;
  ASSERT_EQ(
    multiply(nums, factor),
    std::vector<int>({2, 4, 6, 8})
  );
  ASSERT_EQ(
    nums,
    std::vector<int>({1, 2, 3, 4})
  );
}

TEST(multiply_test, returns_correct_value_rval) {
  ASSERT_EQ(
    multiply(std::vector<int>{1, 2, 3, 4}, 2),
    std::vector<int>({2, 4, 6, 8})
  );
}

TEST(square_test, returns_correct_value_lval) {
  std::vector<int> nums = {1, 2, 3, 4};
  ASSERT_EQ(
    square(nums),
    std::vector<int>({1, 4, 9, 16})
  );
  ASSERT_EQ(
    nums,
    std::vector<int>({1, 2, 3, 4})
  );
}

TEST(square_test, returns_correct_value_rval) {
  ASSERT_EQ(
    square(std::vector<int>{1, 2, 3, 4}),
    std::vector<int>({1, 4, 9, 16})
  );
}

TEST(get_from_toml_test, returns_the_correct_value) {
  auto config = cpptoml::parse_file("../tests/cfg/util.toml");
  ASSERT_EQ(get_from_toml<int>(config, "prop"), 3);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
