#include <iostream>

#include "finite_mixture.hpp"
#include "util.hpp"
#include "gtest/gtest.h"

TEST(get_fsj, returns_correct_values) {
  std::vector<double> p = {.2, .2, .2, .2, .2};
  std::vector<double> v1 = {-0.4472136, 0.4472136, 0., 0., 0.};
  std::vector<double> v2 = {-0.31622777, -0.31622777, 0.63245553, 0., 0.};
  std::vector<double> v3 = {-0.25819889, -0.25819889, -0.25819889, 0.77459667, 0.};
  std::vector<double> v4 = {-0.2236068, -0.2236068, -0.2236068, -0.2236068, 0.89442719};
auto fs = get_fs(p, 1);
  ASSERT_TRUE(approx_equal(fs.begin(), fs.end(), v1.begin(), v1.end()));
  fs = get_fs(p, 2);
  ASSERT_TRUE(approx_equal(fs.begin(), fs.end(), v2.begin(), v2.end()));
  fs = get_fs(p, 3);
  ASSERT_TRUE(approx_equal(fs.begin(), fs.end(), v3.begin(), v3.end()));
  fs = get_fs(p, 4);
  ASSERT_TRUE(approx_equal(fs.begin(), fs.end(), v4.begin(), v4.end()));
}

TEST(get_orthonormal_basis, returns_correct_values) {
  std::vector<double> p = {.2, .2, .2, .2, .2};
  std::vector<double> v1 = {-0.70710678, 0.70710678, 0., 0., 0.};
  std::vector<double> v2 = {-0.40824829, -0.40824829, 0.81649658, 0., 0.};
  std::vector<double> v3 = {-0.28867513, -0.28867513, -0.28867513, 0.8660254, 0.};
  std::vector<double> v4 = {-0.2236068, -0.2236068, -0.2236068, -0.2236068, 0.89442719};

  auto basis = get_orthonormal_basis(p);
  ASSERT_TRUE(
      approx_equal(basis[0].begin(), basis[0].end(), v1.begin(), v1.end()) &&
      approx_equal(basis[1].begin(), basis[1].end(), v2.begin(), v2.end()) &&
      approx_equal(basis[2].begin(), basis[2].end(), v3.begin(), v3.end()) && 
      approx_equal(basis[3].begin(), basis[3].end(), v4.begin(), v4.end()) 
  );
}

TEST(get_gamma, returns_correct_values) {
  std::vector<double> p = {.2, .2, .2, .2, .2};
  std::vector<double> varpi = {1.12674078, 1.05344725, 3.44247279};
  double varphi2 = 0.6819057940917935; 
  std::vector<double> expected = {-0.17977888, 0.32192417, 0.52276042, -0.49311077, -0.17179493};
  std::vector<double> gamma = get_gamma(p, varpi, varphi2);
  ASSERT_TRUE(approx_equal(gamma.begin(), gamma.end(), expected.begin(), expected.end()));
}

TEST(get_eta, returns_correct_values) {
  std::vector<double> p = {.2, .2, .2, .2, .2};
  std::vector<double> xi = {0.52032313, 0.44653358, 0.88552296, 0.41636214};
  double varphi2 = 0.4076792348110412;
  std::vector<double> eta = get_eta(p, xi, varphi2);
  std::vector<double> expected = {0.66777067, 0.34510983, 0.10457411, 0.11700213, 0.05174025};
  ASSERT_TRUE(approx_equal(eta.begin(), eta.end(), expected.begin(), expected.end()));
}

TEST(sample_finite_mixture, returns_valid_distribution) {
  std::vector<double> p = {.2, .2, .2, .2, .2};
  double mean_ = 500;
  double sd_ = 100;

  auto mixture = sample_finite_mixture(p, mean_, sd_);
  std::vector<double> child_means = mixture.first;
  std::vector<double> child_sds = mixture.second;
  
  ASSERT_EQ(mean(child_means), mean_);

  double mixture_variance = 0;
  for (int i = 0; i < p.size(); i++) {
    mixture_variance += p[i] * (std::pow(child_means[i], 2) + std::pow(child_sds[i], 2));
  }
  mixture_variance -= std::pow(mean_, 2);
  ASSERT_EQ(mixture_variance, std::pow(sd_, 2));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
