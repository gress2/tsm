#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "beta_distribution.hpp"
#include "random_engine.hpp"

double get_fsj(const std::vector<double>& p, int s, int j) {
  if (s == 1) {
    if (j == 1) {
      return -std::sqrt(p[1]);
    } else if (j == 2) {
      return std::sqrt(p[0]);
    } else {
      return 0;
    }
  } else {
    if (j <= s) {
      double numer = -std::sqrt(p[j - 1] * p[s]);
      double denom = 0;
      for (int l = 0; l < s; l++) {
        denom += p[l];
      }
      return numer / std::sqrt(denom);
    } else if (j == s + 1) {
      double sum_ps = 0; 
      for (int l = 0; l < s; l++) {
        sum_ps += p[l];
      } 
      return std::sqrt(sum_ps);
    } else {
      return 0;
    }
  }
}

std::vector<double> get_fs(const std::vector<double>& p, int s) {
  int k = p.size();
  std::vector<double> fs;
  for (int i = 1; i < k + 1; i++) {
    fs.push_back(get_fsj(p, s, i));
  }
  return fs;
}

std::vector<double> normalize(std::vector<double> vec) {
  double sum_sq = 0;
  for (auto& elem : vec) {
    sum_sq += std::pow(elem, 2);
  }
  double norm = std::sqrt(sum_sq);
  for (auto& elem : vec) {
    elem /= norm;
  }
  return vec;
}

std::vector<std::vector<double>> get_orthonormal_basis(const std::vector<double>& p) {
  std::vector<std::vector<double>> basis;
  for (int i = 1; i < p.size(); i++) {
    basis.push_back(normalize(get_fs(p, i)));
  }
  return basis;
}

std::vector<double> get_varpi(int k) {
  std::vector<double> varpi;
  std::uniform_real_distribution<> dis1(0.0, M_PI);

  for (int i = 0; i < k - 2; i++) {
    varpi.push_back(dis1(random_engine::generator));
  } 

  std::uniform_real_distribution<> dis2(0.0, 2 * M_PI);
  varpi.push_back(dis2(random_engine::generator));
  return varpi;
}

double get_varphi2(double alpha = 2, double beta = 2) {
  sftrabbit::beta_distribution dist(alpha, beta);
  return dist(random_engine::generator);
}

std::vector<double> get_gamma_for_k2(
    const std::vector<double>& p, double varphi2) {
  std::vector<double> basis = get_orthonormal_basis(p)[0];
  for (auto& elem : basis) {
    elem *= std::sqrt(varphi2);
  }
  return basis;
}

std::vector<double> get_gamma(const std::vector<double>& p, 
    const std::vector<double>& varpi, double varphi2) {
  int k = p.size();
  if (k == 2) {
    return get_gamma_for_k2(p, varphi2);
  } else {
    std::vector<std::vector<double>> basis = get_orthonormal_basis(p);
    std::vector<double> gamma(k, 0);
    std::vector<double> coeffs;

    for (int i = 0; i < k - 1; i++) {
      double mul = std::sqrt(varphi2); 
      for (int j = 0; j < std::min(i, k - 3); j++) {
        mul *= std::sin(varpi[j]);
      } 
      if (i < k - 2) {
        mul *= std::cos(varpi[i]);
      } else {
        mul *= std::sin(varpi[i - 1]);
      }
      coeffs.push_back(mul);
    }

    for (int g = 0; g < basis.size(); g++) {
      for (int h = 0; h < k; h++) {
        gamma[h] += basis[g][h] * coeffs[g];
      } 
    }
    return gamma;
  }
}

std::vector<double> get_eta(const std::vector<double>& p, const std::vector<double>& xi, 
    double varphi2) {
  int k = p.size(); 
  std::vector<double> coeffs;
  for (int i = 0; i < k; i++) {
    double mul = std::sqrt(1 - varphi2);
  } 
}

