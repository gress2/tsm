#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "beta_distribution.hpp"
#include "random_engine.hpp"
#include "util.hpp"

// based on https://arxiv.org/pdf/1601.01178.pdf

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

std::vector<double> get_xi(int k) {
  std::vector<double> xi;
  std::uniform_real_distribution<> dis(0.0, M_PI / 2);
  for (int i = 0; i < k - 1; i++) {
    xi.push_back(dis(random_engine::generator));
  } 
  return xi;
}

std::vector<double> get_eta(const std::vector<double>& p, const std::vector<double>& xi, 
    double varphi2) {
  int k = p.size(); 
  std::vector<double> eta;
  for (int i = 0; i < k; i++) {
    double mul = std::sqrt(1 - varphi2);
    if (i == 0) {
      mul *= std::cos(xi[i]);
    } else if (i > 0 && i < k - 1) {
      for (int j = 0; j < i; j++) {
        mul *= std::sin(xi[j]);
      }
      mul *= std::cos(xi[i]);
    } else if (i == k - 1) {
      for (int j = 0; j < i; j++) {
        mul *= std::sin(xi[j]);
      }
    }
    eta.push_back(mul);
  } 
  return eta;
}

double reverse_to_varphi2(double sd, const std::vector<double>& child_sds) {
  double sum_sq = 0;

  for (auto& elem : child_sds) {
    sum_sq += std::pow(elem, 2);
  }

  double varphi2 = 1 - sum_sq / (child_sds.size() * std::pow(sd, 2));
  return varphi2;
}

std::pair<std::vector<double>, std::vector<double>> sample_finite_mixture(const std::vector<double>& p, 
    double mean, double sd, double beta_a = 2, double beta_b = 2) {
  int k = p.size();
  if (k == 1) {
    return std::make_pair(std::vector<double>{mean}, std::vector<double>{sd});
  }
  double varphi2 = get_varphi2(beta_a, beta_b);
  std::vector<double> varpi = get_varpi(k);
  std::vector<double> gamma = get_gamma(p, varpi, varphi2);
  std::vector<double> xi = get_xi(k);
  std::vector<double> eta = get_eta(p, xi, varphi2); 

  std::vector<double> alpha;
  std::vector<double> tau;
  for (int i = 0; i < k; i++) {
    alpha.push_back(gamma[i] / std::sqrt(p[i]));
    tau.push_back(eta[i] / std::sqrt(p[i]));
  }

  std::vector<double> mu;
  std::vector<double> sigma;
  for (int i = 0; i < k; i++) {
    mu.push_back(alpha[i] * sd + mean);
    sigma.push_back(tau[i] * sd);
  }
  return std::make_pair(std::move(mu), std::move(sigma));
}

