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

std::vector<double> get_gamma(const std::vector<double>& p, double varphi2) {
  int k = p.size();
  if (k == 2) {
    return get_gamma_for_k2(p, varphi2);
  } else {
    std::vector<std::vector<double>> basis = get_orthonormal_basis(p);

    std::vector<double> v;
    for (int i = 0; i < k - 1; i++) {
      v.push_back(sample_gaussian(0, 1));
    }

    for (int j = 0; j < v.size(); j++) {
      for (auto& elem : basis[j]) {
        elem *= v[j]; 
      }
    } 

    std::vector<double> x(k, 0);
    for (int p = 0; p < basis.size(); p++) {
      for (int q = 0; q < basis[p].size(); q++) {
        x[q] += basis[p][q];
      }
    }

    return sample_hypersphere(k, std::sqrt(varphi2), x); 
  }
}
std::vector<double> get_eta(const std::vector<double>& p, double varphi2) {
  int k = p.size(); 

  std::vector<double> eta = sample_hypersphere(k, std::sqrt(1 - varphi2));

  for (auto& elem : eta) {
    elem = std::abs(elem);
  }

  return eta;
}

double reverse_to_varphi2(double sd, const std::vector<double>& child_sds) {
  double sum_sq = 0;

  for (auto& elem : child_sds) {
    sum_sq += std::pow(elem, 2);
  }

  if (sum_sq < 1e-5) {
    return 1;
  }

  double varphi2 = 1 - (sum_sq / (static_cast<double>(child_sds.size()) * std::pow(sd, 2)));
  return varphi2;
}

std::pair<std::vector<double>, std::vector<double>> sample_finite_mixture(const std::vector<double>& p, 
    double mean, double sd, double beta_a = 2, double beta_b = 2) {
  int k = p.size();
  if (k == 1) {
    return std::make_pair(std::vector<double>{mean}, std::vector<double>{sd});
  }

  double varphi2 = get_varphi2(beta_a, beta_b);
  std::vector<double> gamma = get_gamma(p, varphi2);
  std::vector<double> eta = get_eta(p, varphi2); 

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

