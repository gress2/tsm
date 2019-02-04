#pragma once
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <torch/script.h>

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
  for (std::vector<double>::size_type i = 1; i < p.size(); i++) {
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

    using nested_vector = std::vector<std::vector<double>>;

    nested_vector basis = get_orthonormal_basis(p);

    std::vector<double> v;
    for (int i = 0; i < k - 1; i++) {
      v.push_back(sample_gaussian(0, 1));
    }

    for (std::vector<double>::size_type j = 0; j < v.size(); j++) {
      for (auto& elem : basis[j]) {
        elem *= v[j];
      }
    }

    std::vector<double> x(k, 0);
    for (nested_vector::size_type p = 0; p < basis.size(); p++) {
      for (nested_vector::size_type q = 0; q < basis[p].size(); q++) {
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

std::vector<double> get_eta(
  const double mean, const double sd, int d, const std::vector<double>& p,
  double varphi2, std::shared_ptr<torch::jit::script::Module> sd_module = nullptr) {
  int k = p.size();

  auto input_tensor = torch::ones({2, 5}, torch::kFloat64);
  input_tensor[0][0] = mean;
  input_tensor[0][1] = sd;
  input_tensor[0][2] = static_cast<double>(d);
  input_tensor[0][3] = static_cast<double>(k);
  input_tensor[0][4] = varphi2;

  std::vector<torch::jit::IValue> input({input_tensor});
  at::Tensor output = sd_module->forward(input).toTensor();

  double epsilon = *(output.data<double>());
  std::vector<double> x = sample_uniform_dirichlet(k, epsilon);
  double sum_sq = 0;
  for (auto& elem : x) {
    sum_sq += elem * elem;
  }
  double l2_norm = std::sqrt(sum_sq);

  for (auto& elem : x) {
    elem *= std::sqrt(1 - varphi2) / l2_norm;
  }

  return x;
}

std::pair<std::vector<double>, std::vector<double>> sample_finite_mixture(const std::vector<double>& p,
    double mean, double sd, int d, double varphi2,
    std::shared_ptr<torch::jit::script::Module> sd_module = nullptr) {
  int k = p.size();
  if (k == 1) {
    return std::make_pair(std::vector<double>{mean}, std::vector<double>{sd});
  }

  std::vector<double> gamma = get_gamma(p, varphi2);

  double gamma_c1 = 0;
  for (const auto& elem : gamma) {
    gamma_c1 += std::sqrt(1/k) * elem;
  }

  double gamma_c2 = 0;
  for (const auto& elem : gamma) {
    gamma_c2 += elem * elem;
  }

  std::vector<double> eta = get_eta(mean, sd, d, p, varphi2, sd_module);

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
