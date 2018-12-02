#pragma once

#include <cmath>
#include <iostream>
#include <random>

#include "cpptoml.hpp"
#include "logger.hpp"
#include "random_engine.hpp"

namespace generic_game 
{

struct config {
  int depth_r;
  double depth_p;
  double disp_mean_delta;
  double disp_mean_beta;
  double disp_var_delta;
  double disp_var_beta;
  double nc_alpha;
  double nc_beta;
  double root_mean;
  double root_var;
};

template <class T> 
T get_from_toml(std::shared_ptr<cpptoml::table>& tbl, std::string prop) {
  cpptoml::option<T> opt = tbl->get_as<T>(prop);
  assert(opt);
  return *opt;
}

config get_config_from_toml(std::string toml_file_path) {
  auto tbl = cpptoml::parse_file(toml_file_path);
  config cfg;
  cfg.depth_r = get_from_toml<decltype(cfg.depth_r)>(tbl, "depth_r");
  cfg.depth_p = get_from_toml<decltype(cfg.depth_p)>(tbl, "depth_p");
  cfg.disp_mean_delta = get_from_toml<decltype(cfg.disp_mean_delta)>(tbl, "disp_mean_delta");
  cfg.disp_mean_beta = get_from_toml<decltype(cfg.disp_mean_beta)>(tbl, "disp_mean_beta");
  cfg.disp_var_delta = get_from_toml<decltype(cfg.disp_var_delta)>(tbl, "disp_var_delta");
  cfg.disp_var_beta = get_from_toml<decltype(cfg.disp_var_beta)>(tbl, "disp_var_beta");
  cfg.nc_alpha = get_from_toml<decltype(cfg.nc_alpha)>(tbl, "nc_alpha");
  cfg.nc_beta = get_from_toml<decltype(cfg.nc_alpha)>(tbl, "nc_beta");
  cfg.root_mean = get_from_toml<decltype(cfg.root_mean)>(tbl, "root_mean");
  cfg.root_var = get_from_toml<decltype(cfg.root_var)>(tbl, "root_var");
  return cfg;
}

std::vector<double> sample_gaussian(double mean, double sd, int n) {
  std::normal_distribution<double> dist(mean, sd);
  std::vector<double> samples;
  for (int i = 0; i < n; i++) {
    samples.push_back(dist(random_engine::generator));
  }
  return samples;
}

template <class T>
typename std::remove_reference<T>::type::value_type sum(T&& vec) {
  return std::accumulate(vec.begin(), vec.end(), 0);
}

template <class T>
double mean(T&& vec) {
  double sum = sum(std::forward<T>(vec));
  return sum / vec.size(); 
}

template <class T, class V>
T multiply(T&& vec, V factor) {
  T res(std::forward<T>(vec));
  for (auto& elem : res) {
    elem *= factor;
  }
  return res;
}

template <class T>
T square(T&& vec) {
  T res(std::forward<T>(vec));
  for (auto& elem : res) {
    elem *= elem;
  }
  return res;
}

class game {
  public:
  private:
    /* Members */
    const config cfg_;
    const double mean_;
    const double var_;
    const int success_count_;
    const int num_moves_;
    const bool is_game_over_;
    const int num_children_;
    const std::vector<double> child_means_;
    const std::vector<double> child_vars_;
   
    /* Methods */
    int draw_num_children() const {
      double lambda = std::exp(cfg_.nc_alpha + num_moves_ * cfg_.nc_beta);
      std::poisson_distribution<int> distribution(lambda);
      return distribution(random_engine::generator);
    }

    std::vector<double> draw_child_means() const {
      int n = num_children_;
      while (true) {
        std::vector<double> means = sample_gaussian(mean_, std::sqrt(var_), n);
        double scaling_factor = n * mean_ / sum(means);
        means = multiply(means, scaling_factor);
        if (sum(square(means)) <= n * var_ + n * mean_ * mean_) {
          return means;
        }
      }
    }
    
  public:
    game(config cfg)
      : cfg_(cfg),
        mean_(cfg_.root_mean),
        var_(cfg_.root_var),
        success_count_(0),
        num_moves_(1),
        is_game_over_(false),
        num_children_(draw_num_children()),
        child_means_(draw_child_means())
    {
    }

    int get_num_children() const {
      return num_children_;
    }

    std::vector<double> get_child_means() const {
      return child_means_;
    }

};
 
}