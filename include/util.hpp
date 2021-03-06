#pragma once

#include <iostream>
#include <iterator>
#include <vector>
#include <random>
#include <sstream>

#include "beta_distribution.hpp"
#include "cpptoml.hpp"
#include "random_engine.hpp"


template<typename Out>
void split(const std::string &s, char delim, Out result) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        *(result++) = item;
    }
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

/**
 * Writes the values of an iterable container to stdout
 *
 * @param iterable a universal reference to an iterable container
 */
template <class T>
void print(T&& iterable) {
  std::cout << "[";
  for (auto it = iterable.begin(); it != iterable.end(); ++it) {
    std::cout << *it;
    if (std::next(it) != iterable.end()) {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;
}

template <class T, class V>
void print(std::vector<std::pair<T, V>>& pair_vec) {
  std::cout << "[";
  for (auto it = pair_vec.begin(); it != pair_vec.end(); ++it) {
    std::cout << "{" << it->first << ", " << it->second << "}";
    if (std::next(it) != pair_vec.end()) {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;
}

std::vector<double> sample_uniform(double lower, double upper, int n) {
  std::uniform_real_distribution<double> dist(lower, upper);

  std::vector<double> samples;
  for (int i = 0; i < n; i++) {
    samples.push_back(dist(random_engine::generator));
  }

  return samples;
}

/**
 * Draws a random variable from a Gaussian
 *
 * @param mean the mean of the Gaussian
 * @param sd the standard deviation of the Gaussian
 */
double sample_gaussian(const double mean, const double sd) {
  std::normal_distribution dist(mean, sd);
  double draw = dist(random_engine::generator);
  return draw;
}

/**
 * Draws a vector of random variables from a Gaussian distribution
 *
 * @param mean the mean of the Gaussian
 * @param sd the standard deviation of the Gaussian
 * @param n the number of samples to draw (and size of returned vector)
 * @return a vector of random variables
 */
std::vector<double> sample_gaussian(const double mean, const double sd, const int n) {
  std::normal_distribution<double> dist(mean, sd);
  std::vector<double> samples;
  for (int i = 0; i < n; i++) {
    double draw = dist(random_engine::generator);
    samples.push_back(draw);
  }
  return samples;
}

std::vector<double> sample_uniform_dirichlet(int k, double epsilon) {
  std::gamma_distribution<double> dist(epsilon, 1.0);
  std::vector<double> x;
  for (int i = 0; i < k; i++) {
    x.push_back(dist(random_engine::generator));
  }

  double sum = std::accumulate(x.begin(), x.end(), 0.0);
  for (auto& elem : x) {
    elem /= sum;
  }
  return x;
}

std::vector<double> sample_beta(double alpha, double beta, int n) {
  sftrabbit::beta_distribution<double> dist(alpha, beta);
  std::vector<double> x;
  for (int i = 0; i < n; i++) {
    x.push_back(dist(random_engine::generator));
  }
  return x;
}

/**
 * Calculates the arithmetic sum of the elements of an iterable container
 *
 * @param con a universal reference to an iterable container
 * @return the arithmetic sum
 */
template <class T>
typename std::remove_reference<T>::type::value_type sum(T&& con) {
  return std::accumulate(con.begin(), con.end(), 0.0);
}

/**
 * Calculates the arithmetic mean of the elements of an iterable container
 *
 * @param con a universal reference to an iterable container
 * @return the arithmetic sum
 */
template <class T>
double mean(T&& con) {
  double con_sum = sum(std::forward<T>(con));
  return con_sum / con.size();
}

template <class T>
double variance(T&& con) {
  double con_mean = mean(con);
  double var = 0;
  for (auto& elem : con) {
    var += std::pow(elem - con_mean, 2);
  }
  var /= con.size();
  return var;
}

template <class T>
double stddev(T&& con) {
  return std::sqrt(variance(std::forward<T>(con)));
}

std::vector<double> sample_hypersphere(int k, double r, std::vector<double> x) {
  double sum_sq = 0;
  for (const auto& elem : x) {
    sum_sq += elem * elem;
  }
  double x_norm = std::sqrt(sum_sq);
  for (auto& elem : x) {
    elem *= r / x_norm;
  }

  return x;
}

std::vector<double> sample_hypersphere(int k, double r) {
  std::vector<double> x;
  for (int i = 0; i < k; i++) {
    x.push_back(sample_gaussian(0, 1));
  }

  double sum_sq = 0;
  for (auto& elem : x) {
    sum_sq += elem * elem;
  }


  for (auto& elem : x) {
    elem *= r / std::sqrt(sum_sq);
  }

  return x;
}

/**
 * Copies a container and multiplies the value of its elements by a factor.
 *
 * @param con a universal reference to an iterable container
 * @param factor a scalar to multiply the elements of the container by
 * @return a copy of con scaled by factor
 */
template <class T, class V>
typename std::remove_reference<T>::type multiply(T&& con, V factor) {
  typename std::remove_reference<T>::type res(std::forward<T>(con));
  for (auto& elem : res) {
    elem *= factor;
  }
  return res;
}

/**
 * Copies a container and squares the values of its elements
 *
 * @param con a universal reference to an iterable container
 * @return a copy of con with its elements squared
 */
template <class T>
typename std::remove_reference<T>::type square(T&& con) {
  typename std::remove_reference<T>::type res(std::forward<T>(con));
  for (auto& elem : res) {
    elem *= elem;
  }
  return res;
}

/**
 * Retrieves a property from a cpptoml::table as type T. If the property doesn't exist
 * in the table, program execution is stopped.
 *
 * @param tbl reference to a shared pointer pointing to a cpptoml::table
 * @param prop a string representing to property to be retrieved from the toml table
 * @return the value of property prop as type T
 */
template <class T>
T get_from_toml(const std::shared_ptr<cpptoml::table>& tbl, std::string prop) {
  cpptoml::option<T> opt = tbl->get_as<T>(prop);
  assert(opt);
  return *opt;
}

template <class T>
bool is_in_toml(const std::shared_ptr<cpptoml::table>& tbl, std::string prop) {
  cpptoml::option<T> opt = tbl->get_as<T>(prop);
  return static_cast<bool>(opt);
}

template <class T>
bool approx_equal(T lhs, T rhs, double tolerance = 1e-5) {
  double diff = std::abs(lhs - rhs);
  return diff < tolerance;
}

template <class Iter>
bool approx_equal(Iter c1_begin, Iter c1_end, Iter c2_begin,
    Iter c2_end, double tolerance = 1e-5) {
  if (std::distance(c1_begin, c1_end) != std::distance(c2_begin, c2_end)) {
    return false;
  }

  while (c1_begin != c1_end) {
    if (std::abs(*c1_begin - *c2_begin) >= tolerance) {
      return false;
    }
    ++c1_begin;
    ++c2_begin;
  }
  return true;
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
