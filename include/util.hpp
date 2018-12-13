#pragma once

#include <iostream>
#include <vector>
#include <random>

#include "cpptoml.hpp"
#include "random_engine.hpp"

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

