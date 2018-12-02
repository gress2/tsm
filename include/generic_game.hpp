#pragma once

#include <iostream>

#include "cpptoml.hpp"
#include "logger.hpp"

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
  cfg.depth_p = get_from_toml<decltype(cfg.disp_mean_delta)>(tbl, "disp_mean_delta");
  cfg.depth_p = get_from_toml<decltype(cfg.disp_mean_beta)>(tbl, "disp_mean_beta");
  cfg.depth_p = get_from_toml<decltype(cfg.disp_var_delta)>(tbl, "disp_var_delta");
  cfg.depth_p = get_from_toml<decltype(cfg.nc_alpha)>(tbl, "nc_alpha");
  cfg.depth_p = get_from_toml<decltype(cfg.nc_beta)>(tbl, "nc_beta");
  cfg.depth_p = get_from_toml<decltype(cfg.root_mean)>(tbl, "root_mean");
  cfg.depth_p = get_from_toml<decltype(cfg.root_var)>(tbl, "root_var");
  return cfg;
}

class game {
  public:
  private:
    const config cfg_;
  public:
    game(config cfg)
      : cfg_(cfg)
    {}
};
 
}