#pragma once

#include <vector>

#include "cpptoml.hpp"
#include "util.hpp"

namespace roller_ball
{

struct config {
  int width; 
  int height;
  std::string board_layout_file;
};

config get_config_from_toml(std::string toml_file_path) {
  auto tbl = cpptoml::parse_file(toml_file_path);
  config cfg;
  cfg.width = get_from_toml<decltype(cfg.width)>(tbl, "width");
  cfg.height = get_from_toml<decltype(cfg.height)>(tbl, "height");
  
  if (is_in_toml<std::string>(tbl, "board_layout_file")) {
    cfg.board_layout_file = get_from_toml<decltype(cfg.board_layout_file)>
      (tbl, "board_layout_file");
  } else {
    cfg.board_layout_file = "";
  }

  return cfg;
}

enum move {
  UP, RIGHT, DOWN, LEFT 
};

enum tile {
  WALL, VACANT, VISITED
};

class game {
  public:
    using move_type = move;
  private:
    config cfg_;
    using board_type = std::vector<std::vector<tile>>;
    board_type board_;
  public:
    game(config cfg)
      : cfg_(cfg) {
      
    }

    void print() const {


       
    }
};

} // namespace roller_ball
