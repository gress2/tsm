#pragma once

#include <vector>

namespace simulator 
{
template <class Game>
class node {
  public:
    using child_type = node<Game>;
    using move_type = typename Game::move_type;
  private:
    Game game_;
    std::vector<child_type> children_;
    double mean_;
    double sd_;
  public:
    node(Game&& game)
      : game_(std::move(game)), mean_(0), sd_(0)
    {}
    
    void expand() {
      std::vector<move_type> moves = game_.get_available_moves();
      for (auto& move : moves) {
        children_.push_back(node{game_.make_move(move)});
      }
    }

    bool can_expand() const {
      return game_.has_available_moves();
    }

    std::vector<child_type>& get_children() {
      return children_;
    }

    Game get_game() const {
      return game_;
    }

    void set_mean(double mean) {
      mean_ = mean;
    }

    void set_sd(double sd) {
      sd_ = sd;
    }

    double get_mean() const {
      return mean_;
    }

    double get_sd() const {
      return sd_;
    }

    int get_depth() const {
      return game_.get_num_moves_made();
    }
};
}
