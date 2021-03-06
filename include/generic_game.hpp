#pragma once

#include <cmath>
#include <iostream>
#include <random>
#include <stack>

#include <torch/script.h>

#include "cpptoml.hpp"
#include "logger.hpp"
#include "random_engine.hpp"
#include "finite_mixture.hpp"
#include "util.hpp"

namespace generic_game
{

/**
 * @struct config
 *
 * A POD structure for holding generic game configuration
 */
struct config {
  int root_children;
  double root_mean;
  double root_sd;
};

/**
 * Parses a .toml configuration file for a generic game.
 * Creates a config object and sets its fields based on the
 * values that appear in the .toml file.
 *
 * @param toml_file_path the location of the toml file relative to where
 * the executable will be invoked from.
 * @return a generic game config with all fields set
 */
config get_config_from_toml(std::string toml_file_path) {
  auto tbl = cpptoml::parse_file(toml_file_path);
  config cfg;
  cfg.root_children = get_from_toml<decltype(cfg.root_children)>(tbl, "root_children");
  cfg.root_mean = get_from_toml<decltype(cfg.root_mean)>(tbl, "root_mean");
  cfg.root_sd = get_from_toml<decltype(cfg.root_sd)>(tbl, "root_sd");
  return cfg;
}

class game {
  public:
    using move_type = int;
  private:
    /* Members */
    config cfg_;
    double mean_;
    double sd_;
    int success_count_;
    int num_moves_made_;
    int num_siblings_;
    std::shared_ptr<torch::jit::script::Module> delta_module_;
    int num_children_;
    std::vector<double> child_means_;
    std::vector<double> child_sds_;
    std::vector<move_type> available_moves_;
    double cumulative_reward_;
    std::shared_ptr<torch::jit::script::Module> sd_module_;
    std::shared_ptr<torch::jit::script::Module> varphi_module_;
    double varphi2_;

    /* Methods */
    int draw_num_children() const {
      auto input_tensor = torch::ones({2, 2}, torch::kFloat64);
      input_tensor[0][0] = static_cast<double>(num_moves_made_);
      input_tensor[0][1] = static_cast<double>(num_children_);

      std::vector<torch::jit::IValue> input({input_tensor});
      at::Tensor output = delta_module_->forward(input).toTensor();

      auto output_it = output.data<double>();
      double lambda_p = *output_it;
      double lambda_n = *(output_it + 1);
      double p = *(output_it + 2);
      
      std::bernoulli_distribution ber_dist(p);
      int pi = ber_dist(random_engine::generator);
      int delta;
      if (pi == 0) {
        std::poisson_distribution<int> pois_dist(lambda_n);
        delta = -1 * pois_dist(random_engine::generator);
      } else {
        std::poisson_distribution<int> pois_dist(lambda_p);
        delta = pois_dist(random_engine::generator);
      }

      return std::max(0, num_siblings_ + 1 + delta);
    }

    /**
     * Moves for a generic game are simply the indices of children which
     * we have already drawn means and variances for. This method should be
     * called by the constructor to build a vector of these indices.
     *
     * @return a vector of child indices
     */
    std::vector<move_type> find_available_moves() const {
      std::vector<move_type> moves;
      for (int i = 0; i < num_children_; i++) {
        moves.push_back(i);
      }
      return moves;
    }

    /**
     * If this state is a terminal state in the game, go ahead and draw a final
     * reward. Otherwise, reward nothing.
     *
     * @return the game's final reward
     */
    double find_current_reward() const {
      return num_children_ == 0 ? sample_gaussian(mean_, sd_) : 0;
    }

    double calc_varphi2() {
      auto input_tensor = torch::ones({2, 2}, torch::kFloat64);
      input_tensor[0][0] = static_cast<double>(num_moves_made_);
      input_tensor[0][1] = static_cast<double>(num_children_);

      std::vector<torch::jit::IValue> input({input_tensor});
      at::Tensor output = varphi_module_->forward(input).toTensor();

      double mean =  *(output.data<double>());
      double sd = *(output.data<double>() + 1);
      double varphi2 = sample_gaussian(mean, sd);
      return std::min(std::max(varphi2, 0.0), 1.0);
    }

    /**
     * The idea here is that we want game states to be more or less immutable.
     * Additionally, we only want new game states to be made by copying the
     * previous state and applying a move. We only want this constructor to be
     * exposed by proxy of the make_move funciton.
     *
     * @param other a const reference to the game state to copy from
     * @param move the move to be made from the previous g ame state
     * @return a new game state object
     */
    game(const game& other, move_type move)
      : cfg_(other.cfg_),
        mean_(other.child_means_[move]),
        sd_(other.child_sds_[move]),
        success_count_(other.success_count_),
        num_moves_made_(other.num_moves_made_ + 1),
        num_siblings_(other.num_children_ - 1),
        delta_module_(other.delta_module_),
        num_children_(draw_num_children()),
        available_moves_(find_available_moves()),
        cumulative_reward_(other.cumulative_reward_ + find_current_reward()),
        sd_module_(other.sd_module_),
        varphi_module_(other.varphi_module_),
        varphi2_(calc_varphi2())
    {
      std::vector<double> p(num_children_, 1. / num_children_);
      std::pair<std::vector<double>, std::vector<double>> mixture_dist =
        sample_finite_mixture(p, mean_, sd_, num_moves_made_, varphi2_, sd_module_);
      child_means_ = std::move(mixture_dist.first);
      child_sds_ = std::move(mixture_dist.second);
    }

  public:
    /**
     * The constructor to build a root game state based on a config.
     *
     * @param config a config struct which sets various parameters of the game
     */
    game(config cfg, 
      std::string sd_model_path = "../models/sd_model.pt", 
      std::string varphi_model_path = "../models/varphi_model.pt",
      std::string delta_model_path = "../models/delta_model.pt"
    )
      : cfg_(cfg),
        mean_(cfg_.root_mean),
        sd_(cfg_.root_sd),
        success_count_(0),
        num_moves_made_(0),
        num_siblings_(0),
        delta_module_(torch::jit::load(delta_model_path)),
        num_children_(cfg_.root_children),
        available_moves_(find_available_moves()),
        cumulative_reward_(find_current_reward()),
        sd_module_(torch::jit::load(sd_model_path)),
        varphi_module_(torch::jit::load(varphi_model_path)),
        varphi2_(calc_varphi2())
    {
      assert(varphi_module_ != nullptr && sd_module_ != nullptr);

      std::vector<double> p(num_children_, 1. / num_children_);

      std::pair<std::vector<double>, std::vector<double>> mixture_dist =
        sample_finite_mixture(p, mean_, sd_, num_moves_made_, varphi2_, sd_module_);

      child_means_ = std::move(mixture_dist.first);
      child_sds_ = std::move(mixture_dist.second);
    }

    /**
     * Getter for game's mean reward
     *
     * @return mean reward
     */
    double get_mean() const noexcept {
      return mean_;
    }

    /**
     * Getter for game's reward variance
     *
     * @return reward variance
     */
    double get_sd() const noexcept {
      return sd_;
    }

    /**
     * Getter for number of moves which have been made in the
     * game up to this point
     *
     * @return number of moves made in the game so far
     */
    int get_num_moves_made() const noexcept {
      return num_moves_made_;
    }

    /**
     * Getter for the vector of drawn child means of this state
     *
     * @return a vector of size num_children_ of the child means
     */
    std::vector<double> get_child_means() const noexcept {
      return child_means_;
    }

    /**
     * Getter for the vector of drawn child vars of this state
     *
     * @return a vector of size num_children_ of the child vars
     */
    std::vector<double> get_child_sds() const noexcept {
      return child_sds_;
    }

    /**
     * Getter for vector of available moves of move_type. Returns a copy
     *
     * @return a vector of available moves from this game state
     */
    std::vector<move_type> get_available_moves() const noexcept {
      return available_moves_;
    }

    bool has_available_moves() const {
      return !available_moves_.empty();
    }

    /**
     * Getter for the cumulative reward of the game. In this game,
     * the reward will be zero until you hit the terminal state
     *
     * @return the total reward of this game
     */
    double get_cumulative_reward() const noexcept {
      return cumulative_reward_;
    }

    double get_varphi2() const noexcept {
      return varphi2_;
    }

    /**
     * Makes a move in the current game and returns a new game object
     * corresponding to the resulting state
     *
     * @param move the move to be made from this state
     * @return a new game state
     */
    game make_move(move_type move) const {
      return game(*this, move);
    }
};

}
