#include <deque>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "finite_mixture.hpp"
#include "util.hpp"

namespace pts
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
      : game_(std::move(game))
    {}
    void expand() {
      std::vector<move_type> moves = game_.get_available_moves();
      for (auto& move : moves) {
        children_.push_back(node{game_.make_move(move)});
      }
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
};

struct state_statistics {
  double mean;
  double sd;
  double k;
  double d;
};

template <class Game>
class simulator {
  public:
    using node_type = node<Game>;
  private:
    node_type root_;
    std::vector<node_type*> worklist_;
    std::size_t num_unf_nodes_;
    std::size_t max_unf_nodes_;
    std::size_t rollouts_per_node_;
    std::ofstream statistics_file_;
    std::ofstream mixing_data_file_;
    std::mutex io_mutex_;
    std::mutex progress_mutex_;

    void rollout(node_type* node, std::vector<state_statistics>& range_stats) {
      std::vector<double> rewards;
      for (std::size_t i = 0; i < rollouts_per_node_; i++) {
        Game g(node->get_game());
        auto moves = g.get_available_moves();
        while (!moves.empty()) {
          int random_idx = std::rand() % moves.size();
          g = g.make_move(moves[random_idx]);
          moves = g.get_available_moves();
        }
        rewards.push_back(g.get_cumulative_reward());
      } 

      double reward_mean = mean(rewards);
      double reward_sd = stddev(rewards);

      node->set_mean(reward_mean);
      node->set_sd(reward_sd);

      state_statistics stats;
      stats.mean = reward_mean;
      stats.sd = reward_sd; 
      stats.k = node->get_game().get_available_moves().size();
      stats.d = node->get_game().get_num_moves_made();

      range_stats.push_back(stats);
    }

    void rollout_range(std::vector<node_type*>& worklist, std::size_t start_idx, std::size_t end_idx, std::size_t& progress) {
      std::vector<state_statistics> range_stats;
      for (std::size_t i = start_idx; i < worklist.size() && i < end_idx; i++) {
        if (i % 1000 == 0 && i != 0) {
          std::lock_guard progress_guard(progress_mutex_);
          progress += 1000;
          std::cout << "[" << progress << "/" << worklist_.size() << "] (" 
            << (progress / worklist_.size() * 100) << "%)" << std::endl;
        }
        rollout(worklist[i], range_stats);
      }

      std::lock_guard lock_guard(io_mutex_);
      for (auto& stats : range_stats) {
        statistics_file_ << stats.mean << "," << stats.sd << "," << stats.k 
          << "," << stats.d << std::endl;
      }
    }

    std::pair<double, double> mix(node_type* node) {
      auto& children = node->get_children();
      if (children.empty()) {
        return std::make_pair(node->get_mean(), node->get_sd());
      }

      std::size_t n = children.size();
      std::vector<std::pair<double, double>> child_vals;

      for (auto& child : children) {
        child_vals.push_back(mix(&child));
      } 

      std::vector<double> child_sds;
      for (auto& elem : child_vals) {
        child_sds.push_back(elem.second);
      }

      double mean = 0;
      double v1 = 0;
      double v2 = 0;

      for (std::pair<double, double>& p : child_vals) {
        mean += p.first / static_cast<double>(n);
        
        v1 += p.first * p.first / static_cast<double>(n);
        v2 += p.second * p.second / static_cast<double>(n);
      }

      double variance = v1 + v2 - mean * mean;
      double sd = std::sqrt(variance);
      std::size_t depth = node->get_game().get_num_moves_made();
      std::size_t k = child_vals.size();

      double varphi2 = reverse_to_varphi2(sd, child_sds);

      mixing_data_file_ << mean << ", " << sd << ", " 
        << depth << ", " << k << ", " << varphi2 << ", "; 

      for (auto it = child_vals.begin(); it != child_vals.end(); ++it) {
        mixing_data_file_ << "(" << it->first << "," << it->second << ")";
        if (std::next(it) != child_vals.end()) {
          mixing_data_file_ << ", ";
        }
      }
      mixing_data_file_ << std::endl;

      return std::make_pair(mean, sd);
    }

  public:
    simulator(Game game, std::size_t max_unf_nodes) 
      : root_(std::move(game)),
        worklist_({&root_}),
        num_unf_nodes_(0),
        max_unf_nodes_(max_unf_nodes),
        rollouts_per_node_(10),
        statistics_file_("stats.csv"),
        mixing_data_file_("mixing_data.csv")
    {}

    void simulate() {
      while (num_unf_nodes_ < max_unf_nodes_ && !worklist_.empty()) {
        int random_idx = std::rand() % worklist_.size();
        auto it = worklist_.begin();
        std::advance(it, random_idx);
        node_type* cur = *it;

        cur->expand();
        for (auto& child : cur->get_children()) {
          worklist_.push_back(&child);
          num_unf_nodes_++;
        }

        it = worklist_.begin();
        std::advance(it, random_idx);
        worklist_.erase(it);
        num_unf_nodes_--;
      } 

      std::vector<std::thread> threads;
      std::size_t num_threads = std::thread::hardware_concurrency();
      std::size_t workload_per_thread = worklist_.size() / num_threads + 1;
      std::size_t progress = 0;

      for (std::size_t i = 0; i < num_threads; i++) {
        std::size_t start_idx = workload_per_thread * i;
        std::size_t end_idx = start_idx + workload_per_thread;
        auto& wl = worklist_;
        threads.push_back(std::thread([this, &wl, start_idx, end_idx, &progress] {
          this->rollout_range(wl, start_idx, end_idx, progress);
        }));
      }

      for (auto& thread : threads) {
        if (thread.joinable()) {
          thread.join();
        }
      }

      auto parent_stats = mix(&root_);
      std::cout << "Root statistics --- (mean: " << parent_stats.first <<
        ", stddev: " << parent_stats.second << ")" << std::endl;
    }
};

}
