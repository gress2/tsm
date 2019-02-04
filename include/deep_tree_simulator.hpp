#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <queue>
#include <thread>

#include "simulator_node.hpp"
#include "util.hpp"

namespace simulator
{

template <class Node>
struct node_ptr_comparator {
  bool operator() (Node* lhs, Node* rhs) {
    return lhs->get_depth() > rhs->get_depth();
  } 
};

struct state_statistics {
  double mean;
  double sd;
  double k;
  double d;
};

template <class Game>
class deep_tree_simulator {
  public:
    using node_type = simulator::node<Game>;
  private:
    node_type root_;
    int rollouts_per_node_;
    std::size_t num_unf_nodes_;
    std::size_t max_unf_nodes_;
    std::ofstream data_file_;
    std::priority_queue<node_type*, std::vector<node_type*>, 
      node_ptr_comparator<node_type>> pri_q_; 
    std::mutex io_mutex_;
    std::mutex progress_mutex_;
    std::vector<node_type*> worklist_;
  public:

    deep_tree_simulator(Game game, std::size_t max_unf_nodes, std::string data_file_path = "/dev/null") 
      : root_(std::move(game)),
        rollouts_per_node_(100),
        num_unf_nodes_(0),
        max_unf_nodes_(max_unf_nodes),
        data_file_(data_file_path)
    {
      pri_q_.push(&root_);
    }

    void rollout(node_type* node, std::vector<state_statistics>& range_stats) {
      std::vector<double> rewards;
      for (int i = 0; i < rollouts_per_node_; i++) {
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

    void rollout_range(std::vector<node_type*>& wl, std::size_t start_idx, 
        std::size_t end_idx, std::size_t& progress) {
      std::vector<state_statistics> range_stats;
      for (std::size_t i = start_idx; i < wl.size() && i < end_idx; i++) {
        if (i % 1000 == 0 && i != 0) {
          std::lock_guard progress_guard(progress_mutex_);
          progress += 1000;
          std::cout << "[" << progress << "/" << wl.size() << "] (" 
            << (progress / static_cast<float>(wl.size()) * 100) 
            << "%)" << std::endl;
        }
        rollout(wl[i], range_stats);
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

      double variance = std::max(0., v1 + v2 - mean * mean);
      double sd = std::sqrt(variance);
      std::size_t depth = node->get_game().get_num_moves_made();
      std::size_t k = child_vals.size();

      double varphi2 = reverse_to_varphi2(sd, child_sds);

      data_file_ << mean << ", " << sd << ", " 
        << depth << ", " << k << ", " << varphi2 << ", "; 

      for (auto it = child_vals.begin(); it != child_vals.end(); ++it) {
        data_file_ << "(" << it->first << "," << it->second << ")";
        if (std::next(it) != child_vals.end()) {
          data_file_ << ", ";
        }
      }
      data_file_ << std::endl;

      return std::make_pair(mean, sd);
    }

    void simulate() {
      while (pri_q_.size() < max_unf_nodes_ && !pri_q_.empty()) {
        node_type* cur = pri_q_.top();
        while (true) {
          cur->expand();

          std::vector<node_type*> non_terminals;
          for (auto& child : cur->get_children()) {
            if (child.can_expand()) {
              non_terminals.push_back(&child);
            } else {
              Game g = child.get_game(); 
              child.set_mean(g.get_cumulative_reward());
              child.set_sd(0);
            }
          }

          if (non_terminals.empty()) {
            break;
          }

          int random_idx = std::rand() % non_terminals.size();
          auto it = non_terminals.begin();
          std::advance(it, random_idx);
          cur = *it;
          non_terminals.erase(it);
          for (auto& elem : non_terminals) {
            pri_q_.push(elem);
          }
        }
        pri_q_.pop();
      }


      while (!pri_q_.empty()) {
        node_type* cur = pri_q_.top();
        worklist_.push_back(cur);
        pri_q_.pop();
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

      worklist_.clear();
    }

};

}
