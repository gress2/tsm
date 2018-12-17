#include <fstream>
#include <iostream>
#include <map>
#include <queue>

#include "finite_mixture.hpp"
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

template <class Game>
class deep_tree_simulator {
  public:
    using node_type = simulator::node<Game>;
  private:
    node_type root_;
    int rollouts_per_node_;
    std::size_t num_unf_nodes_;
    std::size_t max_unf_nodes_;
    std::ofstream statistics_file_;
    std::ofstream mixing_data_file_;
    std::priority_queue<node_type*, std::vector<node_type*>, 
      node_ptr_comparator<node_type>> pri_q_; 
  public:

    deep_tree_simulator(Game game, std::size_t max_unf_nodes) 
      : root_(std::move(game)),
        rollouts_per_node_(10),
        num_unf_nodes_(0),
        max_unf_nodes_(max_unf_nodes),
        statistics_file_("stats.csv"),
        mixing_data_file_("mixing_data.csv")
    {
      pri_q_.push(&root_);
    }

    void simulate() {
      while (pri_q_.size() < max_unf_nodes_ && !pri_q_.empty()) {
        node_type* cur = pri_q_.top();
        std::cout << cur->get_depth() << std::endl;
        std::cout << pri_q_.size() << std::endl;
        while (true) {
          cur->expand();

          std::vector<node_type*> non_terminals;
          for (auto& child : cur->get_children()) {
            if (child.can_expand()) {
              non_terminals.push_back(&child);
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
    }
};

}
