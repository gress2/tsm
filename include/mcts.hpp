#pragma once

#include <cmath>
#include <deque>
#include <limits>
#include <vector>

namespace mcts
{

template <class Game>
class node {
  public:
    using move_type = typename Game::move_type;
    using state_type = Game;
  private:
    std::size_t n_;
    double q_total_;
    node* parent_;
    std::vector<node> children_;
    Game game_;
    std::deque<move_type> unused_moves_;
    bool is_terminal_;
  public:
    node(Game game, node* parent = nullptr)
     : n_(0),
       q_total_(0),
       parent_(parent),
       game_(game)
    {
      std::vector<move_type> avail_moves = game_.get_available_moves();
      unused_moves_.insert(unused_moves_.end(), avail_moves.begin(), avail_moves.end()); 
      is_terminal_ = avail_moves.empty();
    }

    /**
     * getter for the visit count of this node
     *
     * @return the visit count of the node
     */
    std::size_t get_n() const noexcept {
      return n_;
    }

    /**
     * getter for the sum of the q values propagated through this node
     *
     * @return the sum of q values
     */
    double get_q_total() const noexcept {
      return q_total_;
    }

    /**
     * getter for the parent of this node
     *
     * @return a pointer to the parent node
     */
    node* get_parent() const noexcept {
      return parent_;
    }

    /**
     * getter for detrmining whether or not this node wraps
     * a terminal game state.
     *
     * @return true if terminal, false if not
     */
    bool is_terminal() const noexcept {
      return is_terminal_;
    }


    /**
     * getter for finding the total, cumulative reward of a game
     *
     * @return the total reward
     */
    double get_reward() const noexcept {
      return game_.get_cumulative_reward(); 
    }

    /**
     * gets a copy of the wrapped state (the game)
     *
     * @return copy of the wrapped state
     */
    state_type get_state() const noexcept {
      return game_;
    }

    /**
     * setter for the visit count of this node
     *
     * @param n the new visit count
     */
    void set_n(std::size_t n) noexcept {
      n_ = n;
    }

    /**
     * setter for the total q value of the node
     *
     * @param q_total the new total q value of this node
     */ 
    void set_q_total(double q_total) noexcept {
      q_total_ = q_total;
    } 

    /**
     * appends a child node to this node by taking
     * an unused move from the deque and making a new
     * game state from this node. the new game state is wrapped
     * in a node and appended as a child.
     *
     * @return the newly appended node or a nullptr if nothing was
     * appended.
     */
    node* expand() {
      if (unused_moves_.empty()) {
        return nullptr;
      }
      move_type move = unused_moves_.front();
      unused_moves_.pop_front();

      Game to_append = game_.make_move(move);
      children_.push_back(node(to_append, this));
      return &(children_.back());
    }

    /**
     * Returns the best child of the node according to UCT.
     * If a child is unvisited, n=0 and UCT score is infinite,
     * so go ahead and return it. Otherwise return the child 
     * which maximizes UCT.
     *
     * @return a pointer to the child node which maximizes UCT
     */
    node* best_child() {
      node* best = nullptr;
      double max_uct = std::numeric_limits<double>::min();
      for (auto& child : children_) {
        double n = child.get_n();
        if (!n) {
          return &child;
        }
        double Q = child.get_q_total();
        constexpr double C = 2;

        double uct_score = Q / n + C * std::sqrt(2 * std::log(n_) / n);
        if (uct_score > max_uct) {
          best = &child;
          max_uct = uct_score;
        }
      }
      return best;
    }
};

template <class Node>
class uct {
  public:
  private:
    Node root_;
    double high_score_;
    std::vector<typename Node::move_type> best_seq_; 
    std::size_t num_iterations_;
  public:
    uct(Node root, std::size_t num_iterations = 1e5)
     : root_(root), 
       high_score_(std::numeric_limits<double>::min()),
       num_iterations_(num_iterations) 
    {}

    /**
     * This finds the best descendant of a node according
     * to UCT. It recurses down the tree starting from v0
     * until it finds either a terminal node or a node which
     * may yet be expanded. It uses UCT to determine which
     * child to pick at each level.
     *
     * @param v0 the node to start looking from
     * @return a pointer to the "best" descendant node
     */
    Node* tree_policy(Node* v0) {
      Node* cur = v0;
      while (!cur->is_terminal()) {
        Node* expanded = cur->expand();
        if (expanded) {
          return expanded;
        } else {
          cur = cur->best_child();
        }
      }
      return cur;
    }

    /**
     * Takes random actions in a game starting from some
     * initial state until the game ends.
     * 
     * @param game an initial game state
     * @return the reward after playing the game out randomly from
     * the initial state.
     */
    double default_policy(typename Node::state_type game) {
      auto moves = game.get_available_moves(); 
      while (!moves.empty()) {
        int random_idx = std::rand() % moves.size();
        auto move = moves[random_idx];
        game = game.make_move(move);
        moves = game.get_available_moves();
      }
      return game.get_cumulative_reward();
    } 

    /**
     * Propagates visit counts and deltas (rewards) up
     * the tree until the root.
     *
     * @param v a pointer to the node to start backup from
     * @param delta the reward encountered which we use to increment
     * Q values with.
     */
    void backup(Node* v, double delta) {
      while (v) {
        v->set_n(v->get_n() + 1);
        v->set_total_q(v->get_total_q() + delta);
        v = v->get_parent;
      }
    }

    /**
     * Primary driver for UCT in which we sequentially explore/expand,
     * simulate, and backpropagate findings
     */
    void search() {
      Node* v1 = tree_policy(&root_);
      double delta = default_policy(v1->get_state());
      backup(v1, delta);
    }

}; 

}
