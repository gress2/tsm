#include "generic_game.hpp"
#include "gtest/gtest.h"
#include "mcts.hpp"

class mcts_node_test : public ::testing::Test {
  protected:
    void SetUp() override {}
    generic_game::config cfg_{
      generic_game::get_config_from_toml("../tests/cfg/generic_game.toml")};
    generic_game::game game_{cfg_};
    mcts::node<generic_game::game> node_{game_};  
};

TEST_F(mcts_node_test, expands_correctly_when_able) {
  auto* node = node_.expand();
  ASSERT_EQ(node->get_parent(), &node_);
} 

TEST_F(mcts_node_test, can_expand_until_terminal) {
  auto* node = &node_;
  auto* prev = node;

  while (node) {
    prev = node;
    node = node->expand();
  }

  ASSERT_TRUE(prev->is_terminal());
}

TEST_F(mcts_node_test, depth_is_correct_for_root) {
  ASSERT_EQ(node_.get_depth(), 0);
}

TEST_F(mcts_node_test, depth_is_correct_for_depth_1) {
  auto* node = node_.expand();
  ASSERT_EQ(node->get_depth(), 1);
}

TEST_F(mcts_node_test, depth_is_correct_for_depth_4) {
  auto* node = node_.expand();
  node = node->expand();
  node = node->expand();
  node = node->expand();
  ASSERT_EQ(node->get_depth(), 4);
}

TEST_F(mcts_node_test, get_seq_is_correct_for_root) {
  ASSERT_EQ(node_.get_seq(), std::vector<int>{});
}

TEST_F(mcts_node_test, get_seq_correct_for_depth_1) {
  auto* node = node_.expand();
  ASSERT_EQ(node->get_seq().size(), 1);
}

TEST_F(mcts_node_test, get_seq_is_correct_for_depth_4) {
  auto* node = node_.expand();
  node = node->expand();
  node = node->expand();
  node = node->expand();
  ASSERT_EQ(node->get_seq().size(), 4);
}

class uct_test : public ::testing::Test {
  protected:
    using config_type = generic_game::config;
    using game_type = generic_game::game;
    using node_type = mcts::node<game_type>;
    using uct_type = mcts::uct<node_type>;

    void SetUp() override {}
    
    config_type cfg_{generic_game::get_config_from_toml("../tests/cfg/generic_game.toml")};
    game_type game_{cfg_};
    node_type node_{game_};
    uct_type uct_{node_};
};


TEST_F(uct_test, tree_policy_returns_non_null) {
  auto* node = uct_.tree_policy(&node_);
  ASSERT_TRUE(node);
  ASSERT_NE(node, &node_);
}

TEST_F(uct_test, default_policy_returns_reward) {
  double reward = uct_.default_policy(&node_);
  ASSERT_NE(reward, 0);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
