#include <cstdlib>
#include <ctime>

#include "generic_game.hpp"
#include "gtest/gtest.h"

class generic_game_test : public ::testing::Test {
  protected:
    void SetUp() override {}
    generic_game::config cfg_{generic_game::get_config_from_toml("../tests/cfg/generic_game.toml")};
    generic_game::game game_{cfg_};
};

TEST_F(generic_game_test, correctly_inits_from_toml) {
  EXPECT_EQ(cfg_.depth_r, 130);
  EXPECT_EQ(cfg_.depth_p, 0.6567);
  EXPECT_EQ(cfg_.nc_alpha, 3.966);
  EXPECT_EQ(cfg_.nc_beta, -0.0346);
  EXPECT_EQ(cfg_.root_mean, 213.493);
  EXPECT_EQ(cfg_.root_sd, 65);
}

TEST_F(generic_game_test, drew_correct_number_of_child_means) {
  EXPECT_EQ(game_.get_available_moves().size(), game_.get_child_means().size());
}

TEST_F(generic_game_test, drew_correct_number_of_child_sds) {
  EXPECT_EQ(game_.get_available_moves().size(), game_.get_child_sds().size());
}

TEST_F(generic_game_test, make_move_creates_appropriate_copy) {
  auto avail_moves = game_.get_available_moves();
  auto child_means = game_.get_child_means();
  auto child_sds = game_.get_child_sds();

  int random_move = std::rand() % avail_moves.size();
  auto next_game_state = game_.make_move(random_move);
  EXPECT_EQ(next_game_state.get_mean(), child_means[random_move]);
  EXPECT_EQ(next_game_state.get_sd(), child_sds[random_move]);
  EXPECT_EQ(next_game_state.get_num_moves_made(), 1);
}

TEST_F(generic_game_test, cumulative_reward_non_zero) {
  auto game = game_;
  while (true) {
    auto moves = game.get_available_moves();
    if (moves.empty()) {
      break;
    }
    auto random_move = std::rand() % moves.size();
    game = game.make_move(random_move); 
  }
  ASSERT_NE(game.get_cumulative_reward(), 0);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
