#include "generic_game.hpp"
#include "gtest/gtest.h"

#include <iostream>

class generic_game_test : public ::testing::Test {
  protected:
    void SetUp() override {}

    generic_game::config cfg_{generic_game::get_config_from_toml("../tests/cfg/generic_game.toml")};
    generic_game::game game_{cfg_};

};

TEST_F(generic_game_test, correctly_inits_from_toml) {
  EXPECT_EQ(cfg_.depth_r, 130);
  EXPECT_EQ(cfg_.depth_p, 0.6567);
  EXPECT_EQ(cfg_.disp_mean_delta, 108.382);
  EXPECT_EQ(cfg_.disp_mean_beta, 3.44);
  EXPECT_EQ(cfg_.disp_var_delta, 4.0);
  EXPECT_EQ(cfg_.disp_var_beta, 0.0051);
  EXPECT_EQ(cfg_.nc_alpha, 3.966);
  EXPECT_EQ(cfg_.nc_beta, -0.0346);
  EXPECT_EQ(cfg_.root_mean, 213.493);
  EXPECT_EQ(cfg_.root_var, 4273.46);
}

TEST_F(generic_game_test, drew_correct_number_of_child_means) {
  auto cmeans = game_.get_child_means();
  for (auto& elem : cmeans) {
    std::cout << elem << std::endl;
  }

  EXPECT_EQ(game_.get_num_children(), game_.get_child_means().size());
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}