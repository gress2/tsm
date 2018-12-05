#include "gtest/gtest.h"
#include "same_game.hpp"

class same_game_test : public ::testing::Test {
  protected:
    void SetUp() override {}
    same_game::config cfg_{
      same_game::get_config_from_toml("../tests/cfg/same_game.toml")
    };
};

TEST_F(same_game_test, correctly_inits_from_toml) {
  EXPECT_EQ(cfg_.width, 15);
  EXPECT_EQ(cfg_.height, 15);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
