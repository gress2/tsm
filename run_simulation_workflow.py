#!/usr/bin/env python3

import argparse
import sys
import subprocess

parser = argparse.ArgumentParser(description='Run game simulation workflow for a given game')
parser.add_argument('game', type=str, help='The game to run simulations on. Should be generic or same')
parser.add_argument('--u', type=int, default=10, help='Max number of unfinished nodes in partial tree simulation')
parser.add_argument('--w', type=int, default=10, help='Number of random walks to be done')

args = parser.parse_args()

game = args.game
num_unf_nodes = args.u
num_walks = args.w
build_type = 'Release'
cfg = './cfg/{}_game.toml'.format(game)
sd_model_path = "./models/sd_model.pt"
varphi_model_path = "./models/varphi_model.pt"

pts_exec = './build/{}/bin/{}_game_pts'.format(build_type, game)
rw_exec = './build/{}/bin/{}_game_rw'.format(build_type, game)

subprocess.call([pts_exec, '-c', cfg, '-n', str(num_unf_nodes), '-s', sd_model_path, '-v', varphi_model_path])
subprocess.call([rw_exec, '-c', cfg, '-n', str(num_walks), '-s', sd_model_path, '-v', varphi_model_path])

print(pts_exec)
print(rw_exec)

