#!/usr/bin/env python3

import argparse
import sys
import subprocess

parser = argparse.ArgumentParser(description='Run game simulation workflow for a given game')
parser.add_argument('game', type=str, help='The game to run simulations on. Should be generic or same')
parser.add_argument('-u', type=int, default=10, help='Max number of unfinished nodes in partial tree simulation')
parser.add_argument('-w', type=int, default=10, help='Number of random walks to be done')
parser.add_argument('-d', action='store_true')

args = parser.parse_args()

game = args.game
num_unf_nodes = args.u
num_walks = args.w
build_type = 'Debug' if args.d else 'Release'
cfg = './cfg/{}_game.toml'.format(game)
sd_model_path = "./models/sd_model.pt"
varphi_model_path = "./models/varphi_model.pt"

pts_exec = './build/{}/bin/{}_game_pts'.format(build_type, game)
rw_exec = './build/{}/bin/{}_game_rw'.format(build_type, game)

pts_args = [pts_exec, '-c', cfg, '-n', str(num_unf_nodes)]
rw_args = [rw_exec, '-c', cfg, '-n', str(num_walks)]

if game is 'generic':
    model_path_args = ['-s', sd_model_path, '-v', varphi_model_path]
    pts_args += model_path_args
    rw_args += model_path_args

subprocess.call(pts_args)
subprocess.call(rw_args)