#!/usr/bin/env python3

import argparse
import sys
import subprocess
import pandas as pd
from math import sqrt
from time import strftime, gmtime
from os import path, makedirs
from shutil import copyfile

parser = argparse.ArgumentParser(description='Run game simulation workflow for a given game')
parser.add_argument('game', type=str, help='The game to run simulations on. Should be generic or same')
parser.add_argument('-u', type=int, default=10, help='Max number of unfinished nodes in partial tree simulation')
parser.add_argument('-w', type=int, default=10, help='Number of random walks to be done')
parser.add_argument('-d', action='store_true', help='Use the debug built executables')
parser.add_argument('-a', action='store_true', help='Create an archive of these files')

args = parser.parse_args()

game = args.game
num_unf_nodes = args.u
num_walks = args.w
build_type = 'Debug' if args.d else 'Release'
cfg = './cfg/{}_game.toml'.format(game)
sd_model_path = "./models/sd_model.pt"
varphi_model_path = "./models/varphi_model.pt"
should_archive = args.a

pts_exec = './build/{}/bin/{}_game_pts'.format(build_type, game)
rw_exec = './build/{}/bin/{}_game_rw'.format(build_type, game)

pts_args = [pts_exec, '-c', cfg, '-n', str(num_unf_nodes)]
rw_args = [rw_exec, '-c', cfg, '-n', str(num_walks)]

if game == 'generic':
    model_path_args = ['-s', sd_model_path, '-v', varphi_model_path]
    pts_args += model_path_args
    rw_args += model_path_args

subprocess.call(pts_args)
subprocess.call(rw_args)

main_cols = ['mean', 'sd', 'd', 'k', 'varphi2', 'gammas', 'etas']
main_df = pd.DataFrame(columns=main_cols)

with open('main.{}_game.csv'.format(game), 'r') as main_f:
    for line in main_f:
        split = line.split(',')
        mean = float(split[0])
        sd = float(split[1])
        d = int(split[2])
        k = int(split[3])
        varphi2 = float(split[4])
        cdists = [ float(elem.replace('(', '').replace(')', '')) for elem in split[5:] ]
        means = cdists[0::2]
        sds = cdists[1::2]
        alphas = list(map(lambda x: (x - mean) / sd if sd > 0 else 0, means))
        taus = list(map(lambda x: x / sd if sd > 0 else 0, sds))
        mul_p = lambda x: x * sqrt(1.0 / k)
        gammas = list(map(mul_p, alphas))
        etas = list(map(mul_p, taus))
        main_df.loc[len(main_df)] = [mean, sd, d, k, varphi2, gammas, etas]

main_df.to_pickle('main.{}_game.pkl'.format(game))

td_cols = ['td', 'freq']
td_df = pd.DataFrame(columns=td_cols)
td_freqs = dict()

with open('td.{}_game.csv'.format(game), 'r') as td_f:
    for line in td_f:
        td = int(line)
        if td not in td_freqs:
            td_freqs[td] = 1
        else:
            td_freqs[td] += 1
    for td in td_freqs:
        td_df.loc[len(td_df)] = [td, td_freqs[td]]

td_df.to_pickle('td.{}_game.pkl'.format(game))

dk_cols = ['d', 'k']
dk_df = pd.DataFrame(columns=dk_cols)

with open('dk.{}_game.csv'.format(game), 'r') as dk_f:
    for line in dk_f:
        split = line.split(',')
        d = int(split[0])
        k = int(split[1])
        dk_df.loc[len(dk_df)] = [d, k]

dk_df.to_pickle('dk.{}_game.pkl'.format(game))

if should_archive:
    if not path.exists('archives'):
        makedirs('archives')
    # MM-DD-H-M
    timestamp = strftime('%m-%d-%H-%M', gmtime())
    makedirs('archives/{}'.format(timestamp))
    copyfile('main.{}_game.csv'.format(game), 'archives/{}/main.{}_game.{}.csv'.format(timestamp, game, timestamp))
    copyfile('main.{}_game.pkl'.format(game), 'archives/{}/main.{}_game.{}.pkl'.format(timestamp, game, timestamp))
    copyfile('td.{}_game.csv'.format(game), 'archives/{}/td.{}_game.{}.csv'.format(timestamp, game, timestamp))
    copyfile('td.{}_game.pkl'.format(game), 'archives/{}/td.{}_game.{}.pkl'.format(timestamp, game, timestamp))
    copyfile('dk.{}_game.csv'.format(game), 'archives/{}/dk.{}_game.{}.csv'.format(timestamp, game, timestamp))
    copyfile('dk.{}_game.pkl'.format(game), 'archives/{}/dk.{}_game.{}.pkl'.format(timestamp, game, timestamp))