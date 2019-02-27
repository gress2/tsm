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
parser.add_argument('-deep', action='store_true', help='Run the deep tree simulator rather than the partial tree simulator')

args = parser.parse_args()

game = args.game
num_unf_nodes = args.u
num_walks = args.w
build_type = 'Debug' if args.d else 'Release'
cfg = './cfg/{}_game.toml'.format(game)
should_archive = args.a

simulator_exec = 'dts' if args.deep else 'pts'

sim_exec = './build/{}/bin/{}_game_{}'.format(build_type, game, simulator_exec)
rw_exec = './build/{}/bin/{}_game_rw'.format(build_type, game)

sim_args = ['-c', cfg, '-n', str(num_unf_nodes)]
rw_args = ['-c', cfg, '-n', str(num_walks)]

# if we're dealing with the generic game, we only run random walks
if game == 'generic':
    rw_args += ['-s', './models/sd_model.pt', '-v', './models/varphi_model.pt']
    print('Step 1/2: random walk')
    subprocess.call([rw_exec] + rw_args)
    print('Step 2/2: moving data to dataframes')
else:
    print('Step 1/3: tree simulation')
    # subprocess.call([sim_exec] + sim_args)
    print('Step 2/3: random walk')
    subprocess.call([rw_exec] + rw_args)
    print('Step 3/3: moving data to dataframes')
'''
main_cols = ['mean', 'sd', 'd', 'k', 'varphi2', 'gammas', 'etas']
main_df = pd.DataFrame(columns=main_cols)

ctr = 0
tmp_df = pd.DataFrame(columns=main_cols)
with open('main.{}_game.csv'.format(game), 'r') as main_f:
    for line in main_f:
        if ctr % 1000 == 0:
            if len(tmp_df.index) > 0:
                main_df = main_df.append(tmp_df)
                tmp_df = pd.DataFrame(columns=main_cols)
        split = line.split(',')
        mean = float(split[0])
        sd = float(split[1])
        d = int(split[2])
        k = int(split[3])
        varphi2 = float(split[4])
        cdists = [float(elem.replace('(', '').replace(')', '')) for elem in split[5:]]
        means = cdists[0::2]
        sds = cdists[1::2]
        alphas = list(map(lambda x: (x - mean) / sd if sd > 0 else 0, means))
        taus = list(map(lambda x: x / sd if sd > 0 else 0, sds))
        mul_p = lambda x: x * sqrt(1.0 / k)
        gammas = list(map(mul_p, alphas))
        etas = list(map(mul_p, taus))
        tmp_df.loc[len(tmp_df)] = [mean, sd, d, k, varphi2, gammas, etas]
        ctr += 1
if len(tmp_df.index) > 0:
    main_df = main_df.append(tmp_df)

main_df.to_pickle('main.{}_game.pkl'.format(game))
'''
td_cols = ['td', 'freq']
td_df = pd.DataFrame(columns=td_cols)
td_freqs = dict()

tmp_df = pd.DataFrame(columns=td_cols)
with open('td.{}_game.csv'.format(game), 'r') as td_f:
    for line in td_f:
        td = int(line)
        if td not in td_freqs:
            td_freqs[td] = 1
        else:
            td_freqs[td] += 1
    ctr = 0
    for td in td_freqs:
        if ctr % 1000 == 0:
            if len(tmp_df.index) > 0:
                td_df = td_df.append(tmp_df)
                tmp_df = pd.DataFrame(columns=td_cols)
        tmp_df.loc[len(tmp_df)] = [td, td_freqs[td]]
        ctr += 1 
if len(tmp_df.index) > 0:
    td_df = td_df.append(tmp_df)

td_df.to_pickle('td.{}_game.pkl'.format(game))

dkd_cols = ['d', 'k', 'delta']
dkd_df = pd.DataFrame(columns=dkd_cols)

ctr = 0
tmp_df = pd.DataFrame(columns=dkd_cols)
with open('dkd.{}_game.csv'.format(game), 'r') as dkd_f:
    for line in dkd_f:
        if ctr % 1000 == 0:
            if len(tmp_df.index) > 0:
                dkd_df = dkd_df.append(tmp_df)
                tmp_df = pd.DataFrame(columns=dkd_cols)
        split = line.split(',')
        d = int(split[0])
        k = int(split[1])
        delta = int(split[2])
        tmp_df.loc[len(tmp_df)] = [d, k, delta]
        ctr += 1
if len(tmp_df.index) > 0:
    dkd_df = dkd_df.append(tmp_df)

dkd_df.to_pickle('dkd.{}_game.pkl'.format(game))

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
