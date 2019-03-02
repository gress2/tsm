#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def get_game_title(g):
    if g == 'same':
        return 'Samegame'
    elif g == 'generic':
        return 'Generic game'
    else:
        return g

parser = argparse.ArgumentParser(description='Produce side by sides for simulation data')
parser.add_argument('g1', type=str, default='same', help='The first game to be used')
parser.add_argument('g2', type=str, default='generic', help='The second game to be used')
parser.add_argument('-dk', action='store_true', help='Produce only the d vs k plot') 
parser.add_argument('-dv', action='store_true', help='Produce only the d vs varphi2 plot')
parser.add_argument('-kv', action='store_true', help='Produce only the k vs varphi2 plot')
parser.add_argument('-mean', action='store_true', help='Produce only the mean density plot')
parser.add_argument('-sd', action='store_true', help='Produce only the standard deviation density plot')
parser.add_argument('-k', action='store_true', help='Produce only the k density plot')
parser.add_argument('-td', action='store_true', help='Produce only the td density plot')
parser.add_argument('-save', action='store_true', help='Save all produced figures')

args = parser.parse_args()

flag_ct = int(args.dk) + int(args.dv) + int(args.kv) + int(args.mean) + int(args.sd) + int(args.k) + int(args.td)

if flag_ct == 0:
    args.dk = True
    args.dv = True
    args.kv = True
    args.mean = True
    args.sd = True 
    args.k = True
    args.td = True

g1 = args.g1
g2 = args.g2

g1_dfs = dict()
g1_dfs['main'] = pd.read_pickle('main.{}_game.pkl'.format(g1))
g1_dfs['td'] = pd.read_pickle('td.{}_game.pkl'.format(g1))
g1_dfs['dkd'] = pd.read_pickle('dkd.{}_game.pkl'.format(g1))

g2_dfs = dict()
g2_dfs['main'] = pd.read_pickle('main.{}_game.pkl'.format(g2))
g2_dfs['td'] = pd.read_pickle('td.{}_game.pkl'.format(g2))
g2_dfs['dkd'] = pd.read_pickle('dkd.{}_game.pkl'.format(g2))

sns.set(style='whitegrid')

# d vs k
if args.dk:
    fig, axs = plt.subplots(ncols=2, nrows=1)
    sns.scatterplot(x='d', y='k', marker='.', data=g1_dfs['dkd'].astype(float), ax=axs[0])
    sns.scatterplot(x='d', y='k', marker='.', data=g2_dfs['dkd'].astype(float), ax=axs[1])
    axs[0].set_title(get_game_title(g1))
    axs[1].set_title(get_game_title(g2))
    if args.save:
        plt.savefig('side_by_side/d_vs_k.png')
    plt.show()


# d vs varphi
if args.dv:
    fig, axs = plt.subplots(ncols=2, nrows=1)
    g1_dv_df = g1_dfs['main'][['d', 'varphi2']].astype(float)
    g2_dv_df = g2_dfs['main'][['d', 'varphi2']].astype(float)
    sns.scatterplot(x='d', y='varphi2', marker='.', data=g1_dv_df, ax=axs[0])
    sns.scatterplot(x='d', y='varphi2', marker='.', data=g2_dv_df, ax=axs[1])
    axs[0].set_title(get_game_title(g1))
    axs[1].set_title(get_game_title(g2))
    if args.save:
        plt.savefig('side_by_side/d_vs_varphi2.png')
    plt.show()

# k vs varphi
if args.kv:
    fig, axs = plt.subplots(ncols=2, nrows=1)
    g1_kv_df = g1_dfs['main'][['k', 'varphi2']].astype(float)
    g2_kv_df = g2_dfs['main'][['k', 'varphi2']].astype(float)
    sns.scatterplot(x='k', y='varphi2', marker='.', data=g1_kv_df, ax=axs[0])
    sns.scatterplot(x='k', y='varphi2', marker='.', data=g2_kv_df, ax=axs[1])
    axs[0].set_title(get_game_title(g1))
    axs[1].set_title(get_game_title(g2))
    if args.save:
        plt.savefig('side_by_side/k_vs_varphi2.png')
    plt.show()

# mean density
if args.mean:
    fig, axs = plt.subplots(ncols=1, nrows=1)
    g1_mean_df = g1_dfs['main'][['mean']]
    g2_mean_df = g2_dfs['main'][['mean']]
    sns.kdeplot(g1_mean_df['mean'], ax=axs, label=get_game_title(g1), clip=(-7500, 7500))
    sns.kdeplot(g2_mean_df['mean'], ax=axs, label=get_game_title(g2), clip=(-7500, 7500))
    axs.set_title('Mean density')
    if args.save:
        plt.savefig('side_by_side/mean_density.png')
    plt.show()

# standard deviation density
if args.sd:
    fig, axs = plt.subplots(ncols=1, nrows=1)
    g1_sd_df = g1_dfs['main'][['sd']]
    g2_sd_df = g2_dfs['main'][['sd']]
    sns.kdeplot(g1_sd_df['sd'], ax=axs, label=get_game_title(g1), clip=(-100, 7500))
    sns.kdeplot(g2_sd_df['sd'], ax=axs, label=get_game_title(g2), clip=(-100, 7500))
    axs.set_title('Standard deviation density')
    if args.save:
        plt.savefig('side_by_side/sd_density.png')
    plt.show()

# k density
if args.k:
    fig, axs = plt.subplots(ncols=1, nrows=1)
    g1_k_df = g1_dfs['dkd'][['k']]
    g2_k_df = g2_dfs['dkd'][['k']]
    sns.kdeplot(g1_k_df['k'], ax=axs, label=get_game_title(g1))
    sns.kdeplot(g2_k_df['k'], ax=axs, label=get_game_title(g2))
    axs.set_title('k density')
    if args.save:
        plt.savefig('side_by_side/k_density.png')
    plt.show()

# td density
if args.td:
    fig, axs = plt.subplots(ncols=1, nrows=1)
    g1_td_df = g1_dfs['td'][['td']]
    g2_td_df = g2_dfs['td'][['td']]
    sns.kdeplot(g1_td_df['td'], ax=axs, label=get_game_title(g1))
    sns.kdeplot(g2_td_df['td'], ax=axs, label=get_game_title(g2))
    axs.set_title('td density')
    if args.save:
        plt.savefig('side_by_side/td_density.png')
    plt.show()





