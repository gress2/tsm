#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser(description='Produce visualizations of simulation data')
parser.add_argument('game', type=str, help='The game with which we wish to produce visualizations from')

args = parser.parse_args()

game = args.game

main_df = pd.read_pickle('main.{}_game.pkl'.format(game))
td_df = pd.read_pickle('td.{}_game.pkl'.format(game))
dk_df = pd.read_pickle('dk.{}_game.pkl'.format(game))

# scatter plots
dk = dk_df.loc[:, ['d', 'k']].values
d = dk[:, 0]
k = dk[:, 1]
plt.subplot(3, 3, 1)
plt.xlabel('d')
plt.ylabel('k')
plt.scatter(d, k)

dv = main_df.loc[:, ['d', 'varphi2']].values
d = dv[:, 0]
v = dv[:, 1]
plt.subplot(3, 3, 2)
plt.xlabel('d')
plt.ylabel('v')
plt.scatter(d, v)

kv = main_df.loc[:, ['k', 'varphi2']].values
k = kv[:, 0]
v = kv[:, 1]
plt.subplot(3, 3, 3)
plt.xlabel('k')
plt.ylabel('v')
plt.scatter(k, v)

mean = main_df.loc[:, ['mean']].values
plt.subplot(3, 3, 4)
plt.xlabel('mean')
plt.hist(mean, bins=50, density=True)

sd = main_df.loc[:, ['sd']].values
plt.subplot(3, 3, 5)
plt.xlabel('sd')
plt.hist(sd, bins=50, density=True)

k = dk_df.loc[:, ['k']].values
plt.subplot(3, 3, 6)
plt.xlabel('k')
plt.hist(k, bins=50, density=True)

tdf = td_df.loc[:, ['td', 'freq']].values
td = tdf[:, 0]
freq = tdf[:, 1]
dequantized = td.repeat(freq.astype(int))
dequantized = dequantized.reshape(len(dequantized), 1)
plt.subplot(3, 3, 7)
plt.hist(dequantized, bins=50, density=True)
plt.show()