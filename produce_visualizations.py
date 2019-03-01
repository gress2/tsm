#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns

parser = argparse.ArgumentParser(description='Produce visualizations of simulation data')
parser.add_argument('game', type=str, help='The game with which we wish to produce visualizations from')
parser.add_argument('-o', type=str, default='', help='The location of the image to save')

args = parser.parse_args()

game = args.game
output = args.o

main_df = pd.read_pickle('main.{}_game.pkl'.format(game))
td_df = pd.read_pickle('td.{}_game.pkl'.format(game))
dkd_df = pd.read_pickle('dkd.{}_game.pkl'.format(game))

sns.set(style='whitegrid')
fig, axs = plt.subplots(ncols=3, nrows=3)

sns.scatterplot(x='d', y='k', marker='.', data=dkd_df.astype(float), ax=axs[0][0])

dv_df = main_df[['d', 'varphi2']].astype(float)
sns.scatterplot(x='d', y='varphi2', marker='.', data=dv_df, ax=axs[0][1])

kv_df = main_df[['k', 'varphi2']].astype(float)
sns.scatterplot(x='k', y='varphi2', marker='.', data=kv_df, ax=axs[0][2])

mean_df = main_df[['mean']]
sns.distplot(mean_df, axlabel='mean', ax=axs[1][0])

sd_df = main_df[['sd']]
sns.distplot(sd_df, axlabel='standard deviation', ax=axs[1][1])

k_df = dkd_df[['k']].astype(float)
print(k_df.max() - k_df.min())
sns.distplot(k_df, axlabel='k', bins=int(k_df.max() - k_df.min()), ax=axs[1][2])

td_df = td_df[['td']].astype(float)
sns.distplot(td_df, bins=int(td_df.max() - td_df.min()), axlabel='terminal depth', ax=axs[2][0])

plt.subplots_adjust(hspace=.30,left=.05,bottom=.05,right=.95,top=.95)

if output != '': 
    plt.savefig(output, dpi=180)
else:
    plt.show()
