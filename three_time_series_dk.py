#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns
import numpy as np

dk_df = pd.read_pickle('dk.same_game.pkl')
ts = dk_df[:218].astype(float)

eol_idx = list(ts.loc[ts['k'] == 0].index)
l_idx = list()

ctr = 0
prev = 0
for e in eol_idx:
    if prev == 0:
        l_idx += [ctr] * (e + 1)
    else:
        l_idx += [ctr] * (e - prev)
    prev = e
    ctr += 1

ts['e'] = pd.Series(l_idx, index=ts.index)

sns.set_style('whitegrid', {'axes.grid' : False})
sns.lineplot(x='d', y='k', hue='e', data=ts, palette=sns.color_palette("Blues_d", n_colors=3), legend=False)
plt.show()
