#!/usr/bin/env python3
import argparse
import glob
import os

parser = argparse.ArgumentParser(description='Removes data generated from simulation runs')
parser.add_argument('-p', action='store_true', help='Preserves pickled dataframes (.pkl files)')
parser.add_argument('-c', action='store_true', help='Preserves csvs')

args = parser.parse_args()
should_preserve_pickles = args.p
should_preserve_csvs = args.c

to_be_removed = list()

if not should_preserve_pickles:
    to_be_removed += glob.glob('*.*_game.pkl')
if not should_preserve_csvs:
    to_be_removed += glob.glob('*.*_game.csv')

for file in to_be_removed:
    os.remove(file)