#!/usr/bin/env python3
import argparse

parser = argparse.ArgumentParser(description='Removes data generated from simulation runs')
parser.add_argument('-a', action='store_true', help='Remove all archived data')
parser.add_argument('-p', action='store_true', help='Preserves pickled dataframes (.pkl files)')
parser.add_argument('-c', action='store_true', help='Preserves csvs')

args = parser.parse_args()
should_remove_archives = args.a
should_preserve_pickles = args.p
should_preserve_csvs = args.c