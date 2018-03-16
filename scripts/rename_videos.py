"""
Script for renaming Game 1 Round 1.mov to G1R1.mov
"""
import os
import argparse
import glob
import re

parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str)
args = parser.parse_args()

for video in glob.glob("{}/*".format(args.dir)):
    regex = re.compile('Game (\d+) Round (\d+)\.(.*)')
    file_name = os.path.basename(video)
    match = regex.match(file_name)
    game_id = match.group(1)
    round_id = match.group(2)
    file_type = match.group(3)

    os.rename(video, os.path.join(args.dir, "G{}R{}.{}".format(game_id, round_id, file_type)))
