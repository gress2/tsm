import glob
import os
import subprocess

# want:
# ./build/Release/bin/generic_game_rw -v ./python/pts/varphi_2-4-4-4-1_l0.1_w0.01_b128_e50.pt -c ./cfg/generic_game.toml -s ./models/sd_model.pt

pts = glob.glob('./python/pts/*.pt')
FNULL = open(os.devnull, 'w')

ctr = 0
for pt in pts:
    print('Progress: [{}/{}]'.format(ctr, len(pts)))
    args = ['./build/Release/bin/generic_game_rw', '-v', pt, '-c', './cfg/generic_game.toml', '-s', './models/sd_model.pt']
    subprocess.call(args, stdout=FNULL, stderr=subprocess.STDOUT)
    pt_no_dir = pt[13:]
    pt_no_dir = pt_no_dir[:len(pt_no_dir) - 3]
    args = ['./produce_visualizations.py', 'generic', '-o', 'imgs/' + pt_no_dir + '.png']
    subprocess.call(args, stdout=FNULL, stderr=subprocess.STDOUT)
    ctr += 1
