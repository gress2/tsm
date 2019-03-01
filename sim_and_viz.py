from math import sqrt
import glob
import os
import pandas as pd
import subprocess

# want:
# ./build/Release/bin/generic_game_rw -v ./python/pts/varphi_2-4-4-4-1_l0.1_w0.01_b128_e50.pt -c ./cfg/generic_game.toml -s ./models/sd_model.pt

pts = glob.glob('./python/pts/*.pt')
FNULL = open(os.devnull, 'w')

ctr = 0
for pt in pts:
    print('Progress: [{}/{}]'.format(ctr, len(pts)))

    # Run generic game simulation
    args = ['./build/Release/bin/generic_game_rw', '-v', pt, '-c', './cfg/generic_game.toml', '-s', './models/sd_model.pt']
    subprocess.call(args, stdout=FNULL, stderr=subprocess.STDOUT)

    # move .csv data to .pkl
    main_cols = ['mean', 'sd', 'd', 'k', 'varphi2', 'gammas', 'etas']
    main_df = pd.DataFrame(columns=main_cols)
    df_ctr = 0
    tmp_df = pd.DataFrame(columns=main_cols)
    with open('main.generic_game.csv', 'r') as main_f:
        for line in main_f:
            if df_ctr % 1000 == 0:
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
            df_ctr += 1
    if len(tmp_df.index) > 0:
        main_df = main_df.append(tmp_df)
    main_df.to_pickle('main.generic_game.pkl')

    # produce plot and write to .png
    pt_no_dir = pt[13:]
    pt_no_dir = pt_no_dir[:len(pt_no_dir) - 3]
    args = ['./produce_visualizations.py', 'generic', '-o', 'imgs/' + pt_no_dir + '.png']
    subprocess.call(args, stdout=FNULL, stderr=subprocess.STDOUT)
    ctr += 1
