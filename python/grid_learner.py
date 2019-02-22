import subprocess

l = [.1, .05, .01, .005, .001, .0005, .0001, .00005, .00001, .000005, .000001, .0000005, .0000001, .00000005, .00000001]
w = [.01, .05, .1, .5]
e = [50]
b = [16, 32, 64, 128]

for _l in l:
    for _w in w:
        for _e in e:
            for _b in b:
                arg_list = ['./learn_varphi_model.py', '-l', str(_l), '-w', str(_w), '-e', str(_e), '-b', str(_b)]
                subprocess.call(arg_list)
