ls = list()
with open('mixing_data.csv') as md_f:
    lines = md_f.readlines()
    splits = [line.split(',') for line in lines]
    d_prev = 0
    l = list()
    for split in splits:
        d = int(split[2])
        k = int(split[3])
        if d > d_prev:
            if len(l) > 0:
                ls.append(l)
            l = list()
        l.append([d, k])
        d_prev = d

print(ls[0])

dat = list()
for playout in ls:
    for i in range(1, len(playout)):
        delta = playout[i-1][1] - playout[i][1]
        d = playout[i][0]
        k = playout[i][1]
        dat.append([d, k, delta])

with open('d_k_delta.csv', 'w') as dkd_f:
    for elem in dat:
        dkd_f.write(str(elem[0]) + ', ' + str(elem[1]) + ', ' + str(elem[2]) + '\n')
