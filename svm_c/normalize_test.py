import numpy as np
import sys
lll = None

file_ori = sys.argv[1]
file_stat = sys.argv[2]
file_out = sys.argv[3]
stat = np.load(file_stat)

print 'normalize ', file_ori, ' according to stats file ', file_stat, ', output to ', file_out

with open(sys.argv[1], 'r') as f:
    ids = []
    lstrs = []
    lines = []
    features = []

    lstrs = f.readlines()
    feat_num = len(lstrs[0].split()) - 1
    print 'features num:', feat_num

    for i in xrange(feat_num):
        features.append([])
    for l in lstrs:
        line = l.strip().split(' ')
        ids.append(line[0])
        line = line[1:]
        line = [float(x) for x in line]
        lines.append(line)


        for i in xrange(feat_num):
            features[i].append(line[i])

    means = stat[:,0]
    stds = stat[:,1]

    # for i in range(39):
    #     arr = np.asarray(features[i])
    #     print 'feature ' + str(i)
    #     print 'max:' + str(arr.max())
    #     print 'min:' + str(arr.min())
    #     means[i] = arr.mean()
    #     print 'mean:' + str(means[i])
    #     stds[i] = arr.std()
    #     print 'std:' + str(stds[i])

    with open(file_out, 'w') as f1:
        for i in xrange(len(ids)):
            f1.write(ids[i])
            for j in xrange(feat_num):
                f1.write(' ' + str((lines[i][j] - means[j]) / stds[j]) )
            f1.write('\n')





