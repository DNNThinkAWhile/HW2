import numpy as np
import costfunc
'''
ob:    48 phone

69 dim


tr:    48 phone

48 phone

X:     69 dim

length

P:     length

48 phone

'''
def backtrace(seq_length, tail, parent):
    path = np.ones(seq_length, dtype=np.int)*(-1)
    path[seq_length-1] = tail
    for l in range(seq_length-2, -1, -1):
        path[l] = parent[path[l+1], l+1]
    return path.tolist()

def err(y, l, tmp_next, tmp_path):
    cost = np.zeros((48,1))
    for i in range(48):
        y_bar = list(tmp_path[i])
        y_bar.append(tmp_next[i])
        cost[i] = costfunc.SimpleDiffCost.fn(y[0:l+2], y_bar)
    return cost

def viterbi(w, x, y):
    w = np.asarray(w)
    # observation matrix
    ob = w[0:69*48]
    ob = np.reshape(ob,(69,48))
    obT = ob.transpose()

    # transition matrix
    tr = w[69*48:69*48+48*48]
    tr = np.reshape(tr,(48,48))

    # x
    np_x = np.asarray(x)

    seq_length, dim = np_x.shape
    p = np.zeros((48, seq_length)) # propability along the path
    parent = np.ones((48, seq_length))*(-1) # parent for back tracing

    for s in range(seq_length):
        p[:,s] = np.sum(np.multiply(np_x[s,:], obT), 1)

    tmp_path = [None]*48
    new_tmp_path = [None]*48
    for i in range(48):
        tmp_path[i] = []
        new_tmp_path[i] = []

    for l in range(1, seq_length):
        tmp_p = np.tile(p[:,l-1].reshape(48,1), 48) + tr
        tmp_next = np.argmax(tmp_p, 0)
        # for every phone of next layer, find its err(y,ybar)
        cost = err(y, l, tmp_next, tmp_path)
        tmp_p += cost
        parent[:,l] = np.argmax(tmp_p, 0)
        p[:,l] += np.max(tmp_p, 0)
        for i in range(48):
            new_tmp_path[i] = list(tmp_path[int(parent[i, l])])
            new_tmp_path[i].append(int(parent[i, l]))
        tmp_path = list(new_tmp_path)
    for i in range(48):
        tmp_path[i].append(i)
    # Back Tracing...
    #path = backtrace(seq_length, np.argmax(p[:,seq_length-1]), parent)
    return tmp_path

def main():
    seq_length = 5
    dim = 69

    w = np.random.random(69*48+48*48)
    x= np.random.random((seq_length, dim))
    y = np.random.randint(0,48,seq_length)
    y_bar = viterbi(w, x, y)
#print y_bar

if __name__ == '__main__':
    main()
