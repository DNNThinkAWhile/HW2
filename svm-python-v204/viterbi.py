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

def viterbi(w, x, y):
    w = np.asarray(w)
    # observation matrix
    ob = w[0:69*48]
    ob = np.reshape(ob,(69,48))

    # transition matrix
    tr = w[69*48:69*48+48*48]
    tr = np.reshape(tr,(48,48))

    # x
    np_x = np.asarray(x)

    seq_length, dim = np_x.shape
    p = np.zeros((48, seq_length)) # propability along the path
    parent = np.ones((48, seq_length))*(-1) # parent for back tracing
    loss = np.ones((48,seq_length)) # for loss

    p = np.matrix(np_x)*np.matrix(ob)
    p = np.transpose(p)
    loss[y[0],0] -= 1

    for l in range(1, seq_length):
        tmp_p = np.tile(p[:,l-1].reshape(48,1), 48) + tr
        tmp_next = np.argmax(tmp_p, 0)
        # for every phone of next layer, find its err(y,ybar)
        loss[y[l],l] -= 1
        loss[:,l] += loss[:,l-1]
        tmp_p += loss[:,l]
        parent[:,l] = np.argmax(tmp_p, 0)
        p[:,l] += np.transpose(np.max(tmp_p, 0))
    # Back Tracing...
    path = backtrace(seq_length, np.argmax(p[:,seq_length-1]), parent)
    return path

def main():
    seq_length = 5
    dim = 69

    w = np.random.random(69*48+48*48)
    x= np.random.random((seq_length, dim))
    y = np.random.randint(0,48,seq_length)
    y_bar = viterbi(w, x, y)

if __name__ == '__main__':
    main()
