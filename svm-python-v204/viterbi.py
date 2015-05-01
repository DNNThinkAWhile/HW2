import numpy as np
import costfunc
import random as rd
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
    #np.set_printoptions(threshold=np.nan)
    w = np.asarray(w[1:])
    # observation matrix
    ob = w[0:69*48]
    ob = np.reshape(ob,(48,69)).transpose()

    # transition matrix
    tr = w[69*48:69*48+48*48]
    tr = np.reshape(tr,(48,48))

    # x
    np_x = np.asarray(x)

    seq_length, dim = np_x.shape
    parent = np.ones((48, seq_length))*(-1) # parent for back tracing
    loss = np.ones((48,seq_length)) # for loss
    acc_loss = np.ones((48,seq_length))
    p = np.matrix(np_x)*np.matrix(ob)
    p = np.transpose(p)
    loss[y[0],0] -= 1
    p[:,0] += np.reshape(loss[:,0],(48,1))
    acc_loss[y[0],0] -= 1
    #print 'p', p



    for l in range(1, seq_length):
        tmp_p = np.tile(p[:,l-1].reshape(48,1), 48) + tr
        # for every phone of next layer, find its err(y,ybar)
        loss[y[l],l] -= 1
        acc_loss[y[0],0] -= 1
        acc_loss[:,l] += acc_loss[:,l-1]
        #tmp_p += loss[:,l]

        #print l, 'tmp_p', tmp_p

        parent[:,l] = np.argmax(tmp_p, 0)
        p[:,l] += np.reshape(loss[:,l],(48,1))
        p[:,l] += np.transpose(np.max(tmp_p, 0))
    # Back Tracing...



    #print 'parent', parent
    path = backtrace(seq_length, np.argmax(p[:,seq_length-1]), parent)
    #print 'viterbi loss', acc_loss[np.argmax(p[:,seq_length-1]),(seq_length-1)]
    #print 'viterbi cost', p[np.argmax(p[:,seq_length-1]),(seq_length-1)]
    return path


def main():
    seq_length = 400
    dim = 69

    w = np.random.random(69*48+48*48+1)
    x= np.random.random((seq_length, dim))
    y = np.random.randint(0,48,seq_length)

    for i in xrange(50):
        y_bar = viterbi(w, x, y)
        print y_bar

if __name__ == '__main__':
    main()
