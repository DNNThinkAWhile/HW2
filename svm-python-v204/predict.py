import numpy as np
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

def inference(w, x):
    w = np.asarray(w[1:])    
    # observation matrix
    ob = w[0:69*48] 
    ob = np.reshape(ob,(48,69)).transpose()
    
    # transition matrix
    tr = np.asarray(w[69*48:69*48+48*48])
    tr = np.reshape(tr,(48,48))

    # x
    np_x = np.asarray(x)

    seq_length, dim = np_x.shape
    parent = np.ones((48, seq_length))*(-1) # parent for back tracing

    p = np.matrix(np_x)*np.matrix(ob)
    p = np.transpose(p) # propability along the path
    for l in range(1, seq_length):
        tmp_p = np.tile(p[:,l-1].reshape(48,1), 48) + tr
        parent[:,l] = np.argmax(tmp_p, 0)
        p[:,l] += np.transpose(np.max(tmp_p, 0))
    
    # Back Tracing...
    path = backtrace(seq_length, np.argmax(p[:,seq_length-1]), parent)
    #print 'predict', p[np.argmax(p[:,seq_length-1]), seq_length-1]
    return path

def main():
    seq_length = 5
    dim = 69

    w = np.random.random(69*48+48*48+1)
    x= np.random.random((seq_length, dim))
    ans = inference(w, x)
    print ans

if __name__ == '__main__':
    main()
