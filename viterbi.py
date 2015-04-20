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
def viterbi(w, X):
    ob = w[0]
    tr = w[1]
    seq_length, dim = X.shape
    obT = ob.transpose()
    P = np.zeros((48, seq_length))
    parent = np.ones((48, seq_length))*(-1)

    for s in range(seq_length):
        P[:,s] = np.sum(np.multiply(X[s,:], obT), 1)
    log_P = np.log(P)
    prob = log_P
    for l in range(1, seq_length):
        tmp_prob = np.tile(prob[:,l-1].reshape(48,1), 48) + tr
        parent[:,l] = np.argmax(tmp_prob, 0)
        prob[:,l] += np.max(tmp_prob, 0)
    
    # Back Tracing...
    path = np.ones(seq_length)*(-1)
    for l in range(seq_length-1, -1, -1):
        if l == seq_length-1:
            path[l] = np.argmax(prob[:,l])
        else:
            path[l] = parent[path[l+1], l+1]
    return path

def main():
    seq_length = 5
    dim = 69
    ob = np.random.random((69,48))
    tr = np.random.random((48,48))
    w = []
    w.append(ob)
    w.append(tr)

    X = np.random.random((seq_length, dim))
    ybar = viterbi(w, X)

if __name__ == '__main__':
    main()
