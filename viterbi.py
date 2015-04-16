import numpy as np
ob = np.zeros(69*48)
tr = np.zeros(48*48)
w.append(ob)
w.append(tr)

X = 

def viterbi(w, seq_length, X):
    ob = w[0]
    tr = w[1]

    P = np.zeros((48, seq_length))
    parent = np.zeros((48, seq_length-1))
    

