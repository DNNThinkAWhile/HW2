import numpy as np
import sys

class EditDistanceCost:
    # working matrix
    wk_mat = np.zeros((512, 512))
    weights = (1, 1, 1) # insertion, deletion, substitution

    @staticmethod
    def resize_wk_mat(m, n):
        EditDistanceCost.wk_mat = np.zeros((m, n))

    @staticmethod
    def set_weights(w_ins = 1, w_del = 1, w_subs = 1):
        EditDistanceCost.weights = (w_ins, w_del, w_subs)

    @staticmethod
    def fn(str1, str2):
        clazz = EditDistanceCost
        wk_mat = clazz.wk_mat
        w = clazz.weights

        m = len(str1)
        n = len(str2)

        if m + 1 > wk_mat.shape[0] or \
           n + 1 > wk_mat.shape[1]:
            clazz.resize_wk_mat(m + 1, n + 1)
            wk_mat = clazz.wk_mat

        # initialization
        for i in range(m + 1):
            wk_mat[i][0] = i * w[1] # deletion
        for j in range(n + 1):
            wk_mat[0][j] = j * w[0] # insertion

        # flood-fill
        for j in range(1, n + 1):
            for i in range(1, m + 1):
                if str1[i - 1] == str2[j - 1]:
                    wk_mat[i][j] = wk_mat[i - 1][j - 1]
                else:
                    wk_mat[i][j] = min(
                        wk_mat[i - 1, j] + w[1], # deletion
                        wk_mat[i, j - 1] + w[0], # insertion
                        wk_mat[i - 1, j - 1] + w[2] # substitution
                        )

        return wk_mat[m][n]

class SimpleDiffCost:
    @staticmethod
    def fn(str1, str2):
        return sum([c1 == c2 for c1,c2 in zip(str1, str2)])

def main():
    if len(sys.argv) != 3:
        print 'need 2 arguments: str1 str2'
        sys.exit(-1)

    str1 = sys.argv[1]
    str2 = sys.argv[2]

    cost_clz = EditDistanceCost

    # set the weights for insertion, deletion, substitution
    # default: 1, 1, 1
    cost_clz.set_weights(1, 1, 1) # ins, del, subs
    print 'new weights:', cost_clz.weights

    # usage
    cost = cost_clz.fn(str1, str2)
    print 'EditDistance between "', str1, '" and "', str2, '" is ', cost

    # usage 2
    l1 = [1,2,3,4,5,6]
    l2 = [2,4,5,6,7]
    cost = cost_clz.fn(l1, l2)
    print 'EditDistance between ', l1, ' and ', l2, ' is ', cost

    # usage 2
    l1 = [1,2,3]
    l2 = [1,2]
    cost = cost_clz.fn(l1, l2)
    print 'EditDistance between ', l1, ' and ', l2, ' is ', cost

if __name__ == '__main__':
    main()
