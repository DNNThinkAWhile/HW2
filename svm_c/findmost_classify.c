#include "findmost_classify.h"

#ifndef LENGTH
#define LENGTH  69
#endif
#ifndef WIDTH
#define WIDTH   48
#endif

// W is 1-base
int* inference (double w[],  double x[] ,  int x_length)
{
    double *bigmap = (double*)calloc(x_length*WIDTH, sizeof(double));
    int *backmap = (int*)calloc(x_length*WIDTH, sizeof(int));
    int *path = (int*)calloc(x_length, sizeof(int));
    // Observation array
    double *ob = w + 1;
    // Transition array
    double *tr = ob + LENGTH*WIDTH;
    // Matrix Multiply
    for (int i = 0; i < WIDTH; i ++) {
        for (int j = 0; j < x_length; j ++) {
            double tmp_sum = 0.0;
            for (int i69 = 0; i69 < LENGTH ; i69 ++) {
                tmp_sum += x[j*LENGTH + i69] * ob[i*LENGTH + i69];
            }
            bigmap[WIDTH*j + i] = tmp_sum ;
       }
    }


    // Viterbi
    int MaxEndIndex = 0;
    double MaxEndValue = 0.0;

    for (int j = 1 ; j < x_length; j ++) {
        for (int i = 0 ; i < WIDTH; i ++) {
            double tmpMax = -1 << 20;
            for (int index = 0 ; index < WIDTH ; index ++) {
                double compare = bigmap[WIDTH*(j-1) + index] + tr[index*WIDTH + i];
                if (tmpMax < compare) {
                    tmpMax = compare;
                    backmap[WIDTH*j + i] = index;
                }
            }
            bigmap[WIDTH*j + i] += tmpMax;
            if (j == x_length - 1) {
                if (bigmap[WIDTH*j + i] > MaxEndValue) {
                    MaxEndValue = bigmap[WIDTH*j + i];
                    MaxEndIndex = i;
                }
            }
        }
    }


    // Backtrace
    int maxIdx = MaxEndIndex;
    path[x_length - 1] = maxIdx;
    for (int i = x_length - 1 ; i > 0 ; i --) {
        maxIdx = backmap[WIDTH * i + maxIdx];
        path[i - 1] = maxIdx;
    }

    free(bigmap);
    free(backmap);

    return path;
}

int* find_most_violated (double w[], double x[], int y[], int x_length)
{
    double *bigmap = (double*)calloc(x_length*WIDTH, sizeof(double));
    int *backmap = (int*)calloc(x_length*WIDTH, sizeof(int));
    int *path = (int*)calloc(x_length, sizeof(int));
    int *lossmap = (int*)calloc(x_length*WIDTH, sizeof(int));
    // Observation array
    double *ob = w + 1;
    // Transition array
    double *tr = ob + LENGTH*WIDTH;
    // Matrix Multiply
    for (int i = 0; i < WIDTH; i ++) {
        for (int j = 0; j < x_length; j ++) {
            double tmp_sum = 0.0;
            for (int i69 = 0; i69 < LENGTH ; i69 ++) {
                tmp_sum += x[j*LENGTH + i69] * ob[i*LENGTH + i69];
            }
            lossmap[WIDTH*j + i ] = (i == y[j] ? 0 : 1 );
            bigmap[WIDTH*j + i] = tmp_sum + lossmap[WIDTH*j + i];
       }
    }


    // Viterbi
    int MaxEndIndex = 0;
    double MaxEndValue = 0.0;

    for (int j = 1 ; j < x_length; j ++) {
        for (int i = 0 ; i < WIDTH; i ++) {
            double tmpMax = -1 << 20;
            for (int index = 0 ; index < WIDTH ; index ++) {
                double compare = bigmap[WIDTH*(j-1) + index] + tr[index*WIDTH + i];
                if (tmpMax < compare) {
                    tmpMax = compare;
                    backmap[WIDTH*j + i] = index;
                }
            }
            bigmap[WIDTH*j + i] += tmpMax;
            if (j == x_length - 1) {
                if (bigmap[WIDTH*j + i] > MaxEndValue) {
                    MaxEndValue = bigmap[WIDTH*j + i];
                    MaxEndIndex = i;
                }
            }
        }
    }


    // Backtrace
    int maxIdx = MaxEndIndex;
    path[x_length - 1] = maxIdx;
    for (int i = x_length - 1 ; i > 0 ; i --) {
        maxIdx = backmap[WIDTH * i + maxIdx];
        path[i - 1] = maxIdx;
    }

    free(bigmap);
    free(backmap);

    return path;
}

int main() {

    return 0;
}
