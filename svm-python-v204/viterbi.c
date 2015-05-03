#include<string.h>
#include<stdio.h>
#include<assert.h>
#include<stdlib.h>


#define LENGTH  3
#define WIDTH   2


/*
 * TODO :
 * CUDA style
 * backtrace char*
 */

// Interface between Pythona and C

typedef struct {
    int* data;
    int len;
} Array;


// Cost function
double SimpleDiffCost (char* const A, char* const B)
{
    printf("Input strings: %s", A);
    printf(" and %s\n", B);
    double Diff = 0.0;
    assert(strlen(A) == strlen(B) && "SimpleDiffCost lengh of A != lengh of B");
    for (int i = 0 ; i < strlen(B); i ++) {
        if (A[i] != B[i])
            Diff = Diff + 1.0;
    }
    return Diff;
}


void printArray(double* d, int len) {
    printf("print array:");
    for (int i = 0; i < len; i++) {
        printf("%f ", d[i]);
    }
    puts("");
}

void printArrayI(int* d, int len) {
    printf("print array:");
    for (int i = 0; i < len; i++) {
        printf("%d ", d[i]);
    }
    puts("");
}


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
                tmp_sum += x[j*LENGTH + i69] * ob[WIDTH*i69 + i];
            }
            bigmap[WIDTH*j + i] = tmp_sum;
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
                    printf("set backmap %d, %d\n", WIDTH*j+i, index);
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
        printf("%d, %d\n", WIDTH*i + maxIdx, maxIdx);
        maxIdx = backmap[WIDTH * i + maxIdx];

        path[i - 1] = maxIdx;
    }

    /*for (int i = 0 ; i < LENGTH ; i ++) {
        for (int j = 0 ; j < WIDTH; j ++) {
            printf("%f ",bigmap[LENGTH*i + j]);
        }
        printf("\n");
    }*/

    free(bigmap);
    free(backmap);

    // Interface
    Array PATH;
    PATH.data = path;
    PATH.len = x_length;

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
                tmp_sum += x[j*LENGTH + i69] * ob[WIDTH*i69 + i];
            }
            bigmap[WIDTH*j + i] = tmp_sum;
       }
    }

    // Viterbi
    int MaxEndIndex = 0;
    double MaxEndValue = 0.0;
    for (int i = 0 ; i < WIDTH; i ++) 
        lossmap[i] = (i == y[0] ? 0 : 1);
    
    for (int j = 1 ; j < x_length; j ++) {
        for (int i = 0 ; i < WIDTH; i ++) {
            double tmpMax = -1 << 20;
            for (int index = 0 ; index < WIDTH ; index ++) {
                double compare = bigmap[WIDTH*(j-1) + index] + tr[index*WIDTH + i] + lossmap[WIDTH*(j-1) + index];
                if (tmpMax < compare) {
                    tmpMax = compare;
                    backmap[WIDTH*j + i] = index;
                    lossmap[WIDTH*j + i] = lossmap[WIDTH*(j-1) + index] + (i == y[j] ? 0 : 1);
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
        printf("%d, %d\n", WIDTH*i + maxIdx, maxIdx);
        maxIdx = backmap[WIDTH * i + maxIdx];

        path[i - 1] = maxIdx;
    }

    // for (int i = 0; i < 3; i++) {
    //     printf("%d, %f\n", path[i], bigmap[WIDTH * i + path[i]]);
    // }

    free(bigmap);
    free(backmap);

    return path;
}

int main () {


    double w[] = {
        0.0,
        // obs
        0.1, 0.5,
        0.3, 0.1,
        0.2, 0.2,

        // trans
        0.2, 0.8,
        0.6, 0.4
    };

    double x[] = {
        2, 1, 3,
        1, 3, 1,
        2, 3, 3
    };

    int y[] = {
        1, 1, 1
    };

    // should be 1->0->1
    int* d = inference(w, x, 3);
    int* e = find_most_violated(w, x, y, 3);

    for (int i = 0; i < 3; i++) {
        printf("%d\n", e[i]);
    }


    return 0;
}
