#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

// Cost function
int SimpleDiffCost (char* const A, char* const B)
{
    printf("Input strings: %s", A);
    printf(" and %s\n", B);
    int Diff = 0;
    assert(strlen(A) == strlen(B) && "SimpleDiffCost lengh of A != lengh of B");
    for (int i = 0 ; i < strlen(B); i ++) {
        if (A[i] != B[i])
            Diff++;
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


#define LENGTH  3//69
#define WIDTH   2//48
// W is 1-base
int* inference (double w[],  double x[] ,  int x_length)
{
    double *bigmap = (double*)calloc(x_length*WIDTH, sizeof(double));
    int *backmap = (int*)calloc(x_length*WIDTH, sizeof(int));
    int *path = (int*)calloc(x_length, sizeof(int));
    // Observation array
    double ob[LENGTH*WIDTH];
    for (int i = 0; i < LENGTH*WIDTH; i ++)
        ob[i] = w[i + 1]; // 1-base
    // Transition array
    double tr[WIDTH*WIDTH];
    for (int i = 0; i < WIDTH*WIDTH ; i ++ )
        tr[i] = w[LENGTH*WIDTH + i + 1]; // 1-base
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

    // for (int i = 0; i < 3; i++) {
    //     printf("%d, %f\n", path[i], bigmap[WIDTH * i + path[i]]);
    // }

    free(bigmap);
    free(backmap);

    return path;
}

char* find_most_violated_constraint ()
{
    return NULL;
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

    // should be 1->0->1
    int* d = inference(w, x, 3);

    for (int i = 0; i < 3; i++) {
        printf("%d\n", d[i]);
    }

    return 0;
}
