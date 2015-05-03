#include<string.h>
#include<stdio.h>
#include<assert.h>

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

#define LENGTH  69
#define WIDTH   48
// W is 1-base
char* inference (double w[],  double x[] ,  int x_length)
{
    double *bigmap = (double*)malloc(x_length*WIDTH);
    int *backmap = (int*)malloc(x_length*WIDTH);
    char *path = (char*)malloc(x_length);
    // Observation array
    double ob[LENGTH*WIDTH];
    for (int i = 0; i < LENGTH*WIDTH; i ++) 
        ob[i] = w[i + 1]; // 1-base
    // Transition array
    double tr[WIDTH*WIDTH];
    for (int i = 0; i < WIDTH*WIDTH ; i ++ )
        tr[i] = w[LENGTH*WIDTH + i]; // 1-base
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
            double tmpMax = 0.0;
            for (int index = 0 ; index < WIDTH ; index ++) {
                double compare = bigmap[WIDTH*(j-1) + index] + tr[index*WIDTH + i];
                if (tmpMax < compare) {
                    tmpMax = compare;
                    backmap[x_length*j + i] = index;
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
    for (int i = x_length - 1 ; i >= 0 ; i --) {
        // TODO
    }
    free(bigmap);
    free(backmap);
    return path;
}

char* find_most_violated (double w[], double x[], char* y, int x_length)
{
   double *bigmap = (double*)malloc(x_length*WIDTH);
   int *backmap = (int*)malloc(x_length*WIDTH);
   char* path = (char*)malloc(x_length);
   // Ob
   double ob[LENGTH*WIDTH];
   for (int i = 0;i < LENGTH*WIDTH; i ++)
       ob[i] = w[i + 1];
   // Tr
   double tr[WIDTH*WIDTH];
   for (int i = 0; i < WIDTH*WIDTH; i ++)
       tr[i] = w[LENGTH+WIDTH + i];
   // Matrix Mult.
   for (int i = 0 ; i < WIDTH ; i ++) {
       for (int j = 0 ; j < x_length; j ++) {
           double tmp_sum = 0.0;
           for (int i69 = 0 ; i69 < LENGTH ; i69 ++) 
               tmp_sum += x[j*x_length + i69] * ob[WIDTH*i69 + i];
           bigmap[WIDTH*j + i] = tmp_sum;
       }
   }

   // Viterbi
   int MaxEndIndex = 0;
   double MaxEndValue = 0.0;
   for (int j = 1 ; j < x_length ; j ++) {
       for (int i = 0 ; i < WIDTH ; i ++) {
           double tmpMax =0.0;
           for (int index = 0 ; index < WIDTH ; index ++) {
               double compare = bigmap[WIDTH*(j-1) + index] + tr[index*WIDTH + i]; // + SimpleDiffCost()
               if (tmpMax < compare) {
                   tmpMax = compare;
                   backmap[x_length*j + i] = index;
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
   for (int i = x_length - 1 ; i >= 0 ; i ++) {
       // TODO
   }


   free(bigmap);
   free(backmap);
   return path;
}

int main () {

    return 0;
}
