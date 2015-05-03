#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <Python.h>

#define LENGTH  69
#define WIDTH   48


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
Array inference (double w[],  double x[] ,  int x_length)
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
    Array pathArr;
    pathArr.data = path;
    pathArr.len = x_length;

    return pathArr;
}

Array find_most_violated (double w[], double x[], int y[], int x_length)
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
            //printf("%d %d \n", i, y[j]);
            lossmap[WIDTH*j + i ] = (i == y[j] ? 0 : 1 );
            bigmap[WIDTH*j + i] = tmp_sum + lossmap[WIDTH*j + i];
       }
    }

    //printArray(lossmap, x_length*WIDTH);

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

    // for (int i = 0; i < 3; i++) {
    //     printf("%d, %f\n", path[i], bigmap[WIDTH * i + path[i]]);
    // }

    free(bigmap);
    free(backmap);

    // Interface
    Array pathArr;
    pathArr.data = path;
    pathArr.len = x_length;

    return pathArr;
}

int main2 () {


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
    // int* d = inference(w, x, 3);
    // int* e = find_most_violated(w, x, y, 3);

    // for (int i = 0; i < 3; i++) {
    //     printf("%d\n", e[i]);
    // }


    return 0;
}



static PyObject* find_most_interface(PyObject* self, PyObject* args) {

    PyObject *wobj; // 5617
    PyObject *xobj;
    PyObject *yobj;
    int seqLen;

    double w[(LENGTH + WIDTH) * WIDTH];
    double *x;
    int *y;
    double* darrs[2];

    // parse arguments
    if (!PyArg_ParseTuple(args, "OOOi", &wobj, &xobj, &yobj, &seqLen)) {
        // error
        printf("arguments type (list, list, list, int)");
        Py_RETURN_NONE;
    }

    x = (double*) calloc(LENGTH * seqLen, sizeof(double));
    y = (double*) calloc(WIDTH * seqLen, sizeof(double));
    darrs[0] = &w;
    darrs[1] = x;

    PyObject *witer = PyObject_GetIter(wobj);
    PyObject *xiter = PyObject_GetIter(xobj);
    PyObject *yiter = PyObject_GetIter(yobj);
    PyObject *iters[2];
    iters[0] = witer;
    iters[1] = xiter;

    if (!(witer && xiter && yiter)) {
        // error not iterator
        printf("Not iterator?\n");
    }

    // w and x
    for (int i = 0; i < 2; i++) {
        PyObject* iter = iters[i];
        double* arr = darrs[i];

        PyObject *next = PyIter_Next(iter);
        for (int j = 0; next; j++, next = PyIter_Next(iter)) {
            if (!PyFloat_Check(next) &&
                !PyInt_Check(next) &&
                !PyLong_Check(next)) {
                // error, we were expecting a floating point value
                printf("Item not float\n");
                Py_RETURN_NONE;
            }
            arr[j] = PyFloat_AsDouble(next);
        }
    }



    PyObject *next = PyIter_Next(yiter);
    for (int j = 0; next; j++, next = PyIter_Next(yiter)) {
        if (!PyInt_Check(next)) {
            // error, we were expecting a floating point value
            printf("Item not float\n");
            Py_RETURN_NONE;
        }
        y[j] = (int) PyInt_AsSsize_t(next);
    }




    //printArrayI(y, seqLen);

    //puts("before find_most_violated");
    Array path = find_most_violated(w, x, y, seqLen);
    //puts("after find_most_violated");

    PyObject *lst = PyList_New(path.len);
    for (int i = 0; i < path.len; i++) {
        PyObject *num = PyInt_FromLong((long)path.data[i]);
        if (!num) {
            Py_DECREF(lst);
            return NULL;
        }
        PyList_SET_ITEM(lst, i, num);   // reference to num stolen
    }

    free(path.data);
    return lst;

    //Py_RETURN_NONE;
}




static PyMethodDef HelloMethods[] = {
    {"find_most_interface", find_most_interface, METH_VARARGS, "intereteete"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initfindmost(void) {
     (void) Py_InitModule("findmost", HelloMethods);
}
