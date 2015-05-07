/***********************************************************************/
/*                                                                     */
/*   svm_struct_api.c                                                  */
/*                                                                     */
/*   Definition of API for attaching implementing SVM learning of      */
/*   structures (e.g. parsing, multi-label classification, HMM)        */
/*                                                                     */
/*   Author: Thorsten Joachims                                         */
/*   Date: 03.07.04                                                    */
/*                                                                     */
/*   Copyright (c) 2004  Thorsten Joachims - All rights reserved       */
/*                                                                     */
/*   This software is available for non-commercial use only. It must   */
/*   not be modified and distributed without prior permission of the   */
/*   author. The author is not responsible for implications from the   */
/*   use of this software.                                             */
/*                                                                     */
/***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "svm_struct/svm_struct_common.h"
#include "svm_struct_api.h"
#include "findmost_classify.h"
#include <Python.h>

#ifndef LENGTH
#define LENGTH 69
#endif
#ifndef WIDTH
#define WIDTH 48
#endif

void        svm_struct_learn_api_init(int argc, char* argv[])
{
  /* Called in learning part before anything else is done to allow
     any initializations that might be necessary. */
}

void        svm_struct_learn_api_exit()
{
  /* Called in learning part at the very end to allow any clean-up
     that might be necessary. */
}

void        svm_struct_classify_api_init(int argc, char* argv[])
{
  /* Called in prediction part before anything else is done to allow
     any initializations that might be necessary. */
}

void        svm_struct_classify_api_exit()
{
  /* Called in prediction part at the very end to allow any clean-up
     that might be necessary. */
}



SAMPLE      read_struct_examples(char *file, STRUCT_LEARN_PARM *sparm)
{
  /* Reads struct examples and returns them in sample. The number of
     examples must be written into sample.n */
  SAMPLE   sample;  /* sample */
  EXAMPLE  *examples;

  if (file == NULL) {
    printf("Error: File name = null\n");
    exit(-1);
  }

  FILE* rFILE;
  char* pNAME;
  double data[LENGTH];

  rFILE = fopen(file, "r");

  if (rFILE == NULL) {
    printf("Fail to open file");
    exit(-1);
  }

  //n=100; /* replace by appropriate number of examples */


  Py_Initialize();

  PyObject *module;
  PyObject *readPy, *exPyList;
  PyObject *exIter;
  int r;
  int exNum;

  // need to add the current path to sys path
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("sys.path.append('.')");

  module = PyImport_ImportModule("svm_struct_python");
  readPy = PyObject_GetAttrString(module, "read_examples");

  // exPyList is a python list of examples
  exPyList = PyObject_CallFunction(readPy, "s", file);

  exNum = PyList_Size(exPyList);
  examples=(EXAMPLE *)my_malloc(sizeof(EXAMPLE)*exNum);

  exIter = PyObject_GetIter(exPyList);
  if (!exIter) {
    printf("No iterator?\n");
  }

  // iterate through examples
  PyObject* next = PyIter_Next(exIter);
  for (int i = 0; next; i++, next=PyIter_Next(exIter)) {

    if (!PyTuple_Check(next)) {
      printf("Not tuple\n");
      exit(-1);
    }

    // y is a list of int phonemes
    // x is a list of lists of features
    PyObject *x = PyTuple_GetItem(next, 0);
    PyObject *y = PyTuple_GetItem(next, 1);
    int seqLen = PyList_Size(y);
    EXAMPLE *ex = &examples[i];
    FEATURE *feats = (FEATURE*) calloc(seqLen, sizeof(FEATURE));
    int *seq = (int*) calloc(seqLen, sizeof(int));

    // printf("seqLen %d\n", seqLen);

    // get all pairs of phoneme and feature
    for (int j = 0; j < seqLen; j++) {
      PyObject *phoPy = PyList_GetItem(y, j);
      PyObject *featPy = PyList_GetItem(x, j);
      int phoneme = PyInt_AsLong(phoPy);
      int featSize = PyList_Size(featPy);

      seq[j] = phoneme;
      for (int k = 0; k < featSize; k++) {
        PyObject *featK = PyList_GetItem(featPy, k);
        feats[j].data[k] = PyFloat_AsDouble(featK);
      }
    }

    // print what i read, please don't remove this for debugging
    // for (int m = 0; m < seqLen; m++) {
    //   printf("%d %d ", m, seq[m]);
    //   for (int p = 0; p < 69; p++) {
    //     printf("%f ", feats[m].data[p]);
    //   }
    //   puts("");
    // }

    ex->x.size = seqLen;
    ex->x.features = feats;
    ex->y.size = seqLen;
    ex->y.head = seq;
  }

  Py_DECREF(module);
  Py_DECREF(readPy);
  Py_DECREF(exPyList);
  Py_DECREF(exIter);
  Py_Finalize();


  sample.n=exNum;
  sample.examples=examples;

  return(sample);
}

void        init_struct_model(SAMPLE sample, STRUCTMODEL *sm,
			      STRUCT_LEARN_PARM *sparm, LEARN_PARM *lparm,
			      KERNEL_PARM *kparm)
{
  /* Initialize structmodel sm. The weight vector w does not need to be
     initialized, but you need to provide the maximum size of the
     feature space in sizePsi. This is the maximum number of different
     weights that can be learned. Later, the weight vector w will
     contain the learned weights for the model. */

  sm->sizePsi=LENGTH*WIDTH + WIDTH*WIDTH ; /* replace by appropriate number of features */
}

CONSTSET    init_struct_constraints(SAMPLE sample, STRUCTMODEL *sm,
				    STRUCT_LEARN_PARM *sparm)
{
  /* Initializes the optimization problem. Typically, you do not need
     to change this function, since you want to start with an empty
     set of constraints. However, if for example you have constraints
     that certain weights need to be positive, you might put that in
     here. The constraints are represented as lhs[i]*w >= rhs[i]. lhs
     is an array of feature vectors, rhs is an array of doubles. m is
     the number of constraints. The function returns the initial
     set of constraints. */
  CONSTSET c;
  long     sizePsi=sm->sizePsi;
  long     i;
  WORD     words[2];

  if(1) { /* normal case: start with empty set of constraints */
    c.lhs=NULL;
    c.rhs=NULL;
    c.m=0;
  }
  else { /* add constraints so that all learned weights are
            positive. WARNING: Currently, they are positive only up to
            precision epsilon set by -e. */
    c.lhs=my_malloc(sizeof(DOC *)*sizePsi);
    c.rhs=my_malloc(sizeof(double)*sizePsi);
    for(i=0; i<sizePsi; i++) {
      words[0].wnum=i+1;
      words[0].weight=1.0;
      words[1].wnum=0;
      /* the following slackid is a hack. we will run into problems,
         if we have move than 1000000 slack sets (ie examples) */
      c.lhs[i]=create_example(i,0,1000000+i,1,create_svector(words,"",1.0));
      c.rhs[i]=0.0;
    }
  }
  return(c);
}

LABEL       classify_struct_example(PATTERN x, STRUCTMODEL *sm,
				    STRUCT_LEARN_PARM *sparm)
{
  /* Finds the label yhat for pattern x that scores the highest
     according to the linear evaluation function in sm, especially the
     weights sm.w. The returned label is taken as the prediction of sm
     for the pattern x. The weights correspond to the features defined
     by psi() and range from index 1 to index sm->sizePsi. If the
     function cannot find a label, it shall return an empty label as
     recognized by the function empty_label(y). */
  LABEL y;

  /* insert your code for computing the predicted label y here */

  double *tmp_x = (double*) calloc(x.size * LENGTH, sizeof(double));
  for (int i = 0; i < x.size; i++) {
    for (int j = 0; j < LENGTH; j++) {
      tmp_x[i * LENGTH + j] = x.features[i].data[j];
    }
  }

  y.size = x.size;
  y.head = inference(sm->w, tmp_x, x.size);
  free(tmp_x);
  return(y);
}

LABEL       find_most_violated_constraint_slackrescaling(PATTERN x, LABEL y,
						     STRUCTMODEL *sm,
						     STRUCT_LEARN_PARM *sparm)
{
  /* Finds the label ybar for pattern x that that is responsible for
     the most violated constraint for the slack rescaling
     formulation. For linear slack variables, this is that label ybar
     that maximizes

            argmax_{ybar} loss(y,ybar)*(1-psi(x,y)+psi(x,ybar))

     Note that ybar may be equal to y (i.e. the max is 0), which is
     different from the algorithms described in
     [Tschantaridis/05]. Note that this argmax has to take into
     account the scoring function in sm, especially the weights sm.w,
     as well as the loss function, and whether linear or quadratic
     slacks are used. The weights in sm.w correspond to the features
     defined by psi() and range from index 1 to index
     sm->sizePsi. Most simple is the case of the zero/one loss
     function. For the zero/one loss, this function should return the
     highest scoring label ybar (which may be equal to the correct
     label y), or the second highest scoring label ybar, if
     Psi(x,ybar)>Psi(x,y)-1. If the function cannot find a label, it
     shall return an empty label as recognized by the function
     empty_label(y). */
  LABEL ybar;

  printf("Should not enter scale mode...\n");
  exit(-1);
  /* insert your code for computing the label ybar here */

  return(ybar);
}

LABEL       find_most_violated_constraint_marginrescaling(PATTERN x, LABEL y,
						     STRUCTMODEL *sm,
						     STRUCT_LEARN_PARM *sparm)
{
  /* Finds the label ybar for pattern x that that is responsible for
     the most violated constraint for the margin rescaling
     formulation. For linear slack variables, this is that label ybar
     that maximizes

            argmax_{ybar} loss(y,ybar)+psi(x,ybar)

     Note that ybar may be equal to y (i.e. the max is 0), which is
     different from the algorithms described in
     [Tschantaridis/05]. Note that this argmax has to take into
     account the scoring function in sm, especially the weights sm.w,
     as well as the loss function, and whether linear or quadratic
     slacks are used. The weights in sm.w correspond to the features
     defined by psi() and range from index 1 to index
     sm->sizePsi. Most simple is the case of the zero/one loss
     function. For the zero/one loss, this function should return the
     highest scoring label ybar (which may be equal to the correct
     label y), or the second highest scoring label ybar, if
     Psi(x,ybar)>Psi(x,y)-1. If the function cannot find a label, it
     shall return an empty label as recognized by the function
     empty_label(y). */
  LABEL ybar;
  //printf("Finding most violated constraint...\n");
  double * tmp_x = (double*)calloc(y.size* LENGTH, sizeof(double*));
  for (int i = 0 ; i < y.size ; i ++ ) {
    for ( int j = 0 ; j < LENGTH ; j ++) {
        tmp_x[i*LENGTH + j] = x.features[i].data[j];
    }
  }
  ybar.size = y.size;
  ybar.head = find_most_violated(sm->w, tmp_x, y.head, y.size);

  free(tmp_x);


  /* insert your code for computing the label ybar here */

  return(ybar);
}

int         empty_label(LABEL y)
{
  /* Returns true, if y is an empty label. An empty label might be
     returned by find_most_violated_constraint_???(x, y, sm) if there
     is no incorrect label that can be found for x, or if it is unable
     to label x at all */

  return(y.head==NULL || y.size == 0 ? 1 : 0);
}

SVECTOR     *psi(PATTERN x, LABEL y, STRUCTMODEL *sm,
		 STRUCT_LEARN_PARM *sparm)
{
  /* Returns a feature vector describing the match between pattern x
     and label y. The feature vector is returned as a list of
     SVECTOR's. Each SVECTOR is in a sparse representation of pairs
     <featurenumber:featurevalue>, where the last pair has
     featurenumber 0 as a terminator. Featurenumbers start with 1 and
     end with sizePsi. Featuresnumbers that are not specified default
     to value 0. As mentioned before, psi() actually returns a list of
     SVECTOR's. Each SVECTOR has a field 'factor' and 'next'. 'next'
     specifies the next element in the list, terminated by a NULL
     pointer. The list can be though of as a linear combination of
     vectors, where each vector is weighted by its 'factor'. This
     linear combination of feature vectors is multiplied with the
     learned (kernelized) weight vector to score label y for pattern
     x. Without kernels, there will be one weight in sm.w for each
     feature. Note that psi has to match
     find_most_violated_constraint_???(x, y, sm) and vice versa. In
     particular, find_most_violated_constraint_???(x, y, sm) finds
     that ybar!=y that maximizes psi(x,ybar,sm)*sm.w (where * is the
     inner vector product) and the appropriate function of the
     loss + margin/slack rescaling method. See that paper for details. */
  double *psiArray = (double *)calloc(LENGTH * WIDTH + WIDTH * WIDTH , sizeof(double));

  for (int j = 0 ; j < y.size ; j ++) {
      // Observation
    for (int i = 0 ; i < LENGTH ; i ++)
        psiArray[y.head[j]*LENGTH + i] += x.features[j].data[i];
      // Transition
    if (j > 0)
        psiArray[LENGTH*WIDTH + y.head[j-1]*WIDTH + y.head[j] ] += 1.0;

  }

  SVECTOR* fvec = (SVECTOR*)my_malloc(sizeof(SVECTOR));
  fvec -> words = (WORD*)my_malloc((sm->sizePsi + 1)*sizeof(WORD));

  for (int i = 0 ; i < sm->sizePsi ; i ++) {
    fvec->words[i].wnum = i + 1;
    fvec->words[i].weight = psiArray[i];
  }

  free(psiArray);

  // End fvec
  fvec->words[sm->sizePsi].wnum=0;
  fvec->twonorm_sq = -1;
  fvec->userdefined = NULL;
  fvec->kernel_id = 0;
  fvec->next = NULL;
  fvec->factor=1;

  return(fvec);
}

double      loss(LABEL y, LABEL ybar, STRUCT_LEARN_PARM *sparm)
{
  /* loss for correct label y and predicted label ybar. The loss for
     y==ybar has to be zero. sparm->loss_function is set with the -l option. */
  if(sparm->loss_function == 0) { /* type 0 loss: 0/1 loss */
                                  /* return 0, if y==ybar. return 1 else */
      if (y.size != ybar.size) {
        printf("ysize %d, ybarsize %d\n", y.size, ybar.size);
        exit(-1);
      }
      double error = 0.0;
      for (int i = 0 ; i < y.size ; i ++) {
        if (ybar.head[i] != y.head[i]) return 1;
      }
      return 0;
  }
  else {
      if (y.size != ybar.size) {
          printf("ysize %d, ybarsize %d\n", y.size, ybar.size);
          exit(-1);
      }
      double error = 0.0;
      for (int i = 0 ; i < y.size ; i ++)
          error = error + (y.head[i] == ybar.head[i] ? 0.0 : 1.0 / y.size );
      return error;

    /* Put your code for different loss functions here. But then
       find_most_violated_constraint_???(x, y, sm) has to return the
       highest scoring label with the largest loss. */
  }
}

int         finalize_iteration(double ceps, int cached_constraint,
			       SAMPLE sample, STRUCTMODEL *sm,
			       CONSTSET cset, double *alpha,
			       STRUCT_LEARN_PARM *sparm)
{
  /* This function is called just before the end of each cutting plane iteration. ceps is the amount by which the most violated constraint found in the current iteration was violated. cached_constraint is true if the added constraint was constructed from the cache. If the return value is FALSE, then the algorithm is allowed to terminate. If it is TRUE, the algorithm will keep iterating even if the desired precision sparm->epsilon is already reached. */
  return(0);
}

void        print_struct_learning_stats(SAMPLE sample, STRUCTMODEL *sm,
					CONSTSET cset, double *alpha,
					STRUCT_LEARN_PARM *sparm)
{
  /* This function is called after training and allows final touches to
     the model sm. But primarly it allows computing and printing any
     kind of statistic (e.g. training error) you might want. */
}

void        print_struct_testing_stats(SAMPLE sample, STRUCTMODEL *sm,
				       STRUCT_LEARN_PARM *sparm,
				       STRUCT_TEST_STATS *teststats)
{
  /* This function is called after making all test predictions in
     svm_struct_classify and allows computing and printing any kind of
     evaluation (e.g. precision/recall) you might want. You can use
     the function eval_prediction to accumulate the necessary
     statistics for each prediction. */
}

void        eval_prediction(long exnum, EXAMPLE ex, LABEL ypred,
			    STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm,
			    STRUCT_TEST_STATS *teststats)
{
  /* This function allows you to accumlate statistic for how well the
     predicition matches the labeled example. It is called from
     svm_struct_classify. See also the function
     print_struct_testing_stats. */
  if(exnum == 0) { /* this is the first time the function is
		      called. So initialize the teststats */
  }
}

void        write_struct_model(char *file, STRUCTMODEL *sm,
			       STRUCT_LEARN_PARM *sparm)
{
  /* Writes structural model sm to file file. */
  /* Writes structural model sm to file file. */
  FILE *fMdl;
  long j,i,sv_num;
  MODEL *model=sm->svm_model;
  SVECTOR *v;

  if (!(fMdl = fopen(file, "w"))) {
    perror (file);
    exit (1);
  }
  fprintf(fMdl,"SVM-multiclass Version %s\n",INST_VERSION);
  fprintf(fMdl,"%d # loss function\n", sparm->loss_function);
  fprintf(fMdl,"%ld # kernel type\n", model->kernel_parm.kernel_type);
  fprintf(fMdl,"%ld # kernel parameter -d \n", model->kernel_parm.poly_degree);
  fprintf(fMdl,"%.8g # kernel parameter -g \n", model->kernel_parm.rbf_gamma);
  fprintf(fMdl,"%.8g # kernel parameter -s \n", model->kernel_parm.coef_lin);
  fprintf(fMdl,"%.8g # kernel parameter -r \n", model->kernel_parm.coef_const);
  fprintf(fMdl,"%s# kernel parameter -u \n",model->kernel_parm.custom);
  fprintf(fMdl,"%ld # highest feature index \n",model->totwords);
  fprintf(fMdl,"%ld # number of training documents \n",model->totdoc);
  fprintf(fMdl,"%ld # sizePsi \n", sm->sizePsi);
  int k;
  for (k = 1; k < sm->sizePsi + 1; k++) {
    fprintf(fMdl,"%lf ", sm->w[k]);
  }
  fprintf(fMdl,"# weight vector (index 1 to sizePsi)\n");

  sv_num=1;
  for(i=1;i<model->sv_num;i++) {
    for(v=model->supvec[i]->fvec;v;v=v->next)
      sv_num++;
  }
  fprintf(fMdl,"%ld # number of support vectors plus 1 \n",sv_num);
  fprintf(fMdl,"%.8g # threshold b, each following line is a SV (starting with alpha*y)\n",model->b);

  for(i=1;i<model->sv_num;i++) {
    for(v=model->supvec[i]->fvec;v;v=v->next) {
      fprintf(fMdl,"%.32g ",model->alpha[i]*v->factor);
      fprintf(fMdl,"qid:%ld ",v->kernel_id);
      for (j=0; (v->words[j]).wnum; j++) {
        fprintf(fMdl,"%ld:%.8g ",
              (long)(v->words[j]).wnum,
              (double)(v->words[j]).weight);
      }
      if(v->userdefined)
        fprintf(fMdl,"#%s\n",v->userdefined);
      else
        fprintf(fMdl,"#\n");
    /* NOTE: this could be made more efficient by summing the
       alpha's of identical vectors before writing them to the
       file. */
     }
   }
}

STRUCTMODEL read_struct_model(char *file, STRUCT_LEARN_PARM *sparm)
{
  /* Reads structural model sm from file file. This function is used
     only in the prediction module, not in the learning module. */
  FILE *fMdl;
  STRUCTMODEL sm;
  long i,queryid,slackid;
  double costfactor;
  long max_sv,max_words,ll,wpos;
  char *line,*comment;
  WORD *words;
  char version_buffer[100];
  MODEL *model;

  nol_ll(file,&max_sv,&max_words,&ll); /* scan size of model file */
  max_words+=2;
  ll+=2;

  words = (WORD *)my_malloc(sizeof(WORD)*(max_words+10));
  line = (char *)my_malloc(sizeof(char)*ll);
  model = (MODEL *)my_malloc(sizeof(MODEL));

  if ((fMdl = fopen (file, "r")) == NULL)
  { perror (file); exit (1); }

  fscanf(fMdl,"SVM-multiclass Version %s\n",version_buffer);
  if(strcmp(version_buffer,INST_VERSION)) {
    perror ("Version of model-file does not match version of svm_struct_classify!");
    exit (1);
  }
  // fscanf(fMdl,"%d%*[^\n]\n", &sparm->num_classes);
  // fscanf(fMdl,"%d%*[^\n]\n", &sparm->num_features);
  fscanf(fMdl,"%d%*[^\n]\n", &sparm->loss_function);
  fscanf(fMdl,"%ld%*[^\n]\n", &model->kernel_parm.kernel_type);
  fscanf(fMdl,"%ld%*[^\n]\n", &model->kernel_parm.poly_degree);
  fscanf(fMdl,"%lf%*[^\n]\n", &model->kernel_parm.rbf_gamma);
  fscanf(fMdl,"%lf%*[^\n]\n", &model->kernel_parm.coef_lin);
  fscanf(fMdl,"%lf%*[^\n]\n", &model->kernel_parm.coef_const);
  fscanf(fMdl,"%[^#]%*[^\n]\n", model->kernel_parm.custom);
  fscanf(fMdl,"%ld%*[^\n]\n", &model->totwords);
  fscanf(fMdl,"%ld%*[^\n]\n", &model->totdoc);
  fscanf(fMdl,"%ld%*[^\n]\n", &(sm.sizePsi));
  sm.w = (double *)my_malloc((sm.sizePsi + 1) * sizeof(double));
  double temp = 0.2;
  sm.w[0] = 0.0;
  int k;
  printf("temp:\n");
  for (k = 0; k < sm.sizePsi; k++) {
    fscanf(fMdl,"%lf", &(sm.w[1 + k]));
    printf("%d:%lf ", k, sm.w[1+k]);
  }
  fscanf(fMdl,"%*[^\n]\n");
  fscanf(fMdl,"%ld%*[^\n]\n", &model->sv_num);
  fscanf(fMdl,"%lf%*[^\n]\n", &model->b);

  model->supvec = (DOC **)my_malloc(sizeof(DOC *)*model->sv_num);
  model->alpha = (double *)my_malloc(sizeof(double)*model->sv_num);
  model->index=NULL;
  model->lin_weights=NULL;

  for(i=1;i<model->sv_num;i++) {
    fgets(line,(int)ll,fMdl);
    if(!parse_document(line,words,&(model->alpha[i]),&queryid,&slackid,
           &costfactor,&wpos,max_words,&comment)) {
      printf("\nParsing error while reading model file in SV %ld!\n%s",
       i,line);
      exit(1);
    }
    model->supvec[i] = create_example(-1,0,0,0.0,
              create_svector(words,comment,1.0));
    model->supvec[i]->fvec->kernel_id=queryid;
  }
  fclose(fMdl);
  free(line);
  free(words);
  if(verbosity>=1) {
    fprintf(stdout, " (%d support vectors read) ",(int)(model->sv_num-1));
  }
  sm.svm_model=model;

  printf("~~~~\n");
  return(sm);
}

void        write_label(FILE *fp, LABEL y)
{
  /* Writes label y to file handle fp. */
  int len = y.size;
  fprintf(fp, "[%d", y.head[0]);
  for (int i = 1; i < len; i++) {
    fprintf(fp, ", %d", y.head[i]);
  }
  fprintf(fp, "]\n");
}

void        free_pattern(PATTERN x) {
  /* Frees the memory of x. */
    free(x.features);
}

void        free_label(LABEL y) {
  /* Frees the memory of y. */
    assert(!empty_label(y) && "Error: Freeing empty label...\n");
    free(y.head);
}

void        free_struct_model(STRUCTMODEL sm)
{
  /* Frees the memory of model. */
  /* if(sm.w) free(sm.w); */ /* this is free'd in free_model */
  if(sm.svm_model) free_model(sm.svm_model,1);
  /* add free calls for user defined data here */
}

void        free_struct_sample(SAMPLE s)
{
  /* Frees the memory of sample s. */
  int i;
  for(i=0;i<s.n;i++) {
    free_pattern(s.examples[i].x);
    free_label(s.examples[i].y);
  }
  free(s.examples);
}

void        print_struct_help()
{
  /* Prints a help text that is appended to the common help text of
     svm_struct_learn. */
  printf("         --* string  -> custom parameters that can be adapted for struct\n");
  printf("                        learning. The * can be replaced by any character\n");
  printf("                        and there can be multiple options starting with --.\n");
}

void         parse_struct_parameters(STRUCT_LEARN_PARM *sparm)
{
  /* Parses the command line parameters that start with -- */
  int i;

  for(i=0;(i<sparm->custom_argc) && ((sparm->custom_argv[i])[0] == '-');i++) {
    switch ((sparm->custom_argv[i])[2])
      {
      case 'a': i++; /* strcpy(learn_parm->alphafile,argv[i]); */ break;
      case 'e': i++; /* sparm->epsilon=atof(sparm->custom_argv[i]); */ break;
      case 'k': i++; /* sparm->newconstretrain=atol(sparm->custom_argv[i]); */ break;
      default: printf("\nUnrecognized option %s!\n\n",sparm->custom_argv[i]);
	       exit(0);
      }
  }
}

void        print_struct_help_classify()
{
  /* Prints a help text that is appended to the common help text of
     svm_struct_classify. */
  printf("         --* string -> custom parameters that can be adapted for struct\n");
  printf("                       learning. The * can be replaced by any character\n");
  printf("                       and there can be multiple options starting with --.\n");
}

void         parse_struct_parameters_classify(STRUCT_LEARN_PARM *sparm)
{
  /* Parses the command line parameters that start with -- for the
     classification module */
  int i;

  for(i=0;(i<sparm->custom_argc) && ((sparm->custom_argv[i])[0] == '-');i++) {
    switch ((sparm->custom_argv[i])[2])
      {
      /* case 'x': i++; strcpy(xvalue,sparm->custom_argv[i]); break; */
      default: printf("\nUnrecognized option %s!\n\n",sparm->custom_argv[i]);
	       exit(0);
      }
  }
}

