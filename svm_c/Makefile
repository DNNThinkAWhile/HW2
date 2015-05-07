# Makefile for empty SVM-struct API, 03.10.06

#Call 'make' using the following line to make CYGWIN produce stand-alone Windows executables
#		make 'SFLAGS=-mno-cygwin'

#Use the following to compile under unix or cygwin
CC = gcc
LD = gcc

CFLAGS =   $(SFLAGS) -O3 -fomit-frame-pointer -ffast-math -Wall -std=c11
LDFLAGS =  $(SFLAGS) -O3 -lm -Wall
#CFLAGS =  $(SFLAGS) -pg -Wall -g
#LDFLAGS = $(SFLAGS) -pg -lm -Wall

all: svm_empty_learn svm_empty_classify

.PHONY: clean
clean: svm_light_clean svm_struct_clean
	rm -f *.o *.tcov *.d core gmon.out *.stackdump





#-----------------------#
#----   Python dec  ----#
#-----------------------#

PYTHON := python
# Happily, thanks to the built-in distutils module, the Python
# interpreter is able to return many variables relating to its
# installation including where to look for the library, the library
# name, include files, and the special compiler directives that allow
# for dynamic loading of modules in programs that embed Python!
define PYTHON_COMMAND
-c "from distutils import sysconfig; print sysconfig.get_config_var('R')"
endef

PYTHON_LIB_DIR := $(shell $(PYTHON) $(subst R,LIBPL,$(PYTHON_COMMAND)))
PYTHON_LIB := $(basename $(patsubst lib%,%,$(shell $(PYTHON) $(subst R,LIBRARY,$(PYTHON_COMMAND)))))
PYTHON_INCLUDE := $(shell $(PYTHON) $(subst R,INCLUDEPY,$(PYTHON_COMMAND)))
PYTHON_SHARED := $(shell $(PYTHON) $(subst R,LINKFORSHARED,$(PYTHON_COMMAND)))
# Make it work on OS X with the Apple supplied python.
PYTHON_SHARED := $(PYTHON_SHARED:Python.framework/Versions/%/Python=-framework Python)
PYTHON_OTHER_LIBS := $(shell $(PYTHON) $(subst R,LIBS,$(PYTHON_COMMAND)))
PYTHON_LD_FLAGS := $(PYTHON_SHARED) -L$(PYTHON_LIB_DIR) -l$(PYTHON_LIB) $(PYTHON_OTHER_LIBS)
PYTHON_CC_FLAGS := -I$(PYTHON_INCLUDE)




#-----------------------#
#----   SVM-light   ----#
#-----------------------#
svm_light_hideo_noexe:
	cd svm_light; make svm_learn_hideo_noexe

svm_light_clean:
	cd svm_light; make clean

#----------------------#
#----  STRUCT SVM  ----#
#----------------------#

svm_struct_noexe:
	cd svm_struct; make svm_struct_noexe

svm_struct_clean:
	cd svm_struct; make clean


#-------------------------#
#----  SVM empty API  ----#
#-------------------------#

svm_empty_classify: svm_light_hideo_noexe svm_struct_noexe svm_struct_api.o svm_struct/svm_struct_classify.o svm_struct/svm_struct_common.o svm_struct/svm_struct_main.o findmost_classify.o
	$(LD) $(LDFLAGS) $(PYTHON_LD_FLAGS) svm_struct_api.o svm_struct/svm_struct_classify.o svm_light/svm_common.o svm_struct/svm_struct_common.o findmost_classify.o -o svm_empty_classify $(LIBS) -lpthread

svm_empty_learn: svm_light_hideo_noexe svm_struct_noexe svm_struct_api.o svm_struct_learn_custom.o svm_struct/svm_struct_learn.o svm_struct/svm_struct_common.o svm_struct/svm_struct_main.o findmost_classify.o
	$(LD) $(LDFLAGS) $(PYTHON_LD_FLAGS) svm_struct/svm_struct_learn.o svm_struct_learn_custom.o svm_struct_api.o svm_light/svm_hideo.o svm_light/svm_learn.o svm_light/svm_common.o svm_struct/svm_struct_common.o svm_struct/svm_struct_main.o findmost_classify.o viterbi.o -o svm_empty_learn $(LIBS) -v -lpthread

findmost_classify.o: findmost_classify.c findmost_classify.h
	$(CC) -c $(CFLAGS) findmost_classify.c -o findmost_classify.o
viterbi.o: viterbi.c viterbi.h
	$(CC) -c $(CFLAGS) viterbi.c -o viterbi.o -lpthread

svm_struct_api.o: svm_struct_api.c svm_struct_api.h svm_struct_api_types.h svm_struct/svm_struct_common.h
	$(CC) -c $(CFLAGS) svm_struct_api.c -o svm_struct_api.o $(PYTHON_CC_FLAGS)

svm_struct_learn_custom.o: svm_struct_learn_custom.c svm_struct_api.h svm_light/svm_common.h svm_struct_api_types.h svm_struct/svm_struct_common.h
	$(CC) -c $(CFLAGS) svm_struct_learn_custom.c -o svm_struct_learn_custom.o
