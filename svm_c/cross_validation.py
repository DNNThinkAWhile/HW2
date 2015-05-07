# -*- coding: utf8 -*-
import sys
import time
from subprocess import call

def write_cut_file(filename, parts):
    with open(filename, 'w') as f:
        for lines in parts:
            f.writelines(lines)

def main():
    # args: python cross_validation.py file 5
    if (len(sys.argv) < 4):
        print 'command: python cross_validation.py <train_file> <fold> <regularization parameter c>'
        exit()

    filenm = sys.argv[1]
    fold = int(sys.argv[2])
    c = sys.argv[3]

    name = ''
    if len(sys.argv) > 4:
        name = '.' + sys.argv[4]

    # read train file
    lines = []
    with open(filenm, 'r') as fff:
        lines = fff.readlines()
    linenum = len(lines)

    for i in xrange(1):
        print 'fold ', i, ' starting'

        train_fn = 'cv.' + str(i) + name + '.train.ark'
        test_fn = 'cv.' + str(i) + name + '.test.ark'
        model_fn = 'cv.' + str(i) + name + '.model'
        out_fn = 'cv.' + str(i) + name + '.out'
        result_fn = 'cv.' + str(i) + name + '.result'

        test_size = linenum / fold
        test_st = 0 + i * test_size
        test_end = test_st + test_size

        write_cut_file(test_fn, [lines[test_st: test_end]])
        write_cut_file(train_fn, [lines[0: test_st], lines[test_end:]])

        st = time.time()
        print 'training ...'
        cmd = './svm_empty_learn -c %s %s %s' %(c, train_fn, model_fn)
        print 'cmd: ', cmd
        call(cmd, shell=True)
        print 'train finished. time:', time.time() - st, ' seconds'
        #call(['./svm_python_learn -c',
                #      c,
        #      train_fn,
        #      model_fn])

        st = time.time()
        print 'testing ...'
        cmd = './svm_empty_classify %s %s %s' %(test_fn, model_fn, out_fn)
        print 'cmd: ', cmd
        call(cmd, shell=True)
        print 'test finished. time:', time.time() - st, ' seconds'
        #call(['./svm_python_classify',
        #      test_fn,
        #      model_fn,
        #      out_fn])
        print 'formation and trimming ...'
        cmd = 'python trim.py %s %s %s' %(test_fn, out_fn, result_fn)
        print 'cmd: ', cmd
        call(cmd, shell=True)
        #call(['python trim.py',
        #      test_fn,
        #      out_fn,
        #      result_fn])

if __name__ == '__main__':
    main()
