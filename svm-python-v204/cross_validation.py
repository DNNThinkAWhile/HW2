# -*- coding: utf8 -*-
import sys
from subprocess import call

def write_cut_file(filename, parts):
	with open(filename, 'w') as f:
		for lines in parts:
			f.writelines(lines)

def main():
	# args: python cross_validation.py file 5
	if (len(sys.argv) < 4):
		print 'command: cross_validation.py <train_file> <fold> <regularization parameter c>'
                exit()

	filenm = sys.argv[1]
	fold = int(sys.argv[2])
        c = sys.argv[3]


	# read train file
	lines = []
	with open(filenm, 'r') as fff:
		lines = fff.readlines()
	linenum = len(lines)

	for i in xrange(fold):
		print 'fold ', i, ' starting'

		train_fn = 'cv.' + str(i) + '.normalize.train.ark'
		test_fn = 'cv.' + str(i) + '.normalize.test.ark'
		model_fn = 'cv.' + str(i) + '.normalize.model'
		out_fn = 'cv.' + str(i) + '.normalize.out'
                result_fn = 'cv.' + str(i) + '.normalize.result'

		test_size = linenum / fold
		test_st = 0 + i * test_size
		test_end = test_st + test_size

		write_cut_file(test_fn, [lines[test_st: test_end]])
		write_cut_file(train_fn, [lines[0: test_st], lines[test_end:]])
'''
		print 'training ...'
                cmd = '../svm_c/svm_empty_learn -c %s %s %s' %(c, train_fn, model_fn)
                call(cmd, shell=True)
		#call(['./svm_python_learn -c',
                #      c,
		#      train_fn,
		#      model_fn])

		print 'testing ...'
                cmd = '../svm_c/svm_empty_classify %s %s %s' %(test_fn, model_fn, out_fn)
                call(cmd, shell=True)
		#call(['./svm_python_classify',
		#      test_fn,
		#      model_fn,
		#      out_fn])
                print 'formation and trimming ...'
                cmd = 'python trim.py %s %s %s' %(test_fn, out_fn, result_fn)
                call(cmd, shell=True)
		#call(['python trim.py',
		#      test_fn,
		#      out_fn,
		#      result_fn])
   '''     
if __name__ == '__main__':
	main()
