import numpy as np
import sys

def main(argv):
	if (len(argv) < 3):
		print 'save_stat.py stat.txt stat.npy'
		return


	with open(argv[1], 'r') as f:

		lines = f.readlines()
		stat = []
		for i in range(len(lines) / 2):
			idx = int(lines[i*2])
			stats = lines[i*2+1].strip().split(' ')
			mean = float((stats[0]))
			std = float(stats[1])
			stat.append([mean, std])

		print stat
		statarr = np.array(stat)
		np.save(argv[2], statarr)



if __name__=='__main__':
	main(sys.argv)
