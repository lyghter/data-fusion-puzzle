

import sys
import subprocess

if __name__=='__main__':
	print(sys.argv)
	subprocess.call(['bash','run.sh', '--'.join(sys.argv)])





