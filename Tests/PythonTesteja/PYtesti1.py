import time
import sys

count = int(sys.argv[1])

for i in range(count):
	time.sleep(1)
	print i + 1
