import os
from ukko_runner import ukko_runner

def check(n):
	n = str(n)
	n = (3 -len(n))*'0'+n
	return int(os.popen('ssh othe@ukko{0}.hpc.cs.helsinki.fi "ps -u "othe" | grep "python" -c"'.format(n)).read()) == 0

def get_lines(server, user='othe'):
	return os.popen('ssh ' + get_server(server) + ' "ps -u "' + user + '""').readlines()

def machines():
	runner = ukko_runner.runner()
	runner.remove_bad_hosts()
	return runner.machines

def get_server(server):
	try:
		n = str(int(server))
		return 'othe@ukko{0}.hpc.cs.helsinki.fi'.format( (3 -len(n))*'0'+n)
	except ValueError:
		return server

def check_all(program = 'python', user = 'othe'):
	programs = [program]
	mans = machines()
	for machine in mans:
		n = machine[0][4:7]
		lines = get_lines(n, user=user)
		relevant = [lines[0]]
		for line in lines:
			if line.split()[-1] in programs:
				relevant.append(line)
		if len(relevant) > 1:
			print n
			for line in relevant:
				print line[:-1]
			for line in relevant[1:]:
				cmd = raw_input('Kill "' + line.split()[-1] + '" process ' + line.split()[0] + '? (y/n): ')
				if cmd == 'y':
					kill_process(n, line.split()[0])

def kill_process(server, PID, signal=15):
	os.system('ssh ' + get_server(server) + ' "kill -{0} '.format(str(signal)) + str(PID) + '"')


