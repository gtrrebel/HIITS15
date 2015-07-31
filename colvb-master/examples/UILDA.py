import sys
sys.path.append('/home/othe/Desktop/HIIT/HIITS15/colvb-master/colvb')
from database import database
from LDA3 import LDA3
from data_creator import data_creator
from input_parser import input_parser
from uitils import Help

database = database()
help = Help()

def interpret(cmd, arg = None):
	if cmd == 'help':
		help.help(arg)
	if cmd == 'alias':
		help.alias(arg)

while True:
	cmd = help.interpret(raw_input('(h for help, q to quit): ').split())
	if not cmd:
		print 'invalid command'
	else:
		if cmd != True:
			if cmd[0] == 'quit':
				break
			else:
				interpret(cmd[0], cmd[1:])

