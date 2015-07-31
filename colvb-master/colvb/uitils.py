
class Aliases():

	def __init__(self):
		self.alias_to_cmd = {}
		self.cmd_to_alias = {}
		self.make_basic_aliases()

	def make_basic_aliases(self):
		alias_setup = [('quit', ['q', 'quit']),
					('help', ['h', 'help']),
					('alias', ['a', 'alias'])
					]
		for cmd in alias_setup:
			self.cmd_to_alias[cmd[0]] = cmd[1]
			for alias in cmd[1]:
				self.alias_to_cmd[alias] = cmd[0]

	def aliases_str(self, cmd):
		return ', '.join(self.cmd_to_alias[cmd])

	def get_aliases(self, cmd):
		return self.cmd_to_alias[cmd]

	def all_aliases_str(self):
		return '\n'.join([cmd + ': ' + self.aliases_str(cmd) for cmd in self.cmd_to_alias]) + '\n'

	def get_all_aliases(self):
		return [alias for sublist in [self.get_aliases(cmd) for cmd in self.cmd_to_alias] for alias in sublist]

	def alias(self, arg):
		if not arg:
			print 'available aliases:\n' + \
				self.all_aliases_str(),
		elif len(arg) == 1:
			cmd = arg[0]
			if cmd in self.cmd_to_alias:
				print cmd + ': (c) ' + self.aliases_str(cmd)
			elif cmd in self.alias_to_cmd:
				print cmd + ': (a) ' + self.alias_to_cmd[cmd]
			else:
				print 'alias not known'
		else:
			self.change_aliases(arg)

	def change_aliases(self, arg):
		cmd = arg[0]
		if cmd == 'add':
			self.add_alias(arg[1:])
		elif cmd == 'remove':
			self.remove_alias(arg[1:])
		elif cmd == 'set':
			self.set_aliases(arg[1:])
		else:
			'invalid command'

	def add_alias(self, arg):
		if len(arg) != 2:
			print 'invalid command'
		else:
			cmd = arg[0]
			alias = arg[1]
			if not cmd in self.cmd_to_alias:
				print 'no command called \'{0}\''.format(cmd)
			elif alias in self.get_all_aliases():
				print 'alias already in use: ' + self.alias_to_cmd[alias]
			else:
				self.cmd_to_alias[cmd].append(alias)
				self.alias_to_cmd[alias] = cmd

	def remove_alias(self, arg):
		if len(arg) != 1:
			print 'invalid command'
		else:
			alias = arg[0]
			if alias in self.cmd_to_alias:
				print 'can\'t remove direct commands'
			elif not alias in self.get_all_aliases():
				print 'no alias called \'{0}\''.format(alias)
			else:
				cmd = self.alias_to_cmd[alias]
				self.alias_to_cmd.pop(alias, None)
				self.cmd_to_alias[cmd].remove(alias)

	def set_aliases(self, arg):
		if len(arg) != 1:
			print 'invalid command'
		else:
			if arg[0] == 'default':
				self.make_basic_aliases()
			else:
				print 'alias setup not known'

	def cmd(self, alias):
		try:
			return self.alias_to_cmd[alias]
		except KeyError:
			return False

	def interpret(self, arg):
		if arg:
			return [self.cmd(arg[0])] + arg[1:]
		else:
			return True

class Commands(Aliases):

	def __init__(self):
		self.commands = ['quit', 'help', 'alias', 'lol']
		Aliases.__init__(self)

	def get_commands(self):
		return self.commands

class Help(Commands):

	def __init__(self):
		Commands.__init__(self)
		self.default_help_string = '== {name} == \n' + \
					'{custom}' + \
					'available aliases: ' + \
					'{aliases}'
		self.help_strings = {'quit': 'quits the program\n', 
				'help': 'manuals for commands\n',
				'alias': 'aliases for commands\n'}

	def default_help(self):
		return '\n== LDA3 - user interface ==\n\n' + \
			'type \'help cmd\' for info about \'cmd\'\n\n' + \
			'list of available commands:\n\n' + \
			'\n'.join(self.get_commands()) + '\n'

	def cmd_help(self, arg):
		if len(arg) > 1:
			return 'give one \'help\' argument at a time'
		cmd = self.cmd(arg[0])
		if cmd:
			if not cmd in self.help_strings:
				custom = 'no additional information for \'{0}\'\n'.format(arg[0])
			else:
				custom = self.help_strings[cmd]
			return self.default_help_string.format(name=cmd, custom=custom, aliases=self.aliases(cmd))
		else:
			return 'no command called \'{0}\''.format(arg[0])

	def help(self, arg):
		if not arg:
			print self.default_help()
		else:
			print self.cmd_help(arg)
