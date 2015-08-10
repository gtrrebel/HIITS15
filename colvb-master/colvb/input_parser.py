

class input_parser():

	@staticmethod
	def basic_LDA_parse(arg):
		road_gather = []
		end_gather = []
		basic_data = [3, 10, 5, 2]
		nips_data = [5, 10, 10, 10]
		data_type = 'nips'

		j = 1
		while j < len(arg):
			cmd = arg[j]
			j += 1
			if cmd == 'r':
				k = int(arg[j])
				while k > 0:
					k, j = k - 1, j + 1
					road_gather.append(arg[j])
				j += 1
			elif cmd == 'e':
				k = int(arg[j])
				while k > 0:
					k, j = k - 1, j + 1
					end_gather.append(arg[j])
				j += 1
			elif cmd == 'b':
				k = 0
				data_type = 'basic'
				while k < 4:
					basic_data[k] = int(arg[j])
					k, j = k + 1, j + 1
			elif cmd == 'n':
				k = 0
				data_type = 'nips'
				while k < 4:
					nips_data[k] = int(arg[j])
					k, j = k + 1, j + 1
			else:
				print 'oh no', j
				break

		eps = 1e-14
		rest = 10

		return eps, rest, road_gather, end_gather, basic_data, nips_data, data_type

	@staticmethod
	def LDA_parse2(arg):
		ret = [10, 5, 10, 10, 10]
		if len(arg) != 5:
			arg = ['.']*5
		for i in xrange(5):
			if arg[i] != '.':
				ret[i] = int(arg[i])
		return ret[0], ret[1:]

	@staticmethod
	def LDA_parse3(arg):
		ret = [5, 10, 10, 10]
		if len(arg) != 4:
			arg = ['.']*4
		for i in xrange(4):
			if arg[i] != '.':
				ret[i] = int(arg[i])
		return ret

	@staticmethod
	def LDA_map_parse(arg):
		basic_data = [3, 10, 10, 2]
		nips_data = [5, 10, 20, 30]
		data_type = 'nips'
		run_count = 50
		method = 'steepest'

		j = 1
		while j < len(arg):
			cmd = arg[j]
			j += 1
			if cmd == 'b':
				k = 0
				data_type = 'basic'
				while k < 4:
					basic_data[k] = int(arg[j])
					k, j = k + 1, j + 1
			elif cmd == 'n':
				k = 0
				data_type = 'nips'
				while k < 4:
					nips_data[k] = int(arg[j])
					k, j = k + 1, j + 1
			elif cmd == 'r':
				run_count = int(arg[j])
				j += 1
			else:
				print 'oh no', j
				break
		return method, run_count, basic_data, nips_data, data_type
