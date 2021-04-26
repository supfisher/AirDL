from base import Mpi
import time
import os
import shutil
from tensorboardX import SummaryWriter
import torch


class TaskBase:
	def __init__(self, model=None, log="tf_"):
		self.sys_time_begin = time.time()

		self.global_step = 0
		self.logging_steps = 100

		self.model = model
		self.rank = Mpi.rank
		self.world_size = max(Mpi.world_size - 1, 1)  #this word size does not include the server

		if self.rank==0:
			if os.path.exists(log):
				shutil.rmtree(log)

			self.tb_writer = SummaryWriter(log)
			self.output = os.path.join(log, "time-acc-loss.txt")
	
	# Currently, we're not able to use std::vector<uint64_t> as python input.
	# To avolocal_rank this awkward thing, we use the std::vector<uint32_t> as input, while
	# adjusting it in the python input.
	def long2int(self, longlist):
		intlist = []    
		for l in longlist:
			intlist.append(l%(2**32))
			intlist.append(l//(2**32))
		return intlist

	def addrs(self):
		addr_list = []
		sizes = []
		for params in self.model.parameters():
			addr_list.append(params.data.data_ptr())
			sizes.append(params.data.view(-1).shape[0])

		return self.long2int(addr_list), sizes
	
	@property
	def wall_clock(self):
		return '{} s'.format(time.time() - self.sys_time_begin)

	def get_parameter_number(self):
		total_num = sum(p.numel() for p in self.model.parameters())
		trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
		return {'Total': total_num, 'Trainable': trainable_num}

	def load_state(self, weight):
		self.model.load_state_dict(weight)

	def get_state(self):
		return self.model.state_dict()

	def uniform_partition(self, dataset, size):
		seq_list = [int(len(dataset)/size) for _ in range(size-1)]
		seq_list.append(len(dataset) - sum(seq_list))

		datasets = torch.utils.data.random_split(dataset, seq_list, generator=torch.Generator().manual_seed(42))
		return datasets

	def partition(self, dataset, size, part_ratio=[1,1,1,1]):
		if len(part_ratio) < size:
			part_ratio.extend([0]*(size-len(part_ratio)))
		else:
			part_ratio = part_ratio[:size]

		seq_list = [int(len(dataset)/sum(part_ratio)*part_ratio[i]) for i, _ in enumerate(range(len(part_ratio)))]

		seq_list.append(len(dataset)-sum(seq_list))
		datasets = torch.utils.data.random_split(dataset, seq_list, generator=torch.Generator().manual_seed(42))
		return datasets

	def add_noise(self, noise_ratio=1e-5):
		f = lambda x: x + torch.randn_like(x)*noise_ratio
		new_weight = {name: f(value) for name, value in self.get_state().items()}
		self.load_state(new_weight)

	def multi_noise(self, noise_ratio=1e-5):
		f = lambda x: x*(1+torch.randn_like(x)*noise_ratio)
		new_weight = {name: f(value) for name, value in self.get_state().items()}
		self.load_state(new_weight)
