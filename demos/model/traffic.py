import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

import torch.optim as optim

import torchvision
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR

import random


import ns.distributedml as dml
import time


from model import TaskBase

"""
    Build Neural network
"""
from torch.autograd import Variable

class args:
	input_dim = 1
	hidden_dim = 32
	num_layers = 2
	out_dim = 1
	lr = 0.05
	gpu = False
	file_path = './data/demo_data.h5'
	window_size = 5
	test_days = 7




class LSTM(nn.Module):
	def __init__(self):
		super(LSTM, self).__init__()
		self.criterion = nn.MSELoss()
		self.input_dim = args.input_dim
		self.hidden_dim = args.hidden_dim
		self.out_dim = args.out_dim
		self.num_layers = args.num_layers
		self.device = 'cuda' if args.gpu else 'cpu'

		self.lstm_layer = nn.LSTM(input_size=self.input_dim,
		                          hidden_size=self.hidden_dim,
		                          num_layers=self.num_layers, batch_first=True)

		self.linear_layer = nn.Linear(self.hidden_dim, self.out_dim)

	def forward(self, x):
		bz = x.size(0)
		h0 = Variable(torch.zeros(self.num_layers * 1, bz, self.hidden_dim)).to(self.device)
		c0 = Variable(torch.zeros(self.num_layers * 1, bz, self.hidden_dim)).to(self.device)

		self.lstm_layer.flatten_parameters()
		lstm_out, hn = self.lstm_layer(x, (h0, c0))
		y_pred = self.linear_layer(lstm_out[:, -1, :])
		return y_pred




class AirTask(TaskBase):
	def __init__(self, global_rank=0, global_size=1, log="tf_", \
						local_epochs=1, \
						active_ratio=1, \
						sleeping_time=0, \
						noise_ratio=0, \
						noise_type="add", \
						batch_size=8, \
						part_ratio=[1,1,1,1]):

		super(AirTask, self).__init__(log=log)

		self.global_rank = global_rank
		self.global_size = global_size
		self.batch_size = batch_size
		self.local_epochs = local_epochs

		self.active_ratio = active_ratio
		self.sleeping_time = sleeping_time
		self.noise_ratio = noise_ratio
		self.noise_type = noise_type

		self.part_ratio = part_ratio

		self.model = LSTM()
		
		self.initialize()

		
	def __noise(self):
		if self.noise_type == "add":
			return self.add_noise(self.noise_ratio)
		elif self.noise_type == "multi":
			return self.multi_noise(self.noise_ratio)
		else:
			print("Currently we only implemented add-noise and multi_noise, uses can implement their own noises by themself.")



	def get_dataset(self):
		df = pd.read_csv(args.file_path, header=0, index_col=0)
		df.fillna(0.0, inplace=True)

		train_cells = df.columns[0: self.global_size*3]
		df_train_cells = df[train_cells]


		test_cells = df.columns[-4:]
		df_test_cells = df[test_cells]


		train_data = df_train_cells.iloc[:]
		test_data = df_test_cells.iloc[:]

		# normalize the data to zero mean and unit deviation using only train data
		mean_train = train_data.mean(axis=0)
		std_train = train_data.std(axis=0)
		train_data = (train_data - mean_train) / std_train

		mean_test = test_data.mean(axis=0)
		std_test = test_data.std(axis=0)
		test_data = (test_data - mean_test) / std_test

		train_x, train_y = [], []
		for cell in train_cells:
			cell_data = train_data.loc[:, cell]
			x, y = self.get_data(cell_data)
			train_x.append(x)
			train_y.append(y)
		
		train_x, train_y = torch.cat(train_x, dim=0), torch.cat(train_y, dim=0)

		test_x, test_y = [], []
		for cell in test_cells:
			cell_data = test_data.loc[:, cell]
			x, y = self.get_data(cell_data)
			test_x.append(x)
			test_y.append(y)
		
		test_x, test_y = torch.cat(test_x, dim=0), torch.cat(test_y, dim=0)
		

		train_dataset = list(zip(train_x, train_y))
		test_dataset = list(zip(test_x, test_y))

		return train_dataset, test_dataset


	def get_data(self, dataset):
		train_shifts = [dataset.shift(i) for i in range(1 - args.out_dim, args.window_size + 1, 1)]

		df_train = pd.concat(train_shifts, axis=1, ignore_index=True)
		df_train.dropna(inplace=True)

		x, y = df_train.iloc[:, args.out_dim:].values[:, :, np.newaxis], df_train.iloc[:, :args.out_dim].values

		X = torch.from_numpy(x).type(torch.Tensor)
		Y = torch.from_numpy(y).type(torch.Tensor)

		
		return X, Y


	def initialize(self):

		train_dataset, test_dataset = self.get_dataset()

		if self.rank>0 and self.world_size>=1:
			train_dataset = self.uniform_partition(train_dataset, self.global_size)[self.global_rank]

			# train_dataset = self.partition(train_dataset, self.global_size, part_ratio=self.part_ratio)[self.global_rank]


		train_kwargs = {'batch_size': self.batch_size, 'drop_last': True}

		self.train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)

		if self.rank == 0:
			self.test_loader = torch.utils.data.DataLoader(test_dataset, **train_kwargs)

				
		self.optimizer = optim.Adadelta(self.model.parameters(), lr=args.lr*self.active_ratio*self.global_size)
		self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.9)
	


	def train(self):
		self.__noise()
		self.model.train()
		for i in range(self.local_epochs):
			for batch_idx, (data, target) in enumerate(self.train_loader):
				self.optimizer.zero_grad()
				output = self.model(data)

				loss = self.model.criterion(output, target)
				loss.backward()
				
				self.optimizer.step()

			self.scheduler.step()

		self.global_step += 1		


	def evaluate(self):
		self.model.eval()
		test_loss, mse = 0.0, 0.0
		with torch.no_grad():
			for batch_idx, (data, target) in enumerate(self.test_loader):
				output = self.model(data)
				test_loss += self.model.criterion(output, target).item()  # sum up batch loss
				mse += torch.sum((output - target) ** 2).item()


			test_loss = test_loss / (batch_idx+1)
			mse = mse / (batch_idx+1)
		
		
		if self.rank == 0:
			print("writing into tb_writer... curret time: {}, wall-clock: {}".format(dml.PyTimer.now("s"), self.wall_clock))
			self.tb_writer.add_scalar('loss', test_loss, self.global_step)
			self.tb_writer.add_scalar('mse', mse, self.global_step)
			
			with open(self.output, 'a+') as f:
				out_str = "EVAL:: epoch: {} curret time: {}, wall-clock: {}, loss: {}, mse: {}\n".format(self.global_step, dml.PyTimer.now("s"), self.wall_clock, test_loss, mse)
				print(out_str)
				f.write(out_str)

			self.tb_writer.flush()

		# with open(self.output, 'a+') as f:
		# 	out_str = "EVAL:: epoch: {} curret time: {}, wall-clock: {}, loss: {}, mse: {}\n".format(self.global_step, dml.PyTimer.now("s"), self.wall_clock, test_loss, mse)
		# 	print(out_str)
		# 	f.write(out_str)
		self.global_step += 1

	



		

	