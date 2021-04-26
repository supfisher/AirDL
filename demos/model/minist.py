import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import torch.optim as optim

import torchvision
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR

import random
import os

from tensorboardX import SummaryWriter
import shutil

import ns.distributedml as dml
import time

from model import TaskBase


"""
    Build Neural network
"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output



class AirTask(TaskBase):
	def __init__(self, global_rank=0, global_size=1, log="tf_", \
						local_epochs=1, \
						batch_size=128, \
						active_ratio=1, \
						sleeping_time=0, \
						noise_ratio=0, \
						noise_type="add", \
						part_ratio=[1,1,1,1]
						):
		
		super(AirTask, self).__init__(log=log)

		self.global_rank = global_rank
		self.global_size = global_size
		self.local_epochs = local_epochs
		self.batch_size = batch_size
		self.active_ratio = active_ratio
		self.sleeping_time = sleeping_time
		self.noise_ratio = noise_ratio
		self.noise_type = noise_type
		self.part_ratio = part_ratio

		self.model = Net()

		
		self.initialize()

	

	def get_dataset(self):
		transform=transforms.Compose([
	        transforms.ToTensor(),
	        transforms.Normalize((0.1307,), (0.3081,))
	    ])
		dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
		dataset, test_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)], generator=torch.Generator().manual_seed(42))

		return dataset, test_dataset


	def initialize(self):

		train_dataset, test_dataset = self.get_dataset()

		if self.rank>0 and self.world_size>=1:
			train_dataset = self.uniform_partition(train_dataset, self.global_size)[self.global_rank]

		train_kwargs = {'batch_size': self.batch_size, 'drop_last': True}

		self.train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)

		self.test_loader = torch.utils.data.DataLoader(test_dataset, **train_kwargs)


		self.optimizer = optim.Adadelta(self.model.parameters(), lr=1*self.active_ratio)
		self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.9)
	

	def __noise(self):
		if self.noise_type == "add":
			return self.add_noise(self.noise_ratio)
		elif self.noise_type == "multi":
			return self.multi_noise(self.noise_ratio)
		else:
			print("Currently we only implemented add-noise and multi_noise, uses can implement their own noises by themself.")

	def train(self):
		self.__noise()
		self.model.train()
		logging_steps  = 10
		for i in range(self.local_epochs):
			for batch_idx, (data, target) in enumerate(self.train_loader):
				self.optimizer.zero_grad()
				output = self.model(data)

				loss = F.nll_loss(output, target)
				loss.backward()
				
				self.optimizer.step()	

				# if batch_idx % logging_steps == 0:
				# 	print('TRAIN:: rank: {}\t global_rank: {}\t epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
				#             self.rank, self.global_rank, self.global_step, batch_idx * len(data), len(self.train_loader.dataset),
				#             100. * batch_idx / len(self.train_loader), loss.item()))
				# 	break
				
			self.scheduler.step()
		
		self.global_step += 1		
			

	def evaluate(self):
		self.model.eval()
		test_loss = 0
		correct = 0
		with torch.no_grad():
			for batch_idx, (data, target) in enumerate(self.test_loader):
				output = self.model(data)
				test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
				pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
				correct += pred.eq(target.view_as(pred)).sum().item()

				# if batch_idx % self.logging_steps == 0:
				# 	print('EVAL:: epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \t Acc: {:.6f}'.format(
				#             self.global_step, batch_idx * len(data), len(self.test_loader.dataset),
				#             100. * batch_idx / len(self.test_loader), test_loss/((batch_idx+1)*len(data)), correct/((batch_idx+1)*len(data))))
					# break

			test_loss = test_loss/len(self.test_loader.dataset)
			acc = correct / len(self.test_loader.dataset)

		if self.rank == 0:
			print("writing into tb_writer... curret time: {}, wall-clock: {}".format(dml.PyTimer.now("s"), self.wall_clock))
			self.tb_writer.add_scalar('loss', test_loss, self.global_step)
			self.tb_writer.add_scalar('acc', acc, self.global_step)
			
			with open(self.output, 'a+') as f:
				out_str = "EVAL:: epoch: {} curret time: {}, wall-clock: {}, loss: {}, acc: {}\n".format(self.global_step, dml.PyTimer.now("s"), self.wall_clock, test_loss, acc)
				print(out_str)
				f.write(out_str)


				self.tb_writer.flush()
		

		# with open(self.output, 'a+') as f:
		# 	out_str = "EVAL:: epoch: {} curret time: {}, wall-clock: {}, loss: {}, acc: {}\n".format(self.global_step, dml.PyTimer.now("s"), self.wall_clock, 0, 0)
		# 	print(out_str)
		# 	f.write(out_str)
			
		self.global_step += 1



