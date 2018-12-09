import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import datetime
import pdb
import time
import torchvision.models as torchmodels

class BaseModel(nn.Module):
	def __init__(self):
		super(BaseModel, self).__init__()
		if not os.path.exists('logs'):
			os.makedirs('logs')
		ts = time.time()
		st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H_%M_%S_log.txt')
		self.logFile = open('logs/' + st, 'w')

	def log(self, str):
		print(str)
		self.logFile.write(str + '\n')

	def criterion(self):
		return nn.CrossEntropyLoss()
		#return nn.MSELoss()

	def optimizer(self):
		return optim.SGD(self.parameters(), lr=0.001)

	def adjust_learning_rate(self, optimizer, epoch, args):
		lr = args.lr  # TODO: Implement decreasing learning rate's rules
		if (epoch % 50 == 0):
			lr = lr * 0.9;
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr
	   


class LazyNet(BaseModel):
	def __init__(self):
		super(LazyNet, self).__init__()
		# TODO: Define model here
		self.fc1 = nn.Linear(128 * 128 * 3, 10)

	def forward(self, x):
		# TODO: Implement forward pass for LazyNet
		x = x.view(-1, 128*128*3)
		x = self.fc1(x);
		return x
		

class BoringNet(BaseModel):
	def __init__(self):
		super(BoringNet, self).__init__()
		# TODO: Define model here
		self.fc1 = nn.Linear(256 * 256 * 3, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		# TODO: Implement forward pass for BoringNet
		x = x.view(-1, 256*256*3)
		x = F.relu(self.fc1(x));
		x = F.relu(self.fc2(x));
		x = self.fc3(x);
		return x


class CoolNet(BaseModel):
	def __init__(self):
		super(CoolNet, self).__init__()
		# TODO: Define model here
		self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
		self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

		self.fc1 = nn.Linear(13456, 300)
		self.fc2 = nn.Linear(300, 200)

	def forward(self, x):
		# TODO: Implement forward pass for CoolNet
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2(x), 2))
		
		x = x.view(-1, 13456)
		x = F.relu(self.fc1(x));
		x = self.fc2(x);
		return x
