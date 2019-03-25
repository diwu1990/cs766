import torch
import torch.nn as nn
import torch.nn.init as init

class Net(nn,Module):
	def __init__(self):
		super(Net, self).__init__()
		
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu1 = nn.LeakyReLU(0.1)
		self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
		self.bn2 = nn.BatchNorm2d(64)
		self.relu2 = nn.LeakyReLU(0.1)
		self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
		self.bn3 = nn.BatchNorm2d(64)
		self.relu3 = nn.LeakyReLU(0.1)
		
	def forward(self, x):
		residual = x
		out = self. 
		
	def _initialize_weights(self):