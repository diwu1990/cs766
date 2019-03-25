import torch
import torch.nn as nn
import torch.nn.init as init

class Net(nn,Module):
	def __init__(self):
		super(Net, self).__init__()
        
        self.block1 = nn.Sequential(
            # input image 96x96
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=7, stride=2
            )
            # image 48x48
            nn.BatchNorm2d(64)
            nn.LeakyReLU(0.1) # parameters
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1
            )
            nn.BatchNorm2d(64)
            nn.LeakyReLU(0.1) # parameters
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1
            )
            nn.BatchNorm2d(64)
            nn.LeakyReLU(0.1) # parameters
        )
        
        self.block2 = nn.Sequential(
            # image 48x48
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1
            )
            nn.BatchNorm2d(64)
            nn.LeakyReLU(0.1) # parameters
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1
            )
            nn.BatchNorm2d(64)
            nn.LeakyReLU(0.1) # parameters
        )
        
        # TODO: implement all blocks
        
        
		
        
	def forward(self, x):
        
		residual1 = x # save input as residual
		out = self.block1(x) # run block1
        out = out + residual # add residuak to output
        
        residual2 = out # save block1 output as new residual
        out = sefl.block2(out)
        out = out + residual2
        
        # TODO: implement all block forward
        out = F.sigmoid(out) # fully connect sigmoid, image 96x96      
        
		return out
    
    
	def _initialize_weights(self):