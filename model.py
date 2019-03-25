import torch
import torch.nn as nn
import torch.nn.init as init

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		
		self.block0 = nn.Sequential(
			# input image 96x96
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=7, stride=2
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1), # parameters
            # image 48x48
		)
        
        self.block1 = nn.Sequential(           
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1), # parameters
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1), # parameters
			# image 48x48
        )
        
        self.block2 = nn.Sequential(
            # image 48x48
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1), # parameters
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1), # parameters
			# image 48x48
        )
		
		self.block3 = nn.Sequential(
			# image 48x48
			nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=2
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1), # parameters
			# image 24x24
			nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1), # parameters
			# image 24x24
		)
		
		self.side3 = nn.Sequential(
			# image 48x48
			nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=1, stride=2
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1), # parameters
			# image 24x24
		)
		
		self.block4 = nn.Sequential(
			# image 24x24
			nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=2
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1), # parameters
			# image 12x12			
			nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1), # parameters
		)
		
		self.side4 = nn.Sequential(
			# image 24x24
			nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=1, stride=2
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1), # parameters
			# image 12x12
		
		)
		
		self.block5 = nn.Sequential(
			# image 12x12
			nn.ConvTransposed2d(
				in_channels=256, out_channels=128, kernel_size=3, stride=2
			)
			nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1), # parameters
			# image 24x24
			nn.ConvTransposed2d(
				in_channels=128, out_channels=128, kernel_size=3, stride=1
			)
			nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1), # parameters	
		)
		
		self.side5 = nn.Sequential(
			# image 12x12
			nn.ConvTransposed2d(
				in_channels=256, out_channels=128, kernel_size=1, stride=2
			)
			nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1), # parameters
			# image 24x24
		)
		
		self.block6 = nn.Sequential(
			# image 24x24
			nn.ConvTransposed2d(
				in_channels=128, out_channels=64, kernel_size=3, stride=2
			)
			nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1), # parameters
			# image 48x48
			nn.ConvTransposed2d(
				in_channels=64, out_channels=64, kernel_size=3, stride=1
			)
			nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1), # parameters	
		)
		
		self.side6 = nn.Sequential(
			# image 24x24
			nn.ConvTransposed2d(
				in_channels=128, out_channels=64, kernel_size=1, stride=2
			)
			nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1), # parameters
			# image 48x48
		)
		
		self.block7 = nn.Sequential(
			# image 48x48
			nn.ConvTransposed2d(
				in_channels=64, out_channels=32, kernel_size=3, stride=2
			)
			nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1), # parameters
			# image 96x96
		)
        	
        
	def forward(self, x):
        
		out = self.block(x) # save block0 output, 64 channels, 48x48 image
		residual1 = out # save block0 output as residual, 64 channels, 48x48 image
		out = self.block1(out) # run block1, 64 channels, 48x48 image
        out = out + residual # add residuak to output
        
        residual2 = out # 
        out = self.block2(out) # 64 channels, 48x48 image
        out = out + residual2
		
		residual3 = out
		residual3 = self.side3(residual3) # residual need to be conv in order to add
		out = self.block3(out) # 128 channels, 24x24 image
		out = out + residual3
		
		residual4 = out
		residual4 = self.side4(residual4)
		out = self.block4(out)
		out = out + residual4
        
		residual5 = out
		residual5 = self.side5(residual5)
		out = self.block5(out)
		out = out + residual5
		
		residual6 = out
		residual6 = self.side5(residual6)
		out = self.block6(out)
		out = out + residual6
		
		out = self.block7(out) # image 96x96 
		
        out = nn.sigmoid(out) # fully connect sigmoid, image 96x96      
        
		return out
    
    
	def _initialize_weights(self):