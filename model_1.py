import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
#         self.x = x
        self.block0 = nn.Sequential(
            # input image 96x96
            nn.ReLU(),
            nn.Conv2d(3, 64, (5, 5), (1, 1), (2, 2)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, (3, 3), (2, 2), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 4, (1, 1), (1, 1)),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(4),
            # image 48x48
        )
        
        self.block1 = nn.Sequential(
            # image 48x48
            nn.PixelShuffle(2)
            # image 96x96           
        )
        
        
        self.fc = nn.Sequential(
#             nn.PixelShuffle(96/89)
            nn.Sigmoid()
        )
            
        
    def forward(self, x):
        x=x.float()
        out = self.block0(x) 
        out = self.block1(out)

        return out
    
    
    def _initialize_weights(self):
        pass