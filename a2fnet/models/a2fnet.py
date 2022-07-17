import torch
import torch.nn as nn
import torch.nn.functional as F

from .convs import Up, Down
from models.ips import pixel_unshuffle_height, pixel_unshuffle_width

class A2FNet(nn.Module):
    def __init__(self,init_weights=True): # , num_classes):
        super(A2FNet, self).__init__()
       
    
  
        
        # encoder
        self.down1 = Down(16,64)
        self.down2 = Down(64,128)
        self.down3 = Down(128,256)
        self.down4 = Down(256,256)
        
        #fc 
        self.fc = nn.Sequential ( 
            nn.Linear(4096,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,4096),
        )
        
        # decoder
        self.up1 = Up(512,128)
        self.up2 = Up(256,64)
        self.up3 = Up(128,16)
        self.up4 = Up(32,16)

        self.output = nn.Sequential (  
            nn.Conv2d(16, 1, (1, 1), stride=(1, 1), padding=(0, 0)),
        )

        
        

        
        if init_weights:
            self._initialize_weights()
   
    def forward(self, x):
        # arrange the input
        x1 = pixel_unshuffle_height(x, 16)


        # encoder
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # fc
        x=torch.flatten(x5,start_dim = 1)
        x = self.fc(x)
        
        #decoder
        x6 = x.view(x5.size())
        x = self.up1(x6,x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)


        #output
        x = self.output(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
