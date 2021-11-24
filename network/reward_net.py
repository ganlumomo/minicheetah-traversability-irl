""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class RewardNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, n_states=6, grid_size=80, feat_out_size=25, regression_hidden_size=64):
        super(RewardNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_states = n_states

        scale = 2
        self.inc = DoubleConv(n_channels, 64 // scale)
        self.down1 = Down(64 // scale, 128 // scale)
        self.down2 = Down(128 // scale, 256 // scale)
        self.down3 = Down(256 // scale, 512 // scale)
        factor = 2 if bilinear else 1
        self.down4 = Down(512 // scale, 1024 // scale // factor)
        self.up1 = Up(1024 // scale, 512 //scale // factor, bilinear)
        self.up2 = Up(512 // scale, 256 // scale // factor, bilinear)
        self.up3 = Up(256 // scale, 128 // scale // factor, bilinear)
        self.up4 = Up(128 // scale, feat_out_size, bilinear)
        self.outc = OutConv(feat_out_size, n_classes)

        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=n_states,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(kernel_size=2,
                         stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64,
                      out_channels=6,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(kernel_size=2,
                         stride=2)
        )
        self.fc = nn.ModuleDict({
                '0': nn.Linear(112, 1),
                '1': nn.Linear(112, 1),
                '2': nn.Linear(112, 1),
                '3': nn.Linear(112, 1),
                '4': nn.Linear(112, 1),
                '5': nn.Linear(112, 1)
        })
        self.upsample = nn.Upsample((grid_size, grid_size))
        
        self.regression_block = nn.Sequential(
            nn.Conv2d(feat_out_size + 2 + n_states, regression_hidden_size, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(regression_hidden_size, regression_hidden_size, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(regression_hidden_size, regression_hidden_size, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(regression_hidden_size, 1, 1)
        )
    

    def forward(self, x, state_x):
        kinematic_in = x[:, self.n_channels:, :, :]
        
        # geometric and semantic feature extraction
        x = x[:, :self.n_channels, :, :]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        #out = self.outc(x)

        # robot state feature extraction
        state_x = state_x[:, :, :self.n_states]
        state_x = state_x.permute(0, 2, 1)
        state_x = self.block1(state_x)
        state_x = self.block2(state_x)

        # feature fusing
        x = torch.cat((x, kinematic_in), dim=1)
        for i in range(self.n_states):
            fc_out = self.fc[str(i)](state_x[:,i,:].view(state_x.shape[0], -1))
            fc_out = self.upsample(torch.unsqueeze(torch.unsqueeze(fc_out, 2), 3))
            x = torch.cat((x, fc_out), dim=1)
        out = self.regression_block(x)

        return out
