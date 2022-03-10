import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock


class Net(nn.Module):
    def __init__(self, init_weights=False):
        super(Net, self).__init__()

        self.main = ResNet(BasicBlock, [2, 2, 2, 2], 1440)
        self.fullConnect1 = nn.Linear(1440, 120)
        self.fullConnect2 = nn.Linear(120, 1)
        self.active = nn.Sigmoid()

        if init_weights:
            self.inited_weight()

    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 6 * 12 * 20)
        x = self.fullConnect1(x)
        x = self.fullConnect2(x)
        return self.active(x)

    def inited_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        self.main.load_state_dict(torch.jit.load('../data/weights/resNet34.pth'))
