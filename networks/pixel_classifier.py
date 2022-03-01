import torch.nn as nn


class PixelClassifier(nn.Module):
    def __init__(self, num_classes, dim):
        super(PixelClassifier, self).__init__()
        # self.lin1 = nn.Linear(dim, 2048)
        # self.act1 = nn.ReLU()
        # self.bn1 = nn.BatchNorm1d(num_features=2048)
        # self.drop1 = nn.Dropout(p=0.3)
        # self.lin2 = nn.Linear(2048, 1024)
        # self.act2 = nn.ReLU()
        # self.bn2 = nn.BatchNorm1d(num_features=1024)
        # self.drop2 = nn.Dropout(p=0.3)
        self.lin3 = nn.Linear(dim, 512)
        self.act3 = nn.ReLU()
        self.bn3 = nn.BatchNorm1d(num_features=512)
        self.drop3 = nn.Dropout(p=0.3)
        self.lin4 = nn.Linear(512, 128)
        self.act4 = nn.ReLU()
        self.bn4 = nn.BatchNorm1d(num_features=128)
        self.drop4 = nn.Dropout(p=0.3)
        self.lin5 = nn.Linear(128, 32)
        self.act5 = nn.ReLU()
        self.bn5 = nn.BatchNorm1d(num_features=32)
        self.lin6 = nn.Linear(32, num_classes)

    def forward(self, x):
        # x = self.lin1(x)
        # x = self.act1(x)
        # x = self.bn1(x)
        # x = self.drop1(x)
        # x = self.lin2(x)
        # x = self.act2(x)
        # x = self.bn2(x)
        # x = self.drop2(x)
        x = self.lin3(x)
        x = self.act3(x)
        x = self.bn3(x)
        x = self.drop3(x)
        x = self.lin4(x)
        x = self.act4(x)
        x = self.bn4(x)
        x = self.drop4(x)
        x = self.lin5(x)
        x = self.act5(x)
        x = self.bn5(x)
        x = self.lin6(x)
        return x

    def init_weights(self, init_type='normal', gain=0.02):
        """
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        """
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)
