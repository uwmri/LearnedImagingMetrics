from utils.model_components import *
from utils.utils import *

class L2cnnBlock(nn.Module):
    def __init__(self, channels_in=64, channels_out=64, pool_rate=1, bias=False, norm=False,
                 activation=True, train_on_mag=False):
        super(L2cnnBlock, self).__init__()

        if activation:
            if train_on_mag:
                self.act_func = nn.ReLU(inplace=False)
            else:
                #self.act_func = ComplexReLu()
                #self.act_func = CReLU_bias(channels_in)
                #self.act_func = modReLU(channels_in, ndims=ndims)
                self.act_func = SReLU(channels_in)
                #self.act_func = nn.LeakyReLU(negative_slope=0.01, inplace=False)

        else:
            self.act_func = nn.Identity()

        if train_on_mag:
            self.conv1 = nn.Conv2d( channels_in, channels_out, kernel_size=3, padding=1, stride=1, bias=bias)
            self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1, stride=1, bias=bias)
            self.shortcut = nn.Conv2d( channels_in, channels_out, kernel_size=1, padding=0, stride=1, bias=bias)
            self.pool = nn.AvgPool2d(pool_rate)
        else:
            self.conv1 = ComplexConv2d(channels_in, channels_out, kernel_size=3, padding=1, stride=1, bias=bias)
            self.conv2 = ComplexConv2d(channels_out, channels_out, kernel_size=3, padding=1, stride=1, bias=bias)
            self.shortcut = ComplexConv2d(channels_in, channels_out, kernel_size=1, padding=0, stride=1, bias=bias)
            self.pool = ComplexAvgPool(pool_rate)

        self.bn1 = VarNorm2d(channels_out)
        self.bn2 = VarNorm2d(channels_out)

        self.norm = norm

    def forward(self, x):

        shortcut = self.shortcut(x)
        x = self.conv1(x)
        if self.norm:
            x = self.bn1(x)
        x = self.act_func(x.real) + 1j* self.act_func(x.imag)
        x = self.conv2(x)
        if self.norm:
            x = self.bn2(x)
        x = x + shortcut
        x = self.act_func(x.real) + 1j* self.act_func(x.imag)
        x = self.pool(x)
        return x



class L2cnn(nn.Module):
    def __init__(self, channel_base=32, channels_in=1,  channel_scale=1, group_depth=5, bias=False, init_scale=1.0,
                 train_on_mag=False):

        super(L2cnn, self).__init__()
        pool_rate = 2
        channels_out = channel_base
        self.train_on_mag=train_on_mag

        # Connect to output using a linear combination
        self.weight_mse = torch.nn.Parameter(torch.tensor([1.0]).view(1,1))

        self.layers = nn.ModuleList()
        self.dropout = ComplexDropout2D(p=0.5)
        # self.dropout = torch.nn.Dropout(p=0.5)
        count = 1
        for block in range(group_depth):
            self.layers.append(L2cnnBlock(channels_in, channels_out, pool_rate,
                                          bias=bias,
                                          activation=False,
                                          train_on_mag=self.train_on_mag))

            count = count + channels_out

            # Update channels
            channels_in = channels_out
            channels_out = channels_out * channel_scale

        self.f_end = torch.nn.Parameter(torch.ones((1, count)))
        self.final_dropout = torch.nn.Dropout(p=0.5)

    def channel_mse(self, x):
        return torch.sum(torch.sum((torch.abs(x) + 1e-6) ** 2, dim=-1),dim=-1) ** 0.5

    def forward(self, input, truth):

        if self.train_on_mag:
            diff_mag = torch.abs(input - truth)
        else:
            diff_mag = input - truth

        # Convolutional pathway with MSE at multiple scales
        cnn_score = []
        cnn_score.append(self.weight_mse*self.channel_mse(diff_mag))
        for conv_layer in self.layers:
            diff_mag = conv_layer(diff_mag)
            diff_mag = self.dropout(diff_mag)
            cnn_score.append(self.channel_mse(diff_mag))

        # Create a vector of scores at each level
        f = torch.concat(cnn_score, dim=1)

        # Combine multiple levels
        f = f * torch.abs(self.f_end)
        f = self.final_dropout(f)

        # Sum the scores
        score = torch.sum(f, dim=-1)

        return score


class Classifier(nn.Module):

    def __init__(self, rank):
        super(Classifier,self).__init__()

        self.rank = rank

        self.f = nn.Sequential( nn.Linear(1, 16),
                                 nn.Sigmoid(),
                                 nn.Linear(16, 1))

        self.g = nn.Sequential( nn.Linear(1, 16),
                                 nn.Sigmoid(),
                                 nn.Linear(16, 1))


    def forward(self, image1, image2, imaget):

        # Combine the images for batch norm operations
        images_combined = torch.cat([image1, image2], dim=0)    #(batchsize*2, ch=2, 396, 396)
        truth_combined = torch.cat([imaget, imaget], dim=0)

        # Calculate scores
        scores_combined = self.rank(images_combined, truth_combined)
        scores_combined = scores_combined.view(scores_combined.shape[0],-1) #(batchsize*2,1)
        score1 = scores_combined[:image1.shape[0], ...]
        score2 = scores_combined[image1.shape[0]:, ...]

        # Feed difference to classifier
        d = score1 - score2

        # Train based on symetric assumption
        # P-left = f(x)
        # P-same = g(x), a symetric function g(x) = g(-x)
        # P-right = f(-x)
        px = self.f(d)
        gx = self.g(d) + self.g(-d)
        pnx= self.f(-d)

        d = torch.cat([px, gx, pnx],dim=1)
        d = F.softmax(d, dim=1)      # (BatchSize, 3)

        return d, score1, score2
