from scipy import ndimage

import torch
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.models.mobilenet import _make_divisible, ConvBNReLU, InvertedResidual

from efficientnet_pytorch import EfficientNet

from utils.utils import *



class conv_bn(nn.Module):
    '''conv ->bn + shortcut -> relu'''

    def __init__(self, Nkernels=64, BN=True):
        super(conv_bn, self).__init__()
        self.conv = nn.Conv2d(Nkernels, Nkernels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(Nkernels)
        self.relu = nn.ReLU(inplace=True)
        self.BN = BN

    def forward(self, x):
        identity = x
        x = self.conv(x)
        if self.BN:
            x = self.norm(x)

        x = self.relu(x)
        return x

class L2cnn(nn.Module):

    # ResNet for 2 channel.

    def __init__(self, channel_base = 8, channel_scale = 2):

        super(L2cnn, self).__init__()

        channels = channel_base
        self.block1 = nn.Sequential(nn.Conv2d( 1, channels, kernel_size=3, padding=1, stride=1),
                                    nn.BatchNorm2d(channels),
                                    nn.ReLU(inplace=True),
                                    nn.AvgPool2d(2))

        self.block2 = nn.Sequential(nn.Conv2d(channels, channels*channel_scale, kernel_size=3, padding=1, stride=1),
                                    nn.BatchNorm2d(channels*channel_scale),
                                    nn.ReLU(inplace=True),
                                    nn.AvgPool2d(2))

        self.block3 = nn.Sequential(nn.Conv2d(channels*channel_scale, channels*channel_scale**2, kernel_size=3, padding=1, stride=1),
                                    nn.BatchNorm2d(channels*channel_scale**2),
                                    nn.ReLU(inplace=True),
                                    nn.AvgPool2d(2))

        self.block4 = nn.Sequential(nn.Conv2d(channels*channel_scale**2, channels*channel_scale**3, kernel_size=3, padding=1, stride=1),
                                    nn.BatchNorm2d(channels*channel_scale**3),
                                    nn.ReLU(inplace=True),
                                    nn.AvgPool2d(2))
    def forward(self, x):
        x = x**2
        # x = torch.square(x)
        x = torch.sum(x, dim=-3, keepdim=True)
        x = torch.sqrt(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = torch.reshape(x,(x.shape[0],-1))
        score = torch.sum(x, dim=1)
        return score


class ResNet2(nn.Module):

    # ResNet for 2 channel.

    def __init__(self, block, layers, num_classes=1, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, for_denoise=False):
        super(ResNet2, self).__init__()
        self.for_denoise = for_denoise
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(2, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # hard code for now. For 396*396 images, output of last block should be 25*25*512
        # self.fc = nn.Linear(512 * 25 * 25, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        print(x.shape)
        x = self.layer1(x)
        print(x.shape)
        x = self.layer2(x)
        print(x.shape)
        x = self.layer3(x)
        print(x.shape)
        x = self.layer4(x)
        print((x.shape))
        x = self.avgpool(x)
        print(x.shape)
        if self.for_denoise:
            return x
        else:
            x = torch.flatten(x, 1)
            x = self.fc(x)

            return x

# # lost precision while converting between array and pil?
# transform = transforms.Compose([
#      transforms.ToPILImage(),
#      transforms.RandomHorizontalFlip(p=1),
#      transforms.ToTensor(),
#      ])


class MobileNetV2_2chan(nn.Module):
    def __init__(self,
                 num_classes=1,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet

        """
        super(MobileNetV2_2chan, self).__init__()

        if block is None:
            block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(2, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


class DataGenerator_rank(Dataset):
    def __init__(self, X_1, X_2, Y, ID, flip_prob, trans_prob, rot_prob,  roll_magL=1, roll_magH=15,
                 crop_sizeL=1, crop_sizeH=15):

        '''
        :param X_1: X_1_cnnT/V
        :param X_2: X_2_cnnT/V
        :param Y: labels
        :param transform
        :param flip_prob: prob of getting flipped (50% horizontal/vertical)
        '''
        self.X_1 = X_1
        self.X_2 = X_2
        self.Y = Y
        self.ID = ID
        self.flip_prob = flip_prob

        self.trans_prob = trans_prob
        self.roll_magL = roll_magL
        self.roll_magH = roll_magH

        self.rot_prob = rot_prob
        self.crop_sizeL = crop_sizeL
        self.crop_sizeH = crop_sizeH

    def __len__(self):
        return len(self.ID)

    def __getitem__(self, idx):

        FLIP = np.ndarray.item(np.random.choice([False, True], size=1, p=[1 - self.flip_prob, self.flip_prob]))
        ROLL = np.ndarray.item(np.random.choice([False, True], size=1, p=[1 - self.trans_prob, self.trans_prob]))
        ROT = np.ndarray.item(np.random.choice([False, True], size=1, p=[1 - self.rot_prob, self.rot_prob]))

        IDnum = self.ID[idx]

        x1 = self.X_1[IDnum,...]
        x2 = self.X_2[IDnum,...]

        if ROT:
            angle = np.random.randint(1,359)
            x1 = ndimage.rotate(x1, angle, (1, 2), mode='constant', cval=0.0, reshape=False)
            x2 = ndimage.rotate(x2, angle, (1, 2), mode='constant', cval=0.0, reshape=False)

        x1 = torch.from_numpy(x1)
        x2 = torch.from_numpy(x2)


        if FLIP:
            flip_axis = np.ndarray.item(np.random.choice([0, 1], size=1, p=[0.5, 0.5]))
            x1 = torch.flip(x1, dims=[flip_axis, ])
            x2 = torch.flip(x2, dims=[flip_axis, ])

        if ROLL:
            roll_magLR = np.random.randint(self.roll_magL,self.roll_magH)
            roll_magUD = np.random.randint(self.roll_magL, self.roll_magH)

            x1 = torch.roll(x1, (roll_magLR,roll_magUD),(1,2))
            x2 = torch.roll(x1, (roll_magLR,roll_magUD),(1,2))

        # make images 3 channel.
        zeros = torch.zeros(((1,) + x1.shape[1:]), dtype=x1.dtype)
        x1 = torch.cat((x1, zeros), dim=0)
        x2 = torch.cat((x2, zeros), dim=0)

        y = self.Y[idx]

        return x1, x2, y
        # return x1, x2, y, TRANS, CROP


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier,self).__init__()
        # self.rank = ResNet2(BasicBlock, [1,1,1,1])
        # self.rank = mobilenet_v2(pretrained=False, num_classes=1)
        # self.rank = EfficientNet.from_pretrained('efficientnet-b0', num_classes=1)
        self.rank = L2cnn()
        self.relu6 = nn.ReLU6(inplace=True)
        self.fc1 = nn.Linear(1,8)
        self.fc2 = nn.Linear(8,3)
        self.drop = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, image1,image2, trainOnMSE=False):

        if trainOnMSE:
            d = torch.sum((torch.abs(image1-image2)**2),dim=(1,2,3))/(image1.shape[1]*image1.shape[2]*image1.shape[3])
            d = torch.unsqueeze(d, 1)
            #print(d.shape)

        else:
            score1 = self.rank(image1)
            score1 = score1 * self.relu6(score1+3)/6

            score2 = self.rank(image2)
            score2 = score2 * self.relu6(score2 + 3) / 6

            score1 = score1.view(score1.shape[0], -1)
            score2 = score2.view(score2.shape[0], -1)
            # print(f'shape of score2 after reshape {score2.shape}')
            d = score1 - score2
        # d shape [BatchSize, 1]

        d = self.fc1(d)
        d = self.relu(d)
        d = self.drop(d)
        d = F.softmax(self.fc2(d), dim=1)      # (BatchSize, 3)
        return d






