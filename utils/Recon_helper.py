import pickle
import torch
import torchvision
import torchsummary
from utils.model_helper import *
from utils.CreateImagePairs import get_smaps, add_gaussian_noise
from utils.unet_componets import *


class DataGenerator_recon(Dataset):
    def __init__(self,IMG, KSP, KSP_Full):
        '''
        :param IMG: truth image
        :param KSP: undersampled kspace
        :param KSP_Full: full kspace for smaps

        output: undersampled kspace (rectangular), truth image (square), smaps (square)
        '''
        self.KSP = KSP      # Input kspace needs to be (coil, h, w)
        self.KSP_Full = KSP_Full
        self.IMG = IMG
        self.len = KSP.shape[0]
        self.height = KSP.shape[-3]
        self.width = KSP.shape[-2]
        self.smaps = np.zeros((KSP_Full.shape[0:2]+(self.height, self.width,)+(2,)), dtype=np.float32)

        #print(self.smaps.shape)

        for i in range(KSP_Full.shape[0]):
            ksp = KSP_Full[i,:,:,:,0] + 1j*KSP_Full[i,:,:,:,1]
            kspacesl = sp.to_device(ksp, device=sp.Device(0))       # (20,768, 396)

            mps = get_smaps(kspacesl, device=sp.Device(0), maxiter=50, method='jsense')      # complex64
            #print(mps.shape)

            # temp = np.zeros(mps.shape+(2,),dtype=np.float32)
            # temp[...,0] = np.real(mps)
            # temp[..., 1] = np.imag(mps)

            # mps_crop = mps[:,int(self.width*.5):int(self.width*1.5),:]
            self.smaps[i] = complex_2chan(mps)  # because pytorch generator doesn't like complex

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        kspace = torch.from_numpy(self.KSP[idx,...])
        image = torch.from_numpy(self.IMG[idx,...])

        smapssl = self.smaps[idx, ...]      # (coil, h, w, 2)

        # if self.transform:

        return kspace, image, smapssl


class DataGeneratorRecon(Dataset):
    def __init__(self,path_root, scan_list, h5file, mask_sl):
        '''
        input: mask (768, 396) complex64
        output: all complex numpy array
                fully sampled kspace (rectangular), truth image (square), smaps (rectangular)
        '''

        # scan path+file name
        with open(os.path.join(path_root, scan_list), 'rb') as tf:
            self.scans = pickle.load(tf)
        self.hf = h5py.File(name=os.path.join(path_root,h5file), mode='r')

        # undersampling mask
        self.mask_sl = mask_sl

        # iterate over scans
        self.len = len(self.hf)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        kspace = self.hf[self.scans[idx]]['kspace']
        kspace = kspace[:]                      # array
        kspace = zero_pad4D(kspace)             # array (sl, coil, 768, 396)

        mask = np.broadcast_to(self.mask_sl,self.mask_sl.shape)
        kspace *= mask

        kspace = complex_2chan(kspace)  # (coil, h, w, 2)
        kspace = torch.from_numpy(kspace)


        smaps = self.hf[self.scans[idx]]['smaps']
        smaps = complex_2chan(smaps)        # (coil, h, w, 2)

        truth = self.hf[self.scans[idx]]['truths']
        truth = truth[:]
        truth = complex_2chan(truth)
        truth = torch.from_numpy(truth)

        return kspace, truth, smaps


class DataGeneratorDenoise(Dataset):
    def __init__(self,path_root, scan_list, h5file):

        # scan path+file name
        with open(os.path.join(path_root, scan_list), 'rb') as tf:
            self.scans = pickle.load(tf)
        self.hf = h5py.File(name=os.path.join(path_root,h5file), mode='r')

        # iterate over scans
        self.len = len(self.hf)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        truth = self.hf[self.scans[idx]]['truths']
        truth = truth[:]        #(Nslice, 396,396) complex64
        noisy = np.zeros(truth.shape, dtype=truth.dtype)
        for i in range(truth.shape[0]):
            noisy[i,...] = add_gaussian_noise(truth[i,...], prob=1, level=1e4, mode=0, mean=0)
            # plt.imshow(np.abs(noisy[i,...]))
            # plt.show()

        truth = complex_2chan(truth)
        truth = torch.from_numpy(truth)
        truth = truth.permute(0,-1,1,2)

        noisy = complex_2chan(noisy)
        noisy = torch.from_numpy(noisy)
        noisy = noisy.permute(0,-1,1,2)

        return truth, noisy

# plot a training/val set
def sneakpeek(dataset, Ncoils=20):

    idx = np.random.randint(len(dataset))
    checkkspace, checktruth, checksmaps = dataset[idx]
    Nslice = checktruth.shape[0]
    slice_num = np.random.randint(Nslice)
    coil_num = np.random.randint(20)

    checksmaps = chan2_complex(checksmaps)
    checksmaps = checksmaps[slice_num]
    checktruth = chan2_complex(checktruth.numpy())
    checkkspace = chan2_complex(checkkspace.numpy())



    plt.imshow(np.abs(checktruth[slice_num]), cmap='gray')
    plt.show()
    plt.imshow(np.log(np.abs(checkkspace[slice_num, coil_num, ...])+1e-5), cmap='gray')
    plt.show()

    plt.figure(figsize=(10, 10))
    for m in range(int(np.ceil(Ncoils / 4))):
        for n in range(4):
            plt.subplot(int(np.ceil(Ncoils / 4)), 4, (m * 4 + n + 1))
            plt.imshow(np.abs(checksmaps[(m * 4 + n), :, :]), cmap='gray')
            plt.axis('off')
    plt.show()


# MSE loss
def mseloss_fcn(output, target):
    loss = torch.mean((output - target) ** 2)
    return loss

# TODO: how do i save encoder during training
def loss_fcn_onenet(noisy, output, target, projector, encoder, discriminator, discriminator_l, lam1=1, lam2=1, lam3=1,
                    lam4=-1, lam5=-1):
    loss1 = lam1*torch.mean((target - projector(target)) ** 2)
    loss2 = lam2*torch.mean((target - output) ** 2)
    loss3 = lam3*torch.mean((noisy - output) ** 2)

    cross_entropy = nn.CrossEntropyLoss()
    loss4 = lam4 * cross_entropy(discriminator_l(encoder(target)), discriminator(encoder(noisy)))
    loss5 = lam5*cross_entropy(discriminator(target), discriminator(output))

    return loss1+loss2+loss3+loss4+loss5


# learned metrics loss
def learnedloss_fcn(output, target, scoreModel):
    output = output.permute(0,-1,1,2)
    target = target.permute(0, -1, 1, 2)

    Nslice = output.shape[0]
    # add a zero channel since ranknet expect 3chan
    zeros = torch.zeros(((Nslice,1,) + output.shape[2:]), dtype=output.dtype)
    zeros = zeros.cuda()
    output = torch.cat((output, zeros), dim=1)
    target = torch.cat((target, zeros), dim=1)      # (batch=Nslice, 3, 396, 396)

    # output = output.squeeze()
    # target = target.squeeze()   # (3, 396, 396)

    delta = torch.mean((scoreModel((output - target)) - scoreModel(target)).abs_())
    # loss_fcn = nn.CrossEntropyLoss()
    # loss = loss_fcn(delta, labels)  # labels are 1 (same)

    return delta


# perceptual loss
class PerceptualLoss_VGG16(torch.nn.Module):
    def __init__(self):
        super(PerceptualLoss_VGG16, self).__init__()

        # ReLU_2
        self.net = torchvision.models.vgg16(pretrained=True).features[:9].eval()
        # blocks = []
        # blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        # blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        # blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        # # blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        # for bl in blocks:
        #     for p in bl:
        #         p.requires_grad = False
        # self.blocks = torch.nn.ModuleList(blocks)
        for param in self.net.parameters():
            param.requires_grad = False

        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))


    def forward(self, input, target):
        # input and target both (Slice, 396, 396, 2)
        # make the magnitude as the 3rd channel
        inputabs = torch.sqrt(input[:,:,:,0]**2 +input[:,:,:,1]**2)
        inputabs = torch.unsqueeze(inputabs, dim=3)
        input = torch.cat((input, inputabs), dim=3)
        shape3 = input.shape

        targetabs = torch.sqrt(target[:,:,:,0]**2 +target[:,:,:,1]**2)
        targetabs = torch.unsqueeze(targetabs, dim=3)
        target = torch.cat((target, targetabs), dim=3)

        # normalize to (0,1)
        input = input.view(input.shape[0], -1)
        input -= input.min(1, keepdim=True)[0]
        input /= input.max(1, keepdim=True)[0]
        input = input.view(shape3)

        target = target.view(target.shape[0], -1)
        target -= target.min(1, keepdim=True)[0]
        target /= target.max(1, keepdim=True)[0]
        target = target.view(shape3)

        input = input.permute(0, -1, 1, 2)      # to (slice, 3, 396, 396)
        target = target.permute(0, -1, 1, 2)

        self.mean.cuda()
        self.std.cuda()
        self.net.cuda()

        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std

        # # mean of perceptual loss from j-th ReLU
        # for block in self.blocks:
        #     input = block(input)
        #     target = block(target)
        #     norm = input.shape[1] * input.shape[2] * input.shape[3]
        #     loss = ((input - target)**2)/norm
        #
        #     loss += loss
        # loss /= len(self.blocks)

        loss = (self.net(input) - self.net(target))**2
        norm = loss.shape[1] * loss.shape[2] * loss.shape[3]
        loss /= norm

        loss = torch.sum(loss.contiguous().view(loss.shape[0], -1), -1)     # shape (NSlice)

        return torch.mean(loss)


# patch GAN loss
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()

        # no need to use bias as BatchNorm2d has affine parameters

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):

        return self.model(input)


def loss_GAN(input, target, discriminator):

    Nslice = target.shape[0]
    input = input.permute(0, -1, 1, 2)
    target = target.permute(0, -1, 1, 2)
    Dtruth = torch.sum(torch.mean(discriminator(target.contiguous()).view(Nslice,-1), dim=1))/8
    Drecon = torch.sum(torch.mean(discriminator(input.contiguous()).view(Nslice,-1), dim=1))/8

    return torch.log(Dtruth.abs_()) + torch.log(torch.abs(1.0-Drecon))


class DnCNN(nn.Module):
    def __init__(self, Nconv=4, Nkernels=16):
        super(DnCNN, self).__init__()
        layers = [nn.Sequential(nn.Conv2d(2, Nkernels, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True))]
        for i in range(Nconv - 2):
            layers.append(nn.Sequential(nn.Conv2d(Nkernels, Nkernels, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(Nkernels),
                                        nn.ReLU(inplace=True)))
        layers.append(nn.Conv2d(Nkernels, 2, kernel_size=3, padding=1))
        # layers.append(nn.BatchNorm2d(2))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        y = inputs.clone()
        residual = self.layers(y)
        return y - residual


class DnCNN_dilated(nn.Module):
    def __init__(self, Nconv=7, Nkernels=16, Dilation=None):
        super(DnCNN_dilated, self).__init__()
        if Dilation is None:
            Dilation = [2, 3, 4, 3, 2]
            # Dilation = [2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4, 3, 2]
        layers = [nn.Sequential(nn.Conv2d(2, Nkernels, kernel_size=3, stride=1, padding=1, dilation=1),
                                nn.ReLU(inplace=True))]
        for i in range(Nconv - 2):
            layers.append(nn.Sequential(nn.Conv2d(Nkernels, Nkernels, kernel_size=3, padding=Dilation[i], dilation=Dilation[i]),
                                        nn.BatchNorm2d(Nkernels),
                                        nn.ReLU(inplace=True)))
        layers.append(nn.Conv2d(Nkernels, 2, kernel_size=3, padding=1, dilation=1))
        # layers.append(nn.BatchNorm2d(2))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        y = inputs.clone()
        residual = self.layers(y)
        return y - residual


class Projector(nn.Module):
    # TODO: add channel-wise fully connected.
    # Based on https://arxiv.org/pdf/1703.09912.pdf. Replaced virtual BN with regular BN.

    def __init__(self, ENC=False):
        super(Projector, self).__init__()
        self.layer1 = self._make_layer_enc(2,64, kernel_size=4, stride=1)
        self.layer2 = self._make_layer_enc(64,128, kernel_size=4, stride=1)
        self.layer3 = self._make_layer_enc(128, 256, kernel_size=4, stride=2)
        self.layer4 = self._make_layer_enc(256, 512, kernel_size=4, stride=2)
        self.layer5 = self._make_layer_enc(512, 1024, kernel_size=4, stride=2)
        self.context_cfc = nn.Conv1d(1024,1024,kernel_size=2,groups=1024)
        self.context_conv = self._make_layer_enc(1024, 1024, kernel_size=2, stride=1)

        self.layer6 = self._make_layer_dec(1024,512, kernel_size=4, stride=2)
        self.layer7 = self._make_layer_dec(512,256, kernel_size=4, stride=2)
        self.layer8 = self._make_layer_dec(256,128, kernel_size=4, stride=2)
        self.layer9 = self._make_layer_dec(128,64, kernel_size=4, stride=1)
        self.layer10 = self._make_layer_dec(64,2, kernel_size=4, stride=1)

        self.ENC = ENC

    def _make_layer_enc(self, in_channels, out_channels, kernel_size,  stride, padding=(0,0)):
        layers = []
        layers.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride,padding=padding),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ELU(inplace=True)))
        return nn.Sequential(*layers)

    def _make_layer_dec(self, in_channels, out_channels, kernel_size,  stride, padding=(0,0)):
        layers = []
        layers.append(nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=padding),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ELU(inplace=True)))
        return nn.Sequential(*layers)

    def forward(self, x):
        # encoder
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        # x = self.context_cfc(x)
        # x = self.context_conv(x)
        if self.ENC:
            return x
        else:
            # decoder
            x = self.layer6(x)
            x = self.layer7(x)
            x = self.layer8(x)
            x = self.layer9(x)
            x = self.layer10(x)

        return x


class Bottleneck(nn.Module):

    def __init__(self, type, inplanes, stride):
        super(Bottleneck, self).__init__()
        self.shortcut = self._make_shortcut(type, inplanes=inplanes)
        self.conv = self._make_conv_layer(type, inplanes=inplanes, stride=stride, channel_compress_ratio=4)

    def _make_conv_layer(self, type, inplanes, stride, channel_compress_ratio=4):
        if type=='same' or type=='quarter':
            output_channel = inplanes
        else:
            output_channel = int(inplanes * 2)
        bottleneck_channel = int(output_channel / channel_compress_ratio)

        layers = []
        layers.append(nn.Sequential(nn.BatchNorm2d(inplanes),
                                    nn.ELU(inplace=True),
                                    nn.Conv2d(inplanes, bottleneck_channel, kernel_size=1, stride=stride),
                                    nn.BatchNorm2d(bottleneck_channel),
                                    nn.ELU(inplace=True),
                                    nn.Conv2d(bottleneck_channel, bottleneck_channel, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(bottleneck_channel),
                                    nn.ELU(inplace=True),
                                    nn.Conv2d(bottleneck_channel, output_channel, kernel_size=1, stride=1)))
        return nn.Sequential(*layers)

    def _make_shortcut(self, type, inplanes):
        if type == 'same':
            return nn.Identity()
        elif type == 'quarter':
            return nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=2)
        else:
            output_channel = inplanes * 2
            return nn.Conv2d(inplanes, output_channel, kernel_size=1, stride=2)

    def forward(self, x):
        short = self.shortcut(x)
        print(short.shape)
        conv = self.conv(x)
        print(conv.shape)
        return short + conv


class ClassifierD(nn.Module):
    # Based on https://arxiv.org/pdf/1703.09912.pdf
    '''
    Input: layers: length-4 list, default is [3,4,6,3] according to paper
    TODO: HOW to make fc dynamically change the number of input features based on number of slices.
            Right now need to pass it.
    '''
    def __init__(self, Nlayers=None, Nslices=2):
        super(ClassifierD, self).__init__()
        if Nlayers is None:
            Nlayers = [3, 4, 6, 3]
        self.net = self._build_block(Nlayers=Nlayers)
        self.Nslices = Nslices
        self.fc = nn.Linear(1024*25*25*self.Nslices,1)

    def _build_block(self, Nlayers, inplanes=64):
        layers = []
        layers.append(nn.Conv2d(2, inplanes, kernel_size=4, stride=1,padding=2))
        for i in range(len(Nlayers)):
            # print(f'half, inplanes={int(inplanes*(2**i))}')
            layers.append(Bottleneck(type='half', inplanes=int(inplanes*(2**i)), stride=2))
            for j in range(Nlayers[i]):
                # print(f'same, inplanes={int(inplanes * (2 ** (i + 1)))}')
                layers.append(Bottleneck(type='same', inplanes=int(inplanes*(2**(i+1))), stride=1))
        layers.append(nn.Sequential(nn.BatchNorm2d(1024),
                                    nn.ELU(inplace=True),
                                    ))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 1024*25*25*self.Nslices)
        x = self.fc(x)
        return x


class ClassifierD_l(nn.Module):
    # Based on https://arxiv.org/pdf/1703.09912.pdf
    '''
    Input: layers: length-4 list, default is [3,4,6,3] according to paper
    TODO: HOW to make fc dynamically change the number of input features based on number of slices.
            Right now need to pass it.
    '''
    def __init__(self, Nslices=2):
        super(ClassifierD_l, self).__init__()
        self.net = self._build_block()
        self.Nslices = Nslices
        self.fc = nn.Linear(1024*13*13*self.Nslices,1)

    def _build_block(self, inplanes=1024):
        layers = []
        for j in range(3):
            # print(f'same, inplanes={int(inplanes * (2 ** (i + 1)))}')
            layers.append(Bottleneck(type='same', inplanes=inplanes, stride=1))
        layers.append(Bottleneck(type='quarter', inplanes=inplanes, stride=2))
        for j in range(2):
            # print(f'same, inplanes={int(inplanes * (2 ** (i + 1)))}')
            layers.append(Bottleneck(type='same', inplanes=inplanes, stride=1))
        layers.append(nn.Sequential(nn.BatchNorm2d(inplanes),
                                    nn.ELU(inplace=True),
                                    ))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 1024*13*13*self.Nslices)
        x = self.fc(x)
        return x


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.conv_in = DoubleConv(2,64)
        self.down1 = Down(64,128)
        self.down2 = Down(128,256)
        self.down3 = Down(256,512)
        self.down4 = Down(512,1024)

        self.up1 = Up(1024, 512)
        self.up2 = Up(512,256)
        self.up3 = Up(256,128)
        self.up4 = Up(128,64)
        self.conv_out = nn.Conv2d(64,2,kernel_size=1)

    def forward(self,x):
        x1 = self.conv_in(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.conv_out(x)
        return x


class ScaleLayer(nn.Module):

   def __init__(self, init_value=1e-3):
       super().__init__()
       self.scale = nn.Parameter(torch.FloatTensor([init_value]),requires_grad=True)

   def forward(self, input):
       print(self.scale)
       return input * self.scale



class MoDL(nn.Module):
    # TODO: projector has dimension issue. 768*396 -> 764*396 ->764*396...
    # TODO: How to save the encoder and projector during training MoDL for loss calc
    # def __init__(self, inner_iter=10):
    def __init__(self, scale_init=1e-3, inner_iter=1):
        super(MoDL, self).__init__()

        self.inner_iter = inner_iter
        self.scale_layers = nn.Parameter(torch.ones([inner_iter]), requires_grad=True)
        self.lam = nn.Parameter(torch.ones([inner_iter]), requires_grad=True)
        self.denoiser = DnCNN_dilated()
        #self.denoiser = Projector(ENC=False)

    def forward(self, x, encoding_op, decoding_op):

        # Initial guess
        image = decoding_op.apply(x)
        # image = torch.zeros([1, 768, 396, 2], dtype=torch.float32)
        # image = image.cuda()

        zeros = torch.zeros(((1,) + image.shape[:-1]), dtype=image.dtype)  # torch.Size([1, slice, 768, 396])
        zeros = zeros.permute(1,0,2,3)
        zeros = zeros.cuda()

        for i in range(self.inner_iter):

            image = image.permute(0, -1, 1, 2)
            # # make images 3 channel.
            # image = torch.cat((image, zeros), dim=1)        # torch.Size([slice, 3, 768, 396])

            # denoised

            image = self.denoiser(image)

            # # back to 2 channel
            image = image.permute(0, 2, 3, 1)
            # image = image[:, :, :, :-1]

            # Steepest descent
            # Ex
            kspace = encoding_op.apply(image)

            # Ex - d
            kspace -= x

            # alpha * E.H *(Ex - d + lambda * image)
            # print(f'step is {self.scale_layers[i]}')
            # print(f'lambda is {self.lam[i]}')
            image = image - self.scale_layers[i]*(decoding_op.apply(kspace))


        # crop to square here to match ranknet
        idxL = int((image.shape[1] - image.shape[2]) / 2)
        idxR = int(idxL + image.shape[2])
        image = image[:, idxL:idxR, ...]

        return image


class MoDL_CG(nn.Module):
    # def __init__(self, inner_iter=10):
    def __init__(self, scale_init=1e-3, inner_iter=1):
        super(MoDL_CG, self).__init__()

        # self.encoding_op = encoding_op
        # self.decoding_op = decoding_op

        self.inner_iter = inner_iter
        self.scale_layers = nn.Parameter(torch.ones([inner_iter]), requires_grad=True)
        self.lam = nn.Parameter(torch.ones([inner_iter]), requires_grad=True)
        self.denoiser = DnCNN()


        # nn.ModuleList(self.scale_layers)    # a list of scale layers


    def forward(self, b, encoding_op, decoding_op, encoding_op_torch, decoding_op_torch):

        # Initial guess
        b = torch.squeeze(b)
        image = decoding_op_torch.apply(b)
        # image = torch.zeros([1, 768, 396, 2], dtype=torch.float32)
        # image = image.cuda()

        zeros = torch.zeros(((1,) + image.shape[:-1]), dtype=image.dtype)  # torch.Size([1, 1, 396, 396])
        zeros = zeros.cuda()

        def AhA_I_torch_op(encode, decode, lam):
            AhA = decode * encode
            AhA_I = AhA + lam * sp.linop.Identity(AhA.oshape)
            AhA_I_torch = sp.to_pytorch_function(AhA_I, input_iscomplex=True, output_iscomplex=True)

            return AhA_I_torch

        def grad(b, im, AhA_I_torch, decode_torch):
            # im should be torch tensor ([1, 768, 396, 2])
            return AhA_I_torch.apply(im) - decode_torch.apply(b)

        for i in range(self.inner_iter):

            image = image.permute(0, -1, 1, 2)
            # make images 3 channel.
            image = torch.cat((image, zeros), dim=1)

            # denoised
            print(image.dtype)
            image = self.denoiser(image)  # torch.Size([1, 3, 396, 396])

            # back to 2 channel
            image = image.permute(0, 2, 3, 1)
            image = image[:, :, :, :-1]

            AhA_I_torch = AhA_I_torch_op(encoding_op, decoding_op, lam=self.scale_layers[i])

            d = - grad(b, image, AhA_I_torch, decoding_op_torch)    # tensor (1, 768, 396, 2)

            Qd = AhA_I_torch.apply(d)
            Qdcpu = d.cpu().numpy()

            dcpu = d.cpu().numpy()
            gH = - np.transpose(np.conj(np.squeeze(chan2_complex(dcpu))))
            gHd = gH @ np.squeeze(chan2_complex(dcpu))      # array (396, 396)

            dHQd = np.transpose(np.conj(np.squeeze(chan2_complex(dcpu)))) @ np.squeeze(chan2_complex(Qdcpu))    # array (396, 396)

            alpha = np.sum(-gHd/dHQd)   # should be a number?

            image = image + alpha * d   # should be tensor (1, 768, 396, 2)


        # crop to square here to match ranknet
        idxL = int((image.shape[1] - image.shape[2]) / 2)
        idxR = int(idxL + image.shape[2])
        image = image[:, idxL:idxR, ...]

        #print(image.shape)
        #print('Done')
        return image

