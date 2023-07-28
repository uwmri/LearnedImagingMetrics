import logging
from skimage.metrics import structural_similarity
from random import randrange
import sigpy.mri as mri
import pywt
import time
import matplotlib
matplotlib.use('TKAgg')
import scipy

from utils.utils import *
from utils.CreateImagePairs import *
from utils.ISOResNet import *
from utils.utils_DL import *

spdevice = sp.Device(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



train_folder = Path("D:/NYUbrain/brain_multicoil_train/multicoil_train")
val_folder = Path("D:/NYUbrain/brain_multicoil_val/multicoil_val")
test_folder = Path("D:/NYUbrain/brain_multicoil_test/multicoil_test")
files = find("*.h5", train_folder)

# CORRUPTIONS = ['PE Motion Corrupted (%)', 'PE removed randomly (%)',
#              'Blurring (a.u.)', 'Gaussian noise level(a.u.)']
CORRUPTIONS = [ 'Gaussian noise level(a.u.)']
SAME_IMAGE = '(the same image)'
# out_name = os.path.join(f'corrupted_images_{WHICH_CORRUPTION}.h5')
# try:
#     os.remove(out_name)
# except OSError:
#     pass

Ntrials = 50
Nxmax = 320
Nymax = 640
# Nxmax = 396
# Nymax = 768



#for index_file in range(1):
#file = 'D:\\NYUbrain\\brain_multicoil_train\\multicoil_train\\file_brain_AXT1PRE_205_6000160.h5'
file = 'D:\\NYUbrain\\brain_multicoil_train\\multicoil_train\\file_brain_AXT1PRE_206_6000196.h5'
#while count == 0:

    # file = files[np.random.randint(len(os.listdir(train_folder)))]


hf = h5py.File(file, mode='r')

ksp = hf['kspace'][()]

# Hard code zero padding kspace
padyU = int(.5 * (Nymax - ksp.shape[2]))
padxU = int(.5 * (Nxmax - ksp.shape[3]))
ksp = np.pad(ksp,
             ((0, 0), (0, 0), (padyU, Nymax - ksp.shape[2] - padyU), (padxU, Nxmax - ksp.shape[3] - padxU)),
             'constant', constant_values=0 + 0j)
tot_slices = np.size(ksp, 0)
tot_coils = np.size(ksp, 1)


# # Get list of slice numbers without duplicates
# mu, sigma = 0, 0.7 * tot_slices
# for ii in range(10000):  # Not ideal...
#     num_duplicates = 0
#     num_outrange = 0
#     index_duplicates = []
#     slice_nums = np.floor(np.abs(np.random.normal(mu, sigma, num_slices)))
#     for i in range(0, num_slices):
#         if slice_nums[i] >= tot_slices:
#             num_outrange += 1
#         for j in range(i + 1, num_slices):
#             if slice_nums[i] == slice_nums[j]:
#                 index_duplicates.append(j)
#                 num_duplicates += 1
#     if num_duplicates == 0 and num_outrange == 0:
#         break

#for s in slice_nums:
    #s = int(s)
#s = np.random.randint(tot_slices)
s = 2
print(f'{file}, slice{s}')
ksp_full = ksp[s]
ksp_full_gpu = sp.to_device(ksp_full, device=spdevice)

smaps = get_smaps(ksp_full_gpu, device=spdevice, maxiter=50)
smaps = sp.to_device(smaps, device=spdevice)

# Get Truth
image_truth = mri.app.SenseRecon(ksp_full_gpu, smaps, lamda=0.005, device=spdevice, max_iter=100).run()

# send to cpu and normalize
image_truth = sp.to_device(image_truth, sp.cpu_device)
image_truth /= np.max(np.abs(image_truth))
image_truthSQ = crop_flipud(image_truth)
# with h5py.File(out_name, 'a') as hf:
#     name = 'EXAMPLE_%07d_TRUTH' % count
#     hf.create_dataset(name, data=image_truth)


filepath_rankModel = Path(r'I:\code\LearnedImagingMetrics_pytorch\Rank_NYU\ImagePairs_Pack_04032020\rank_trained_L2cnn\CV-5')
filepath_train = Path("I:/NYUbrain")
filepath_val = Path("I:/NYUbrain")

cor = 0
for WHICH_CORRUPTION in CORRUPTIONS:

    try:
        os.remove(os.path.join(filepath_rankModel, f'example_images_{WHICH_CORRUPTION}.h5'))
    except OSError:
        pass

    scoreList = []
    mseList = []
    ssimList = []
    corruption_magList = []
    for i in range(Ntrials):

        if WHICH_CORRUPTION =='Linear phase (pixel)':
            kshift_max=1
            corruption_mag = np.random.randint(-kshift_max, kshift_max)
            ksp2 = np.roll(ksp_full, corruption_mag, axis=-1)

        if WHICH_CORRUPTION == 'PE Motion Corrupted (%)':
            # corruption_mag is from which PE line the motion started/total PEs
            maxshift=30
            ksp2, corruption_mag = trans_motion(ksp_full, dir_motion=2, maxshift=maxshift, prob=1, startPE=180,
                                                fix_shift=True, fix_start=False, low=0.0, high=0.82)
        if WHICH_CORRUPTION == 'Total shift (pixels)':
            # corruption_mag is magnitude of the motion
            startPE=203
            ksp2, corruption_mag = trans_motion(ksp_full, dir_motion=2, maxshift=5, prob=1,
                                                startPE=startPE,fix_shift=False, fix_start=True)

        elif WHICH_CORRUPTION == 'Gaussian noise level(a.u.)':
            gaussian_ulim = 8
            gaussian_level = np.random.randint(0, gaussian_ulim)
            ksp2, sigma_real, sigma_imag = add_gaussian_noise(ksp_full, 1, kedge_len=30,level=gaussian_level, mode=1, mean=0)
            #corruption_mag = (sigma_real**2 + sigma_imag**2)**0.5
            corruption_mag = gaussian_level
        elif WHICH_CORRUPTION == 'Blurring (a.u.)':
            ksp2, corruption_mag = blurring(ksp_full, 3.2)
        elif WHICH_CORRUPTION == 'Random undersampling (%)':
            ksp2,_,_, corruption_mag= add_incoherent_noise(ksp_full, prob=1,
                                                       central=0.1, mode=0,
                                                       num_corrupted=0, dump=0.1)
        elif WHICH_CORRUPTION == 'PE removed randomly (%)':
            ksp2,_,_, corruption_mag = add_incoherent_noise(ksp_full, prob=1,
                                                       central=.05, mode=1,
                                                       num_corrupted=0, dump=0.27)

        elif WHICH_CORRUPTION =='none':
            ksp2 = ksp_full
            corruption_mag = 0

        corruption_magList.append(corruption_mag)
        ksp2_gpu = sp.to_device(ksp2, device=spdevice)

        image = mri.app.SenseRecon(ksp2_gpu, smaps, lamda=0.005, device=spdevice, max_iter=100).run()
        # print('corrupted recon done')
        image = sp.to_device(image, sp.cpu_device)
        imageSQ = crop_flipud(image)
        scale = np.sum(np.conj(imageSQ).T * image_truthSQ) / np.sum(np.conj(imageSQ).T * imageSQ)
        image *= scale      #imageSQ is scaled here

        mse = np.sum((np.abs(imageSQ) - np.abs(image_truthSQ)) ** 2)** 0.5
        ssim = structural_similarity(np.abs(imageSQ), np.abs(image_truthSQ))
        image_tensor = torch.unsqueeze(torch.from_numpy(imageSQ.copy()),0)
        image_tensor = image_tensor.unsqueeze(0)
        image_truth_tensor = torch.unsqueeze(torch.from_numpy(image_truthSQ.copy()),0)
        image_truth_tensor = image_truth_tensor.unsqueeze(0)
        image_truth_tensor, image_tensor = image_truth_tensor.cuda(), image_tensor.cuda()
        scores = []
        metric_files = os.listdir(filepath_rankModel)
        for i_cv in range(5):

            file_rankModel = os.path.join(filepath_rankModel, metric_files[i_cv])
            log_dir = filepath_rankModel

            rank_channel = 1
            # ranknet = ISOResNet2(BasicBlock, [2,2,2,2], for_denoise=False)
            ranknet = L2cnn(channels_in=rank_channel, channel_base=8)
            classifier = Classifier(ranknet)

            state = torch.load(file_rankModel)
            classifier.load_state_dict(state['state_dict'], strict=True)
            classifier.eval()
            for param in classifier.parameters():
                param.requires_grad = False
            scoreNet = classifier.rank
            scoreNet.cuda()
            score = scoreNet(image_tensor, image_truth_tensor)
            scores.append(score.squeeze().cpu().numpy())
        scores = np.array(scores)
        scoreMean = np.mean(scores)


        # if i%150 ==0:
        #     fig = plt.figure()
        #     fig.suptitle(f'{mse}, {ssim}, {score.cpu().numpy().item()}')
        #     plt.subplot(121)
        #     plt.title('corrupted')
        #     plt.imshow(np.abs(imageSQ), cmap='gray')
        #     plt.axis('off')
        #     plt.subplot(122)
        #     plt.title('truth')
        #     plt.imshow(np.abs(image_truthSQ), cmap='gray')
        #     plt.axis('off')
        #     plt.show()

        scoreList.append(scoreMean)
        mseList.append(mse)
        ssimList.append(ssim)

        # save example images

        # with h5py.File(os.path.join(r'D:\LearnedImagingMetrics_figs_docs\new_cv5_perturbation', f'example_images_{cor}.h5'), 'a') as hf:
        #     hf.create_dataset(f'{i}_{corruption_mag}', data=np.abs(np.squeeze(image_tensor.cpu().numpy())))
        #     hf.create_dataset(f'{i}_truth', data=np.abs(np.squeeze(image_truth_tensor.cpu().numpy())))

    cor += 1
    # if index_file == 25:
    #     break

    scoreList = np.asarray(scoreList).squeeze()
    mseList = np.asarray(mseList).squeeze()
    ssimList = 1 - np.asarray(ssimList).squeeze()
    corruption_magList = np.asarray(corruption_magList)

# scoreAll = np.reshape(scoreListAll,(len(RankID),len(CORRUPTIONS),Ntrials))
# mseAll = np.reshape(mseListAll,(len(RankID),len(CORRUPTIONS),Ntrials))
# ssimAll = 1 - np.asarray(np.reshape(ssimListAll,(len(RankID),len(CORRUPTIONS),Ntrials))).squeeze()
# corruption_magAll = np.reshape(corruption_magListAll,(len(RankID),len(CORRUPTIONS),Ntrials))
#
# scoreMean = np.mean(scoreAll, axis=0)
# mseMean = np.mean(mseAll, axis=0)
# ssimMean = np.mean(ssimAll, axis=0)
# corruption_magMean = np.mean(corruption_magAll, axis=0)



    # plt.figure()
    # plt.scatter(corruption_magList, scoreList, alpha=0.5)
    # plt.xlabel(f'{WHICH_CORRUPTION}')
    # plt.ylabel('score')
    # plt.show()
    # plt.figure()
    # plt.scatter(corruption_magList, mseList, alpha=0.5)
    # plt.xlabel(f'{WHICH_CORRUPTION} ')
    # plt.ylabel('mse')
    # plt.show()
    # plt.figure()
    # plt.scatter(corruption_magList, ssimList, alpha=0.5)
    # plt.xlabel(f'{WHICH_CORRUPTION} ')
    # plt.ylabel('ssim')
    # plt.show()
    fig_ssim, ax1 = plt.subplots(figsize=(7,5))
    color = 'tab:orange'
    ax1.set_xlabel(f'{WHICH_CORRUPTION}', fontsize=28)
    ax1.set_ylabel('1-ssim', color=color, fontsize=26)
    ax1.scatter(corruption_magList, ssimList, color=color, alpha=0.2, marker="^", s=120)
    ax1.set_ylim([0,0.25])
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.tick_params(axis='both', labelsize=22)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('score', color=color, fontsize=26)  # we already handled the x-label with ax1
    ax2.scatter(corruption_magList, scoreList, color=color, alpha=0.15, s=100)
    # ax2.set_ylim([0,125])
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.tick_params(axis='both', labelsize=22)
    fig_ssim.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    # fig_ssim.savefig(f'cv5Mean_{WHICH_CORRUPTION}_ssim-score-corruption_test.png')

    fig_mse, ax1 = plt.subplots(figsize=(7,5))
    color = 'tab:orange'
    ax1.set_xlabel(f'{WHICH_CORRUPTION}', fontsize=28)
    ax1.set_ylabel('mse', color=color, fontsize=26)
    ax1.scatter(corruption_magList, mseList, color=color, alpha=0.2, marker='x', s=120)
    ax1.set_ylim([0,12])
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.tick_params(axis='both', labelsize=22)
    #
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('score', color=color, fontsize=26)  # we already handled the x-label with ax1
    ax2.scatter(corruption_magList, scoreList, color=color, alpha=0.15, s=100)
    # ax2.set_ylim([0,125])
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.tick_params(axis='both', labelsize=22)
    fig_mse.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    # fig_mse.savefig(f'cv5Mean_{WHICH_CORRUPTION}_mse-score-corruption_test.png')

    fig_mseVscore, ax3 = plt.subplots(figsize=(7,5))
    ax3.scatter(mseList, scoreList, alpha=0.5)
    ax3.set_title(f'{WHICH_CORRUPTION}', fontsize=18)
    ax3.set_xlabel('MSE', fontsize=18)
    ax3.set_ylabel('score', fontsize=18)
    ax3.tick_params(axis='both', labelsize=14)
    fig_mseVscore.tight_layout()
    fig_mseVscore.savefig(f'cv5Mean_{WHICH_CORRUPTION}_mse-score.png')

    fig_ssimVscore, ax4 = plt.subplots(figsize=(7,5))
    ax4.scatter(ssimList, scoreList, alpha=0.5)
    ax4.set_title(f'{WHICH_CORRUPTION}', fontsize=18)
    ax4.set_xlabel('SSIM', fontsize=18)
    ax4.set_ylabel('score', fontsize=18)
    ax4.tick_params(axis='both', labelsize=14)
    fig_ssimVscore.tight_layout()
    # fig_ssimVscore.savefig(f'cv5Mean_{WHICH_CORRUPTION}_ssim-score.png')
    #
