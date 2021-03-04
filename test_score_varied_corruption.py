import logging
from skimage.metrics import structural_similarity
from random import randrange
import sigpy.mri as mri
import pywt
import time

from utils.utils import *
from utils.CreateImagePairs import *
from utils.model_helper import *
from utils.utils_DL import *

spdevice = sp.Device(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

filepath_rankModel = Path('I:\code\LearnedImagingMetrics_pytorch\Rank_NYU\ImagePairs_Pack_04032020')
filepath_train = Path("I:/NYUbrain")
filepath_val = Path("I:/NYUbrain")
file_rankModel = os.path.join(filepath_rankModel, "RankClassifier4217_pretrained.pt")
log_dir = filepath_rankModel

rank_channel =1
ranknet = L2cnn(channels_in=rank_channel)
classifier = Classifier(ranknet)

state = torch.load(file_rankModel)
classifier.load_state_dict(state['state_dict'], strict=True)
classifier.eval()
for param in classifier.parameters():
    param.requires_grad = False
scoreNet = classifier.rank
scoreNet.cuda()

train_folder = Path("D:/NYUbrain/brain_multicoil_train/multicoil_train")
val_folder = Path("D:/NYUbrain/brain_multicoil_val/multicoil_val")
test_folder = Path("D:/NYUbrain/brain_multicoil_test/multicoil_test")
files = find("*.h5", train_folder)

WHICH_CORRUPTION = 'incoherent_lines'
out_name = os.path.join(f'corrupted_images_{WHICH_CORRUPTION}.h5')
try:
    os.remove(out_name)
except OSError:
    pass

# number of slices to grab from each scan
num_slices = 4
Nxmax = 396
Nymax = 768
count = 1
scoreList = []
mseList = []
ssimList = []
corruption_magList = []
for index_file in range(len(os.listdir(train_folder))):

    file=files[np.random.randint(len(os.listdir(train_folder)))]

    file_size = os.path.getsize(file)
    if file_size < 300000000:
        pass
    else:
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

        # Get list of slice numbers without duplicates
        mu, sigma = 0, 0.7 * tot_slices
        for ii in range(10000):  # Not ideal...
            num_duplicates = 0
            num_outrange = 0
            index_duplicates = []
            slice_nums = np.floor(np.abs(np.random.normal(mu, sigma, num_slices)))
            for i in range(0, num_slices):
                if slice_nums[i] >= tot_slices:
                    num_outrange += 1
                for j in range(i + 1, num_slices):
                    if slice_nums[i] == slice_nums[j]:
                        index_duplicates.append(j)
                        num_duplicates += 1
            if num_duplicates == 0 and num_outrange == 0:
                break

        for s in slice_nums:
            s = int(s)
            ksp_full = ksp[s]
            ksp_full_gpu = sp.to_device(ksp_full, device=spdevice)

            smaps = get_smaps(ksp_full_gpu, device=spdevice, maxiter=50)
            smaps = sp.to_device(smaps, device=spdevice)

            # Get Truth
            image_truth = mri.app.SenseRecon(ksp_full_gpu, smaps, lamda=0.005, device=spdevice, max_iter=20).run()

            # send to cpu and normalize
            image_truth = sp.to_device(image_truth, sp.cpu_device)
            image_truth /= np.max(np.abs(image_truth))
            image_truthSQ = crop_flipud(image_truth)

            with h5py.File(out_name, 'a') as hf:
                name = 'EXAMPLE_%07d_TRUTH' % count
                hf.create_dataset(name, data=image_truth)

            # Get corrupted
            if WHICH_CORRUPTION == 'motion':
                ksp2, corruption_mag = trans_motion(ksp_full, 2, 150, 1)
            elif WHICH_CORRUPTION == 'gaussian':
                gaussian_ulim = 12
                gaussian_level = np.random.randint(0, gaussian_ulim)
                ksp2, sigma_real, sigma_imag = add_gaussian_noise(ksp_full, 1, kedge_len=30,level=gaussian_level, mode=1, mean=0)
                corruption_mag = (sigma_real**2 + sigma_imag**2)**0.5
            elif WHICH_CORRUPTION == 'blurring':
                ksp2, corruption_mag = blurring(ksp_full, 15)
            elif WHICH_CORRUPTION == 'incoherent_pts':
                ksp2, corruption_mag, _ = add_incoherent_noise(ksp_full, prob=1,
                                                           central=np.random.uniform(0.2, 0.4), mode=0,
                                                           num_corrupted=0, dump=0.5)
            elif WHICH_CORRUPTION == 'incoherent_lines':
                ksp2, corruption_mag, _ = add_incoherent_noise(ksp_full, prob=1,
                                                           central=np.random.uniform(0.2, 0.4), mode=1,
                                                           num_corrupted=0, dump=0.5)

            corruption_magList.append(corruption_mag)
            ksp2_gpu = sp.to_device(ksp2, device=spdevice)

            image = mri.app.SenseRecon(ksp2_gpu, smaps, lamda=0.005, device=spdevice, max_iter=20).run()
            image = sp.to_device(image, sp.cpu_device)
            imageSQ = crop_flipud(image)
            scale = np.sum(np.conj(imageSQ).T * image_truthSQ) / np.sum(np.conj(imageSQ).T * imageSQ)
            image *= scale

            mse = np.sum((np.abs(image) - np.abs(image_truth)) ** 2)** 0.5
            ssim = structural_similarity(np.abs(image), np.abs(image_truth))
            image_tensor = torch.unsqueeze(torch.from_numpy(complex_2chan(image)),0)
            image_truth_tensor = torch.unsqueeze(torch.from_numpy(complex_2chan(image_truth)),0)
            image_tensor = image_tensor.permute(0,-1,1,2).cuda()
            image_truth_tensor = image_truth_tensor.permute(0, -1, 1, 2).cuda()
            score = scoreNet(image_tensor, image_truth_tensor)

            scoreList.append(score.detach().cpu().numpy())
            mseList.append(mse)
            ssimList.append(ssim)

            count += 1

    if index_file == 100:
        break
scoreList = np.asarray(scoreList)
mseList = np.asarray(mseList)
ssimList = np.asarray(ssimList)

plt.scatter(corruption_magList, scoreList, alpha=0.5)
plt.xlabel(f'{WHICH_CORRUPTION} magnitude')
plt.ylabel('score')
plt.show()
plt.scatter(corruption_magList, mseList, alpha=0.5)
plt.xlabel(f'{WHICH_CORRUPTION} magnitude')
plt.ylabel('mse')
plt.show()
plt.scatter(corruption_magList, ssimList, alpha=0.5)
plt.xlabel(f'{WHICH_CORRUPTION} magnitude')
plt.ylabel('ssim')
plt.show()