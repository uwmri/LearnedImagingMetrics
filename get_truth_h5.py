import pickle
from utils.utils import *
from utils.CreateImagePairs import get_truth

filepath_raw = Path("D:/NYUbrain/brain_multicoil_val/multicoil_val")
files_raw = find("*.h5", filepath_raw)


file_smaps = 'D:\\NYUbrain\\brain_multicoil_val\\smaps_val_espirit.h5'
hf_smaps = h5py.File(name=file_smaps, mode='r')

# save
out_name = os.path.join('truths_val.h5')
try:
    os.remove(out_name)
except OSError:
    pass

# Only choose 20-coil data for now
Ncoils = 20
xres = 768
yres = 396

Nfile = 0
scan_names = []

for file in files_raw:

    hf = h5py.File(file, mode='r')
    ksp = hf['kspace'][()]

    if ksp.shape[1] == Ncoils:
        print(f'{Nfile+1} scans has {Ncoils} coils')

        ksp = zero_pad4D(ksp)

        smaps = hf_smaps[file]
        Nslice = smaps.shape[0]

        image_truth = np.zeros((smaps.shape[0], smaps.shape[-2], smaps.shape[-1]), dtype=smaps.dtype)
        for sl in range(Nslice):
            mps = sp.to_device(smaps[sl], sp.Device(0))
            image_truth[sl] = get_truth(ksp, sl, device=sp.Device(0), lamda=0.005, smaps=mps, forRecon=True)

        # crop truth
        idxL = int((image_truth.shape[1] - image_truth.shape[2]) / 2)
        idxR = int(idxL + image_truth.shape[2])
        image_truth = image_truth[:, idxL:idxR, ...]

        with h5py.File(out_name, 'a') as hf:
            hf.create_dataset(f'{file}', data=image_truth)

        scan_names.append(file)

        Nfile += 1

    else:
        pass


with open('val_20coil.txt','wb') as tf:
    pickle.dump(scan_names, tf)

