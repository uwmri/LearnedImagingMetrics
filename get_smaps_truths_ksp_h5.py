import pickle
from utils.utils import *
from utils.CreateImagePairs import get_smaps, get_truth

filepath_raw = Path("D:/NYUbrain/brain_multicoil_val/multicoil_val")
files_raw = find("*.h5", filepath_raw)

# save
out_name = os.path.join('ksp_truths_smaps_val.h5')
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
    scanname = os.path.split(Path(file))[1]

    hf_ksp = h5py.File(file, mode='r')

    ksp = hf_ksp['kspace'][()]
    Nslice = ksp.shape[0]

    if ksp.shape[1] == Ncoils:
        print(f'{Nfile+1} scans has {Ncoils} coils')

        ksp = zero_pad4D(ksp)
        smaps = np.zeros(ksp.shape, dtype=ksp.dtype)
        image_truth = np.zeros((ksp.shape[0], ksp.shape[-2], ksp.shape[-1]), dtype=ksp.dtype)

        # seems jsense and espirit wants (coil,h,w), can't do (sl, coil, h, w)
        for sl in range(Nslice):
            ksp_gpu = sp.to_device(ksp[sl], device=sp.Device(0))
            mps = get_smaps(ksp_gpu, device=sp.Device(0), maxiter=30, method='jsense')
            image_truth[sl] = get_truth(ksp, sl, device=sp.Device(0), lamda=0.005, smaps=mps, forRecon=True)
            smaps[sl] = sp.to_device(mps, sp.cpu_device)

        # crop truth
        idxL = int((image_truth.shape[1] - image_truth.shape[2]) / 2)
        idxR = int(idxL + image_truth.shape[2])
        image_truth = image_truth[:, idxL:idxR, ...]

        with h5py.File(out_name, 'a') as hf:
            hf.create_dataset(f'{scanname}/kspace', data=ksp, compression="gzip")
            hf.create_dataset(f'{scanname}/truths', data=image_truth, compression="gzip")
            hf.create_dataset(f'{scanname}/smaps', data=smaps, compression="gzip")

        Nfile += 1

        scan_names.append(scanname)


    else:
        pass

with open('val_20coil.txt','wb') as tf:
    pickle.dump(scan_names, tf)