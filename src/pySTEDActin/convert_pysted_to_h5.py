import os.path
import sys
import glob
import h5py
import numpy as np
#### import cv2

IMG_WIDTH = 30
IMG_HEIGHT = 30

h5file = os.path.expanduser("~/Downloads/MICRA-Net/training_pySTED.hdf5")
pySTED_acqs_path = os.path.expanduser(
    "~/Documents/a22_docs/micranet_dataset_test/datamaps_processed/acqs_bckgrnd_1000000/*.ppm"
)

nfiles = len(glob.glob(pySTED_acqs_path))
print(f'count of image files nfiles={nfiles}')

# resize all images and load into a single dataset
with h5py.File(h5file,'w') as  h5f:
    img_ds = h5f.create_dataset('images',shape=(nfiles, IMG_WIDTH, IMG_HEIGHT,3), dtype=int)
    for cnt, ifile in enumerate(glob.iglob(pySTED_acqs_path)):
        img = np.load(ifile)
        img_ds[cnt:cnt + 1:, :, :] = img
        #### img = cv2.imread(ifile, cv2.IMREAD_COLOR)
        # or use cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED
        #### img_resize = cv2.resize( img, (IMG_WIDTH, IMG_HEIGHT) )
        #### img_ds[cnt:cnt+1:,:,:] = img_resize