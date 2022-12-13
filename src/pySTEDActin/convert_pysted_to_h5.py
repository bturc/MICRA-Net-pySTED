import os.path
import datetime
import glob
import h5py
import torch
import numpy as np
import loader
#### import cv2
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('tkagg')

IMG_WIDTH = 256
IMG_HEIGHT = 256

h5file = os.path.expanduser("~/Downloads/MICRA-Net/datasets/training_pySTED.hdf5")
pySTED_acqs_path = os.path.expanduser(
    "~/Documents/a22_docs/micranet_dataset_test/datamaps_processed/acqs_bckgrnd_1000000/*.npy"
)

nfiles = len(glob.glob(pySTED_acqs_path))
print(f'count of image files nfiles={nfiles}')

# load the corresponding normal_acqs loader, make sure the data matches, add the label/mask data to the pySTED dataset
current_datetime = datetime.datetime.today().strftime("%Y%m%d-%H%M%S")
PATH = os.path.expanduser("~/Downloads/MICRA-Net")
hdf5_training_path = f"{PATH}/datasets/training_01-04-19.hdf5"
hdf5_validation_path = f"{PATH}/datasets/validation_01-04-19.hdf5"
hdf5_testing_path = f"{PATH}/datasets/testing_01-04-19.hdf5"
pos_weight = [3.3, 1.6]
dry_run = True
lr, epochs, min_valid_loss = 1e-3, 250 if not dry_run else 1, np.inf
seed = 0
add_to_seed = -1
trainer_params = {
    "model_name" : "_".join((current_datetime, str(seed + add_to_seed))),
    "savefolder" : f"{PATH}/Results",
    "datetime" : current_datetime,
    "dry_run" : dry_run,
    "size" : 256,
    "seed" : seed + add_to_seed,
    "lr" : lr,
    "epochs" : epochs,
    "cuda" : torch.cuda.is_available(),
    "data_aug" : 0.5,
    "step" : 0.75,
    "pos_weight" : pos_weight,
    "hdf5_training_path" : hdf5_training_path,
    "hdf5_validation_path" : hdf5_validation_path,
    "hdf5_testing_path" : hdf5_testing_path,
    "dataloader_params" : {
        "shuffle" : True,
        "num_workers" : 4,
        "pin_memory" : True,
        "drop_last" : True,
        "batch_size" : 32,
    },
    "model_params" : {
        "num_classes" : 2,
        "num_input_images" : 1,
    },
    "scheduler" : {
        "patience" : 10,
        "threshold" : 0.01,
        "min_lr" : 1e-5,
        "factor" : 0.1,
        "verbose" : True
    }
}
train_dataset = loader.ActinHDF5Dataset(trainer_params["hdf5_training_path"], **trainer_params)

# resize all images and load into a single dataset
with h5py.File(h5file, 'w') as h5f:
    img_ds = h5f.create_dataset("data", shape=(nfiles, IMG_WIDTH, IMG_HEIGHT, 2), dtype=float)
    for cnt, ifile in enumerate(glob.iglob(pySTED_acqs_path)):
        img = np.load(ifile)
        img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img))

        og_img_data = train_dataset[cnt]
        og_img = og_img_data[0].numpy()

        fig, axes = plt.subplots(1, 2)

        axes[0].imshow(og_img, cmap="hot")
        axes[1].imshow(img_normalized, cmap="hot")

        plt.show()

        img_ds[cnt:cnt + 1:, :, :, 0] = img_normalized

        label = None

        img_ds[cnt:cnt + 1:, :, :, 1] = label
