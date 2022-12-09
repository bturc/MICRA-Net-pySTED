import numpy
import os
import torch
import datetime
import h5py

from src.Actin import loader
from torch.utils.data import DataLoader


if __name__ == "__main__":
    import argparse
    import matplotlib
    from matplotlib import pyplot as plt
    matplotlib.use('tkagg')

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dry-run", action="store_true",
                        help="Performs a dry run")
    parser.add_argument("-s", "--size", type=str, default="256",
                        help="The size of crops to use.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Sets the default random seed")
    parser.add_argument("--num", type=int, default=1,
                        help="Sets the number of repetitions")
    args = parser.parse_args()

    PATH = os.path.expanduser("~/Downloads/MICRA-Net")

    add_to_seed = -1

    hdf5_training_path = f"{PATH}/datasets/training_01-04-19.hdf5"
    hdf5_validation_path = f"{PATH}/datasets/validation_01-04-19.hdf5"
    hdf5_testing_path = f"{PATH}/datasets/training_01-04-19.hdf5"

    lr, epochs, min_valid_loss = 1e-3, 250 if not args.dry_run else 1, numpy.inf
    pos_weight = [3.3, 1.6]
    current_datetime = datetime.datetime.today().strftime("%Y%m%d-%H%M%S")
    trainer_params = {
        "model_name": "_".join((current_datetime, str(args.seed + add_to_seed))),
        "savefolder": f"{PATH}/Results",
        "datetime": current_datetime,
        "dry_run": args.dry_run,
        "size": eval(args.size),
        "seed": args.seed + add_to_seed,
        "lr": lr,
        "epochs": epochs,
        "cuda": torch.cuda.is_available(),
        # "data_aug": 0.5,
        "step": 0.75,
        "pos_weight": pos_weight,
        "hdf5_training_path": hdf5_training_path,
        "hdf5_validation_path": hdf5_validation_path,
        "hdf5_testing_path": hdf5_testing_path,
        "dataloader_params": {
            "shuffle": False,
            "num_workers": 4,
            "pin_memory": True,
            "drop_last": True,
            "batch_size": 1,
        },
        "model_params": {
            "num_classes": 2,
            "num_input_images": 1,
        },
        "scheduler": {
            "patience": 10,
            "threshold": 0.01,
            "min_lr": 1e-5,
            "factor": 0.1,
            "verbose": True
        }
    }

    train_dataset = loader.HDF5Dataset(trainer_params["hdf5_training_path"], **trainer_params)
    train_loader = DataLoader(train_dataset, **trainer_params["dataloader_params"])

    print(train_loader[0])
    print(train_dataset[0])
    exit()

    it = iter(train_loader)
    first = next(it)
    print(len(first))
    img = first[0].numpy()
    segmentation = first[1].numpy()
    print(img.shape)
    for i in range(img.shape[0]):
        print(segmentation[i])
        print(numpy.max(img[i]), numpy.min(img[i]))
        fig, axes = plt.subplots(1, 1)
        axes.imshow(img[i], cmap="hot")
        # axes[1].imshow(segmentation[i])
        plt.show()
