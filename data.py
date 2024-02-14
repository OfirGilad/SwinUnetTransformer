import os
import numpy as np

from monai.transforms import (
    AsDiscrete,
    # AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
)

from monai.data import (
    DataLoader,
    CacheDataset,
)

import glob
import BMIC_Utils as bmu


def data_filepaths():
    # Get filepaths
    root_dir = "../data/data_ct/train/"
    train_images = sorted(glob.glob(os.path.join(root_dir, 'raw', "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(root_dir, 'seg', "*.nii.gz")))

    data_dicts = [{"image": images_name, "label": label_name} for images_name, label_name in zip(train_images, train_labels)]
    train_filepaths, val_filepaths = data_dicts[:-10], data_dicts[-10:]

    # root_dir = "../data/data_ct/train/"
    # train_images = sorted(glob.glob(os.path.join(root_dir, 'raw', "*.nii.gz")))[:10]
    # train_labels = sorted(glob.glob(os.path.join(root_dir, 'seg', "*.nii.gz")))[:10]

    # data_dicts = [{"image": images_name, "label": label_name} for images_name, label_name in zip(train_images, train_labels)]
    # train_filepaths, val_filepaths = data_dicts[:9], data_dicts[-1:]

    print(f"Train Data: {len(train_filepaths)} files")
    print(f"Val Data: {len(val_filepaths)} files")

    # Open files in numpy
    train_files = []
    val_files = []
    for idx, file_name in enumerate(train_filepaths):
        print(f"train: {idx}")
        image, _ = bmu.load_NII(file_name["image"], with_affine=True)
        image = image.astype(np.float32)
        image = np.expand_dims(image, axis=0)
        label, _ = bmu.load_NII(file_name["label"], with_affine=True)
        label = label.astype(np.float32)
        label = np.expand_dims(label, axis=0)
        train_files.append({"image": image, "label": label})

    for idx, file_name in enumerate(val_filepaths):
        print(f"val: {idx}")
        image = bmu.load_NII(file_name["image"], with_affine=False)
        image = image.astype(np.float32)
        image = np.expand_dims(image, axis=0)
        label = bmu.load_NII(file_name["label"], with_affine=False)
        label = label.astype(np.float32)
        label = np.expand_dims(label, axis=0)
        val_files.append({"image": image, "label": label})

    return train_files, val_files


def data_transforms():
    train_transforms = Compose([
        # LoadImaged(keys=["image", "label"]),
        # # AddChanneld(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="LPS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000,
            a_max=1000,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
        ToTensord(keys=["image", "label"]),
    ])

    val_transforms = Compose([
        # LoadImaged(keys=["image", "label"]),
        # # AddChanneld(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="LPS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ])

    return train_transforms, val_transforms


def data_dataloaders():
    train_files, val_files = data_filepaths()
    train_transforms, val_transforms = data_transforms()
    # print(f"train: {train_files}")
    # print(f"val: {val_files}")

    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_num=24,
        cache_rate=1.0,
        num_workers=8,
    )
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, num_workers=8, pin_memory=True
    )
    val_ds = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_num=6,
        cache_rate=1.0,
        num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader
