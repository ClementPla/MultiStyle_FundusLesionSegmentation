import os

import albumentations as A
import cv2
import nntools as nt
import nntools.dataset as D
import numpy as np
import torch
from nntools.dataset.tools import CacheBullet

from fundseg.data.utils import Dataset, Lesions


@D.nntools_wrapper
def process_masks_multiclass(Exudates=None, Microaneurysms=None, Hemorrhages=None, Cotton_Wool_Spot=None):
    mask = np.zeros_like(Exudates, dtype=np.uint8)
    mask[Cotton_Wool_Spot != 0] = 1
    mask[Exudates != 0] = 2
    mask[Hemorrhages != 0] = 3
    mask[Microaneurysms != 0] = 4
    return {"mask": mask}


@D.nntools_wrapper
def autocrop(image, mask=None):
    threshold = 25
    r_img = np.max(image, axis=2)
    _, roi = cv2.threshold(r_img, threshold, 1, cv2.THRESH_BINARY)
    not_null_pixels = cv2.findNonZero(roi)

    if not_null_pixels is None:
        if mask is None:
            return {"image": image, "roi": roi}
        else:
            return {"image": image, "mask": mask, "roi": roi}

    x_range = (np.min(not_null_pixels[:, :, 0]), np.max(not_null_pixels[:, :, 0]))
    y_range = (np.min(not_null_pixels[:, :, 1]), np.max(not_null_pixels[:, :, 1]))

    d = {
        "image": image[y_range[0] : y_range[1], x_range[0] : x_range[1]],
        "roi": roi[y_range[0] : y_range[1], x_range[0] : x_range[1]],
    }
    if mask is not None:
        d["mask"] = mask[y_range[0] : y_range[1], x_range[0] : x_range[1]]

    return d


def get_masks_paths(ex, ctw, he, ma, labels):
    masks = {}
    if Lesions.EXUDATES in labels:
        masks["Exudates"] = ex
    if Lesions.COTTON_WOOL_SPOT in labels:
        masks["Cotton_Wool_Spot"] = ctw
    if Lesions.HEMORRHAGES in labels:
        masks["Hemorrhages"] = he
    if Lesions.MICROANEURYSMS in labels:
        masks["Microaneurysms"] = ma
    return masks


def get_generic_dataset(
    root_img, masks_paths, dataset_id, input_shape, sort_func=None, task="multiclass", use_cache=False
):
    segmentDataset = D.SegmentationDataset(
        img_root=root_img,
        mask_root=masks_paths,
        shape=input_shape,
        filling_strategy=nt.NN_FILL_UPSAMPLE,
        keep_size_ratio=True,
        auto_pad=True,
        recursive_loading=False,
        extract_image_id_function=sort_func,
        id=dataset_id,
        use_cache=use_cache,
    )
    composer = D.Composition()
    resizing = A.Compose(
        [
            A.LongestMaxSize(max_size=max(input_shape), always_apply=True),
            A.PadIfNeeded(
                min_height=input_shape[0],
                min_width=input_shape[1],
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                always_apply=True,
            ),
        ],
        additional_targets={"roi": "mask"},
    )

    if task == "multiclass":
        composer.add(process_masks_multiclass)
    else:
        raise NotImplementedError("Multilabel function can be added here")

    composer.add(autocrop, resizing)

    if use_cache:
        composer.add(CacheBullet())

    segmentDataset.composer = composer

    segmentDataset.tag = {"Dataset": dataset_id.value}
    return segmentDataset


def get_image_dataset(root_img, input_shape):
    dataset = D.ImageDataset(root_img, shape=input_shape, keep_size_ratio=True, auto_pad=True)
    dataset.composer = get_resize_composer(input_shape)
    return dataset


def get_resize_composer(input_shape):
    composer = D.Composition()
    resizing = A.Compose(
        [
            A.LongestMaxSize(max_size=max(input_shape), always_apply=True),
            A.PadIfNeeded(
                min_height=input_shape[0], min_width=input_shape[1], border_mode=cv2.BORDER_CONSTANT, value=0
            ),
        ],
        additional_targets={"roi": "mask"},
    )
    composer.add(autocrop, resizing)
    return composer


def sort_func_idrid(x):
    return "_".join(x.split(".")[0].split("_")[:2])


def get_idrid_dataset(root_img, root_mask, input_shape, labels, task, use_cache):
    input_shape = tuple(input_shape)
    ma = os.path.join(root_mask, "1. Microaneurysms/")
    he = os.path.join(root_mask, "2. Haemorrhages/")
    ex = os.path.join(root_mask, "3. Hard Exudates/")
    ctw = os.path.join(root_mask, "4. Soft Exudates/")
    masks = get_masks_paths(ex, ctw, he, ma, labels)
    return get_generic_dataset(
        root_img, masks, Dataset.IDRID, input_shape, sort_func_idrid, task=task, use_cache=use_cache
    )


def get_messidor_dataset(root_img, root_mask, input_shape, labels, task, use_cache):
    input_shape = tuple(input_shape)
    ma = os.path.join(root_mask, "Red/Microaneurysms/")
    he = os.path.join(root_mask, "Red/Hemorrhages/")
    ex = os.path.join(root_mask, "Bright/Exudates/")
    ctw = os.path.join(root_mask, "Bright/Cotton Wool Spots/")
    masks = get_masks_paths(ex, ctw, he, ma, labels)
    return get_generic_dataset(root_img, masks, Dataset.MESSIDOR, input_shape, task=task, use_cache=use_cache)


def get_fgadr_dataset(root_img, root_mask, input_shape, labels, task, use_cache):
    input_shape = tuple(input_shape)
    ma = os.path.join(root_mask, "Microaneurysms_Masks/")
    he = os.path.join(root_mask, "Hemohedge_Masks/")
    ex = os.path.join(root_mask, "HardExudate_Masks/")
    ctw = os.path.join(root_mask, "SoftExudate_Masks/")
    masks = get_masks_paths(ex, ctw, he, ma, labels)

    return get_generic_dataset(root_img, masks, Dataset.FGADR, input_shape, task=task, use_cache=use_cache)


def get_retlesions_dataset(root_img, root_mask, input_shape, labels, task, use_cache):
    input_shape = tuple(input_shape)
    ma = os.path.join(root_mask, "microaneurysm/")
    he = os.path.join(root_mask, "retinal_hemorrhage/")
    ex = os.path.join(root_mask, "hard_exudate/")
    ctw = os.path.join(root_mask, "cotton_wool_spots/")
    masks = get_masks_paths(ex, ctw, he, ma, labels)

    return get_generic_dataset(root_img, masks, Dataset.RETINAL_LESIONS, input_shape, task=task, use_cache=use_cache)


def get_DDR_dataset(root_img, root_mask, input_shape, labels, task, use_cache):
    input_shape = tuple(input_shape)
    ma = os.path.join(root_mask, "MA/")
    he = os.path.join(root_mask, "HE/")
    ex = os.path.join(root_mask, "EX/")
    ctw = os.path.join(root_mask, "SE/")
    masks = get_masks_paths(ex, ctw, he, ma, labels)

    return get_generic_dataset(root_img, masks, Dataset.DDR, input_shape, task=task, use_cache=use_cache)


def split_dataset(dataset, ratio_split, seed):
    if ratio_split:
        train_len = int(len(dataset) * (1 - ratio_split))
        valid_len = len(dataset) - train_len
        train_dataset, valid_dataset = D.random_split(
            dataset, [train_len, valid_len], generator=torch.Generator().manual_seed(seed)
        )
        return train_dataset, valid_dataset
    else:
        return dataset, None


def get_datasets(
    labels,
    roots_idrid=(None, None),
    roots_messidor=(None, None),
    roots_fgadr=(None, None),
    root_retlesion=(None, None),
    root_ddr=(None, None),
    train_or_test="train",
    shape=(1024, 1024),
    split_ratio=0.15,
    use_cache=False,
    seed=1234,
    task="multiclass",
):
    outputs = {"core": [], "split": []}
    roots = [
        (roots_idrid, get_idrid_dataset, Dataset.IDRID),
        (roots_messidor, get_messidor_dataset, Dataset.MESSIDOR),
        (roots_fgadr, get_fgadr_dataset, Dataset.FGADR),
        (root_retlesion, get_retlesions_dataset, Dataset.RETINAL_LESIONS),
        (root_ddr, get_DDR_dataset, Dataset.DDR),
    ]

    for r, func, flag in roots:
        if all(r):
            dataset = func(
                root_img=r[0], root_mask=r[1], input_shape=shape, labels=labels, task=task, use_cache=use_cache
            )
            if flag == Dataset.RETINAL_LESIONS or flag == Dataset.MESSIDOR or flag == Dataset.FGADR:
                # These datasets do not have a predefined test set.
                # We extract 30% of the train set as test set.
                train, test = split_dataset(dataset, 0.3, seed)
                if train_or_test == "train":
                    dataset = train
                else:
                    dataset = test

            if train_or_test == "train":
                if flag == Dataset.DDR:
                    # DDR has a predefined validation set.
                    train = dataset
                    root_mask_val = r[1].replace("/train/", "/valid/")
                    root_mask_val = root_mask_val.replace("label/", "segmentation label/")
                    val = get_DDR_dataset(
                        root_img=r[0].replace("/train/", "/valid/"),
                        root_mask=root_mask_val,
                        input_shape=shape,
                        labels=labels,
                        task=task,
                        use_cache=use_cache,
                    )

                else:
                    # For the others, we split part of the train set.
                    train, val = split_dataset(dataset, split_ratio, seed)

                outputs["core"].append(train)
                if val is not None:
                    outputs["split"].append(val)

            elif train_or_test == "test":
                outputs["core"].append(dataset)

    return outputs


def get_datasets_from_config(
    root,
    config,
    sets,
    labels,
    seed=1234,
    shape=None,
    split_ratio=0,
    train_or_test="train",
    task="multiclass",
    use_cache=False,
):
    root_path = root

    img_idrid_root = config.get("img_idrid_url", None) if Dataset.IDRID in sets else None
    mask_idrid_root = config.get("mask_idrid_url", None) if Dataset.IDRID in sets else None
    img_idrid_root = os.path.join(root_path, img_idrid_root) if img_idrid_root is not None else None
    mask_idrid_root = os.path.join(root_path, mask_idrid_root) if mask_idrid_root is not None else None

    img_messidor_root = config.get("img_messidor_url", None) if Dataset.MESSIDOR in sets else None
    mask_messidor_root = config.get("mask_messidor_url", None) if Dataset.MESSIDOR in sets else None
    img_messidor_root = os.path.join(root_path, img_messidor_root) if img_messidor_root is not None else None
    mask_messidor_root = os.path.join(root_path, mask_messidor_root) if mask_messidor_root is not None else None

    img_fgadr_root = config.get("img_fgadr_url", None) if Dataset.FGADR in sets else None
    mask_fgadr_root = config.get("mask_fgadr_url", None) if Dataset.FGADR in sets else None
    img_fgadr_root = os.path.join(root_path, img_fgadr_root) if img_fgadr_root is not None else None
    mask_fgadr_root = os.path.join(root_path, mask_fgadr_root) if mask_fgadr_root is not None else None

    img_retles_root = config.get("img_retles_url", None) if Dataset.RETINAL_LESIONS in sets else None
    mask_retles_root = config.get("mask_retles_url", None) if Dataset.RETINAL_LESIONS in sets else None
    img_retles_root = os.path.join(root_path, img_retles_root) if img_retles_root is not None else None
    mask_retles_root = os.path.join(root_path, mask_retles_root) if mask_retles_root is not None else None

    img_ddr_root = config.get("img_ddr_url", None) if Dataset.DDR in sets else None
    mask_ddr_root = config.get("mask_ddr_url", None) if Dataset.DDR in sets else None
    img_ddr_root = os.path.join(root_path, img_ddr_root) if img_ddr_root is not None else None
    mask_ddr_root = os.path.join(root_path, mask_ddr_root) if mask_ddr_root is not None else None

    if shape is None:
        shape = config["shape"]
    shape = tuple(shape)
    return get_datasets(
        roots_idrid=(img_idrid_root, mask_idrid_root),
        roots_fgadr=(img_fgadr_root, mask_fgadr_root),
        roots_messidor=(img_messidor_root, mask_messidor_root),
        root_retlesion=(img_retles_root, mask_retles_root),
        root_ddr=(img_ddr_root, mask_ddr_root),
        shape=shape,
        split_ratio=split_ratio,
        labels=labels,
        use_cache=use_cache,
        seed=seed,
        train_or_test=train_or_test,
        task=task,
    )


def add_operations_to_dataset(datasets, aug=None):
    if aug is None:
        return
    if not isinstance(datasets, list):
        datasets = [datasets]
    for d in datasets:
        if isinstance(aug, list):
            d.composer.add(*aug)
        else:
            d.composer.add(aug)
