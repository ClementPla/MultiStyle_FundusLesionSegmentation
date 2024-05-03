<p align="center">
    <img src="imgs/header.png" width="800px"/>
</p>

This repository contains the code associated to the paper "Multi-style conversion for semantic segmentation of lesions in fundus images by adversarial attacks"

## Installation

```bash
pip install .
```

To train the model, you will have to adjust the [config file](configs/data_config.yaml) to correspond the paths to the five datasets:

    1. IDRID
    2. MESSIDOR
    3. FGADR
    4. DDR
    5. RETINAL LESIONS


## Semantic Segmentation

We focus on the semantic segmentations of:

    1. Cotton Wool Spot
    2. Exudates
    3. Hemmorrhages
    4. Microaneurysms

For an easier use of the models, we refer to cleaned-up version of the code provided in the [fundus lesions toolkit](https://github.com/ClementPla/fundus-lesions-toolkit/tree/main/).



## Style conversion

Modifying an image by adversarial attacks to move from RETINAL LESIONS to IDRiD style.
<p align="center">
    <img src="imgs/soft_adversarial.gif" width="800px"/>
</p>

Visualization of the values added by gradient attack.
<p align="center">
    <img src="imgs/diff_adversarial.gif" width="800px"/>
</p>