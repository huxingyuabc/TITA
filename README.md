# TITA
This repository contains the official implementation of the ICCV 2025 paper  [" Balancing Task-invariant Interaction and Task-specific Adaptation for Unified Image Fusion"](https://arxiv.org/pdf/2504.05164) .


## Setup

### Environment

 - [ ] python 3.9
 - [ ] torch 1.12.1
 - [ ] cudatoolkit 11.3
 - [ ] torchvision 0.13.1
 - [ ] numpy 1.23.5
 - [ ] opencv-python 4.7.0
 - [ ] ...

### Datasets

The datasets is constructed following [TC-MoA](https://github.com/YangSun22/TC-MoA).

The data structure is like this:

```
dataset_name
├── subdir1
│  ├── xxx.png
│  ├── ...
├── subdir2
│  ├── xxx.png
│  ├── ...
```

**For training**:

- [ ] LLVIP
- [ ] SCIE
- [ ] RealMFF, MFI-WHU

**For testing**:

- [ ] LLVIP
- [ ] MEFB
- [ ] MFFB (Lytro, MFFW, MFI-WHU)

## Results

TITA results are available on [google drive](https://drive.google.com/file/d/1K66Km2i7mACXex_3ZdhkXFkCQx132rv0/view?usp=sharing).

## Testing
Download the pretrained checkpoint from [google drive](https://drive.google.com/file/d/17mjdsybIVN1sonfDHpUf_Jazuk4pAlZW/view?usp=sharing), and put it under `./logs/mixed/models/`. Change the dataset name and path in `test.py`. And the results can be found in `./results_tita/`.

    bash test.bash


## Training
Change the dataset name and path in `config.json` and run:

    bash train.bash

## Citation

```
@inproceedings{hu2025balancing,
  title={Balancing Task-invariant Interaction and Task-specific Adaptation for Unified Image Fusion},
  author={Hu, Xingyu and Jiang, Junjun and Wang, Chenyang and Jiang, Kui and Liu, Xianming and Ma, Jiayi},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2025}
}
```

## Acknowledgements

This code is mainly built upon [SwinFusion](https://github.com/Linfeng-Tang/SwinFusion) and [FAMO](https://github.com/Cranial-XIX/FAMO). Thanks for their excellent work!

 