FHDR: HDR Image Reconstruction from a SingleLDR Image using Feedback Network
========================================

### [Project page](https://mukulkhanna.github.io/projects/FHDR) |   [Presentation](https://sigport.org/documents/fhdr-hdr-image-reconstruction-single-ldr-image-using-feedback-network)  |  Paper | [Code](https://github.com/mukulkhanna/FHDR)

This repository contains the code for the paper's implementation.

**"FHDR: HDR Image Reconstruction from a SingleLDR Image using Feedback Network"**
*Zeeshan Khan, Mukul Khanna, Shanmuganathan Raman*
**Presented at [Global SIP 2019](http://2019.ieeeglobalsip.org)**


<img src="https://user-images.githubusercontent.com/24846546/71309080-99a96380-23fb-11ea-94b3-2384eca101dd.png">

About the project
-----------

This project was a part of my research internship at IIT Gandhinagar’s Computer Vision lab, with Prof Shanmuganathan Raman and fellow Research Assistant, Zeeshan Khan. The aim of the project was to generate High Dynamic Range (HDR) content from Low Dynamic Range (LDR) images captured from off-the-shelf consumer cameras using Deep Learning. This research was supported by the Science and Engineering Research Board (SERB) Core Research Grant.

Introduction
------------

High dynamic range (HDR) image generation from a single exposure low dynamic range (LDR) image has been made possible due to the recent advances in Deep Learning. Various feed-forward Convolutional Neural Networks (CNNs) have been proposed for learning LDR to HDR representations. 

To better utilize the power of CNNs, we exploit the idea of feedback, where the initial low level features are guided by the high level features using a hidden state of a Recurrent Neural Network. Unlike a single forward pass in a conventional feed-forward network, the reconstruction from LDR to HDR in a feedback network is learned over multiple iterations. This enables us to create a coarse-to-fine representation, leading to an improved reconstruction at every iteration. Various advantages over standard feed-forward networks include early reconstruction ability and better reconstruction quality with fewer network parameters. We design a dense feedback block and propose an end-to-end feedback network- FHDR for HDR image generation from a single exposure LDR image. Qualitative and quantitative evaluations show the superiority of our approach over the state of-the-art methods.

<img src="https://user-images.githubusercontent.com/24846546/71309203-2274cf00-23fd-11ea-8fbb-fcd0fb36ec1d.png" width="75%">

Results
-----

#### Qualitative

<img src="https://user-images.githubusercontent.com/24846546/71311632-645f3e80-2418-11ea-9bcc-70e8fdc24e1e.png">
<img src="https://user-images.githubusercontent.com/24846546/71311566-c0759300-2417-11ea-91d2-e3bac56843b6.png">
<img src="https://user-images.githubusercontent.com/24846546/71311567-c10e2980-2417-11ea-91fa-d9d871ea1a45.png">
<img src="https://user-images.githubusercontent.com/24846546/71311565-bf446600-2417-11ea-9e08-96ba7a182531.png">


#### Quantitative

<img src="https://user-images.githubusercontent.com/24846546/71311656-b7d18c80-2418-11ea-8acf-238e0b257f6f.jpg" width="70%">

Setup
-----

### Pre-requisites

- Python3
- [PyTorch](https://pytorch.org/)
- CUDA, CUDNN
- [OpenCV](https://opencv.org)
- [PIL](https://pypi.org/project/Pillow/)
- [Numpy](https://numpy.org/)
- [scikit-image](https://scikit-image.org/)
- [tqdm](https://pypi.org/project/tqdm/)


Dataset
--------

The dataset is to comprise of LDR (input) and HDR (ground truth) image pairs. The network is trained to learn the mapping from LDR images to their corresponding HDR ground truth counterparts.

The dataset should follow the following folder structure - 

```
> dataset

    > train

        > LDR

            > ldr_image_1.jpg/png
            > ldr_image_2.jpg/png
            .
            .

        > HDR

            > hdr_image_1.hdr/exr
            > hdr_image_2.hdr/exr
            .
            .

    > test

```

**Note:** The pre-trained models were trained on 256x256 size images.

Training
--------

After the dataset has been prepared, the model can be trained using the **`train.py`** file.

```sh
python3 train.py
```

The corresponding parameters/options for training have been specified in the **`options.py`** file and can be easily altered. They can be logged using -

```sh
python3 train.py --help
```
- **`--iter`** param is used to specify the number of feedback iterations for global and local feedback mechanisms (refer to paper/architecture diagram)
- Checkpoints of the model are saved in the **`checkpoints`** directory. (Saved after every 2 epochs by default)
- GPU is used for training. Specify GPU IDs using **`--gpu_ids`** param.

Evaluation
----------

The performance of the network can be evaluated using the **`test.py`** file - 

```sh
python3 test.py --ckpt_path /path/to/pth/checkpoint
```

- Test results (LDR input, HDR prediction and HDR ground truth) are stored in the **`test_results`** directory.
- If checkpoint path is not specified, it defaults to `checkpoints/latest.ckpt` for evaluating the model.
- PSNR and SSIM scores are provided for quantitative evaluation.


Pretrained models [Coming soon]
---------------------------

<!-- Pre-trained models can be downloaded from the below-mentioned links. 

These models have been trained with the default options, on 256x256 size images for 200 epochs, in accordance with the paper.

- [Feed-forward model]()
- [2-Iterations model]()
- [3-Iterations model]()
- [4-Iterations model]()

Here is a graph plotting performance vs iteration count. 

<img src="https://user-images.githubusercontent.com/24846546/71311250-ed28ab00-2415-11ea-9842-f84b5999161e.png" width="40%"> -->


