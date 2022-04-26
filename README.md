# D-GaussianNet (Retinal Vessel Segmentation)
PyTorch implementation of D-GaussianNet, based on conference paper [D-GaussianNet: Adaptive Distorted Gaussian Matched Filter with Convolutional Neural Network for Retinal Vessel Segmentation](https://link.springer.com/chapter/10.1007%2F978-3-030-72073-5_29), presented in The International Symposium on Geometry and Vision (ISGV 2021). 

![](figures/graphical_abstract.png)

If you use this code please cite the paper using the bibtex reference below.

```
@InProceedings{Alvarado2021_dgaussiannet, 
  title={D-GaussianNet: Adaptive Distorted Gaussian Matched Filter with Convolutional Neural Network for Retinal Vessel Segmentation},
  author={Alvarado-Carrillo, Dora E and Ovalle-Magallanes, Emmanuel and Dalmau-Cede{\~n}o, Oscar S},
  editor="Nguyen, Minh and Yan, Wei Qi and Ho, Harvey",
  booktitle="Geometry and Vision",
  year="2021",
  publisher="Springer International Publishing",
  address="Cham",
  pages="378--392",
  organization={Springer}, 
  isbn="978-3-030-72073-5", 
  doi={10.1007/978-3-030-72073-5_29}
}
```

## Abstract

Automating retinal vessel segmentation is a primary element of computer-aided diagnostic systems for many retinal diseases. It facilitates the inspection of shape, width, tortuosity, and other blood vessel characteristics. In this paper, a new method that incorporates Distorted Gaussian Matched Filters (D-GMFs) with adaptive parameters as part of a Deep Convolutional Architecture is proposed. The D-GaussianNet includes D-GMF units, a variant of the Gaussian Matched Filter that considers curvature, 
placed at the beginning and end of the network to implicitly indicate that spatial attention should focus on curvilinear structures in the image. Experimental results on datasets DRIVE, STARE, and CHASE show state-of-the-art performance with an accuracy of 0.9565, 0.9647, and 0.9609 and a F1-score of 0.8233, 0.8141, and 0.8077, respectively.

## Prerequisities
The neural network is mainly developed with Pytorch, Kornia and PennyLane libraries, we refer to [Pytorch website](https://pytorch.org/), [Kornia Repository](https://github.com/kornia/kornia) for the installation.
This code has been tested with pytorch 1.7.0, torchvision 0.8.1, pennylane 0.12.0, kornia 0.6.1, and CUDA 10.2. 

The following dependencies are needed:
- pytorch >=1.7.0
- torchvision >= 0.8.1
- pennylane >=0.12.0
- kornia > = 0.6.1
- numpy >= 1.18.1
- Pillow >=7.0.0
- opencv >= 4.2.0
- scikit-learn >= 0.22.1

Additionally, the [DRIVE](https://drive.grand-challenge.org/) dataset is needed.  We are not allowed to provide the dataset here, but it can be freely downloaded from the official website by joining the challenge. 
The images should be included in the path "./datasets/DRIVE". This folder should have the following structure:

```
DRIVE
│
└───test
|    ├───1st_manual
|    └───2nd_manual
|    └───images
|    └───mask
│
└───training
    ├───1st_manual
    └───images
    └───mask
```

This structure can be modified in the file "./trainer/dataset_config_file.py".

## Training

The model can be trained with:

```
python main_train.py 
python main_train.py -c config.txt
python main_train.py --param param_value
```

Execute "main_train.py -h" for more information on available parameters. First option use default parameter values, which can be modified in the file "./utils/settings.py", second option overrides default parameters with the specified in a configfile. Third option overrides default and configfile values. 

If available, a GPU will be used. The following files will be saved in the folder "exp1" :
- best model weights (model_best.pth.tar)
- last model weights (model_checkpoint.pth.tar)

*Please notice that the quantum preprocessing requires more time in a non-quantum computer. Each image from DRIVE dataset (i.e. height: 584 pixels, width:565 pixels) took around an hour and a half to be preprocessed in a regular computer (Processor: AMD Ryzen 5 2.10GHz, RAM: 8GB).*




