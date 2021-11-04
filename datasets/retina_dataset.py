import os
import numpy as np
import random
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from .preprocessing import read_grayscale_img, read_preproc_img, save_img, clahe_equalized, adjust_gamma, change_range, qrandc
from torch import squeeze
import pickle

class RetinaDataset(Dataset):

    def __init__(self,
                 img_dpath,
                 gt_dpath,
                 img_fnames,
                 n_patches=20,
                 img_ext = '.tif',
                 gt_ext ='_manual1.gif',
                 patch_size = 48,
                 transform = False,
                 fov_dpath = None,
                 fov_ext='.png',
                 preproc_name = 'qrandc',
                 save_preproc = True
                 ):

        self.img_dpath = img_dpath
        self.gt_dpath = gt_dpath
        self.fov_dpath = fov_dpath
        self.preproc_name = preproc_name
        self.preproc = self.preproc_func(preproc_name)
        self.img_fnames = img_fnames
        self.gt_ext = gt_ext
        self.fov_ext = fov_ext
        self.patch_size = patch_size
        self.n_patches = n_patches

        self.lst_orig_imgs = []
        self.lst_imgs = []
        self.lst_gts = []
        self.lst_fovs = []
        self.n_channels = None
        self.flag_transform = transform
        self.flag_save = save_preproc

        self.lst_img_filenames = [name for name in img_fnames if name.endswith(img_ext)]

        #if preproc_name == 'qrandc':
        self.qparams_path = self.img_dpath + '/qrandc/qparams.pkl'
        if os.path.exists(self.qparams_path):
            with open(self.qparams_path, 'rb') as file:
                self.qparams = pickle.load(file)
        else:
            self.qparams =None

        for fname_img in self.lst_img_filenames:
            # get name and extension
            fname, fname_ext = os.path.splitext(fname_img)
            # get full path to original image
            img_fpath = os.path.join(self.img_dpath, fname_img)
            # get full path to groud-truth mask
            fname_gt = fname+self.gt_ext
            gt_fpath = os.path.join(self.gt_dpath, fname_gt)
            # read ground-truth mask
            mask = read_grayscale_img(gt_fpath)/255.
            # read image
            img = read_grayscale_img(img_fpath)
            h, w, c = img.shape
            img = self.data_normalization(img, max_val =255, dtype=np.uint8)

            # preprocessing step
            preproc_img_path = self.img_dpath + '/' + self.preproc_name + '/' + fname + '.png'
            if os.path.exists(preproc_img_path):
                to_load = read_preproc_img(preproc_img_path, h, w)
                to_load = to_load.reshape((h, -1, w))
                preproc_img = np.moveaxis(to_load, 1, -1)
            else:
                preproc_img = self.preproc(img[:, :, 0], **{'qparams': self.qparams, 'qparams_path': self.qparams_path})
                preproc_img = preproc_img.reshape((h, w,-1))

            if self.flag_save:
                to_save = np.moveaxis(preproc_img, -1, 1)
                to_save = to_save.reshape((h, -1))
                save_img(to_save, preproc_img_path)

            #patch_ = self.preproc_qrandc(patch)


            if self.fov_dpath is not None:
                fname_fov  = fname + self.fov_ext
                fov_fpath = os.path.join(self.fov_dpath, fname_fov)
                fov = read_grayscale_img(fov_fpath)/255.
                self.lst_fovs.append(fov)

            # save in RAM
            self.lst_orig_imgs.append(img)
            self.lst_imgs.append(preproc_img)
            self.lst_gts.append(mask)

        self.lst_imgs = np.asarray(self.lst_imgs)
        self.lst_gts = np.asarray(self.lst_gts)
        self.lst_fovs = np.asarray(self.lst_fovs)
        # global normalization
        self.lst_imgs = self.data_normalization(self.lst_imgs, max_val=1., dtype=np.float)
        self.n_imgs = len(self.lst_img_filenames)

    def preproc_grayscale(self, image, **kargs):
        image = clahe_equalized(image)
        image = adjust_gamma(image, 1.2)
        return image

    def preproc_qrandc(self, image, qparams = None, qparams_path = './qparams.pkl'):
        image = clahe_equalized(image)
        image = adjust_gamma(image, 1.2)
        image = qrandc(image / 255., qparams=qparams, qparams_path = qparams_path)
        image = (image*255).astype(np.uint8)
        return image

    def preproc_func(self, name):
        return dict(grayscale = self.preproc_grayscale, qrandc = self.preproc_qrandc)[name]

    def data_normalization(self, img, max_val=255., dtype=np.uint8):
        img_std = np.std(img)
        img_mean = np.mean(img)
        img_normalized = (img - img_mean) / img_std
        img_normalized = change_range(img_normalized, img_normalized.min(), img_normalized.max(), 0., max_val)

        return img_normalized.astype(dtype)

    def transform(self, image, mask, prob=0.):
        # Border crop
        h, w, _ = image.shape
        min_dim2 = (np.max([h, w]) - np.min([h, w])) // 2
        if w < h:
            image = image[min_dim2:-min_dim2]  # cut bottom and top
            mask = mask[min_dim2:-min_dim2]  # cut bottom and top
        else:
            image = image[:, min_dim2:-min_dim2]  # cut left and right
            mask = mask[:, min_dim2:-min_dim2]  # cut left and right

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        # Random patch crop
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.patch_size, self.patch_size))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        # Random horizontal flipping
        if random.random() > (1.-prob):
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > (1.-prob):
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        return image, mask

    def __getitem__(self, i):

        i = i % self.n_imgs  # equally extracting patches from each image
        img = self.lst_imgs[i]
        mask = self.lst_gts[i]
        img_h, img_w, _ = img.shape
        self.img_h = img_h
        self.img_w = img_w

        prob = 0.3
        patch, patch_mask = self.transform(img, mask, prob=self.flag_transform * prob)
        # remove dim for patch_mask
        patch_mask = squeeze(patch_mask, dim=0)
        return patch, patch_mask

    def __len__(self):
        return self.n_patches

