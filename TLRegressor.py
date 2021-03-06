from __future__ import print_function, division
import os
import torch
import pandas as pd
import time
from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, sampler
import torch.optim as optim

from torchvision import transforms, models
from torchvision.utils import save_image, make_grid

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

##
from Classifier.torchImbalancedDatasetSampler import ImbalancedDatasetSampler

from Classifier.regression_evaluation import regression_metrics
from Classifier.regression_visualization import plotDist, plotTruePredScatter

##

def fixSeed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False
    return None


##
def plotCSVData(alldata, nc, img_size, nsamples=100):

    topo_array = alldata.iloc[:nsamples, 0:img_size**2].values
    topo_tensor = np.reshape(topo_array, (nsamples, nc, img_size, img_size), order='C')
    topo_tensor = torch.from_numpy(topo_tensor).type(torch.FloatTensor)

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Images in CSV")
    plt.imshow(np.transpose(make_grid(topo_tensor[:64], padding=5, pad_value=1, nrow=int(np.sqrt(nsamples)), normalize=True).numpy(), (1, 2, 0)))
    plt.show()

    return None

##
class TopoDataset(Dataset):
    """Topo dataset."""

    def __init__(self, csv_file, img_size, transform=None):
        """
        Args:
            csv_file (Pandas Dataframe): csv file read by Pandas.
            img_size (int): image size.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.topo_data = csv_file
        self.img_size = img_size
        self.transform = transform

        self.topos = self.topo_data.iloc[:, :img_size**2].values
        self.bandgaps = self.topo_data.iloc[:, self.img_size ** 2].values
        self.labels = self.topo_data.iloc[:, self.img_size ** 2 + 1].values

        self.original_bandgaps = self.bandgaps.copy()

    def __len__(self):
        return len(self.topo_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.topos[idx, 0:self.img_size ** 2].reshape(self.img_size, self.img_size, order='C').astype(float)
        bandgap = self.bandgaps[idx]
        label = self.labels[idx]
        sample = {'image': image, 'bandgap': bandgap, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def staticAugMinority(self):
        class_sample_counts = np.bincount(self.labels)  # 0: success, 1: failure
        n_negative = class_sample_counts[0]
        n_positive = class_sample_counts[1]
        candidate_indices = np.random.choice(np.where(self.labels == 1)[0], size=(n_negative - n_positive), replace=True)
        candidates = []
        for idx in candidate_indices:
            image = self.topos[idx, 0:self.img_size ** 2].reshape(self.img_size, self.img_size, order='C').astype(float)
            bandgap = self.bandgaps[idx]
            label = self.labels[idx]
            sample = {'image': image, 'bandgap': bandgap, 'label': label}

            sample = QuarterCrop(img_size / 2)(sample)
            sample = ToPILImage()(sample)
            if random.random() < 0.5:
                sample = RandomFlip(p=1)(sample)  # data augmentation for training set
            else:
                sample = RandomRotation([90, 180, 270])(sample)

            # plt.figure()
            # plt.imshow(sample['image'], cmap='gray')
            # plt.show()

            sample['image'] = np.array(sample['image'])/255
            sample['image'] = sample['image'][:, :, 0]

            # plt.figure()
            # plt.imshow(sample['image'], cmap='gray', vmin=0, vmax=1)
            # plt.show()

            image_rt = sample['image']
            image_lt = np.flip(image_rt, 1)
            image_rb = np.flip(image_rt, 0)
            image_lb = np.flip(image_lt, 0)
            sample['image'] = np.block([[image_lt, image_rt], [image_lb, image_rb]])

            # plt.figure()
            # plt.imshow(sample['image'], cmap='gray', vmin=0, vmax=1)
            # plt.show()

            sample['image'] = sample['image'].ravel()
            original_dict = {}
            for i in range(len(sample['image'])):
                original_dict[str(i)] = sample['image'][i]
            original_dict['bandgap'] = sample['bandgap']
            original_dict['label'] = sample['label']
            candidates.append(original_dict)

        candidates = pd.DataFrame(candidates)
        self.topo_data = self.topo_data.append(candidates)
        self.topos = self.topo_data.iloc[:, :img_size ** 2].values
        self.bandgaps = self.topo_data.iloc[:, self.img_size ** 2].values
        self.labels = self.topo_data.iloc[:, self.img_size ** 2 + 1].values

        return None


class QuarterCrop(object):
    """Crop the image to the quarter size"""

    def __init__(self, crop_size):
        assert isinstance(crop_size, (int, float, tuple))
        if isinstance(crop_size, (int, float)):
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            assert len(crop_size) == 2
            self.crop_size = (int(crop_size[0]), int(crop_size[1]))

    def __call__(self, sample):
        image, bandgap, label = sample['image'], sample['bandgap'], sample['label']

        return {'image': image[:self.crop_size[0], self.crop_size[1]:],
                'bandgap': bandgap,
                'label': label}


class ToPILImage(object):
    def __call__(self, sample):
        image, bandgap, label = sample['image'], sample['bandgap'], sample['label']

        # plt.figure()
        # plt.imshow(image, cmap='gray', vmin=0, vmax=1)
        # plt.show()

        if len(image.shape) == 2:
            image = np.array([image, image, image])

        if image.shape[0] in (1, 3):  # single channel or 3-channels, n_channels x H x W -> H x W x n_channels
            image = np.transpose(image, (1, 2, 0))

        image = Image.fromarray(255*image.astype(np.uint8))

        # plt.figure()
        # plt.imshow(image, cmap='gray')
        # plt.show()

        return {'image': image,
                'bandgap': bandgap,
                'label': label}


class RandomRotation(object):
    """Rotate the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
        fill (3-tuple or int): RGB pixel fill value for area outside the rotated image.
            If int, it is used for all channels respectively.

    """

    def __init__(self, degrees, resample=False, expand=False, center=None, fill=0):

        if len(degrees) < 2:
            raise ValueError("If degrees is a sequence, it must be greater than len 2.")
        self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center
        self.fill = fill

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.choice(degrees)

        return angle

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """

        image, bandgap, label = sample['image'], sample['bandgap'], sample['label']

        angle = self.get_params(self.degrees)

        image = transforms.functional.rotate(image, angle, self.resample, self.expand, self.center, self.fill)

        # p = 0.8
        # if label == 1:
        #     if random.random() <= p:
        #         image = transforms.functional.rotate(image, angle, self.resample, self.expand, self.center, self.fill)
        # else:
        #     if random.random() > p:
        #         image = transforms.functional.rotate(image, angle, self.resample, self.expand, self.center, self.fill)

        return {'image': image,
                'bandgap': bandgap,
                'label': label}


class RandomFlip(object):
    """Random vertical or horizaontal flip"""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, bandgap, label = sample['image'], sample['bandgap'], sample['label']

        # plt.figure()
        # plt.imshow(image, cmap='gray', vmin=0, vmax=1)
        # plt.show()


        flip_list = [
            transforms.RandomHorizontalFlip(p=self.p),
            transforms.RandomVerticalFlip(p=self.p),
        ]

        image = transforms.RandomChoice(flip_list)(image)

        # p = 0.8
        # if label == 1:
        #     if random.random() <= p:
        #         image = transforms.RandomChoice(flip_list)(image)
        #
        # else:
        #     if random.random() > p:
        #         image = transforms.RandomChoice(flip_list)(image)

        # plt.figure()
        # plt.imshow(image, cmap='gray')
        # plt.show()

        return {'image': image,
                'bandgap': bandgap,
                'label': label}


class Resize(object):
    """Resize"""

    def __init__(self, size, interpolation=2, zero_padding=False):

        # 'Image.NEAREST (0)', 'Image.LANCZOS (1)', 'Image.BILINEAR (2)',
        # 'Image.BICUBIC (3)', 'Image.BOX (4)', 'Image.HAMMING (5)'

        assert isinstance(size, (int, float, tuple))
        if isinstance(size, (int, float)):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2
            self.size = (int(size[0]), int(size[1]))

        self.interpolation = interpolation
        self.zero_padding = zero_padding

    def __call__(self, sample):
        image, bandgap, label = sample['image'], sample['bandgap'], sample['label']

        # plt.figure()
        # plt.imshow(image, cmap='gray')
        # plt.show()

        h_diff = self.size[0] - image.height
        w_diff = self.size[1] - image.width

        if self.zero_padding and h_diff > 0 and w_diff > 0:
            # Hashemi, M. (2019). “Enlarging smaller images before inputting into convolutional neural network:
            #        zero-padding vs. interpolation.” J. Big Data, Springer International Publishing, 6(1).
            pad = (w_diff//2, w_diff - w_diff//2, h_diff//2, h_diff - h_diff//2)
            image_tensor = torch.from_numpy(np.transpose(np.array(image), (2, 0, 1))/255).unsqueeze(0)
            # 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
            image_tensor_padded = torch.nn.functional.pad(image_tensor, pad=pad, mode='constant')
            image = Image.fromarray(255*np.transpose(image_tensor_padded.squeeze(0).numpy(), (1, 2, 0)).astype(np.uint8))
        else:
            image = transforms.Resize(self.size, interpolation=self.interpolation)(image)

        # plt.figure()
        # plt.imshow(image, cmap='gray')
        # plt.show()

        return {'image': image,
                'bandgap': bandgap,
                'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, bandgap, label = sample['image'], sample['bandgap'], sample['label']

        image = transforms.ToTensor()(image)

        return {'image': image,
                'bandgap': torch.FloatTensor([bandgap]),
                'label': torch.FloatTensor([label])}


class Normalize(object):
    """Normalize tensors from [0, 1] to [-1, 1]"""

    def __init__(self, mean, std):
        assert isinstance(mean, (int, float, tuple, list))
        assert isinstance(std, (int, float, tuple, list))

        if isinstance(mean, (int, float)):
            nc = 1
        else:
            nc = len(mean)

        self.mean = torch.tensor(mean).view(nc, 1, 1)
        self.std = torch.tensor(std).view(nc, 1, 1)

    def __call__(self, sample):
        image, bandgap, label = sample['image'], sample['bandgap'], sample['label']

        image = (image - self.mean) / self.std

        return {'image': image,
                'bandgap': bandgap,
                'label': label}


def UnnormalizeTensor(img_tensor, mean, std):
    # transform tensor to [0, 1]
    if isinstance(mean, (int, float)):
        nc = 1
    else:
        nc = len(mean)

    if len(img_tensor.shape) == 3:
        # single image, 3 dims
        unnorm_img_tensor = img_tensor * torch.tensor(std).view(nc, 1, 1) + torch.tensor(mean).view(nc, 1, 1)
    else:
        # a batch of images, 4 dims
        unnorm_img_tensor = img_tensor * torch.tensor(std).view(nc, 1, 1).expand(img_tensor.size(0), nc, 1, 1) + torch.tensor(mean).view(nc, 1, 1).expand(img_tensor.size(0), nc, 1, 1)

    return unnorm_img_tensor


##
class TargetScaler(object):
    """normalize target variabel for regression"""

    def __init__(self, mode='Standard', scalers=None):

        if scalers is None:
            scalers = {
                'Standard':  preprocessing.StandardScaler(),
                'MinMax': preprocessing.MinMaxScaler(),
                'MaxAbs': preprocessing.MaxAbsScaler(),
                'Robust': preprocessing.RobustScaler(quantile_range=(25, 75)),
                'Power_Yeo-Johnson': preprocessing.PowerTransformer(method='yeo-johnson'),
                'Power_Box-Cox': preprocessing.PowerTransformer(method='box-cox'),  # only for strictly positive data
                'Quantile_gaussian': preprocessing.QuantileTransformer(output_distribution='normal'),
                'Quantile_uniform': preprocessing.QuantileTransformer(output_distribution='uniform'),
                'L2Norm': preprocessing.Normalizer(),
            }

        self.scaler = scalers[mode]
        self.fitted_scaler = None

    def __call__(self, x):
        # x is the data
        # return the fitted_scaler, we can use .transform() and .inverse_transform()

        # .fit() expects 2D array
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        self.fitted_scaler = self.scaler.fit(x)
        return None

    def transform(self, x):
        # x is the data
        # .transform() expects 2D array
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        x_transformed = self.fitted_scaler.transform(x)
        if any(np.array(x_transformed.shape) == 1):
            x_transformed = x_transformed.squeeze()
        return x_transformed

    def inverse_transform(self, x):
        # x is the data
        # .transform() expects 2D array
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        x_inverse_transformed = self.fitted_scaler.inverse_transform(x)
        if any(np.array(x_inverse_transformed.shape) == 1):
            x_inverse_transformed = x_inverse_transformed.squeeze()
        return x_inverse_transformed


##
def plotDataset(topoDataset, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), nsamples=4):
    fig = plt.figure()
    for i in range(len(topoDataset)):
        sample = topoDataset[i]

        print('the ', i, ' image: shape ', sample['image'].shape, ', bandgap: ', sample['bandgap'].shape,
              ', label: ', sample['label'].shape, ', class: ', classes[int(sample['label'].item())])

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        unnorm_sample = UnnormalizeTensor(sample['image'], mean, std)
        plt.imshow(np.transpose(unnorm_sample, (1, 2, 0)).squeeze(), cmap='gray', vmin=0, vmax=1)

        if i == nsamples - 1:
            fig.show()
            break
    return None


## Helper function to show a batch
def show_batch(sample_batched, i_batch):
    """Show image for a batch of samples."""
    images_batch, bandgaps_batch, lables_batch = \
            sample_batched['image'], sample_batched['bandgap'], sample_batched['label']

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    ax.set_title("Batch {} of Images".format(i_batch))
    ax.imshow(np.transpose(make_grid(images_batch, padding=5, pad_value=1, nrow=int(np.sqrt(sample_batched['image'].shape[0])), normalize=True).numpy(), (1, 2, 0)))
    fig.show()
    return None


def plotDataLoader(train_loader, val_loader, n_batch=4):

    for i_batch, sample_batched in enumerate(train_loader):
        print('the batch', i_batch, ' image batch: ', sample_batched['image'].shape,
              ', bandgap: ', sample_batched['bandgap'].shape, ', label: ', sample_batched['label'].shape)

        print('class: ', ', '.join(
            '{}'.format(classes[int(sample_batched['label'][j, 0].item())]) for j in range(sample_batched['image'].shape[0])))

        # observe 4th batch and stop.
        if i_batch < n_batch:
            show_batch(sample_batched, i_batch)
            if i_batch == n_batch - 1:
                break

    for i_batch, sample_batched in enumerate(val_loader):
        print('the batch', i_batch, ' image batch: ', sample_batched['image'].shape,
              ', bandgap: ', sample_batched['bandgap'].shape, ', label: ', sample_batched['label'].shape)

        print('class: ', ', '.join(
            '{}'.format(classes[int(sample_batched['label'][j, 0].item())]) for j in range(sample_batched['image'].shape[0])))

        # observe 4th batch and stop.
        if i_batch < n_batch:
            show_batch(sample_batched, i_batch)
            if i_batch == n_batch - 1:
                break

    return None


##
class LeNet(nn.Module):
    def __init__(self, nc, n_gpu):
        super(LeNet, self).__init__()
        self.n_gpu = n_gpu

        self.conv = nn.Sequential(
            # w_out=floor((w_in - kernel_size + 2 * padding) / stride + 1)

            # input state size. nc x 32 x 32
            nn.Conv2d(nc, 6, kernel_size=5, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            # state size. 6 x 28 x 28

            nn.Conv2d(6, 6, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, 2),
            # nn.Dropout2d(0.5),
            # state size. 6 x 14 x 14

            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(16),
            # nn.Dropout2d(0.5),
            nn.ReLU(inplace=True),
            # state size. 16 x 10 x 10

            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, 2),
            # nn.Dropout2d(0.5),
            # state size. 16 x 5 x 5
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.5),

            nn.Linear(16 * 5 * 5, 120),
            nn.BatchNorm1d(120),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),

            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),

            nn.Linear(84, 1),
        )

    def forward(self, x):
        output = self.conv(x)
        # print(output.shape)
        output = output.view(output.shape[0], -1)
        # print(output.shape)
        output = self.fc(output)
        return output


class LeNet_small(nn.Module):
    def __init__(self, nc, n_gpu):
        super(LeNet_small, self).__init__()
        self.n_gpu = n_gpu

        self.conv = nn.Sequential(
            # w_out=floor((w_in - kernel_size + 2 * padding) / stride + 1)

            # input state size. nc x 32 x 32
            nn.Conv2d(nc, 6, kernel_size=5, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            # state size. 6 x 28 x 28

            nn.MaxPool2d(2, 2),
            # nn.Dropout2d(0.5),
            # state size. 6 x 14 x 14

        )

        self.fc = nn.Sequential(
            nn.Linear(6 * 14 * 14, 120),
            nn.BatchNorm1d(120),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),  # dropout is usually applied to the fully connected layers

            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),

            nn.Linear(84, 1),
        )

    def forward(self, x):
        output = self.conv(x)
        # print(output.shape)
        output = output.view(output.shape[0], -1)
        # print(output.shape)
        output = self.fc(output)
        return output


class LeNet_gap(nn.Module):
    def __init__(self, nc, n_gpu):
        super(LeNet_gap, self).__init__()
        self.n_gpu = n_gpu

        self.conv = nn.Sequential(
            # w_out=floor((w_in - kernel_size + 2 * padding) / stride + 1)

            # input state size. nc x 32 x 32
            nn.Conv2d(nc, 6, kernel_size=5, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            # state size. 6 x 28 x 28

            nn.Conv2d(6, 16, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # state size. 16 x 14 x 14

            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # state size. 32 x 10 x 10

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # state size. 64 x 5 x 5

            # nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
            # state size. 128 x 5 x 5
        )

        # global average pooling
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5),  # fc is prone to overfitting
            # state size. 64 x 1 x 1
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 1)
        )

    def forward(self, x):
        output = self.conv(x)
        output = self.gap(output)
        # print(output.shape)
        output = output.view(output.shape[0], -1)
        # print(output.shape)
        output = self.fc(output)
        return output


class MyNet(nn.Module):
    def __init__(self, nc, n_gpu):
        super(MyNet, self).__init__()
        self.n_gpu = n_gpu

        self.conv = nn.Sequential(
            # w_out=floor((w_in - kernel_size + 2 * padding) / stride + 1)

            # input state size. nc x 5 x 5
            nn.ReplicationPad2d((1, 1, 1, 1)),
            nn.Conv2d(nc, 6, kernel_size=3, stride=1, padding=0, bias=True),
            # nn.Conv2d(nc, 6, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            # state size. 6 x 5 x 5

            nn.Conv2d(6, 16, kernel_size=4, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # state size. 16 x 2 x 2

            # nn.AvgPool2d(kernel_size=2, stride=None, padding=0),
            # state size. 16 x 1 x 1
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(16*2*2, 1)
        )

    def forward(self, x):
        output = self.conv(x)
        # print(output.shape)
        output = output.view(output.shape[0], -1)
        # print(output.shape)
        output = self.fc(output)
        return output


##
class HuberLoss(torch.nn.Module):
    def __init__(self, HuberDelta=1, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.delta = HuberDelta

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        result = torch.empty_like(ey_t)
        result[torch.abs(ey_t) <= self.delta] = 0.5 * ey_t[torch.abs(ey_t) <= self.delta] ** 2
        result[torch.abs(ey_t) > self.delta] = self.delta * torch.abs(ey_t[torch.abs(ey_t) > self.delta]) - 0.5 * self.delta ** 2

        if self.reduction == 'sum':
            return torch.sum(result)
        elif self.reduction == 'mean':
            return torch.mean(result)
        elif self.reduction == 'none':
            return result
        else:
            raise ValueError("the input reduction mode is not implemented.")


class LogCoshLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        result = torch.log(torch.cosh(ey_t + 1e-12))
        if self.reduction == 'sum':
            return torch.sum(result)
        elif self.reduction == 'mean':
            return torch.mean(result)
        elif self.reduction == 'none':
            return result
        else:
            raise ValueError("the input reduction mode is not implemented.")


class XTanhLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        result = ey_t * torch.tanh(ey_t)
        if self.reduction == 'sum':
            return torch.sum(result)
        elif self.reduction == 'mean':
            return torch.mean(result)
        elif self.reduction == 'none':
            return result
        else:
            raise ValueError("the input reduction mode is not implemented.")


class XSigmoidLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        result = 2 * ey_t / (1 + torch.exp(-ey_t)) - ey_t
        if self.reduction == 'sum':
            return torch.sum(result)
        elif self.reduction == 'mean':
            return torch.mean(result)
        elif self.reduction == 'none':
            return result
        else:
            raise ValueError("the input reduction mode is not implemented.")


def chooseRegressionCriterion(loss_name='MSE', reduction='mean', HuberDelta=1):
    loss_funcs = {
        'MAE': nn.L1Loss(reduction=reduction),
        'MSE': nn.MSELoss(reduction=reduction),
        'HuberLoss': HuberLoss(HuberDelta=HuberDelta, reduction=reduction),
        'LogCoshLoss': LogCoshLoss(reduction=reduction),
        'XTanhLoss': XTanhLoss(reduction=reduction),
        'XSigmoidLoss': XSigmoidLoss(reduction=reduction),

    }
    return loss_funcs[loss_name]


##
def train_val_NN(net, phase, train_loader, val_loader, optimizer, scheduler, criterion,
                 train_size, val_size, batch_size, n_epochs, device, model_savepath, use_target_scaler, target_scaler,
                 use_manual_reg, LAMBDA_conv, LAMBDA_fc, fc_reg_form, use_fc_weight_clip, SparsificationFactor,
                 metric_names, name_funcs_dict):

    since = time.time()

    ## Training or validation
    if phase == 'train':
        net.train()
        dataset_size = train_size
        data_loader = train_loader
    else:  # validation
        net.eval()  # in evaluation model and deactivate batchnorm and dropout layers
        dataset_size = val_size
        data_loader = val_loader

    ##
    running_loss_list = []
    scores_list = []

    iters = 0
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        if phase != 'train':
            net.load_state_dict(torch.load(model_savepath + '/net_epoch_{}.pth'.format(epoch)))

        running_loss = 0.0
        ypred = np.zeros(dataset_size)
        ytrue = np.zeros(dataset_size)

        for i, sample_batched in enumerate(data_loader, 0):

            images_batch, bandgaps_batch, labels_batch = \
                sample_batched['image'].to(device), \
                sample_batched['bandgap'].to(device), \
                sample_batched['label'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # optimize if only in phase == 'train'
            with torch.set_grad_enabled(phase == 'train'):  # comparable to torch.no_grad()

                # forward
                outputs = net(images_batch)
                # regression, scale up the loss since the bandgap value is small
                loss = criterion(outputs, bandgaps_batch)

                # backward + optimize
                if phase == 'train':
                    if use_manual_reg:
                        reg = cal_regPenalty(net, LAMBDA_conv=LAMBDA_conv, LAMBDA_fc=LAMBDA_fc, fc_reg_form=fc_reg_form)
                        loss_total = loss + reg
                        loss_total.backward()
                    else:
                        loss.backward()
                    optimizer.step()

                    if epoch < n_epochs and use_fc_weight_clip:
                        clip_fc_weight(net, SparsificationFactor=SparsificationFactor)

            # print statistics
            # the default reduction mode of loss function is 'mean'
            running_loss += loss.item() * images_batch.shape[0]

            ypred[i * batch_size: i * batch_size + images_batch.shape[0]] = outputs.detach().squeeze().numpy()
            ytrue[i * batch_size: i * batch_size + images_batch.shape[0]] = bandgaps_batch.squeeze().numpy()

            if i == len(data_loader) - 1:
                assert i * batch_size + images_batch.shape[0] == dataset_size

                if use_target_scaler:
                    ypred = target_scaler.inverse_transform(ypred)
                    ytrue = target_scaler.inverse_transform(ytrue)

                scores = regression_metrics(y_true=ytrue, y_pred=ypred,
                                            metric_names=metric_names, name_funcs_dict=name_funcs_dict)
                scores_list.append(scores)

                running_loss_list.append(running_loss / dataset_size)
                running_loss = 0.0

                print('[{:d}, {:5d}] loss: {:.3f}  '.format(epoch + 1, i + 1, running_loss_list[-1]) +
                      ' '.join('{}: {:.3f} '.format(metric_name, scores[metric_name]) for metric_name in metric_names))

                if phase == 'train':
                    torch.save(net.state_dict(), model_savepath + '/net_epoch_{}.pth'.format(epoch))
                    torch.save(optimizer.state_dict(), model_savepath + '/opti_epoch_{}.pth'.format(epoch))

            iters += 1

        if phase == 'train':
            scheduler.step()
            # adjust_learning_rate(epoch, optimizer)

    time_elapsed = time.time() - since
    print('************* Finished {} *************'.format(phase))
    print('Finish {} in {:.0f}m {:.0f}s'.format(phase, time_elapsed // 60, time_elapsed % 60))
    return running_loss_list, scores_list


##
def cal_regPenalty(net, LAMBDA_conv=1e-3, LAMBDA_fc=1e-3, fc_reg_form=2):
    reg_conv = None
    reg_fc = None
    for name, p in net.named_parameters():
        if 'weight' in name:

            if 'conv' in name:
                # only L2 regularization works well for conv layers (squared L2 norm)
                if reg_conv is None:
                    reg_conv = p.norm(2) ** 2
                else:
                    # cannot write as in-place operation reg_conv += p.norm(2)**2
                    # because it will change auto-gradient behavior
                    reg_conv = reg_conv + p.norm(2) ** 2

            elif 'fc' in name:
                if reg_fc is None:
                    reg_fc = p.norm(fc_reg_form) ** fc_reg_form
                else:
                    # cannot write as in-place operation reg_fc += p.norm(fc_reg_form)**fc_reg_form
                    # because it will change auto-gradient behavior
                    reg_fc = reg_fc + p.norm(fc_reg_form) ** fc_reg_form

            else:
                continue

    reg = LAMBDA_conv * (reg_conv if reg_conv else 0) + LAMBDA_fc * (reg_fc if reg_fc else 0)
    return reg


def clip_fc_weight(net, SparsificationFactor=0.2):
    for name, p in net.named_parameters():
        if 'fc' in name and 'weight' in name:
            # print(p)
            threshold = np.quantile(p.data.abs().flatten().numpy(), SparsificationFactor)
            # print(threshold)
            p.data[p.data.abs() < threshold] = 0
            # print(p)
            # print(list(net.named_parameters()))
            break  # only modify the first fc layer after conv layers

    return None


def adjust_learning_rate(epoch, optimizer):

    lr = 0.001

    if epoch >= 20:
        lr = 0.0001
    elif epoch >= 70:
        lr = 0.00005
    elif epoch >= 90:
        lr = 0.000005


    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return None



##
def plotRunningLoss(running_loss_train_list, running_loss_val_list, n_epochs, result_savepath):
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(111)
    ax1.set_title("Running Loss During Training and Validation")
    ax1.plot(range(1, n_epochs + 1), running_loss_train_list, label="train")
    ax1.plot(range(1, n_epochs + 1), running_loss_val_list, label="validation")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    fig.show()
    fig.savefig(result_savepath + './compare_runningloss.png', bbox_inches='tight')
    return None


def plotScore(scores_train_df, scores_val_df, metric_name, n_epochs):
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(111)
    ax1.set_title("{} During Training and Validation".format(metric_name))
    ax1.plot(range(1, n_epochs + 1), scores_train_df[metric_name], label="train")
    ax1.plot(range(1, n_epochs + 1), scores_val_df[metric_name], label="validation")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("{}".format(metric_name))
    ax1.legend()

    fig.show()
    fig.savefig(result_savepath + './compare_{}.png'.format(metric_name), bbox_inches='tight')
    return None

##
def modelOutput(iepoch, net, test_loader, test_size, model_savepath, use_target_scaler, target_scaler):
    net.eval()
    ypred = np.zeros(test_size)
    ytest = np.zeros(test_size)
    with torch.no_grad():
        net.load_state_dict(torch.load(model_savepath + '/net_epoch_{}.pth'.format(iepoch)))
        for i, sample_batched in enumerate(test_loader, 0):
            images_batch, bandgaps_batch, labels_batch = \
                sample_batched['image'].to(device), \
                sample_batched['bandgap'].to(device), \
                sample_batched['label'].to(device)
            # forward
            outputs = net(images_batch)

            ypred[i * batch_size: i * batch_size + images_batch.shape[0]] = outputs.detach().squeeze().numpy()
            ytest[i * batch_size: i * batch_size + images_batch.shape[0]] = bandgaps_batch.squeeze().numpy()

        if use_target_scaler:
            ypred = target_scaler.inverse_transform(ypred)
            ytest = target_scaler.inverse_transform(ytest)
    return ytest, ypred


##
if __name__ == '__main__':

    seed_value = 999
    top_percentile = 0.2
    use_rawdata = False

    result_savepath = './TLRegressor_result'
    model_savepath = result_savepath + '/model'

    classes = ('success', 'failure')  # 0: success, 1: failure

    img_size = 10
    nc = 1
    n_cpu = 0   # set this to 0 to use GPU in Windows
    n_gpu = 0

    # Resize interpolation mode: 'Image.NEAREST (0)', 'Image.LANCZOS (1)', 'Image.BILINEAR (2)',
    #                            'Image.BICUBIC (3)', 'Image.BOX (4)', 'Image.HAMMING (5)'
    interpolation = 4
    zero_padding = True  # if True, interpolation is ignored, use zero to enlarge small images

    use_pretrained = False

    # 'Standard', 'MinMax', 'MaxAbs', 'Robust', 'Power_Yeo-Johnson', 'Power_Box-Cox',
    # 'Quantile_gaussian', 'Quantile_uniform', 'L2Norm'
    use_target_scaler = True
    target_scaler = TargetScaler(mode='Power_Yeo-Johnson')  # 'Robust', 'Power_Yeo-Johnson'

    n_epochs = 300
    batch_size = 300  # batch_size should be large enough to ensure that each batch contains some minority class
    lr = 0.001  # 0.001
    momentum = 0.9  # SGD
    betas = (0.9, 0.999)  # betas for Adam

    loss_name = 'HuberLoss'  # MSE, MAE, HuberLoss, LogCoshLoss, XTanhLoss, XSigmoidLoss
    HuberDelta = 1  # for HuberLoss
    reduction_mode = 'mean'

    use_manual_reg = False
    LAMBDA_conv = 1e-2
    LAMBDA_fc = 2e-5
    fc_reg_form = 1

    weight_decay = 0  # PyTorch L2 regularization for both weights and biases

    use_fc_weight_clip = False
    SparsificationFactor = 0.2

    use_ImbalancedDataSampler = -1  # -1: no sampler, 0: PyTorch weighted random sampler, 1: imbalanced dataset sampler

    metric_names = ('explained_variance', 'neg_max_error', 'neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2')
    name_funcs_dict = None

    train_metric_df = pd.DataFrame(columns=metric_names)
    val_metric_df = pd.DataFrame(columns=metric_names)

    ##
    os.makedirs(result_savepath, exist_ok=True)
    os.makedirs(model_savepath, exist_ok=True)

    cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda:0" if (cuda and n_gpu > 0) else "cpu")
    fixSeed(seed_value, cuda)


    ##
    if use_rawdata:
        data = pd.read_excel('data.xlsx', sheet_name=['x1', 'y1', 'x2', 'y2'],  header=None)

        data['x1']['bandgap'] = data['y1'].loc[:, 0]
        data['x2']['bandgap'] = data['y2'].loc[:, 0]

        data['x1'] = data['x1'].set_index(list(range(img_size**2)))
        data['x2'] = data['x2'].set_index(list(range(img_size**2)))

        alldata = data['x1'].merge(data['x2'], how='outer', left_index=True, right_index=True, suffixes=('_1', '_2'))
        alldata['bandgap'] = alldata.loc[:, ['bandgap_1', 'bandgap_2']].min(axis=1)
        alldata = alldata.drop(['bandgap_1', 'bandgap_2'], axis=1)
        alldata = alldata.reset_index()

        threshold = alldata['bandgap'].quantile(1-top_percentile)
        alldata['label'] = alldata['bandgap'].apply(lambda x: 0 if x < threshold else 1)

        alldata.to_csv('./cleandata.csv', header=True, index=False)
    else:

        alldata = pd.read_csv('./cleandata.csv', header=0, index_col=None)

        if 'label' not in alldata.columns:
            threshold = alldata['bandgap'].quantile(1 - top_percentile)
            alldata['label'] = alldata['bandgap'].apply(lambda x: 0 if x < threshold else 1)
            alldata.to_csv('./cleandata.csv', header=True, index=False)

    ## plot some samples
    plotCSVData(alldata, nc, img_size, nsamples=100)

    ## plot hist of bandgap
    plotDist(alldata['bandgap'], y_pred=None, bins=10, norm_hist=False, histtype='bar',
             result_savepath=result_savepath + '/bandgap_dist_true_alldata.png', model_name='Neural Network')

    ## create dataset and iterate it one by one and plot some samples

    if use_pretrained:
        input_size = (224, 224)  # input_size = 224, pretrained models
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        input_size = (32, 32)  # MyNet (5, 5) # LeNet (32, 32)  # input_size, model from scratch
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    topoDataset = TopoDataset(alldata, img_size,
                              transform=transforms.Compose([QuarterCrop(img_size / 2),
                                                            ToPILImage(),
                                                            Resize(input_size, interpolation=interpolation, zero_padding=zero_padding),  # LeNet
                                                            ToTensor(),
                                                            Normalize(mean, std)]))

    plotDataset(topoDataset, mean, std, nsamples=4)
    plotDist(topoDataset.bandgaps, y_pred=None, bins=10, norm_hist=False, histtype='bar',
             result_savepath=None, model_name='Neural Network')

    ## divide it as training and validation datasets and iterate them and plot some batches

    train_idx, val_idx = train_test_split(np.arange(len(topoDataset.labels)),
                                          test_size=0.2,
                                          stratify=None,  # topoDataset.labels,
                                          random_state=seed_value)

    train_size = len(train_idx)
    val_size = len(val_idx)

    ## augment on the fly
    # train_set = TopoDataset(alldata.iloc[train_idx, :], img_size,
    #                         transform=transforms.Compose([QuarterCrop(img_size / 2),
    #                                                       ToPILImage(),
    #                                                       RandomFlip(0.5),  # data augmentation for training set
    #                                                       RandomRotation([0, 90, 180, 270]),  # data augmentation for training set
    #                                                       # transforms.RandomChoice([RandomRotation([0, 90, 180, 270]), RandomFlip()]),
    #                                                       Resize(input_size, interpolation=interpolation, zero_padding=zero_padding),  # LeNet
    #                                                       ToTensor(),
    #                                                       Normalize(mean, std)]))
    # plotDataset(train_set, mean, std, nsamples=4)
    # plotDist(train_set.bandgaps, y_pred=None, bins=10, norm_hist=False, histtype='bar',
    #          result_savepath=result_savepath + '/bandgap_dist_true_train.png', model_name='Neural Network')

    ## augment statically on minority
    train_set = TopoDataset(alldata.iloc[train_idx, :], img_size, transform=None)
    # train_set.staticAugMinority()  # the length and train_idx are both changed
    # train_size = len(train_set)
    train_set.transform = transforms.Compose([QuarterCrop(img_size / 2),
                                              ToPILImage(),
                                              Resize(input_size, interpolation=interpolation, zero_padding=zero_padding),  # LeNet
                                              ToTensor(),
                                              Normalize(mean, std)])
    plotDataset(train_set, mean, std, nsamples=4)
    plotDist(train_set.bandgaps, y_pred=None, bins=10, norm_hist=False, histtype='bar',
             result_savepath=result_savepath + '/bandgap_dist_true_train.png', model_name='Neural Network')

    ##
    val_set = TopoDataset(alldata.iloc[val_idx, :], img_size,
                            transform=transforms.Compose([QuarterCrop(img_size / 2),
                                                          ToPILImage(),
                                                          Resize(input_size, interpolation=interpolation, zero_padding=zero_padding),  # LeNet
                                                          ToTensor(),
                                                          Normalize(mean, std)]))
    plotDataset(val_set, mean, std, nsamples=4)

    if use_target_scaler:
        target_scaler(train_set.bandgaps)  # get fitted_scaler
        train_set.bandgaps = target_scaler.transform(train_set.bandgaps)
        val_set.bandgaps = target_scaler.transform(val_set.bandgaps)
    plotDist(train_set.bandgaps, val_set.bandgaps, bins=10, norm_hist=False, histtype='bar',
             result_savepath=result_savepath + '/normalized_bandgap_dist_true_train_vs_val.png', model_name='True validation set')
    plotDist(train_set.original_bandgaps, val_set.original_bandgaps, bins=10, norm_hist=False, histtype='bar',
             result_savepath=result_savepath + '/original_bandgap_dist_true_train_vs_val.png', model_name='True validation set')

    ## train_set and val_set are Subset objects and contain 'indices' attribute
    # they can be directly indexed like train_set[i], i is bounded by the len(train_set)

    # train_set, val_set = random_split(topoDataset, [train_size, val_size])
    # train_set.indices = train_idx
    # val_set.indices = val_idx
    #
    # assert train_set[0]['bandgap'] == topoDataset[train_set.indices[0]]['bandgap']

    ## the failure smaples are far less than the sucess sample, we define a weighted random sampler

    # works for Subset object
    # class_sample_counts = np.bincount([int(train_set[i]['label'].item()) for i in range(train_size)])

    class_sample_counts = np.bincount(train_set.labels)

    if use_ImbalancedDataSampler == 1:
        # use ImbalancedDatasetSampler
        # https://github.com/ufoym/imbalanced-dataset-sampler/blob/master/examples/imagefolder.ipynb
        sampler = ImbalancedDatasetSampler(train_set,
                                           indices=None,
                                           num_samples=None,
                                           callback_get_label=lambda dataset, idx: dataset[idx]['label'].type(torch.int).item())

        shuffle = False

    elif use_ImbalancedDataSampler == 0:
        # oversampling through WeightedRandomSampler(replacement=True)
        # just adds exact copies of the minority to reach the target balanced weights

        weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
        # works for Subset object
        # train_targets = [int(train_set[i]['label'].item()) for i in range(train_size)]
        train_targets = train_set.labels

        samples_weights = weights[train_targets]

        sampler = sampler.WeightedRandomSampler(
            weights=samples_weights,
            num_samples=len(samples_weights),
            replacement=True)

        shuffle = False

    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=n_cpu, sampler=sampler)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=n_cpu)

    ## test and visualize train and val dataloader
    plotDataLoader(train_loader, val_loader, n_batch=4)

    ## create NN
    ## alexnet: 3 x 224 x 224
    # net_alex = models.alexnet(pretrained=True)
    # for param in net_alex.parameters():
    #     param.requires_grad = False
    #
    # class Alex_gap(nn.Module):
    #     def __init__(self, originalModel):
    #         super(Alex_gap, self).__init__()
    #
    #         self.features = originalModel.features
    #         self.avgpool = originalModel.avgpool
    #
    #         # input = 256 x 6 x 6
    #         self.gap = nn.AvgPool2d(kernel_size=6)
    #
    #         self.classifier = nn.Sequential(
    #             nn.Dropout(0.5),
    #             nn.Linear(256, 64),
    #             nn.BatchNorm1d(64),
    #             nn.ReLU(inplace=True),
    #             nn.Linear(64, 1),
    #         )
    #
    #     def forward(self, x):
    #         # print(x.shape)
    #         x = self.features(x)
    #         # print(x.shape)
    #         x = self.avgpool(x)
    #         x = self.gap(x).squeeze()
    #         # print(x.shape)
    #         x = self.classifier(x)
    #         # print(x.shape)
    #         return x
    #
    # net = Alex_gap(net_alex)
    # optimizer = optim.Adam(net.classifier.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)   # weight_decay is the L2 penalty coefficient

    ## resnet 18: 3 x 224 x 224
    # net = models.resnet18(pretrained=True)
    # for param in net.parameters():
    #     param.requires_grad = False
    #
    # # Parameters of newly constructed modules have requires_grad=True by default
    # num_ftrs = net.fc.in_features
    # net.fc = nn.Linear(num_ftrs, 1)
    #
    # optimizer = optim.Adam(net.fc.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)  # weight_decay is the L2 penalty coefficient

    ## squeezenet: 3 x 224 x 224
    # net = models.squeezenet1_1(pretrained=True)
    # for param in net.parameters():
    #     param.requires_grad = False
    #
    # # Parameters of newly constructed modules have requires_grad=True by default
    # net.classifier = nn.Sequential(
    #     nn.Dropout(p=0.5),
    #     nn.Conv2d(512, 1, kernel_size=1),
    #     nn.ReLU(inplace=True),
    #     nn.AdaptiveAvgPool2d((1, 1))
    # )
    #
    # optimizer = optim.Adam(net.classifier.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)  # weight_decay is the L2 penalty coefficient

    ## lenet: 3 x 32 x 32
    net = LeNet(3, n_gpu).to(device)
    # net = LeNet_small(3, n_gpu).to(device)
    # net = LeNet_gap(3, n_gpu).to(device)

    optimizer = optim.Adam(net.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)  # weight_decay is the L2 penalty coefficient

    ## MyNet: 3 x 5 x 5
    # net = MyNet(3, n_gpu).to(device)
    #
    # optimizer = optim.Adam(net.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)  # weight_decay is the L2 penalty coefficient

    ##
    # use previously trained model as a start point
    # net.load_state_dict(torch.load(model_savepath + '/net_epoch_{}.pth'.format(499)))

    net = net.to(device)

    criterion = chooseRegressionCriterion(loss_name=loss_name, reduction=reduction_mode, HuberDelta=HuberDelta).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (n_gpu > 1):
        net = torch.nn.DataParallel(net, list(range(n_gpu)))

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)  # 0.99
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    ## Training

    running_loss_train_list, scores_train_list = \
        train_val_NN(net, 'train', train_loader, val_loader, optimizer, scheduler, criterion,
                     train_size, val_size, batch_size, n_epochs, device, model_savepath, use_target_scaler, target_scaler,
                     use_manual_reg, LAMBDA_conv, LAMBDA_fc, fc_reg_form, use_fc_weight_clip, SparsificationFactor,
                     metric_names, name_funcs_dict)

    # manually break the training, use the saved model
    # running_loss_train_list, scores_train_list = \
    #     train_val_NN(net, 'validation', train_loader, train_loader, optimizer, scheduler, criterion,
    #                  train_size, train_size, batch_size, n_epochs, device, model_savepath, use_target_scaler, target_scaler,
    #                  use_manual_reg, LAMBDA_conv, LAMBDA_fc, fc_reg_form, use_fc_weight_clip, SparsificationFactor,
    #                  metric_names, name_funcs_dict)

    ##
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(111)
    ax1.set_title("Running Loss During Training")
    ax1.plot(range(1, n_epochs + 1), running_loss_train_list, label="train")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    fig.show()
    fig.savefig(result_savepath + '/train_runningloss.png', bbox_inches='tight')

    scores_train_df = pd.DataFrame(scores_train_list)
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(111)
    ax1.set_title("RMSE During Training")
    ax1.plot(range(1, n_epochs + 1), - scores_train_df['neg_root_mean_squared_error'], label="train")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("RMSE")
    ax1.legend()

    fig.show()
    fig.savefig(result_savepath + '/train_rmse.png', bbox_inches='tight')


    ## validation
    running_loss_val_list, scores_val_list = \
        train_val_NN(net, 'validation', train_loader, val_loader, optimizer, scheduler, criterion,
                     train_size, val_size, batch_size, n_epochs, device, model_savepath, use_target_scaler, target_scaler,
                     use_manual_reg, LAMBDA_conv, LAMBDA_fc, fc_reg_form, use_fc_weight_clip, SparsificationFactor,
                     metric_names, name_funcs_dict)

    ##
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(111)
    ax1.set_title("Running Loss During Validation")
    ax1.plot(range(1, n_epochs + 1), running_loss_val_list, label="validation")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    fig.show()
    fig.savefig(result_savepath + '/val_runningloss.png', bbox_inches='tight')

    scores_val_df = pd.DataFrame(scores_val_list)
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(111)
    ax1.set_title("RMSE During Validation")
    ax1.plot(range(1, n_epochs + 1), - scores_val_df['neg_root_mean_squared_error'], label="validation")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("RMSE")
    ax1.legend()

    fig.show()
    fig.savefig(result_savepath + '/val_rmse.png', bbox_inches='tight')

    ## compare running loss
    plotRunningLoss(running_loss_train_list, running_loss_val_list, n_epochs, result_savepath)

    ## compare metrics
    for metric_name in metric_names:
        plotScore(scores_train_df, scores_val_df, metric_name, n_epochs)

    ##
    # optimal_epoch = scores_val_df['neg_root_mean_squared_error'].idxmax()
    optimal_epoch = (scores_val_df['r2']-1).abs().idxmin()
    optimal_r2 = scores_val_df['r2'][optimal_epoch]
    ytest, ypred = modelOutput(optimal_epoch, net, val_loader, val_size, model_savepath, use_target_scaler, target_scaler)

    ## plot hist
    plotDist(ytest, ypred, bins=10, norm_hist=False, histtype='bar',
             result_savepath=result_savepath + '/bandgap_dist_true_vs_val.png', model_name='Neural Network')

    ## plot scatter
    plotTruePredScatter(ytest, ypred, r2=optimal_r2,
                        result_savepath=result_savepath + '/nn_scatter.png', model_name='Neural Network')

##

