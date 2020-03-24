from __future__ import print_function, division
import os
import torch
import pandas as pd
import time
from skimage import io, transform
import random
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, sampler
import torch.optim as optim

from torchvision import transforms
from torchvision.utils import save_image, make_grid

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns

##
from Classifier.torchImbalancedDatasetSampler import ImbalancedDatasetSampler
from Classifier.classification_evaluation import confusionMatrix, classification_metrics
from Classifier.classification_visualization import plotConfusionMatrix, plotPRCurve, plotROCCurve

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
            csv_file (string): csv file read by Pandas.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.topo_data = csv_file
        self.img_size = img_size
        self.transform = transform

        self.topos = self.topo_data.iloc[:, :img_size**2].values
        self.bandgaps = self.topo_data.iloc[:, self.img_size ** 2]
        self.labels = self.topo_data.iloc[:, self.img_size ** 2 + 1]

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


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, bandgap, label = sample['image'], sample['bandgap'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).type(torch.FloatTensor),
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
def plotDataset(topoDataset, nsamples=4):
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
class Net(nn.Module):
    def __init__(self, nc, n_gpu):
        super(Net, self).__init__()
        self.n_gpu = n_gpu

        # self.conv = nn.Sequential(
        #     # w_out=floor((w_in - kernel_size + 2 * padding) / stride + 1)
        #     # input state size. nc x 10 x 10
        #     nn.Conv2d(nc, 6, kernel_size=3, stride=1, padding=0, bias=True),
        #     nn.BatchNorm2d(6, 0.8),
        #     nn.Dropout2d(0.5),
        #     nn.ReLU(),
        #     # state size. 6 x 8 x 8
        #     nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=0, bias=True),
        #     nn.BatchNorm2d(16, 0.8),
        #     nn.Dropout2d(0.5),
        #     nn.ReLU(),
        #     # state size. 16 x 6 x 6
        #     nn.MaxPool2d(2, 2),
        #     # state size. 16 x 3 x 3
        # )
        #
        # self.fc = nn.Sequential(
        #     nn.Linear(16 * 3 * 3, 100),
        #     nn.ReLU(),
        #     nn.Linear(100, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1),
        # )

        self.fc = nn.Sequential(
            nn.Linear(25, 64),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        # output = self.conv(x)
        # # print(output.shape)
        # output = output.view(output.shape[0], -1)
        # # print(output.shape)
        # output = self.fc(output)

        x = x.view(x.shape[0], -1)
        output = self.fc(x)

        return output


##
def train_val_NN(net, phase, train_loader, val_loader, optimizer, scheduler, criterion,
                 train_size, val_size, batch_size, n_epochs, device, model_savepath, use_manual_l2_reg, LAMBDA,
                 positive_prob_treshold, metric_names, name_funcs_dict):

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
        yprob = np.zeros(dataset_size)
        yscore = np.zeros(dataset_size)
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
                loss = criterion(outputs, labels_batch)

                # backward + optimize
                if phase == 'train':
                    if use_manual_l2_reg:
                        l2_reg = None
                        for p in net.parameters():
                            if l2_reg is None:
                                l2_reg = p.norm(2)
                            else:
                                # cannot write as in-place operation l2_reg += p.norm(2)
                                # because it will change auto-gradient behavior
                                l2_reg = l2_reg + p.norm(2)
                        loss_total = loss + l2_reg * LAMBDA
                        loss_total.backward()
                    else:
                        loss.backward()
                    optimizer.step()

            # print statistics
            # the default reduction mode of loss function is 'mean'
            running_loss += loss.item() * images_batch.shape[0]

            prob_outputs = nn.Sigmoid()(outputs.detach().squeeze()).numpy()
            yprob[i * batch_size: i * batch_size + images_batch.shape[0]] = prob_outputs
            yscore[i * batch_size: i * batch_size + images_batch.shape[0]] = prob_outputs
            ypred[i * batch_size: i * batch_size + images_batch.shape[0]] = prob_outputs >= positive_prob_treshold
            ytrue[i * batch_size: i * batch_size + images_batch.shape[0]] = labels_batch.squeeze().numpy()

            if i == len(data_loader) - 1:
                assert i * batch_size + images_batch.shape[0] == dataset_size

                scores = classification_metrics(y_true=ytrue, y_pred=ypred, y_prob=yprob, y_score=yscore,
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
def adjust_learning_rate(epoch, optimizer):

    lr = 0.001

    if epoch > 180:
        lr = lr / 1000000
    elif epoch > 150:
        lr = lr / 100000
    elif epoch > 120:
        lr = lr / 10000
    elif epoch > 90:
        lr = lr / 1000
    elif epoch > 60:
        lr = lr / 100
    elif epoch > 30:
        lr = lr / 10

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
def modelOutput(iepoch, net, test_loader, test_size, model_savepath):
    net.eval()
    ypred = np.zeros(test_size)
    yprob = np.zeros(test_size)
    yscore = np.zeros(test_size)
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
            prob_outputs = nn.Sigmoid()(outputs.detach().squeeze()).numpy()
            yprob[i * batch_size: i * batch_size + images_batch.shape[0]] = prob_outputs
            yscore[i * batch_size: i * batch_size + images_batch.shape[0]] = prob_outputs
            ypred[i * batch_size: i * batch_size + images_batch.shape[0]] = prob_outputs >= positive_prob_treshold
            ytest[i * batch_size: i * batch_size + images_batch.shape[0]] = labels_batch.squeeze().numpy()
    return ytest, yscore, yprob, ypred

##
if __name__=='__main__':

    seed_value = 85
    top_percentile = 0.2
    use_rawdata = False

    result_savepath = './NNClassifier_result'
    model_savepath = result_savepath + '/model'

    classes = ('success', 'failure')  # 0: success, 1: failure

    positive_prob_treshold = 0.5

    img_size = 10
    nc = 1
    n_cpu = 0   # set this to 0 to use GPU in Windows
    n_gpu = 0

    n_epochs = 1000
    batch_size = 1300  # batch_size should be large enough to ensure that each batch contains some minority class
    lr = 0.003
    momentum = 0.9

    use_manual_l2_reg = False
    LAMBDA = 0.003

    use_ImbalancedDataSampler = True

    metric_names = ('accuracy', 'precision_binary', 'recall_binary', 'f1_binary', 'roc_auc_binary')
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

    ## create dataset and iterate it one by one and plot some samples
    mean = (0.5)
    std = (0.5)

    topoDataset = TopoDataset(alldata, img_size, transform=transforms.Compose([QuarterCrop(img_size/2),
                                                                               ToTensor(),
                                                                               Normalize(mean, std)]))

    plotDataset(topoDataset, nsamples=4)

    ## divide it as training and validation datasets and iterate them and plot some batches

    train_idx, val_idx = train_test_split(np.arange(len(topoDataset.labels)), test_size=0.2,
                                          stratify=topoDataset.labels, random_state=seed_value)

    train_size = len(train_idx)
    val_size = len(val_idx)

    # train_set and val_set are Subset objects and contain 'indices' attribute
    # they can be directly indexed like train_set[i], i is bounded by the len(train_set)
    train_set, val_set = random_split(topoDataset, [train_size, val_size])
    train_set.indices = train_idx
    val_set.indices = val_idx

    assert train_set[0]['bandgap'] == topoDataset[train_set.indices[0]]['bandgap']

    ## manually oversampling failure samples

    # train_data = train_set.dataset.topo_data.loc[train_set.indices, :]
    # class_sample_counts = np.bincount(train_data['label'])
    # n_failure = class_sample_counts[1]
    # n_success = class_sample_counts[0]
    # moredata = train_data.loc[train_data['label'] == 1, :].sample(n=n_success-n_failure,
    #                                                               replace=True, random_state=seed_value)
    # train_set.indices += moredata.index.to_list()
    #
    # traindata_oversampling = train_data.append(moredata, ignore_index=True)
    # print('oversampling --- n_failure: {}, n_success: {}'.format(sum(traindata_oversampling['label'] == 1),
    #                                                              sum(traindata_oversampling['label'] == 0)))

    ## the failure smaples are far less than the sucess sample, we define a weighted random sampler

    class_sample_counts = np.bincount(train_set.dataset.topo_data.loc[train_set.indices, 'label'])

    if use_ImbalancedDataSampler:
        # use ImbalancedDatasetSampler
        # https://github.com/ufoym/imbalanced-dataset-sampler/blob/master/examples/imagefolder.ipynb
        sampler = ImbalancedDatasetSampler(train_set,
                                           indices=None,
                                           num_samples=None,
                                           callback_get_label=lambda dataset, idx: dataset[idx]['label'].type(torch.int).item())
    else:
        weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
        train_targets = train_set.dataset.topo_data.loc[train_set.indices, 'label'].values
        samples_weights = weights[train_targets]

        sampler = sampler.WeightedRandomSampler(
            weights=samples_weights,
            num_samples=len(samples_weights),
            replacement=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=n_cpu, sampler=sampler)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=n_cpu)

    ## test and visualize train and val dataloader
    plotDataLoader(train_loader, val_loader, n_batch=4)

    ## create NN

    net = Net(nc, n_gpu).to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (n_gpu > 1):
        net = torch.nn.DataParallel(net, list(range(n_gpu)))

    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=LAMBDA)   # weight_decay is the L2 penalty coefficient
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)  # 0.99

    ## Training
    running_loss_train_list, scores_train_list = \
        train_val_NN(net, 'train', train_loader, val_loader, optimizer, scheduler, criterion,
                     train_size, val_size, batch_size, n_epochs, device, model_savepath, use_manual_l2_reg, LAMBDA,
                     positive_prob_treshold, metric_names, name_funcs_dict)

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
    ax1.set_title("Accuracy During Training")
    ax1.plot(range(1, n_epochs + 1), scores_train_df['accuracy'], label="train")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    fig.show()
    fig.savefig(result_savepath + '/train_accuracy.png', bbox_inches='tight')


    ## validation
    running_loss_val_list, scores_val_list = \
        train_val_NN(net, 'validation', train_loader, val_loader, optimizer, scheduler, criterion,
                     train_size, val_size, batch_size, n_epochs, device, model_savepath, use_manual_l2_reg, LAMBDA,
                     positive_prob_treshold, metric_names, name_funcs_dict)

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
    ax1.set_title("Accuracy During Validation")
    ax1.plot(range(1, n_epochs + 1), scores_val_df['accuracy'], label="validation")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    fig.show()
    fig.savefig(result_savepath + '/val_accuracy.png', bbox_inches='tight')

    ## compare running loss
    plotRunningLoss(running_loss_train_list, running_loss_val_list, n_epochs, result_savepath)

    ## compare metrics
    # for metric_name in metric_names:
    #     plotScore(scores_train_df, scores_val_df, metric_name, n_epochs)

    ## print classification_report
    ytest, yscore, yprob, ypred = \
        modelOutput(999, net, val_loader, val_size, model_savepath)

    print(classification_report(ytest, ypred, labels=[0, 1], target_names=classes, digits=3))

    ## plot confusion matrix
    sns.set(font_scale=1.4)
    cm, tn, fp, fn, tp = confusionMatrix(ytest, ypred, labels=(0, 1))
    plotConfusionMatrix(cm, classes, normalize=False, result_savepath=result_savepath + '/nn_confusion_matrix.png')

    ## plot ROC and PR curve and find optimal positive_prob_treshold
    # calculate roc and pr curves
    curves = classification_metrics(y_true=ytest, y_pred=ypred, y_prob=yprob, y_score=yscore,
                              metric_names=('roc_curve', 'pr_curve'), name_funcs_dict=name_funcs_dict)
    fpr, tpr, thresholds_roc = curves['roc_curve']
    plotROCCurve(fpr, tpr, close_default_clf=None, result_savepath=result_savepath + '/nn_ROC.png', model_name='Neural Network')

    precision, recall, thresholds_pr = curves['pr_curve']
    no_skill = len(ytest[ytest == 1]) / len(ytest)
    plotPRCurve(precision, recall, no_skill, close_default_clf=None, result_savepath=result_savepath + '/nn_PR.png', model_name='Neural Network')

##

