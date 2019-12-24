import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch import Tensor
from imageio import imread
from Util.util import print_msg

class BatchIterator:
    def __init__(self, task_data, batch_size=1, flatten=False, shuffle=False):
        if flatten:
            self.task_data_x = task_data['x'].reshape((task_data['x'].shape[0], -1))
        else:
            self.task_data_x = task_data['x']
        self.task_data_y = task_data['y']
        self.data_size = self.task_data_x.shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        if shuffle:
            self.perm = np.random.permutation(self.data_size)
        else:
            self.perm = np.arange(self.data_size)

    def __iter__(self):
        self.minibatch_id = 0
        self.start_sample_id = 0
        self.end_sample_id = self.start_sample_id + self.batch_size
        if self.shuffle:
            self.perm = np.random.permutation(self.data_size)
        else:
            self.perm = np.arange(self.data_size)
        return self

    def __next__(self):
        if self.end_sample_id < self.data_size:
            indices = self.perm[self.start_sample_id:self.end_sample_id]
            result = (self.minibatch_id,
                      self.task_data_x[indices],
                      self.task_data_y[indices])
            self.start_sample_id += self.batch_size
            self.end_sample_id += self.batch_size
            self.minibatch_id += 1
            return result
        elif self.start_sample_id < self.data_size:
            indices = self.perm[self.start_sample_id:-1]
            result = (self.minibatch_id,
                      self.task_data_x[indices],
                      self.task_data_y[indices])
            self.start_sample_id = self.data_size
            self.minibatch_id += 1
            return result
        else:
            raise StopIteration


# get_"dataset" functions are from https://github.com/csm9493/UCL/blob/master/dataloaders/
def get_split_mnist(datapath=".", tasknum = 5):
    split_mnist_datapath = os.path.join(datapath, "binary_split_mnist")
    if tasknum >5:
        tasknum = 5
    data = {}
    taskcla = []
    size = [1, 28, 28]

    # Pre-load
    # MNIST
    mean = (0.1307,)
    std = (0.3081,)
    if not os.path.isdir(split_mnist_datapath):
        os.makedirs(split_mnist_datapath)
        dat = {}
        dat['train'] = datasets.MNIST(datapath, train=True, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        dat['test'] = datasets.MNIST(datapath, train=False, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        for i in range(5):
            data[i] = {}
            data[i]['name'] = 'split_mnist-{:d}'.format(i)
            data[i]['ncla'] = 2
            data[i]['train'] = {'x': [], 'y': []}
            data[i]['test'] = {'x': [], 'y': []}
        for s in ['train', 'test']:
            loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
            for image, target in loader:
                task_idx = target.numpy()[0] // 2
                data[task_idx][s]['x'].append(image)
                data[task_idx][s]['y'].append(target.numpy()[0 ] %2)

        for i in range(5):
            for s in ['train', 'test']:
                data[i][s]['x'] = torch.stack(data[i][s]['x'])
                data[i][s]['y'] = torch.LongTensor(np.array(data[i][s]['y'], dtype=int)).view(-1)
                torch.save(data[i][s]['x']
                           ,os.path.join(os.path.expanduser(split_mnist_datapath), 'data '+ str(i) + s + 'x.bin'))
                torch.save(data[i][s]['y']
                           ,os.path.join(os.path.expanduser(split_mnist_datapath), 'data '+ str(i) + s + 'y.bin'))
    else:
        # Load binary files
        for i in range(5):
            data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
            data[i]['ncla'] = 2
            data[i]['name'] = 'split_mnist-{:d}'.format(i)

            # Load
            for s in ['train', 'test']:
                data[i][s] = {'x': [], 'y': []}
                data[i][s]['x'] = torch.load \
                    (os.path.join(os.path.expanduser(split_mnist_datapath), 'data '+ str(i) + s + 'x.bin'))
                data[i][s]['y'] = torch.load \
                    (os.path.join(os.path.expanduser(split_mnist_datapath), 'data '+ str(i) + s + 'y.bin'))

    for t in range(tasknum):
        valid_ratio = 0.15
        train_ind = int(data[t]['train']['x'].shape[0] * (1 - valid_ratio))

        data[t]['valid'] = {}
        # data[t]['valid']['x'] = data[t]['train']['x'].clone()
        # data[t]['valid']['y'] = data[t]['train']['y'].clone()
        data[t]['valid']['x'] = data[t]['train']['x'][train_ind:]
        data[t]['valid']['y'] = data[t]['train']['y'][train_ind:]
        data[t]['train']['x'] = data[t]['train']['x'][:train_ind]
        data[t]['train']['y'] = data[t]['train']['y'][:train_ind]

    # Others
    n = 0
    for t in range(tasknum):
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, size


def get_split_notmnist(datapath=".", tasknum=5):
    split_notmnist_datapath = os.path.join(datapath, "binary_split_notmnist")
    if tasknum > 5:
        tasknum = 5

    data = {}
    taskcla = []
    size = [1, 28, 28]

    # Pre-load
    # notMNIST
    #     mean = (0.1307,)
    #     std = (0.3081,)
    if not os.path.isdir(split_notmnist_datapath):
        os.makedirs(split_notmnist_datapath)

        # the path of the notMnist_large folder where the downloaded notMNIST_large was unzipped
        data = split_notMNIST_loader(os.path.join(datapath, "notMNIST_large"))

        for i in range(5):
            for s in ['train', 'test']:
                data[i][s]['x'] = torch.stack(data[i][s]['x'])
                data[i][s]['y'] = torch.LongTensor(np.array(data[i][s]['y'], dtype=int)).view(-1)
                torch.save(data[i][s]['x'], os.path.join(os.path.expanduser(split_notmnist_datapath),
                                                         'data' + str(i) + s + 'x.bin'))
                torch.save(data[i][s]['y'], os.path.join(os.path.expanduser(split_notmnist_datapath),
                                                         'data' + str(i) + s + 'y.bin'))
    else:
        # Load binary files
        for i in range(5):
            data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
            data[i]['ncla'] = 2
            data[i]['name'] = 'split_notmnist-{:d}'.format(i)

            # Load
            for s in ['train', 'test']:
                data[i][s] = {'x': [], 'y': []}
                data[i][s]['x'] = torch.load(os.path.join(os.path.expanduser(split_notmnist_datapath),
                                                          'data' + str(i) + s + 'x.bin'))
                data[i][s]['y'] = torch.load(os.path.join(os.path.expanduser(split_notmnist_datapath),
                                                          'data' + str(i) + s + 'y.bin'))

    for t in range(tasknum):
        valid_ratio = 0.15
        train_ind = int(data[t]['train']['x'].shape[0] * (1 - valid_ratio))

        data[t]['valid'] = {}
        # data[t]['valid']['x'] = data[t]['train']['x'].clone()
        # data[t]['valid']['y'] = data[t]['train']['y'].clone()
        data[t]['valid']['x'] = data[t]['train']['x'][train_ind:]
        data[t]['valid']['y'] = data[t]['train']['y'][train_ind:]
        data[t]['train']['x'] = data[t]['train']['x'][:train_ind]
        data[t]['train']['y'] = data[t]['train']['y'][:train_ind]

    # Others
    n = 0
    for t in range(tasknum):
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, size


def split_notMNIST_loader(root):
    data = {}
    for i in range(5):
        data[i] = {}
        data[i]['name'] = 'split_notmnist-{:d}'.format(i)
        data[i]['ncla'] = 2
        data[i]['train'] = {'x': [], 'y': []}
        data[i]['test'] = {'x': [], 'y': []}

    folders = os.listdir(root)
    task_cnt = 0
    for folder in folders:
        folder_path = os.path.join(root, folder)
        cnt = 0
        print_msg(folder)
        for ims in os.listdir(folder_path):
            s = 'train'
            if cnt >= 40000:
                s = 'test'
            try:
                img_path = os.path.join(folder_path, ims)
                img = imread(img_path) / 255.0
                img_tensor = Tensor(img).float()
                task_idx = (ord(folder) - 65) % 5
                label = (ord(folder) - 65) // 5
                data[task_idx][s]['x'].append(img_tensor)
                data[task_idx][s]['y'].append(label)  # Folders are A-J so labels will be 0-9
                cnt += 1

            except:
                # Some images in the dataset are damaged
                print_msg("File {}/{} is datbroken".format(folder, ims))
        task_cnt += 1

    return data

