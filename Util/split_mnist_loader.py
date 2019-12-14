import os, sys
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.utils import shuffle

# code from https://github.com/csm9493/UCL/blob/master/dataloaders/split_mnist.py
def get_data(seed=0, fixed_order=False, pc_valid=0, datapath=".", tasknum = 5):
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
        data[t]['valid'] = {}
        data[t]['valid']['x'] = data[t]['train']['x'].clone()
        data[t]['valid']['y'] = data[t]['train']['y'].clone()

    # Others
    n = 0
    for t in range(tasknum):
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, size


class BatchIterator:
    def __init__(self, task_data, batch_size=1, flatten=False):
        if flatten:
            self.task_data_x = task_data['x'].reshape((task_data['x'].shape[0], -1))
        else:
            self.task_data_x = task_data['x']
        self.task_data_y = task_data['y']
        self.data_size = self.task_data_x.shape[0]
        self.batch_size = batch_size

    def __iter__(self):
        self.minibatch_id = 0
        self.start_sample_id = 0
        self.end_sample_id = self.start_sample_id + self.batch_size
        return self
    def __next__(self):
        if self.end_sample_id < self.data_size:
            result = (self.minibatch_id,
                      self.task_data_x[self.start_sample_id:self.end_sample_id],
                      self.task_data_y[self.start_sample_id:self.end_sample_id])
            self.start_sample_id += self.batch_size
            self.end_sample_id += self.batch_size
            self.minibatch_id += 1
            return result
        elif self.start_sample_id < self.data_size:
            result = (self.minibatch_id,
                      self.task_data_x[self.start_sample_id:-1],
                      self.task_data_y[self.start_sample_id:-1])
            self.start_sample_id = self.data_size
            self.minibatch_id += 1
            return result
        else:
            raise StopIteration

# class TasksIterator:
#     def __init__(self, data):
#         self.data = data
#         self.tasks_num = len(data)
#     def __iter__(self):
#         self.task_id = 0
#         return self
#     def __next__(self):
#         if self.task_id < self.tasks_num:
#             x = self.task_id
#             self.task_id += 1
#             return x, BatchIterator(self.data[x])
#         else:
#             raise StopIteration
