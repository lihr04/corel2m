import torch
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from torch.utils.data import TensorDataset, Dataset, Subset
from copy import deepcopy


class Lines():
    def __init__(self, max_iter=5, num_samples=2000,test_train_ratio=.2, option=0):

        self.offset = 5  # Offset when loading data in next_task()

        # Generate data
        if option == 0:
            # Standard settings
            centers = [[0, 0.2], [0.6, 0.9], [1.3, 0.4], [1.6, -0.1], [2.0, 0.3],
                    [0.45, 0], [0.7, 0.45], [1., 0.1], [1.7, -0.4], [2.3, 0.1]]
            std = [[0.08, 0.22], [0.24, 0.08], [0.04, 0.2], [0.16, 0.05], [0.05, 0.16],
                [0.08, 0.16], [0.16, 0.08], [0.06, 0.16], [0.24, 0.05], [0.05, 0.22]]

        elif option == 1:
            # Six tasks
            centers = [[0, 0.2], [0.6, 0.9], [1.3, 0.4], [1.6, -0.1], [2.0, 0.3], [1.65, 0.1],
                    [0.45, 0], [0.7, 0.45], [1., 0.1], [1.7, -0.4], [2.3, 0.1], [0.7, 0.25]]
            std = [[0.08, 0.22], [0.24, 0.08], [0.04, 0.2], [0.16, 0.05], [0.05, 0.16], [0.14, 0.14],
                [0.08, 0.16], [0.16, 0.08], [0.06, 0.16], [0.24, 0.05], [0.05, 0.22], [0.14, 0.14]]

        elif option == 2:
            # All std devs increased
            centers = [[0, 0.2], [0.6, 0.9], [1.3, 0.4], [1.6, -0.1], [2.0, 0.3],
                    [0.45, 0], [0.7, 0.45], [1., 0.1], [1.7, -0.4], [2.3, 0.1]]
            std = [[0.12, 0.22], [0.24, 0.12], [0.07, 0.2], [0.16, 0.08], [0.08, 0.16],
                [0.12, 0.16], [0.16, 0.12], [0.08, 0.16], [0.24, 0.08], [0.08, 0.22]]

        elif option == 3:
            # Tougher to separate
            centers = [[0, 0.2], [0.6, 0.65], [1.3, 0.4], [1.6, -0.22], [2.0, 0.3],
                       [0.45, 0], [0.7, 0.55], [1., 0.1], [1.7, -0.3], [2.3, 0.1]]
            std = [[0.08, 0.22], [0.24, 0.08], [0.04, 0.2], [0.16, 0.05], [0.05, 0.16],
                   [0.08, 0.16], [0.16, 0.08], [0.06, 0.16], [0.24, 0.05], [0.05, 0.22]]

        elif option == 4:
            # Two tasks, of same two gaussians
            centers = [[0, 0.2], [0, 0.2],
                       [0.45, 0], [0.45, 0]]
            std = [[0.08, 0.22], [0.08, 0.22],
                   [0.08, 0.16], [0.08, 0.16]]

        else:
            # If new / unknown option
            centers = [[0, 0.2], [0.6, 0.9], [1.3, 0.4], [1.6, -0.1], [2.0, 0.3],
                    [0.45, 0], [0.7, 0.45], [1., 0.1], [1.7, -0.4], [2.3, 0.1]]
            std = [[0.08, 0.22], [0.24, 0.08], [0.04, 0.2], [0.16, 0.05], [0.05, 0.16],
                [0.08, 0.16], [0.16, 0.08], [0.06, 0.16], [0.24, 0.05], [0.05, 0.22]]

        if option != 1 and max_iter > 5:
            raise Exception("Current toydatagenerator only supports up to 5 tasks.")

        self.X, self.y = make_blobs(num_samples*2*max_iter, centers=centers, cluster_std=std)
        self.X = self.X.astype('float32')

        self.X_test, self.y_test = make_blobs(int(test_train_ratio*num_samples*2*max_iter), centers=centers, cluster_std=std)
        self.X_test = self.X_test.astype('float32')
        
        self.max_iter = max_iter
        self.num_samples = num_samples  # number of samples per task

        if option == 1:
            self.offset = 6
        elif option == 4:
            self.offset = 2

        self.cur_iter = 0

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception("Number of tasks exceeded!")
        else:
            x_train_0 = self.X[self.y == self.cur_iter]
            x_train_1 = self.X[self.y == self.cur_iter + self.offset]
            y_train_0 = np.zeros_like(self.y[self.y == self.cur_iter])
            y_train_1 = np.ones_like(self.y[self.y == self.cur_iter + self.offset])
            x_train = np.concatenate([x_train_0, x_train_1], axis=0)
            y_train = np.concatenate([y_train_0, y_train_1], axis=0)
            y_train = y_train.astype('int64')
            x_train = torch.from_numpy(x_train)
            y_train = torch.from_numpy(y_train)
            
            x_test_0 = self.X_test[self.y_test == self.cur_iter]
            x_test_1 = self.X_test[self.y_test == self.cur_iter + self.offset]
            y_test_0 = np.zeros_like(self.y_test[self.y_test == self.cur_iter])
            y_test_1 = np.ones_like(self.y_test[self.y_test == self.cur_iter + self.offset])
            x_test = np.concatenate([x_test_0, x_test_1], axis=0)
            y_test = np.concatenate([y_test_0, y_test_1], axis=0)
            y_test = y_test.astype('int64')
            x_test = torch.from_numpy(x_test)
            y_test = torch.from_numpy(y_test)
            self.cur_iter += 1
            return TensorDataset(x_train, y_train), TensorDataset(x_test, y_test)

    def full_data(self):
        x_train_list = []
        y_train_list = []
        for i in range(self.max_iter):
            x_train_list.append(self.X[self.y == i])
            x_train_list.append(self.X[self.y == i+self.offset])
            y_train_list.append(np.zeros_like(self.y[self.y == i]))
            y_train_list.append(np.ones_like(self.y[self.y == i+self.offset]))
        x_train = np.concatenate(x_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        y_train = y_train.astype('int64')
        x_train = torch.from_numpy(x_train)
        y_train = torch.from_numpy(y_train)
        
        x_test_list = []
        y_test_list = []
        for i in range(self.max_iter):
            x_test_list.append(self.X_test[self.y_test == i])
            x_test_list.append(self.X_test[self.y_test == i+self.offset])
            y_test_list.append(np.zeros_like(self.y_test[self.y_test == i]))
            y_test_list.append(np.ones_like(self.y_test[self.y_test == i+self.offset]))
        x_test = np.concatenate(x_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
        y_test = y_test.astype('int64')
        x_test = torch.from_numpy(x_test)
        y_test = torch.from_numpy(y_test)
        return TensorDataset(x_train, y_train), TensorDataset(x_test, y_test)
    
    def get_sequential_lines(self,n_task=5,batch_size=100):
        train_loader = {}
        test_loader = {}

        self.reset()
        for i in range(n_task):
            train, test = self.next_task()
            train_loader[i] = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
            test_loader[i] = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

    def reset(self):
        self.cur_iter = 0
