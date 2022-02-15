import os, scipy.io 
from operator import itemgetter

import pandas as pd
import numpy as np

from torch.utils.data import Dataset

class TorchDataset(Dataset):

    def __init__(self, train:tuple, val:tuple, model_name:str=None, mode:str='train'):

        self.train_data, self.train_labels = train

        self.val_data, self.val_labels = val

        self.train_ratio = len(self.train_labels[self.train_labels==1]) / len(self.train_labels)
        self.val_ratio = len(self.val_labels[self.val_labels==1]) / len(self.val_labels)
        self.model_name = model_name

        self.mode = mode

    def __len__(self):
        if self.mode=='train':
            return len(self.train_data)
        if self.mode=='val':
            return len(self.val_data)

    def num_features(self):
        return self.train_data.shape[1]

    def __getitem__(self, index):

        if self.mode=='train':
            sample = np.array(self.train_data[index]).astype(float)
            return sample
        if self.mode=='val':
            sample = np.array(self.val_data[index]).astype(float)
            return sample, index, self.val_labels[index]


    def get_validation_data(self, index):

        sample = np.array(self.val_data[index]).astype(float)

        return sample, list(itemgetter(*index)(self.val_labels))

    def get_test_data(self, index):

        sample = np.array(self.val_data[index]).astype(float)

        return sample, list(itemgetter(*index)(self.val_labels))

class BaselineDataset():

    def __init__(self, dataset:str, root_dir:str, train_test_val:bool=True,
                 model_name:str=None, mode:str='train', true_label:int=1):

        self.dataset = dataset
        self.data_path = root_dir
        self.train_test_val = train_test_val
        self.model_name = model_name
        self.true_label = true_label

        self.mode = mode

    def norm_kdd_data(self, train_real, val_real, val_fake, cont_indices):
        symb_indices = np.delete(np.arange(train_real.shape[1]), cont_indices)
        mus = train_real[:, cont_indices].mean(0)
        sds = train_real[:, cont_indices].std(0)
        sds[sds == 0] = 1

        def get_norm(xs, mu, sd):
            bin_cols = xs[:, symb_indices]
            cont_cols = xs[:, cont_indices]
            cont_cols = np.array([(x - mu) / sd for x in cont_cols])
            return np.concatenate([bin_cols, cont_cols], 1)

        train_real = get_norm(train_real, mus, sds)
        val_real = get_norm(val_real, mus, sds)
        val_fake = get_norm(val_fake, mus, sds)
        return train_real, val_real, val_fake

    def norm_data(self, train_data, val_real, val_fake):
        mus = train_data.mean(0)
        sds = train_data.std(0)
        sds[sds == 0] = 1

        def get_norm(xs, mu, sd):
            return np.array([(x - mu) / sd for x in xs])

        train_data = get_norm(train_data, mus, sds)
        val_real = get_norm(val_real, mus, sds)
        val_fake = get_norm(val_fake, mus, sds)
        
        return train_data, val_real, val_fake

    def return_torch_dataset(self, norm_samples:np.array, 
                             anom_samples:np.array, cont_indices:list=None)->TorchDataset:

        if self.dataset in ['thyroid', 'arrhythmia']:
            n_train = len(norm_samples) // 2
            train_data = norm_samples[:n_train]

            val_real = norm_samples[n_train:]
            val_fake = anom_samples

            train_data, val_real, val_fake = self.norm_data(train_data, val_real, val_fake)
        else:
            n_norm = norm_samples.shape[0]
            ranidx = np.random.permutation(n_norm)
            n_train = n_norm // 2
            train_data = norm_samples[ranidx[:n_train]]

            val_real = norm_samples[ranidx[n_train:]]
            val_fake = anom_samples

            train_data, val_real, val_fake = self.norm_kdd_data(train_data, val_real, val_fake, cont_indices)
            
        val_data = np.concatenate([val_real, val_fake])
        val_label = np.concatenate([np.zeros(len(val_real)), np.ones(len(val_fake))])

        train_label = np.zeros(len(train_data))

        return TorchDataset((train_data, train_label), (val_data, val_label), self.model_name,
                            self.d_out, self.n_rots, self.mode)

    def Thyroid_train_valid_data(self,):
    
        data = scipy.io.loadmat(os.path.join(self.data_path, 'thyroid'))

        samples = data['X']
        labels = ((data['y']).astype(np.int32)).reshape(-1)

        norm_samples = samples[labels == 0]
        anom_samples = samples[labels == 1]

        return self.return_torch_dataset(norm_samples, anom_samples,)


    def Arrhythmia_train_valid_data(self,):
        data = scipy.io.loadmat(os.path.join(self.data_path, 'arrhythmia'))
        samples = data['X']  # 518
        labels = ((data['y']).astype(np.int32)).reshape(-1)

        norm_samples = samples[labels == 0]
        anom_samples = samples[labels == 1]

        return self.return_torch_dataset(norm_samples, anom_samples,)


    def KDD99_train_valid_data(self,):
        samples, labels, cont_indices = self.KDD99_preprocessing()
        anom_samples = samples[labels == 1]

        norm_samples = samples[labels == 0]

        return self.return_torch_dataset(norm_samples, anom_samples, cont_indices,)


    def KDD99_preprocessing(self,):

        names_file = os.path.join(self.data_path, 'kdd_names.csv')
        data_file = os.path.join(self.data_path, 'kddcup.data_10_percent.gz')

        df_colnames = pd.read_csv(names_file, skiprows=1, sep=':', names=['f_names', 'f_types'])
        df_colnames.loc[df_colnames.shape[0]] = ['status', ' symbolic.']
        df = pd.read_csv(data_file, header=None, names=df_colnames['f_names'].values)
        df_symbolic = df_colnames[df_colnames['f_types'].str.contains('symbolic.')]
        df_continuous = df_colnames[df_colnames['f_types'].str.contains('continuous.')]
        samples = pd.get_dummies(df.iloc[:, :-1], columns=df_symbolic['f_names'][:-1])

        smp_keys = samples.keys()
        cont_indices = []
        for cont in df_continuous['f_names']:
            cont_indices.append(smp_keys.get_loc(cont))

        labels = np.where(df['status'] == 'normal.', 1, 0)
        return np.array(samples), np.array(labels), cont_indices


    def KDD99Rev_train_valid_data(self,):
        samples, labels, cont_indices = self.KDD99_preprocessing()

        norm_samples = samples[labels == 1] 
        anom_samples = samples[labels == 0]

        return self.return_torch_dataset(norm_samples, anom_samples, cont_indices,)

    def norm(self, data, mu=1):
        return 2 * (data / 255.) - mu

    def get_dataset(self,):
        if self.dataset == 'kdd':
            return self.KDD99_train_valid_data()
        if self.dataset == 'kddrev':
            return self.KDD99Rev_train_valid_data()
        if self.dataset == 'thyroid':
            return self.Thyroid_train_valid_data()
        if self.dataset == 'arrhythmia':
            return self.Arrhythmia_train_valid_data()