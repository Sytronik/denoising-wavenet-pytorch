"""
data_manager.py

A file that loads saved features and convert them into PyTorch DataLoader.
"""
import multiprocessing as mp
from copy import copy
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union
import os

import numpy as np
import scipy.io as scio
import torch
from numpy import ndarray
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm import tqdm

from generic import DataPerDevice

from hparams import hp


class Normalization:
    """
    Calculating and saving mean/std of all mel spectrogram with respect to time axis,
    applying normalization to the spectrogram
    This is need only when you don't load all the data on the RAM
    """

    @staticmethod
    def _sum(a: ndarray) -> ndarray:
        return a.sum()

    @staticmethod
    def _sq_dev(a: ndarray, mean_a: ndarray) -> ndarray:
        return ((a - mean_a)**2).sum()

    @staticmethod
    def _load_data(fname: Union[str, Path], key: str, queue: mp.Queue) -> None:
        x = np.load(fname, allow_pickle=True)[key]
        queue.put(x)

    @staticmethod
    def _calc_per_data(data,
                       list_func: Sequence[Callable],
                       args: Sequence = None,
                       ) -> Dict[Callable, Any]:
        """ gather return values of functions in `list_func`

        :param list_func:
        :param args:
        :return:
        """

        if args:
            result = {f: f(data, arg) for f, arg in zip(list_func, args)}
        else:
            result = {f: f(data) for f in list_func}
        return result

    def __init__(self, mean, std):
        self.mean = DataPerDevice(mean)
        self.std = DataPerDevice(std)

    @classmethod
    def calc_const(cls, all_files: List[Path], key: str):
        """

        :param all_files:
        :param key: data name in npz file
        :rtype: Normalization
        """

        # Calculate summation & size (parallel)
        list_fn = (np.size, cls._sum)
        pool_loader = mp.Pool(2)
        pool_calc = mp.Pool(min(mp.cpu_count() - 2, 6))
        with mp.Manager() as manager:
            queue_data = manager.Queue()
            pool_loader.starmap_async(cls._load_data,
                                      [(f, key, queue_data) for f in all_files])
            result: List[mp.pool.AsyncResult] = []
            for _ in tqdm(range(len(all_files)), desc='mean', dynamic_ncols=True):
                data = queue_data.get()
                result.append(pool_calc.apply_async(
                    cls._calc_per_data,
                    (data, list_fn)
                ))

        result: List[Dict] = [item.get() for item in result]
        print()

        sum_size = np.sum([item[np.size] for item in result])
        sum_ = np.sum([item[cls._sum] for item in result], axis=0)
        mean = sum_ / (sum_size // sum_.size)

        print('Calculated Size/Mean')

        # Calculate squared deviation (parallel)
        with mp.Manager() as manager:
            queue_data = manager.Queue()
            pool_loader.starmap_async(cls._load_data,
                                      [(f, key, queue_data) for f in all_files])
            result: List[mp.pool.AsyncResult] = []
            for _ in tqdm(range(len(all_files)), desc='std', dynamic_ncols=True):
                data = queue_data.get()
                result.append(pool_calc.apply_async(
                    cls._calc_per_data,
                    (data, (cls._sq_dev,), (mean,))
                ))

        pool_loader.close()
        pool_calc.close()
        result: List[Dict] = [item.get() for item in result]
        print()

        sum_sq_dev = np.sum([item[cls._sq_dev] for item in result], axis=0)

        std = np.sqrt(sum_sq_dev / (sum_size // sum_sq_dev.size) + 1e-5)
        print('Calculated Std')

        return cls(mean, std)

    def astuple(self):
        return self.mean.data[ndarray], self.std.data[ndarray]

    # normalize and denormalize functions can accept a ndarray or a tensor.
    def normalize(self, a):
        return (a - self.mean.get_like(a)) / (2 * self.std.get_like(a))

    def normalize_(self, a):  # in-place version
        a -= self.mean.get_like(a)
        a /= 2 * self.std.get_like(a)

        return a

    def denormalize(self, a):
        return a * (2 * self.std.get_like(a)) + self.mean.get_like(a)

    def denormalize_(self, a):  # in-place version
        a *= 2 * self.std.get_like(a)
        a += self.mean.get_like(a)

        return a


class MulchWavDataset(Dataset):
    def __init__(self, kind_data: str,
                 n_file: int,
                 random_by_utterance=False,
                 norm_in: Normalization = None,
                 norm_out: Normalization = None):
        self._PATH: Path = hp.dict_path[f'feature_{kind_data}']

        self.norm_in = None
        self.norm_out = None
        path_normconst = Path(hp.dict_path[f'form_normconst_{kind_data}'].format(n_file))
        if path_normconst.exists():
            dict_normconst = scio.loadmat(path_normconst,
                                          chars_as_strings=True, squeeze_me=True)
            s_all_files = dict_normconst['s_all_files']
        else:
            dict_normconst = None
            s_all_files = self.search_files(n_file, random_by_utterance, hp.n_loc[kind_data])

        if s_all_files[0].endswith('.h5'):
            s_all_files = [s.replace('.h5', '.npz') for s in s_all_files]

        if kind_data == 'train':
            self.all_files = [self._PATH / f for f in s_all_files]
            if dict_normconst is not None and not hp.refresh_const:
                self.norm_in = Normalization(*dict_normconst['normconst_in'])
                self.norm_out = Normalization(*dict_normconst['normconst_out'])
            else:
                self.norm_in \
                    = Normalization.calc_const(self.all_files, key=hp.mulchwav_names['x'])
                self.norm_out \
                    = Normalization.calc_const(self.all_files, key=hp.mulchwav_names['y'])
                scio.savemat(path_normconst,
                             dict(s_all_files=s_all_files,
                                  normconst_in=self.norm_in.astuple(),
                                  normconst_out=self.norm_out.astuple(),
                                  ),
                             )
        else:
            assert norm_in, norm_out
            self.all_files = [self._PATH / f for f in s_all_files]
            self.norm_in = norm_in
            self.norm_out = norm_out
            if dict_normconst is None:
                scio.savemat(path_normconst, dict(s_all_files=s_all_files))

    def search_files(self, n_file: int, random_by_utterance=False, n_loc=1):
        s_all_files = [
            f.name for f in os.scandir(self._PATH) if hp.is_featurefile(f)
        ]
        s_all_files = sorted(s_all_files)
        if n_file != -1:
            if random_by_utterance:
                utterances = np.random.randint(
                    len(s_all_files) // n_loc,
                    size=n_file // n_loc
                )
                utterances = [f'{u:4d}_' for u in utterances]
                s_all_files = [
                    f for f in s_all_files if f.startswith(utterances)
                ]
            else:
                s_all_files = np.random.permutation(s_all_files)
                s_all_files = s_all_files[:n_file]
        return s_all_files

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = dict()
        with np.load(self.all_files[idx], mmap_mode='r') as npz_data:
            for xy in ('x', 'y'):
                if not hp.channels[xy].value:
                    continue
                data = npz_data[hp.mulchwav_names[xy]]
                data = data[hp.channels[xy].value]

                if data.ndim == 1:
                    data = data[np.newaxis, :]

                sample[xy] = torch.tensor(data, dtype=torch.float32)
                sample[f'T_{xy}'] = sample[xy].shape[-1]

        return sample

    def __len__(self):
        return len(self.all_files)

    @staticmethod
    @torch.no_grad()
    def pad_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ return data with zero-padding

        Important data like x, y are all converted to Tensor(cpu).
        :param batch:
        :return:
            Values can be an Tensor(cpu), list of str, ndarray of int.
        """
        result = dict()
        T_xs = np.array([item.pop('T_x') for item in batch])
        idxs_sorted = np.argsort(T_xs)
        T_xs = T_xs[idxs_sorted] + 2 * hp.l_diff
        batch = [batch[i] for i in idxs_sorted]
        T_ys = np.array([item.pop('T_y') for item in batch])

        result['T_xs'], result['T_ys'] = T_xs, T_ys

        for key, value in batch[0].items():
            if type(value) == str:
                list_data = [batch[idx][key] for idx in idxs_sorted]
                set_data = set(list_data)
                if len(set_data) == 1:
                    result[key] = set_data.pop()
                else:
                    result[key] = list_data
            else:
                # B, T, C
                data = [item[key].permute(-1, -2) for item in batch]
                data = pad_sequence(data, batch_first=True)
                # B, C, T
                data = data.permute(0, -1, -2)

                result[key] = data

        result['x'] = F.pad(result['x'], [hp.l_diff, hp.l_diff])

        return result

    @staticmethod
    @torch.no_grad()
    def decollate_padded(batch: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """ select the `idx`-th data, get rid of padded zeros and return it.

        Important data like x, y are all converted to ndarray.
        :param batch:
        :param idx:
        :return: DataDict
            Values can be an str or ndarray.
        """
        result = dict()
        for key, value in batch.items():
            if not key.startswith('T_'):
                T_xy = f'T_{key}s'
                result[key] = value[idx, :, :batch[T_xy][idx]].numpy()

        result['x'] = result['x'][..., hp.l_diff:-hp.l_diff]
        return result

    @classmethod
    def split(cls, dataset, ratio: Sequence[float]) -> Sequence:
        """ Split the dataset into `len(ratio)` datasets.

        The sum of elements of ratio must be 1,
        and only one element can have the value of -1 which means that
        it's automaticall set to the value so that the sum of the elements is 1

        :type dataset: SALAMIDataset
        :type ratio: Sequence[float]

        :rtype: Sequence[Dataset]
        """
        if not isinstance(dataset, cls):
            raise TypeError
        n_split = len(ratio)
        ratio = np.array(ratio)
        mask = (ratio == -1)
        ratio[np.where(mask)] = 0

        assert (mask.sum() == 1 and ratio.sum() < 1
                or mask.sum() == 0 and ratio.sum() == 1)
        if mask.sum() == 1:
            ratio[np.where(mask)] = 1 - ratio.sum()

        idx_data = np.cumsum(np.insert(ratio, 0, 0) * len(dataset.all_files),
                             dtype=int)
        result = [copy(dataset) for _ in range(n_split)]
        # all_f_per = np.random.permutation(a._all_files)

        for ii in range(n_split):
            result[ii].all_files = dataset.all_files[idx_data[ii]:idx_data[ii + 1]]

        return result
