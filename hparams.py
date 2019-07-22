import os
from argparse import ArgumentParser, Namespace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np
import scipy.io as scio
# noinspection PyCompatibility
from dataclasses import asdict, dataclass, field
from numpy import ndarray


# noinspection PyArgumentList
class Channel(Enum):
    ALL = slice(None)
    LAST = slice(-1, None)
    NONE = None


@dataclass
class _HyperParameters:
    # devices
    device: Union[int, str, Sequence[str], Sequence[int]] = (0, 1, 2, 3)
    out_device: Union[int, str] = 3
    num_disk_workers: int = 4

    # select dataset
    # room_create: str = 'room1'
    room_train: str = 'room1'
    room_test: str = 'room1'
    room_create: str = ''

    model_name: str = 'DWaveNet'
    criterion_names: str = ('L1Loss', 'MSELoss')
    l_target: int = 8192

    # stft parameters
    fs: int = 16000
    n_fft: int = 512
    l_frame: int = 512
    n_freq: int = 257
    l_hop: int = 256

    # log & normalize
    refresh_const: bool = False

    # training
    n_file: int = 20 * 50
    train_ratio: float = 0.7
    n_epochs: int = 150
    batch_size: int = 4 * 4
    learning_rate: float = 5e-4
    weight_decay: float = 0  # Adam weight_decay
    weight_loss: tuple = (0.1, 1)  # L1, MSE

    # reconstruction
    do_bnkr_eq: bool = True

    # paths
    logdir: str = f'./result/test'
    path_speech: Path = Path('./backup/TIMIT')
    path_feature: Path = Path('./backup')
    form_path_normconst: str = 'normconst_{}.mat'

    # file names
    form_feature: str = '{:04d}_{:02d}.npz'
    form_result: str = 'mulchwav_{}'

    channels: Dict[str, Channel] = field(init=False)
    DWaveNet: Dict[str, Any] = field(init=False)
    scheduler: Dict[str, Any] = field(init=False)
    mulchwav_names: Dict[str, str] = field(init=False)

    l_diff: int = None
    l_input: int = None
    dummy_input_size: Tuple = None
    dict_path: Dict[str, Path] = None
    kwargs_stft: Dict[str, Any] = None
    kwargs_istft: Dict[str, Any] = None
    n_loc: Dict[str, int] = None
    period_save_state: int = None
    bnkr_inv0: ndarray = None

    def __post_init__(self):
        self.channels = dict(x=Channel.ALL,
                             # x=Channel.LAST,
                             y=Channel.ALL,
                             )
        self.DWaveNet = dict(  # in_channels is set in init_dependent_vars
            out_channels=1, bias=False,
            num_layers=30, num_stacks=3,
            kernel_size=3,
            residual_channels=128, gate_channels=256, skip_out_channels=128,
            last_channels=(2048, 256),
        )
        self.scheduler = dict(T_0=10,
                              T_mult=2,
                              )

        self.mulchwav_names = dict(x='reverberant', y='clean',
                                   path_speech='path_speech',
                                   )

    def init_dependent_vars(self):
        self.logdir = Path(self.logdir)
        # nn
        ch_in = 32 if self.channels['x'] == Channel.ALL else 1

        self.DWaveNet['in_channels'] = ch_in
        num_layers_per_stack = self.DWaveNet['num_layers'] // self.DWaveNet['num_stacks']
        self.l_diff = self.DWaveNet['num_stacks'] * (2**num_layers_per_stack - 1)
        self.l_input = self.l_diff * 2 + self.l_target

        self.dummy_input_size = (ch_in, self.l_input)

        # path
        if self.room_create:
            self.room_train = self.room_create
            self.room_test = self.room_create
        path_feature_train = self.path_feature / f'mulchwav_{self.room_train}/TRAIN'
        path_feature_test = self.path_feature / f'mulchwav_{self.room_test}/TEST'
        self.dict_path = dict(
            sft_data=self.path_feature / 'bEQf.mat',
            RIR=self.path_feature / f'RIR_{self.room_create}.mat',

            speech_train=self.path_speech / 'TRAIN',
            speech_test=self.path_speech / 'TEST',

            # dirspec_train=_PATH_DIRSPEC / 'TRAIN',
            feature_train=path_feature_train,
            feature_seen=path_feature_test / 'SEEN',
            feature_unseen=path_feature_test / 'UNSEEN',

            form_normconst_train=str(path_feature_train / self.form_path_normconst),
            form_normconst_seen=str(path_feature_test / self.form_path_normconst),
            form_normconst_unseen=str(path_feature_test / self.form_path_normconst),

            figures=Path('./figures'),
        )

        # dirspec parameters
        self.kwargs_stft = dict(n_fft=self.n_fft,
                                hop_length=self.l_hop,
                                win_length=self.l_frame,
                                window='hann',
                                center=True,
                                dtype=np.complex64)
        self.n_loc = dict()
        for kind in ('train', 'seen', 'unseen'):
            path_metadata = self.dict_path[f'feature_{kind}'] / 'metadata.mat'
            if path_metadata.exists():
                self.n_loc[kind] = scio.loadmat(
                    str(path_metadata), variable_names=('n_loc',)
                )['n_loc'].item()
            else:
                print(f'n_loc of "{kind}" not loaded.')

        # training
        if not self.period_save_state:
            self.period_save_state = self.scheduler['T_0'] // 2

        sft_dict = scio.loadmat(str(self.dict_path['sft_data']),
                                variable_names=('bEQf',))
        self.bnkr_inv0 = sft_dict['bEQf'][:, 0]
        self.bnkr_inv0 = np.concatenate(
            (self.bnkr_inv0, self.bnkr_inv0[-2:0:-1].conj())
        )  # N_fft

    @staticmethod
    def is_featurefile(f: os.DirEntry) -> bool:
        return f.name.endswith('.npz')

    # Function for parsing argument and set hyper parameters
    def parse_argument(self, parser=None, print_argument=True) -> Namespace:
        if not parser:
            parser = ArgumentParser()
        args_already_added = [a.dest for a in parser._actions]
        dict_self = asdict(self)
        for k in dict_self:
            if hasattr(args_already_added, k):
                continue
            parser.add_argument(f'--{k}', default='')

        args = parser.parse_args()
        for k in dict_self:
            parsed = getattr(args, k)
            if parsed == '':
                continue
            if isinstance(dict_self[k], str):
                if (parsed.startswith("'") and parsed.endwith("'")
                        or parsed.startswith('"') and parsed.endwith('"')):
                    parsed = parsed[1:-1]
                setattr(self, k, parsed)
            else:
                v = eval(parsed)
                # if isinstance(v, type(dict_self[k])):
                setattr(self, k, v)

        self.init_dependent_vars()
        if print_argument:
            print(repr(self))

        return args

    def __repr__(self):
        result = ('-------------------------\n'
                  'Hyper Parameter Settings\n'
                  '-------------------------\n')

        result += '\n'.join(
            [f'{k}: {v}' for k, v in asdict(self).items() if not isinstance(v, ndarray)])
        result += '\n-------------------------'
        return result


hp = _HyperParameters()
