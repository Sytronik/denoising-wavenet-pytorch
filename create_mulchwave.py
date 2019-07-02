""" create directional spectrogram.

--init option forces to start from the first data.
--dirac option is for using dirac instead of spatially average intensity.

Ex)
python create_mulchwave.py TRAIN
python create_mulchwave.py UNSEEN --init
python create_mulchwave.py SEEN --dirac
python create_mulchwave.py TRAIN --dirac --init
"""

# noinspection PyUnresolvedReferences
import logging
import multiprocessing as mp
import os
from argparse import ArgumentParser, ArgumentError
from collections import defaultdict
from pathlib import Path
from typing import Tuple, TypeVar, Optional

import numpy as np
import scipy.io as scio
import scipy.signal as scsig
import soundfile as sf
from tqdm import tqdm
from numpy import ndarray

from hparams import hp
from audio_utils import apply_iir_filter


def process():
    print_save_info(idx_start)
    num_propagater = min(mp.cpu_count() - hp.num_disk_workers - 1,
                         hp.num_disk_workers * 3)
    pool_propagater = mp.Pool(num_propagater)
    pool_saver = mp.Pool(hp.num_disk_workers)
    with mp.Manager() as manager:
        # apply propagater
        # propagater sends data to q_data
        q_data = manager.Queue(hp.num_disk_workers * 3)
        range_file = range(idx_start, num_speech)
        pbar = tqdm(range(num_speech),
                    desc='apply', dynamic_ncols=True, initial=idx_start)
        for i_speech, f_speech in zip(range_file, all_files[idx_start:]):
            speech, _ = sf.read(str(f_speech))

            for i_loc in range(RIRs.shape[0]):
                pool_propagater.apply_async(
                    propagate,
                    (i_speech, f_speech, speech, i_loc, q_data)
                )
            pbar.update()
        pool_propagater.close()

        # apply saver
        # saver gets dict_to_save from q_data
        q_done = manager.Queue()
        for idx in range(hp.num_disk_workers):
            n_per_saver = len(range(idx, len(range_file) * n_loc, hp.num_disk_workers))
            pool_saver.apply_async(save_dirspec, (q_data, n_per_saver, q_done))
        pool_saver.close()

        dict_count = defaultdict(lambda: 0)
        pbar = tqdm(range(num_speech),
                    desc='create', dynamic_ncols=True, initial=idx_start)
        for _ in range(len(range_file) * n_loc):
            i_speech, i_loc = q_done.get()
            dict_count[i_speech] += 1
            if dict_count[i_speech] >= n_loc:
                pbar.update()
            pbar.set_postfix_str(f'{q_data.qsize()}')

        pool_propagater.join()
        pool_saver.join()
        print()

    print_save_info(idx_start
                    + sum([1 for v in dict_count.values() if v >= n_loc]))


def propagate(i_speech: int, f_speech: Path, speech: ndarray, i_loc: int, queue: mp.Queue):
    # RIR Filtering
    reverberant = scsig.fftconvolve(speech[np.newaxis, :], RIRs[i_loc])

    # Propagation
    clean = np.append(np.zeros(t_peak[i_loc]), speech * amp_peak[i_loc])

    dict_to_save = dict(path_speech=f_speech,
                        clean=clean,
                        reverberant=reverberant,
                        )
    queue.put((i_speech, i_loc, dict_to_save))


def save_dirspec(q_data: mp.Queue, n_per_saver: int, q_done: mp.Queue):
    for _ in range(n_per_saver):
        i_speech, i_loc, dict_to_save = q_data.get()
        np.savez(path_result / (hp.form_feature.format(i_speech, i_loc)), **dict_to_save)

        q_done.put((i_speech, i_loc))


def print_save_info(i_speech: int):
    """ Print and save metadata.

    """
    print(f'Wave Files Processed/Total: {i_speech}/{len(all_files)}\n'
          f'Number of source location: {n_loc}\n')

    metadata = dict(fs=hp.fs,
                    n_fft=hp.n_fft,
                    n_freq=hp.n_freq,
                    l_frame=hp.l_frame,
                    l_hop=hp.l_hop,
                    n_loc=n_loc,
                    path_all_speech=[str(p) for p in all_files],
                    )

    scio.savemat(f_metadata, metadata)


if __name__ == '__main__':
    # determined by sys argv
    parser = ArgumentParser()
    parser.add_argument('room_create')
    parser.add_argument('kind_data',
                        choices=('TRAIN', 'train',
                                 'SEEN', 'seen',
                                 'UNSEEN', 'unseen',
                                 ),
                        )
    parser.add_argument('-t', dest='target_folder', default='')
    parser.add_argument('--from', type=int, default=-1,
                        dest='from_idx')
    args = hp.parse_argument(parser)
    is_train = args.kind_data.lower() == 'train'

    # Paths
    path_speech = hp.dict_path['speech_train' if is_train else 'speech_test']

    if args.target_folder:
        path_result = hp.path_feature / args.target_folder
        if not is_train:
            path_result = path_result / 'TEST'
        path_result = path_result / args.kind_data.upper()
    else:
        path_result = hp.dict_path[f'feature_{args.kind_data.lower()}']

    os.makedirs(path_result, exist_ok=True)

    f_metadata = path_result / 'metadata.mat'
    if f_metadata.exists():
        all_files = scio.loadmat(str(f_metadata),
                                 variable_names=('path_all_speech',),
                                 chars_as_strings=True,
                                 squeeze_me=True)['path_all_speech']
        all_files = [Path(p) for p in all_files]
    else:
        all_files = list(path_speech.glob('**/*.WAV')) + list(path_speech.glob('**/*.wav'))

    num_speech = len(all_files)
    if num_speech < args.from_idx:
        raise ArgumentError

    # RIR Data
    transfer_dict = scio.loadmat(str(hp.dict_path['RIR_Ys']), squeeze_me=True)
    kind_RIR = 'TEST' if args.kind_data.lower() == 'unseen' else 'TRAIN'
    RIRs = transfer_dict[f'RIR_{kind_RIR}'].transpose((2, 0, 1))
    n_loc, n_mic, len_RIR = RIRs.shape
    del transfer_dict

    # SFT Data
    sft_dict = scio.loadmat(str(hp.dict_path['sft_data']),
                            variable_names=('bEQf',),
                            squeeze_me=True
                            )
    bnkr_inv = sft_dict['bEQf'].T[..., np.newaxis]  # Order x N_freq x 1
    bnkr_inv = np.concatenate(
        (bnkr_inv, bnkr_inv[:, -2:0:-1].conj()), axis=1
    )  # Order x N_fft x 1
    del sft_dict

    # propagation
    p00_RIRs = RIRs.mean(1)  # n_loc x time
    a00_RIRs = apply_iir_filter(p00_RIRs, bnkr_inv[0, :, 0])

    t_peak = a00_RIRs.argmax(axis=1)
    amp_peak = a00_RIRs.max(axis=1)
    # t_peak = np.round(RIRs.argmax(axis=2).mean(axis=1)).astype(int)
    # amp_peak = RIRs.max(axis=2).mean(axis=1)
    del bnkr_inv, p00_RIRs, a00_RIRs

    # The index of the first speech file that have to be processed
    idx_exist = -2
    should_ask_cont = False
    for i_speech_ in range(num_speech):
        if len(list(path_result.glob(f'{i_speech_:04d}_*.npz'))) < n_loc:
            idx_exist = i_speech_ - 1
            break
    if args.from_idx == -1:
        if idx_exist == -2:
            print_save_info(num_speech)
            exit(0)
        idx_start = idx_exist + 1
    else:
        idx_start = args.from_idx
        should_ask_cont = True

    print(f'Start processing from the {idx_start}-th speech file.')
    if should_ask_cont:
        ans = input(f'{idx_exist} speech files were already processed. continue? (y/n)\n')
        if not ans.startswith('y'):
            exit(0)

    del args

    process()
