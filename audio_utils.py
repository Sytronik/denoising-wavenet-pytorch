from collections import OrderedDict as ODict
from typing import IO, Sequence, Tuple, Union
from itertools import repeat

import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tckr
from numpy import ndarray
import scipy.signal as scsig

from hparams import hp
import generic as gen
from matlab_lib import Evaluation as EvalModule


EVAL_METRICS = EvalModule.metrics


def calc_snrseg_time(clean: ndarray, est: ndarray, l_frame: int, l_hop: int,
                     T_ys: Sequence[int] = None) \
        -> float:
    _LIM_UPPER = 35. / 10.  # clip at 35 dB
    _LIM_LOWER = -10. / 10.  # clip at -10 dB
    if clean.ndim == 1:
        clean = clean[np.newaxis, ...]
        est = est[np.newaxis, ...]
    if T_ys is None:
        T_ys = (clean.shape[-1],)

    win = scsig.windows.hann(l_frame, False)[:, np.newaxis]

    sum_result = 0.
    for T, item_clean, item_est in zip(T_ys, clean, est):
        l_pad = l_frame - (T - l_frame) % l_hop
        item_clean = np.pad(item_clean[:T], (0, l_pad), 'constant')
        item_est = np.pad(item_est[:T], (0, l_pad), 'constant')
        clean_frames = librosa.util.frame(item_clean, l_frame, l_hop) * win
        est_frames = librosa.util.frame(item_est, l_frame, l_hop) * win

        # T
        norm_clean = np.linalg.norm(clean_frames, ord=2, axis=0)
        norm_err = (np.linalg.norm(est_frames - clean_frames, ord=2, axis=0)
                    + np.finfo(np.float32).eps)

        snrseg = np.log10(norm_clean / norm_err + np.finfo(np.float32).eps)
        np.minimum(snrseg, _LIM_UPPER, out=snrseg)
        np.maximum(snrseg, _LIM_LOWER, out=snrseg)
        sum_result += snrseg.mean()
    sum_result *= 10

    return sum_result


def calc_using_eval_module(y_clean: ndarray, y_est: ndarray,
                           T_ys: Sequence[int] = (0,)) -> ODict:
    """ calculate metric using EvalModule. y can be a batch.

    :param y_clean:
    :param y_est:
    :param T_ys:
    :return:
    """

    if y_clean.ndim == 1:
        y_clean = y_clean[np.newaxis, ...]
        y_est = y_est[np.newaxis, ...]
    if T_ys == (0,):
        T_ys = (y_clean.shape[1],) * y_clean.shape[0]

    keys = None
    sum_result = None
    for T, item_clean, item_est in zip(T_ys, y_clean, y_est):
        # noinspection PyArgumentList,PyTypeChecker
        temp: ODict = EvalModule(item_clean[:T], item_est[:T], hp.fs)
        result = np.array(list(temp.values()))
        if not keys:
            keys = temp.keys()
            sum_result = result
        else:
            sum_result += result

    return ODict(zip(keys, sum_result.tolist()))


def draw_spectrogram(data: gen.TensArr, fs: int, to_db=True, show=False, **kwargs):
    """
    
    :param data: 
    :param to_db:
    :param show: 
    :param kwargs: vmin, vmax
    :return: 
    """

    data = data.squeeze()
    data = gen.convert(data, astype=ndarray)
    if to_db:
        data = librosa.amplitude_to_db(data)

    fig, ax = plt.subplots(dpi=150)
    ax.imshow(data,
              cmap=plt.get_cmap('CMRmap'),
              extent=(0, data.shape[1], 0, fs // 2),
              origin='lower', aspect='auto', **kwargs)
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Frequency (Hz)')
    fig.colorbar(ax.images[0], format='%+2.0f dB')

    fig.tight_layout()
    if show:
        fig.show()

    return fig


def draw_audio(data: gen.TensArr, fs: int, show=False, xlim=None, ylim=(-1, 1)):
    data = data.squeeze()
    data = gen.convert(data, astype=ndarray)
    t_axis = np.arange(len(data)) / fs
    if xlim is None:
        xlim = (0, t_axis[-1])

    fig, ax = plt.subplots(figsize=(xlim[1] * 10, 2), dpi=150)
    ax.plot(t_axis, data)
    ax.set_xlabel('time')
    ax.xaxis.set_major_locator(tckr.MultipleLocator(0.5))

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    fig.tight_layout()
    if show:
        fig.show()

    return fig


def principle_(angle):
    angle += np.pi
    angle %= (2 * np.pi)
    angle -= np.pi
    return angle


def apply_freq_domain_filter(wave: ndarray, filter_fft: ndarray):
    # bnkr equalization in frequency domain
    filtered = []
    if wave.ndim == 1:
        channel = 1
        wave = (wave,)
    else:
        channel = wave.shape[0]
    if filter_fft.ndim == 1:
        iter_filter = repeat(filter_fft, channel)
    else:
        assert filter_fft.shape[0] == channel
        iter_filter = filter_fft
    for item_wave, item_filter in zip(wave, iter_filter):
        spec = librosa.stft(item_wave,
                            hp.n_fft, hp.l_hop, hp.l_frame,
                            dtype=np.complex128)
        filtered.append(
            librosa.istft(spec * item_filter[:hp.n_freq, np.newaxis],
                          hp.l_hop, hp.l_frame,
                          dtype=item_wave.dtype,
                          length=len(item_wave))
        )

    if channel == 1:
        filtered = filtered[0]
    else:
        filtered = np.stack(filtered, axis=0)

    return filtered


def bnkr_equalize_time(wave: ndarray) -> ndarray:
    """compensate modal strength $b_n(kr)$ in the 0-th order time-domain SHD signal

    :param wave:
    :return:
    """
    return apply_freq_domain_filter(wave, hp.bnkr_inv0)
