from collections import OrderedDict as ODict
from typing import IO, Sequence, Tuple, Union
from itertools import repeat

import librosa
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as tckr
from numpy import ndarray
from scipy.linalg import toeplitz
import scipy.signal as scsig

from hparams import hp
import generic as gen
from matlab_lib import Evaluation as EvalModule


# class SNRseg(nn.Module):
#     EINEXPR = 'ftc,ftc->t'
#
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, y_clean: torch.Tensor, y_est: torch.Tensor,
#                 T_ys: ndarray) -> torch.Tensor:
#         if not T_ys:
#             T_ys = (y_est.shape[-2],) * y_est.shape[0]
#         sum_result = torch.zeros(1, device=y_est.device)
#         for i_b, (T, item_clean, item_est) in enumerate(zip(T_ys, y_clean, y_est)):
#             # T
#             norm_y = torch.einsum(SNRseg.einexpr,
#                                   [item_clean[:, :T, :]] * 2)
#             norm_err = torch.einsum(SNRseg.einexpr,
#                                     [item_est[:, :T, :] - item_clean[:, :T, :]]*2)
#
#             sum_result += torch.log10(norm_y / norm_err).mean(dim=1)
#         sum_result *= 10
#         return sum_result

# class Measurement:
#     __slots__ = ('__len',
#                  '__METRICS',
#                  '__sum_values',
#                  )
#
#     DICT_CALC = dict(SNRseg='Measurement.calc_snrseg',
#                      # STOI=Measurement.calc_stoi,
#                      PESQ_STOI='Measurement.calc_pesq_stoi',
#                      )
#
#     def __init__(self, *metrics):
#         self.__len = 0
#         self.__METRICS = metrics
#         # self.__seq_values = {metric: torch.empty(max_size) for metric in self.__METRICS}
#         self.__sum_values: ODict = None
#
#     def __len__(self):
#         return self.__len
#
#     def __str__(self):
#         return self._str_measure(self.average())
#
#     # def __getitem__(self, idx):
#     #     return [self.__seq_values[metric][idx] for metric in self.__METRICS]
#
#     def average(self):
#         """
#
#         :rtype: OrderedDict[str, torch.Tensor]
#         """
#         return ODict([(metric, sum_ / self.__len)
#                       for metric, sum_ in self.__sum_values.items()])
#
#     def append(self, y: ndarray, out: ndarray,
#                T_ys: Union[int, Sequence[int]]) -> str:
#         values = ODict([(metric, eval(self.DICT_CALC[metric])(y, out, T_ys))
#                         for metric in self.__METRICS])
#         if self.__len:
#             for metric, v in values.items():
#                 # self.__seq_values[metric][self.__len] = self.DICT_CALC[metric](y, out, T_ys)
#                 self.__sum_values[metric] += v
#         else:
#             self.__sum_values = values
#
#         self.__len += len(T_ys) if hasattr(T_ys, '__len__') else 1
#         return self._str_measure(values)
#
#     @staticmethod
#     def _str_measure(values: ODict) -> str:
#         return '\t'.join(
#             [f"{metric}={arr2str(v, 'f')} " for metric, v in values.items()]
#         )
#
#
# def calc_stoi(y_clean: ndarray, y_est: ndarray):
#     sum_result = 0.
#     for item_clean, item_est in zip(y_clean, y_est):
#         sum_result += stoi(item_clean, item_est, hp.Fs)
#     return sum_result


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


def reconstruct_wave(*args: ndarray, n_iter=0, n_sample=-1) -> ndarray:
    """ reconstruct time-domain wave from spectrogram

    :param args: can be (mag_spectrogram, phase_spectrogram) or (complex_spectrogram,)
    :param n_iter: no. of iteration of griffin-lim
    :param n_sample: number of samples of output wave
    :return:
    """

    if len(args) == 1:
        spec = args[0].squeeze()
        mag = None
        phase = None
        assert np.iscomplexobj(spec)
    elif len(args) == 2:
        spec = None
        mag = args[0].squeeze()
        phase = args[1].squeeze()
        assert np.isrealobj(mag) and np.isrealobj(phase)
    else:
        raise ValueError

    for _ in range(n_iter - 1):
        if mag is None:
            mag = np.abs(spec)
            phase = np.angle(spec)
            spec = None
        wave = librosa.core.istft(mag * np.exp(1j * phase), **hp.kwargs_istft)

        phase = np.angle(librosa.core.stft(wave, **hp.kwargs_stft))

    kwarg_len = dict(length=n_sample) if n_sample != -1 else dict()
    if spec is None:
        spec = mag * np.exp(1j * phase)
    wave = librosa.core.istft(spec, **hp.kwargs_istft, **kwarg_len)

    return wave


def delta(*data: gen.TensArr, axis: int, L=2) -> gen.TensArrOrSeq:
    dim = gen.ndim(data[0])
    dtype = gen.convert_dtype(data[0].dtype, np)
    if axis < 0:
        axis += dim

    max_len = max([item.shape[axis] for item in data])

    # Einsum expression
    # ex) if the member of a has the dim (b,c,f,t), (thus, axis=3)
    # einxp: ij,abcd -> abci
    str_axes = ''.join([chr(ord('a') + i) for i in range(dim)])
    str_new_axes = ''.join([chr(ord('a') + i) if i != axis else 'i'
                            for i in range(dim)])
    ein_expr = f'ij,{str_axes}->{str_new_axes}'

    # Create Toeplitz Matrix (T-2L, T)
    col = np.zeros(max_len - 2 * L, dtype=dtype)
    col[0] = -L

    row = np.zeros(max_len, dtype=dtype)
    row[:2 * L + 1] = range(-L, L + 1)

    denominator = np.sum([ll**2 for ll in range(1, L + 1)])
    tplz_mat = toeplitz(col, row) / (2 * denominator)

    # Convert to Tensor
    if type(data[0]) == torch.Tensor:
        if data[0].device == torch.device('cpu'):
            tplz_mat = torch.from_numpy(tplz_mat)
        else:
            tplz_mat = torch.tensor(tplz_mat, device=data[0].device)

    # Calculate
    result = [type(data[0])] * len(data)
    for idx, item in enumerate(data):
        length = item.shape[axis]
        result[idx] = gen.einsum(ein_expr,
                                 (tplz_mat[:length - 2 * L, :length], item))

    return result if len(result) > 1 else result[0]


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

    fig, ax = plt.subplots(dpi=300)
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

    fig, ax = plt.subplots(figsize=(xlim[1] * 10, 2), dpi=300)
    ax.plot(t_axis, data)
    ax.set_xlabel('time')
    ax.xaxis.set_major_locator(tckr.MultipleLocator(0.5))

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    fig.tight_layout()
    if show:
        fig.show()

    return fig


def apply_iir_filter(wave: ndarray, filter_fft: ndarray):
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


def bnkr_equalize_(*args: ndarray) \
        -> Union[ndarray, Tuple[ndarray, Union[ndarray, None]]]:
    """ divide spectrogram into bnkr with regularization

    :param args: (complex_spectrogram,), (magnitude_spectrogram,),
        or (magnitude_spectrogram, phase_spectrogram)
    :return: same as args
    """
    if len(args) == 1:
        if np.iscomplexobj(args[0]):
            spec = args[0]
            mag = None
            phase = None
        else:
            spec = None
            mag = args[0]
            phase = None
    elif len(args) == 2:
        spec = None
        mag, phase = args
    else:
        raise ValueError

    if mag is None:
        assert spec.shape[0] == hp.bnkr_inv0.shape[0]
        bnkr_inv0 = hp.bnkr_inv0.copy()
        while spec.ndim < bnkr_inv0.ndim:
            bnkr_inv0 = bnkr_inv0[..., 0]

        spec *= bnkr_inv0

        return spec
    else:
        assert mag.shape[0] == hp.bnkr_inv0_mag.shape[0]
        bnkr_inv0_mag = hp.bnkr_inv0_mag.copy()
        while mag.ndim < bnkr_inv0_mag.ndim:
            bnkr_inv0_mag = bnkr_inv0_mag[..., 0]

        mag *= bnkr_inv0_mag

        if phase is not None:
            assert phase.shape[0] == mag.shape[0]
            bnkr_inv0_angle = hp.bnkr_inv0_angle.copy()
            while phase.ndim < bnkr_inv0_mag.ndim:
                bnkr_inv0_angle = bnkr_inv0_angle[..., 0]

            phase += bnkr_inv0_angle
            phase = principle_(phase)

            return mag, phase
        else:
            if len(args) == 1:
                return mag
            else:
                return mag, None


def bnkr_equalize(*args: ndarray) -> Union[ndarray, Tuple[ndarray, ndarray]]:
    args = [item.copy() for item in args]
    return bnkr_equalize_(*args)


def bnkr_equalize_time(wave: ndarray) -> ndarray:
    return apply_iir_filter(wave, hp.bnkr_inv0)
