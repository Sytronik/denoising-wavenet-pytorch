from pathlib import Path

import numpy as np
import torch
from numpy import ndarray
from tensorboardX import SummaryWriter
import librosa

from hparams import hp
from audio_utils import (bnkr_equalize_time,
                         calc_snrseg_time,
                         calc_using_eval_module,
                         EVAL_METRICS,
                         draw_spectrogram,
                         draw_audio,
                         )


class CustomWriter(SummaryWriter):
    def __init__(self, *args, group='', **kwargs):
        super().__init__(*args, **kwargs)
        self.group = group
        if group == 'valid':
            dict_custom_scalars = dict(loss=['Multiline', ['loss/train',
                                                           'loss/valid']])
        else:
            dict_custom_scalars = dict()

        dict_custom_scalars['1_SNRseg'] = ['Multiline', [f'{group}/1_SNRseg/Reverberant',
                                                         f'{group}/1_SNRseg/Proposed']]

        for i, m in enumerate(EVAL_METRICS):
            dict_custom_scalars[f'{i + 2}_{m}'] = [
                'Multiline', [f'{group}/{i + 2}_{m}/Reverberant',
                              f'{group}/{i + 2}_{m}/Proposed']
            ]

        self.add_custom_scalars({group: dict_custom_scalars})
        self.reused_sample = dict()
        self.measure_x = dict()
        self.kwargs_fig = dict()
        self.y_max = 1.
        self.xlim = None

    def write_one(self, step: int, out: ndarray = None,
                  **kwargs: ndarray) -> ndarray:
        """ write summary about one sample of output(and x and y optionally).

        :param step:
        :param out:
        :param kwargs: keywords can be [x, y]

        :return: evaluation result
        """

        assert out is not None
        if kwargs:
            x, y = kwargs['x'], kwargs['y']
            do_reuse = False
        else:
            assert self.reused_sample
            x, y = None, None
            do_reuse = True

        if do_reuse:
            y = self.reused_sample['y']
            pad_one = self.reused_sample['pad_one']

            snrseg_x = self.measure_x['SNRseg']
            odict_eval_x = self.measure_x['odict_eval']
        else:
            # T,
            x = x.mean(0)
            y = y.squeeze()

            if hp.do_bnkr_eq:
                x = bnkr_equalize_time(x)

            # x *= np.linalg.norm(y, ord=2) / np.linalg.norm(x, ord=2)

            snrseg_x = calc_snrseg_time(y, x[:len(y)], hp.l_frame, hp.l_hop)
            odict_eval_x = calc_using_eval_module(y, x[:len(y)])

            # draw
            self.xlim = (0, len(x) / hp.fs)
            self.y_max = np.abs(y).max()

            fig_x = draw_audio(x, hp.fs,
                               xlim=self.xlim,
                               ylim=(-self.y_max * 1.4, self.y_max * 1.4))
            fig_y = draw_audio(y, hp.fs,
                               xlim=self.xlim,
                               ylim=(-self.y_max * 1.4, self.y_max * 1.4))

            x_spec = librosa.amplitude_to_db(np.abs(librosa.stft(x, **hp.kwargs_stft)))
            y_spec = librosa.amplitude_to_db(np.abs(librosa.stft(y, **hp.kwargs_stft)))

            vmin, vmax = y_spec.min(), y_spec.max()
            pad_one = np.ones((y_spec.shape[0], x_spec.shape[1] - y_spec.shape[1]))
            y_spec = np.append(y_spec, y_spec.min() * pad_one, axis=1)

            fig_x_spec = draw_spectrogram(x_spec, hp.fs, to_db=False)
            fig_y_spec = draw_spectrogram(y_spec, hp.fs, to_db=False)

            self.add_figure(f'{self.group}/1_Anechoic Spectrum', fig_y_spec, step)
            self.add_figure(f'{self.group}/2_Reverberant Spectrum', fig_x_spec, step)

            self.add_figure(f'{self.group}/4_Anechoic Wave', fig_y, step)
            self.add_figure(f'{self.group}/5_Reverberant Wave', fig_x, step)

            self.add_audio(f'{self.group}/1_Anechoic Wave',
                           torch.from_numpy(y / self.y_max * 0.707),
                           step,
                           sample_rate=hp.fs)
            self.add_audio(f'{self.group}/2_Reverberant Wave',
                           torch.from_numpy(x / np.abs(x).max() * 0.707),
                           step,
                           sample_rate=hp.fs)

            self.reused_sample = dict(x=x, y=y,
                                      pad_one=pad_one,
                                      )
            self.measure_x = dict(SNRseg=snrseg_x, odict_eval=odict_eval_x)
            self.kwargs_fig = dict(vmin=vmin, vmax=vmax)

        out = out.squeeze()

        snrseg = calc_snrseg_time(y, out, hp.l_frame, hp.l_hop)

        odict_eval = calc_using_eval_module(y, out)

        fig_out = draw_audio(out, hp.fs,
                             xlim=self.xlim,
                             ylim=(-self.y_max * 1.4, self.y_max * 1.4))

        out_spec = librosa.amplitude_to_db(np.abs(librosa.stft(out, **hp.kwargs_stft)))
        out_spec = np.append(out_spec, self.kwargs_fig['vmin'] * pad_one, axis=1)
        fig_out_spec = draw_spectrogram(out_spec, hp.fs, to_db=False,
                                        **self.kwargs_fig)

        self.add_scalar(f'{self.group}/1_SNRseg/Reverberant', snrseg_x, step)
        self.add_scalar(f'{self.group}/1_SNRseg/Proposed', snrseg, step)
        for i, m in enumerate(odict_eval.keys()):
            self.add_scalar(f'{self.group}/{2 + i}_{m}/Reverberant', odict_eval_x[m], step)
            self.add_scalar(f'{self.group}/{2 + i}_{m}/Proposed', odict_eval[m], step)

        self.add_figure(f'{self.group}/3_Estimated Anechoic Spectrum', fig_out_spec, step)
        self.add_figure(f'{self.group}/6_Estimated Anechoic Wave', fig_out, step)

        self.add_audio(f'{self.group}/3_Estimated Anechoic Wave',
                       out / self.y_max * 0.707,
                       step,
                       sample_rate=hp.fs)

        return np.array([[snrseg, *(odict_eval.values())],
                         [snrseg_x, *(odict_eval_x.values())]])
