import csv
import json
import os
from argparse import ArgumentParser
from itertools import product as iterprod
from pathlib import Path
from typing import Any, List, NamedTuple, Sequence, Union

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from numpy import ndarray
from tensorboard.backend.event_processing.event_multiplexer import EventMultiplexer
from tqdm import tqdm

from hparams import hp

PathLike = Union[str, Path]
StrOrSeq = Union[str, Sequence[str]]
PathOrSeq = Union[str, Path, Sequence[str], Sequence[Path]]

parser = ArgumentParser()
parser.add_argument('--save-scalars', '-s', action='store_true')
parser.add_argument('--force', '-f', action='store_true')
parser.add_argument('--merge-scalars', '-m', action='store_true')
parser.add_argument('--tbcommand', '-t', action='store_true')
ARGS = parser.parse_args()

PATH_ROOT = Path('./result')
ROOM = 'room1'
TITLE_BASELINE: str = 'No-DF'
BASELINE: str = f'UNet (p00 {ROOM})'
TITLE_PRIMARY: StrOrSeq = ['DV', 'SIV']
PRIMARY: PathOrSeq = [f'UNet (DirAC+p00 {ROOM})', f'UNet (IV+p00 {ROOM})']

# TITLE_BASELINE: str = 'baseline'
# BASELINE: str = 'UNet (p00 room2)'
# TITLE_PRIMARY: StrOrSeq = ['CBAM']
# PRIMARY: PathOrSeq = ['UNet-CBAM (p00 room2)']

COND_PRIMARY = lambda p: True

# SUFFIX = ''
# SUFFIX = 'train'
# SUFFIX = 'seen'
# SUFFIX = 'unseen'
# SUFFIX = 'unseen_room1'
# SUFFIX = 'unseen_room2'
SUFFIX = ['seen', 'unseen', 'unseen_room1', 'unseen_room2']
# SUFFIX = ['seen', 'unseen', 'unseen_room2']
# SUFFIX = ['seen', 'unseen', 'unseen_room1']

# NEED_REVERB = 'delta'
NEED_REVERB = 'sep'
# NEED_REVERB = False

if '_' in ROOM:
    ROOM = 'mixed'
names_RIR = dict(seen=f'{ROOM}\n(seen)', unseen=f'{ROOM}\n(unseen)')

need_all_scalars = False if SUFFIX == 'train' else True

Y_MAX = dict(PESQ=4.5, STOI=1., )


class IdxSuffixCol(NamedTuple):
    idx: int
    suffix: str
    col: str

    def __contains__(self, item):
        if super().__contains__(item):
            return True
        else:
            return item in self.col

    def to(self, *, idx=0, suffix='', col=''):
        idx = idx if idx else self.idx
        suffix = suffix if suffix else self.suffix
        col = col if col else self.col
        return IdxSuffixCol(idx, suffix, col)


def make_command():
    global paths, SUFFIX
    if type(SUFFIX) != str:
        SUFFIX = ''
    paths = [Path(p.name) / SUFFIX for p in paths]
    if len(paths) == 1:
        logdir = str(paths[0])
    else:
        list_logdir: List[str] = [f'{t}:{p}' for t, p in zip(titles, paths)]
        logdir = ','.join(list_logdir)

    command = f'tensorboard --logdir="{logdir}"'
    if need_all_scalars:
        command += ' --samples_per_plugin "scalars=10000,images=1,audio=1"'

    print(command)
    # for item in dir():
    #     if not item.startswith('_') and item != 'command':
    #         exec(f'del {item}')
    #
    # exec('del item')
    # os.system(f'echo \'{command}\' | pbcopy')


def save_scalars(path: PathLike, suffix: str):
    DIR_EVENTS = Path(path, suffix)
    # suffix = suffix.split('_')[0]

    dict_logdirs = {}
    for file in DIR_EVENTS.glob('**/events.out.tfevents.*'):
        key = str(file.parent.absolute()).replace(str(DIR_EVENTS.absolute()), '')[1:]
        if not key:
            key = '.'
        dict_logdirs[key] = str(file)

    events = EventMultiplexer(dict_logdirs, size_guidance=dict(scalars=10000))
    events.Reload()

    scalars = {}
    step_longest = None
    for key in dict_logdirs:
        event = events.GetAccumulator(key)
        for tag in event.Tags()['scalars']:
            _, step, value = zip(*event.Scalars(tag))
            if tag.replace('_', ' ') in key:
                key = key.replace(f'{suffix.split("_")[0]}{os.sep}', '')
                scalars[key] = value
            else:
                tag = tag.replace('_', ' ').replace(f'{suffix.split("_")[0]}{os.sep}', '')
                scalars[tag] = value
            if step_longest is None or len(step) > len(step_longest):
                step_longest = step

    fjson = DIR_EVENTS / 'scalars.json'
    if not ARGS.force and fjson.exists():
        print(f'{fjson} already exists.')
    with fjson.open('w') as f:
        json.dump(dict(step=step_longest, **scalars), f)


def _graph_initialize(_titles: Union[str, Sequence[str]],
                      means: ndarray, stds: ndarray,
                      xticklabels: Sequence,
                      ylabel: str):
    plt.rc('font', family='Arial', size=18)

    fig, ax = plt.subplots(figsize=(5.3, 4))

    # args
    if means.ndim == 1:
        if type(_titles) == str:
            _titles = (_titles,)
        means = means[np.newaxis, :]
        stds = stds[np.newaxis, :] if stds is not None else (None,)
    if stds is None:
        stds = (None,) * len(_titles)

    common = None
    if xticklabels:
        xticklabels = xticklabels[:]
        common = {x.split('_')[0] for x in xticklabels}
        if len(common) == 1:
            xticklabels = ['_'.join(x.split('_')[1:]) for x in xticklabels]

        for ii, label in enumerate(xticklabels):
            if label in names_RIR:
                xticklabels[ii] = names_RIR[label]
            elif '_' in label:
                xticklabels[ii] = label.split('_')[1]
            else:
                raise NotImplementedError

    if len(_titles) == 1:
        if _titles[0] == '.' and common and len(common) == 1:
            _titles = tuple(common)
            draw_legend = True
        else:
            draw_legend = False
    else:
        draw_legend = True

    if 'PESQ' not in ylabel:
        draw_legend = False

    # colors
    cmap = plt.get_cmap('tab20c')
    colors = [cmap.colors[(1 + 4 * i // 16) % 4 + (4 * i) % 16] for i in range(len(_titles))]
    if NEED_REVERB == 'sep':
        colors[-1] = cmap.colors[17]

    # ylim
    # ylim = list(ax.get_ylim())
    # max_ = (means + stds).max()
    # min_ = (means - stds).min()
    # if NEED_REVERB == 'delta' and min_ >= 0:
    #     ylim[0] = 0
    # else:
    #     ylim[0] = min_ - (max_ - min_) * 0.2
    #
    # ylim[1] = ylim[1] + (max_ - ylim[0]) * 0.25
    # if ylabel in Y_MAX:
    #     ylim[1] = min(Y_MAX[ylabel], ylim[1])

    if ylabel == 'SegSNR [dB]':
        ylim = (-2.5, 12.5)
        ax.set_yticks(np.linspace(*ylim, num=7))
    elif ylabel == 'fwSegSNR [dB]':
        ylim = (6, 16)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    elif ylabel == 'PESQ':
        ylim = (2, 4.5)
        # ax.set_yticks(np.linspace(*ylim, num=7))
    elif ylabel == 'STOI':
        ylim = (0.7, 1)
        ax.set_yticks(np.linspace(*ylim, num=7))
    elif ylabel == 'ΔSegSNR [dB]':
        ylim = (0, 15)
        ax.set_yticks(np.linspace(*ylim, num=9))
    elif ylabel == 'ΔfwSegSNR [dB]':
        ylim = (-1, 8)
    elif ylabel == 'ΔPESQ':
        ylim = (0, 1.5)
    elif ylabel == 'ΔSTOI':
        ylim = (0, 0.3)
    else:
        raise NotImplementedError

    return fig, ax, means, stds, xticklabels, draw_legend, cmap, colors, ylim


def draw_lineplot(_titles: Union[str, Sequence[str]],
                  means: ndarray, stds: ndarray = None,
                  xticklabels: Sequence = None,
                  ylabel: str = ''):
    ax: plt.Axes = None
    fig, ax, means, stds, xticklabels, draw_legend, cmap, colors, ylim \
        = _graph_initialize(_titles, means, stds, xticklabels, ylabel)

    # draw
    range_ = np.arange(means.shape[1])
    for ii, (title, mean, std) in enumerate(zip(_titles, means, stds)):
        ax.plot(range_, mean,
                label=title,
                color=colors[ii],
                marker='o')

    ax.set_xticks(range_)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(range_[0] - 0.5, range_[-1] + 0.5)

    ax.grid(True, axis='y')
    ax.set_ylim(*ylim)

    # axis label
    # ax.set_xlabel('RIR')
    ax.set_ylabel(ylabel)

    # ticks
    ax.tick_params('x', direction='in')
    ax.tick_params('y', direction='in')

    # legend
    if draw_legend:
        # ax.legend(loc='lower right', bbox_to_anchor=(1, 1),
        #           ncol=4, fontsize='small', columnspacing=1)
        ax.legend(loc='upper center',
                  ncol=2, fontsize='small', columnspacing=1)

    fig.tight_layout()
    return fig


# noinspection PyUnusedLocal
def draw_bar_graph(_titles: Union[str, Sequence[str]],
                   means: ndarray, stds: ndarray = None,
                   xticklabels: Sequence = None,
                   ylabel: str = ''):
    # constants
    bar_width = 0.5 / len(_titles)
    ndigits = 3 if means.max() - means.min() < 0.1 else 2

    fig, ax, means, stds, xticklabels, draw_legend, cmap, colors, ylim \
        = _graph_initialize(_titles, means, stds, xticklabels, ylabel)

    # draw bar & text
    range_ = np.arange(means.shape[1])
    for ii, (title, mean, std) in enumerate(zip(_titles, means, stds)):
        bar = ax.bar(range_ + bar_width * ii, mean,
                     bar_width,
                     yerr=std,
                     error_kw=dict(capsize=5),
                     label=title,
                     color=colors[ii])

        # for b in bar:
        #     x = b.get_x() + b.get_width() * 0.55
        #     y = b.get_height()
        #     ax.text(x, y, f' {b.get_height():.{ndigits}f}',
        #             # horizontalalignment='center',
        #             rotation=40,
        #             rotation_mode='anchor',
        #             verticalalignment='center')

    ax.set_xticklabels(xticklabels)

    ax.grid(True, axis='y')
    ax.set_axisbelow(True)

    # xlim
    xlim = list(ax.get_xlim())
    xlim[0] -= bar_width
    xlim[1] += bar_width
    ax.set_xlim(*xlim)

    ax.set_ylim(*ylim)

    # axis label
    ax.set_xlabel('RIR')
    ax.set_ylabel(ylabel)

    # ticks
    ax.set_xticks(range_ + bar_width * (len(_titles) - 1) / 2)

    ax.tick_params('x', length=0)
    ax.tick_params('y', direction='in')

    # legend
    if draw_legend:
        ax.legend(loc='lower right', bbox_to_anchor=(1, 1),
                  ncol=4, fontsize='small', columnspacing=1)

    fig.tight_layout()
    return fig


def merge_scalars():
    global titles
    # gather data
    data = {}
    sfxs: list = None
    for idx, path in enumerate(paths):
        def cond(p):
            return (p.is_dir()
                    and p.name != 'train'
                    # and p.name.startswith('unseen')  # warning
                    and Path(p, 'scalars.json').exists())

        if SUFFIX:
            paths_sfx, sfxs = zip(*((path / sfx, sfx) for sfx in SUFFIX if cond(path / sfx)))
        else:
            paths_sfx, sfxs = zip(
                *((p.path, p.name) for p in os.scandir(path) if cond(p))
            )
        paths_sfx = sorted(paths_sfx)
        sfxs = sorted(sfxs)
        assert sfxs, f'{path} is empty.'
        # suffixes = [os.path.basename(p) for p in paths_suffix]
        for suffix, p in zip(sfxs, paths_sfx):
            # noinspection PyBroadException
            try:
                with Path(p, 'scalars.json').open('r') as f:
                    temp = json.loads(f.read())
                    for k, v in temp.items():
                        data[IdxSuffixCol(idx, suffix, k)] = v
            except:
                pass

    # arrange data
    data_arranged = {}
    cols_set = set()
    step_longest = None
    for key, value in data.items():
        if key.col == 'step':
            if step_longest is None or len(step_longest) < len(value):
                step_longest = value
        else:
            col_new = key.col.split('/')[0].split('. ')[-1]
            cols_set.add(col_new)
            if 'Reverberant' in key:
                if NEED_REVERB:
                    # key_rev = IdxSuffixCol(-1, key.suffix.split('_')[0], col_new)
                    key_rev = IdxSuffixCol(-1, key.suffix, col_new)
                    if key_rev not in data_arranged:
                        data_arranged[key_rev] = value
            else:
                data_arranged[key.to(col=col_new)] = value

    cols: list = sorted(cols_set)
    if NEED_REVERB == 'delta':
        for key, value in data_arranged.items():
            if key.col != 'step' and key.idx > -1:
                data_arranged[key] = np.subtract(value,
                                                 data_arranged[key.to(idx=-1)])
        data_arranged \
            = {k.to(col='Δ' + k.col): v
               for k, v in data_arranged.items()
               if k.idx != -1}
        cols = ['Δ' + c for c in cols]
    data = data_arranged
    idxs = [*range(length)]
    if NEED_REVERB == 'sep':
        idxs[-1] = -1
    del data_arranged

    # calculate mean, std
    means = {}
    stds = {}
    for key, value in data.items():
        means[key] = np.mean(value)
        stds[key] = np.std(value, ddof=1)

    # save data, mean and std to csv
    def make_rows(stats):
        _rows: List[List[Any]] = [None] * (len(cols) + 2)
        _rows[0] = [''] * (length * len(sfxs) + 1)
        for i_s, s in enumerate(sfxs):
            _rows[0][length * i_s + 1] = s
        _rows[1] = [''] + titles * len(sfxs)
        for i_c, c in zip(range(2, 2 + len(cols)), cols):
            _rows[i_c] = [c]
            for s in sfxs:
                _rows[i_c] += [stats[IdxSuffixCol(ii, s, c)] for ii in idxs]
        return _rows

    if len(paths) > 1:
        fresult = ', '.join(
            [f'{p.name}' + (f' ({t})' if t not in p.name else '')
             for p, t in zip(paths,
                             titles[:-1] if NEED_REVERB == 'sep' else titles)
             ]
        )
    else:
        fresult = f'{paths[0].name} ' \
                  + ', '.join(['_'.join(s.split('_')[1:]) for s in sfxs])
    fresult += f' rev={NEED_REVERB}'
    fmerged = (PATH_ROOT / fresult).with_suffix('.csv')
    with fmerged.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        rows = []
        for idx, title in zip(idxs, titles):
            rows.append([title])
            for suffix, (i_col, col) in iterprod(sfxs, enumerate(cols)):
                rows.append([suffix if i_col == 0 else '',
                             col,
                             *data[IdxSuffixCol(idx, suffix, col)],
                             ]
                            )
        rows += [['means'], *make_rows(means)]
        rows += [['stds'], *make_rows(stds)]
        for r in rows:
            writer.writerow(r)

    # draw bar graph
    ffig = str(Path(hp.dict_path['figures'], fresult)) + ' ({}).png'
    for col in cols:
        mean = np.empty((length, len(sfxs)))
        std = np.empty((length, len(sfxs)))
        for (i, idx), (j, suffix) in iterprod(enumerate(idxs), enumerate(sfxs)):
            mean[i, j] = means[IdxSuffixCol(idx, suffix, col)]
            std[i, j] = stds[IdxSuffixCol(idx, suffix, col)]

        if 'SNRseg' in col:
            col = col.replace('SNRseg', 'SegSNR')
        if 'SNR' in col:
            col += ' [dB]'
        # figbar = draw_bar_graph(titles, mean, std, sfxs, col)
        if 'ROOM' in globals():
            titles = [f'R{ROOM[1:]}-{t}' if t != 'Unproc.' else t for t in titles]
        figbar = draw_lineplot(titles, mean, std, sfxs, col)
        figbar.savefig(ffig.format(col.replace('Δ', '')), dpi=300)


if __name__ == '__main__':
    if TITLE_PRIMARY:
        if type(PRIMARY) == str:
            paths = [p
                     for p in PATH_ROOT.glob(PRIMARY)
                     if p.is_dir() and COND_PRIMARY(p)
                     ]
            paths = sorted(paths)
        else:
            paths = [PATH_ROOT / p
                     for p in PRIMARY
                     if Path(PATH_ROOT / p).is_dir() and COND_PRIMARY(p)
                     ]
            if len(PRIMARY) == 1:
                if type(TITLE_PRIMARY) != str:
                    TITLE_PRIMARY = TITLE_PRIMARY[0]
    else:
        paths = []
    if len(paths) > 1:
        titles = [TITLE_BASELINE] if TITLE_BASELINE else []
        if type(TITLE_PRIMARY) == str:
            titles += [f'{TITLE_PRIMARY}{i_p}' for i_p in range(len(paths))]
        else:
            titles += [f'{TITLE_PRIMARY[i_p]}' for i_p in range(len(paths))]
    else:
        titles = [TITLE_BASELINE, *TITLE_PRIMARY]
        titles = [t for t in titles if t]

    if TITLE_BASELINE:
        paths.insert(0, PATH_ROOT / BASELINE)

    if ARGS.tbcommand:
        make_command()
    else:
        if NEED_REVERB == 'sep':
            titles += ['Unproc.']

        length = len(titles)

        if SUFFIX and type(SUFFIX) == str:
            SUFFIX = (SUFFIX,)
        if ARGS.save_scalars:
            for p, s in tqdm(iterprod(paths, SUFFIX)):
                save_scalars(p, s)
        elif ARGS.merge_scalars:
            merge_scalars()
