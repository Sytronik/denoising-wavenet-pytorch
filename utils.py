import contextlib
import gc
import os
from pathlib import Path
from typing import Callable, List, Union

import numpy as np
import torch
import torch.optim


def static_vars(**kwargs):
    """ decorator to make static variables in function

    :param kwargs:
    :return:
    """
    def decorate(func: Callable):
        for k, a in kwargs.items():
            setattr(func, k, a)
        return func

    return decorate


class MultipleOptimizer(object):
    def __init__(self, *op: torch.optim.Optimizer):
        self._optimizers = [item for item in op if item]

    def zero_grad(self):
        for op in self._optimizers:
            op.zero_grad()

    def step(self):
        for op in self._optimizers:
            # noinspection PyArgumentList
            op.step()

    def state_dict(self) -> List[dict]:
        return [op.state_dict() for op in self._optimizers]

    def load_state_dict(self, state_dicts: List[dict]):
        for op, st in zip(self._optimizers, state_dicts):
            op.load_state_dict(st)

    def __len__(self):
        return len(self._optimizers)

    def __getitem__(self, idx: int) -> torch.optim.Optimizer:
        return self._optimizers[idx]


class MultipleScheduler(object):
    def __init__(self, cls_scheduler: type,
                 optimizers: MultipleOptimizer, *args, **kwargs):
        self._schedulers = [cls_scheduler(op, *args, **kwargs) for op in optimizers]

    def step(self):
        for sch in self._schedulers:
            sch.step()

    def batch_step(self):
        for sch in self._schedulers:
            sch.batch_step()

    def __len__(self):
        return len(self._schedulers)

    def __getitem__(self, idx: int):
        return self._schedulers[idx]


def arr2str(a: np.ndarray, format_='e', ndigits=2) -> str:
    """convert ndarray of floats to a string expression.

    :param a:
    :param format_:
    :param ndigits:
    :return:
    """
    return np.array2string(
        a,
        formatter=dict(
            float_kind=(lambda x: f'{x:.{ndigits}{format_}}' if x != 0 else '0')
        )
    )


# deprecated. Use tqdm
def print_progress(iteration: int, total: int, prefix='', suffix='',
                   decimals=1, len_bar=0):
    percent = f'{100 * iteration / total:>{decimals + 4}.{decimals}f}'
    if len_bar == 0:
        len_bar = (min(os.get_terminal_size().columns, 80)
                   - len(prefix) - len(percent) - len(suffix) - 11)

    len_filled = len_bar * iteration // total
    bar = '#' * len_filled + '-' * (len_bar - len_filled)

    print(f'{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print('')


def print_cuda_tensors():
    """ Print all cuda Tensors """
    for obj in gc.get_objects():
        try:
            if (torch.is_tensor(obj)
                    or (hasattr(obj, 'data') and torch.is_tensor(obj.data))):
                print(type(obj), obj.size(), obj.device)
        finally:
            pass


def print_to_file(fname: Union[str, Path], fn: Callable, args=None, kwargs=None):
    """ All `print` function calls in `fn(*args, **kwargs)`
      uses a text file `fname`.

    :param fname:
    :param fn:
    :param args: args for fn
    :param kwargs: kwargs for fn
    :return:
    """
    if fname:
        fname = Path(fname).with_suffix('.txt')

    if args is None:
        args = tuple()
    if kwargs is None:
        kwargs = dict()

    with (fname.open('w') if fname else open(os.devnull, 'w')) as file:
        with contextlib.redirect_stdout(file):
            fn(*args, **kwargs)
