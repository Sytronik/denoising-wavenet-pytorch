"""
Generic type & functions for torch.Tensor and np.ndarray
"""
from typing import Sequence, TypeVar, Union

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

TensArr = TypeVar('TensArr', Tensor, ndarray)
TensArrOrSeq = Union[TensArr, Sequence[TensArr]]

dict_package = {Tensor: torch, ndarray: np}
dict_cat_stack_fn = {(Tensor, 'cat'): torch.cat,
                     (ndarray, 'cat'): np.concatenate,
                     (Tensor, 'stack'): torch.stack,
                     (ndarray, 'stack'): np.stack,
                     }


class DataPerDevice:
    __slots__ = ('data',)

    def __init__(self, data_np: ndarray):
        self.data = {ndarray: data_np}

    def __getitem__(self, typeOrtup):
        if type(typeOrtup) == tuple:
            _type, device = typeOrtup
        elif typeOrtup == ndarray:
            _type = ndarray
            device = None
        else:
            raise IndexError

        if _type == ndarray:
            return self.data[ndarray]
        else:
            if typeOrtup not in self.data:
                self.data[typeOrtup] = convert(self.data[ndarray].astype(np.float32),
                                               Tensor,
                                               device=device)
            return self.data[typeOrtup]

    def get_like(self, other: TensArr):
        if type(other) == Tensor:
            return self[Tensor, other.device]
        else:
            return self[ndarray]


def convert_dtype(dtype: type, pkg) -> type:
    if hasattr(dtype, '__name__'):
        if pkg == np:
            return dtype
        else:
            return eval(f'torch.{dtype.__name__}')
    else:
        if pkg == np:
            return eval(f'np.{str(dtype).split(".")[-1]}')
        else:
            return dtype


def copy(a: TensArr, requires_grad=True) -> TensArr:
    if type(a) == Tensor:
        return a.clone() if requires_grad else torch.tensor(a)
    elif type(a) == ndarray:
        return np.copy(a)
    else:
        raise TypeError


def convert(a: TensArr, astype: type, device: Union[int, torch.device] = None) -> TensArr:
    if astype == Tensor:
        if type(a) == Tensor:
            return a.to(device)
        else:
            return torch.as_tensor(a, dtype=torch.float32, device=device)
    elif astype == ndarray:
        if type(a) == Tensor:
            return a.cpu().numpy()
        else:
            return a
    else:
        raise ValueError(astype)


def convert_like(a: TensArr, b: TensArr) -> TensArr:
    if type(b) == Tensor:
        if type(a) == Tensor:
            return a.to(b.device)
        else:
            return torch.as_tensor(a, device=b.device)
    elif type(b) == ndarray:
        if type(a) == Tensor:
            return a.cpu().numpy()
        else:
            return a
    else:
        raise ValueError(type(b))


def ndim(a: TensArr) -> int:
    if type(a) == Tensor:
        return a.dim()
    elif type(a) == ndarray:
        return a.ndim
    else:
        raise TypeError


def transpose(a: TensArr, axes: Union[int, Sequence[int]] = None) -> TensArr:
    if type(a) == Tensor:
        if not axes:
            if a.dim() >= 2:
                return a.permute((1, 0) + tuple(range(2, a.dim())))
            else:
                return a
        else:
            return a.permute(axes)

    elif type(a) == ndarray:
        if a.ndim == 1 and not axes:
            return a
        else:
            return a.transpose(axes)
    else:
        raise TypeError


def einsum(subscripts: str,
           operands: Sequence[TensArr],
           astype: type = None) -> TensArr:
    if not astype:
        astype = type(operands[0])
        if astype != Tensor and astype != ndarray:
            raise TypeError
    else:
        types = [type(item) for item in operands]
        for idx, type_ in enumerate(types):
            if type_ != astype:
                if type(operands) != list:
                    operands = list(operands)
                operands[idx] = convert(operands[idx], astype)

    return dict_package[astype].einsum(subscripts, operands)


def arctan2(a: TensArr, b: TensArr, out: TensArr = None) -> TensArr:
    if type(a) == Tensor:
        return torch.atan2(a, b, out=out)
    else:
        return np.arctan2(a, b, out=out)


def unwrap(a: TensArr, axis: int = -1) -> TensArr:
    if type(a) == Tensor:
        if axis < 0:
            axis += a.dim()

        def get_slice(*args, **kwargs):
            return ((slice(None),) * axis
                    + (slice(*args, **kwargs),)
                    + (slice(None),) * (a.dim() - axis - 1)
                    )

        diff = a[get_slice(1, None)] - a[get_slice(None, -1)]
        dps = (diff + np.pi) % (2 * np.pi) - np.pi
        dps[(dps == -np.pi) & (diff > 0)] = np.pi
        dps -= diff
        corr = dps
        corr[torch.abs(diff) < np.pi] = 0
        cumsum = torch.cumsum(corr, dim=axis, out=corr)
        result = torch.empty_like(a)
        result[get_slice(0, 1)] = a[get_slice(0, 1)]
        result[get_slice(1, None)] = a[get_slice(1, None)] + cumsum

        return result
    elif type(a) == ndarray:
        return np.unwrap(a, axis=axis)
    else:
        raise TypeError


def where(a: TensArr):
    if type(a) == Tensor:
        return tuple([item.squeeze() for item in a.nonzero().split(1, 1)])
    elif type(a) == ndarray:
        return np.where(a)
    else:
        raise TypeError


def expand_dims(a: TensArr, axis: int) -> TensArr:
    if type(a) == Tensor:
        return a.unsqueeze_(axis)
    elif type(a) == ndarray:
        return np.expand_dims(a, axis)
    else:
        raise TypeError


def _cat_stack(fn: str,
               a: Sequence[TensArr],
               axis=0,
               astype: type = None) -> TensArr:
    types = [type(item) for item in a]
    if not astype:
        astype = types[0]
    for idx, type_ in enumerate(types):
        if type_ != astype:
            if type(a) == tuple:
                a = list(a)
            a[idx] = convert(a[idx], astype)

    return dict_cat_stack_fn[(astype, fn)](a, axis)


def cat(*args, **kargs) -> TensArr:
    """
    <parameters>
    a: Iterable[TensArr]
    axis=0
    astype: type=None
    """
    return _cat_stack('cat', *args, **kargs)


def stack(*args, **kargs) -> TensArr:
    """
    <parameters>
    a: Iterable[TensArr]
    axis=0
    astype: type=None
    """
    return _cat_stack('stack', *args, **kargs)
