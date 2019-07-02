

import inspect
import io
from collections import OrderedDict as ODict
from typing import Union

from numpy import ndarray


class CallableSingletonMeta(type):
    def __call__(cls, *args, **kwargs):
        if hasattr(cls, 'instance') and cls.instance:
            if (args or kwargs) and callable(cls.instance):
                return cls.instance(*args, **kwargs)
            else:
                return cls.instance
        else:
            if len(inspect.getfullargspec(cls.__init__)[0]) == 1:
                instance = type.__call__(cls)
                if (args or kwargs) and callable(instance):
                    return instance(*args, **kwargs)
                else:
                    return instance
            else:
                return type.__call__(cls, *args, **kwargs)


class Evaluation(metaclass=CallableSingletonMeta):
    __slots__ = ('eng', 'strio')

    instance = None
    metrics = ('fwSegSNR', 'PESQ', 'STOI')

    def __init__(self):
        import matlab
        import matlab.engine
        self.eng = matlab.engine.start_matlab('-nojvm')
        self.eng.addpath(self.eng.genpath('./matlab_lib'))
        self.strio = io.StringIO()
        Evaluation.instance: Evaluation = self

    def __call__(self, clean: ndarray, noisy: ndarray, fs: int) -> ODict:
        import matlab
        clean = matlab.double(clean.tolist())
        noisy = matlab.double(noisy.tolist())
        fs = matlab.double([fs])
        results = self.eng.se_eval(clean, noisy, fs, nargout=3, stdout=self.strio)

        return ODict([(m, r) for m, r in zip(Evaluation.metrics, results)])

    def _exit(self):
        self.eng.quit()

        # fname = datetime.now().strftime('log_matlab_eng_%Y-%m-%d %H.%M.%S.txt')
        # with io.open(fname, 'w') as flog:
        #     self.strio.seek(0)
        #     shutil.copyfileobj(self.strio, flog)

        self.strio.close()

    def __del__(self):
        self._exit()
