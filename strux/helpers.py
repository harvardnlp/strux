import jax.numpy as np
import math
from .semirings import LogSemiring
import jax

class _Struct:
    length_axes = (0,)
    def __init__(self, semiring=LogSemiring):
        self.semiring = semiring

        if False:
            self.sum = (jax.vmap(self._dp, (0, 0)))
            self.marginals = (jax.grad(lambda *args: self.sum(*args).sum(0), 0))
            def fp(*args):
                v, extra = (jax.vmap(self._from_parts, (0,)))(*args)
                return v, extra[0]
            self.from_parts = fp
            self.to_parts = (jax.vmap(self._to_parts, (0, None, 0)))

        else:
            self.sum = jax.jit(jax.vmap(self._dp, (0, 0)))
            self.marginals = jax.jit(jax.grad(lambda *args: self.sum(*args).sum(0), 0))
            def fp(*args):
                v, extra = jax.jit(jax.vmap(self._from_parts, (0,)))(*args)
                return v, extra[0]
            self.from_parts = fp
            self.to_parts = jax.jit(jax.vmap(self._to_parts, (0, None, 0)),
                                    static_argnums=1)

    def _dp(log_potentials, length):
        pass 

    @classmethod
    def resize(cls, log_potentials, batch=1):
        for j in cls.length_axes:
            log_potentials = pad_to_pow2(log_potentials, batch + j)
        return log_potentials
        
    def score(self, potentials, parts, batch_dims=[0]):
        score = potentials * parts
        batch = tuple((score.shape[b] for b in batch_dims))
        return self.semiring.prod(score.reshape(batch + (-1,)))

    @staticmethod
    def _to_parts(spans, extra, lengths):
        return spans

    @staticmethod
    def _from_parts(spans):
        return spans, 0

def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0):
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)

def pad_to_pow2(tensor, axis):
    size = tensor.shape[axis]
    new_size = int(np.power(2, np.ceil(size)))
    return pad_along_axis(tensor, new_size, axis)
    
