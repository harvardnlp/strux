import jax.numpy as np
from .helpers import _Struct
import jax

A, B = 0, 1

class CKY_CRF(_Struct):
    length_axes = (0, 1)
    log_scale = False        
    def _dp(self, log_potentials, length):
        semiring = self.semiring
        print(log_potentials.shape)
        N, N2, NT = log_potentials.shape
        assert N == N2
        reduced_scores = semiring.sum(log_potentials)
        term = np.diagonal(reduced_scores, 0, 0, 1)
        ns = np.arange(N)
        
        chart = np.full((2, N, N), semiring.zero, log_potentials.dtype)
        chart = jax.ops.index_update(chart, jax.ops.index[A, ns, 0], term)
        chart = jax.ops.index_update(chart, jax.ops.index[B, ns, N - 1], term)

        # Run
        for w in range(1, N):
            left = slice(None, N - w)
            right = slice(w, None)
            Y = chart[A, left, :w]
            Z = chart[B, right, N - w :]
            score = np.diagonal(reduced_scores, w, 0, 1)
            new = semiring.times(semiring.dot(Y, Z), score)
            chart = jax.ops.index_update(chart, jax.ops.index[A, left, w], new)
            chart = jax.ops.index_update(chart, jax.ops.index[B, right, N - w - 1], new)
            
        # chart = jax.lax.fori_loop(1, N, loop, chart)        
        return chart[A, 0, length-1]

