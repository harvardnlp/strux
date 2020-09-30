import jax.numpy as np
from .helpers import _Struct
import jax

class LinearChain(_Struct):    
    log_scale = True
    def _dp(self, log_potentials, length):
        semiring = self.semiring
        N, C, C2 = log_potentials.shape
        assert C == C2, "Transition shape doesn't match"
        log_N = np.log2(N)
        #assert log_N % 1 == 0.0

        extra = np.where(np.eye(C, C), semiring.one, semiring.zero)
        chart = np.where(np.arange(N).reshape(N, 1, 1) < length - 1, log_potentials, extra)    
        for _ in range(int(log_N)):
            chart = semiring.matmul(chart[1::2], chart[0::2])
        return semiring.sum(semiring.sum(chart[0]))

    @staticmethod
    def _to_parts(sequence, C, length):
        """
        Convert a sequence representation to edges

        Parameters:
            sequence : N long tensor in [0, C)
            C : number of states
            length: int of [0, N) values
        Returns:
            edge : (N-1) x C x C markov indicators
                      (t x z_t x z_{t-1})
        """
        N, = sequence.shape
        labels = np.zeros((N - 1, C, C))
        labels = jax.ops.index_update(labels, jax.ops.index[np.arange(N-1), sequence[1:N], sequence[:N-1]], 1)
        return np.where(np.arange(N-1).reshape(N-1, 1, 1) < length - 1, labels, 0)

    @staticmethod
    def _from_parts(edge):
        """
        Convert edges to sequence representation.

        Parameters:
            edge : (N-1) x C x C markov indicators
                        (t x z_t x z_{t-1})
        Returns:
            sequence :  N long tensor in [0, C-1]
        """
        N_1, C, _ = edge.shape
        N = N_1 + 1
        labels = edge.max(-1).argmax(-1)
        start = edge[:1].max(-2).argmax(-1)
        labels = np.concatenate([start, labels])
        return labels, C
