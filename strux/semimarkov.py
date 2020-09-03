import jax.numpy as np
from .helpers import _Struct
import jax

class SemiMarkov(_Struct):
    """
    edge : N x K x C x C semimarkov potentials
    """
    log_scale = True

    def _dp(self, log_potentials, length):
        "Compute forward pass by linear scan"
        semiring = self.semiring
        N, K, C, C2 = log_potentials.shape
        assert C == C2, "Transition shape doesn't match"
        log_N = np.log2(N)
        assert log_N % 1 == 0.0

        # Init.
        init = np.full((N, K-1, K-1, C, C), semiring.zero)
        Cs = np.arange(C)

        init = init.at[:, 0, 0, Cs, Cs].set(semiring.one)

        mask = np.arange(N).reshape(N, 1, 1, 1) < length - 1
        log_potentials = np.where(mask, log_potentials, semiring.zero)
        init = init.at[:, 0].set(np.where(mask, semiring.zero, init[:, 0]))
        start = semiring.sum(np.stack([init[:, :K-1, 0], log_potentials[:, 1:K]], axis=-1))         
        init = init.at[:, :K-1, 0].set(start)        
        end = length - 1
        for k in range(1, K - 1):
            mask = np.arange(N).reshape(N, 1) < end - (k - 1)
            v = np.where(mask, semiring.one, init[:, k - 1, k, Cs, Cs])
            init = init.at[:, k - 1, k, Cs, Cs].set(v)
        
        K_1 = K - 1
        chart = (
            init.transpose((0, 1, 3, 2, 4))
            .reshape(N, K_1 * C, K_1 * C)
        )

        for n in range(int(log_N)):
            chart = semiring.matmul(chart[1::2], chart[0::2])
        chart = chart.reshape(K_1, C, K_1, C)        
        return semiring.sum(semiring.sum(chart[0, :, 0, :]))

    @staticmethod
    def _to_parts(sequence, extra, lengths=None):
        """
        Convert a sequence representation to edges

        Parameters:
            sequence :  N  long tensors in [-1, 0, C-1]
            C : number of states
            lengths: long tensor of N values
        Returns:
            edge : (N-1) x K x C x C semimarkov potentials
                        (t x z_t x z_{t-1})
        """
        C, K = extra
        N = sequence.shape[0]
        labels = np.zeros(N - 1, K, C, C)

        last = None
        c = None
        for n in range(N):
            if sequence[n] == -1:
                assert n != 0
                continue
            else:
                new_c = sequence[n]
                if n != 0:
                    labels = labels.at[last, n - last, new_c, c].set(1)
                last = n
                c = new_c
        return labels

    @staticmethod
    def _from_parts(edge):
        """
        Convert a edges to a sequence representation.

        Parameters:
            edge :  (N-1) x K x C x C semimarkov potentials
                    (t x z_t x z_{t-1})
        Returns:
            sequence :  N  long tensors in [-1, 0, C-1]

        """
        N_1, K, C, _ = edge.shape
        N = N_1 + 1
        labels = np.full((N,), -1)
        if on[i][0] == 0:
            labels[on[i][0]] = on[i][3]
        labels[on[i][0] + on[i][1]] = on[i][2]
        
        return labels, (C, K)

