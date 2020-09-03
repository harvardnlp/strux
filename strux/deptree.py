
import jax.numpy as np
from .helpers import _Struct
import jax

def convert(logits):
    "move root arcs from diagonal"
    N = logits.shape[0]
    new_logits = np.full((N+1, N+1), -1e9, dtype=logits.dtype)
    new_logits = new_logits.at[1:, 1:].set(logits)

    Ns = np.arange(N)
    new_logits = new_logits.at[0, 1:].set(logits[Ns, Ns])
    return new_logits.at[Ns+1, Ns+1].set(-1e9)


def _unconvert(logits):
    "Move root arcs to diagonal"
    new_logits[:, :] = logits[ 1:, 1:]
    Ns = np.arange(new_logits.shape[0])
    return new_logits.at[Ns, Ns].set( logits[0, 1:])


# Constants
A, B, R, C, L, I = 0, 1, 1, 1, 0, 0


class DepTree(_Struct):
    """
    A projective dependency CRF.

    Parameters:
        log_potentials: Arc scores of shape (B, N, N) or (B, N, N, L) with root scores on
        diagonal.

    """
    length_axes = (0, 1)
    log_scale = False    
    def _dp(self, log_potentials, length):
        semiring = self.semiring
        log_potentials = convert(semiring.sum(log_potentials))
        N, N2 = log_potentials.shape
        assert N == N2

        chart = np.full((2, 2, 2, N, N), semiring.zero, log_potentials.dtype)

        for dir in [L, R]: 
            chart = chart.at[A, C, dir, :, 0].set(semiring.one)
        for dir in [L, R]: 
            chart = chart.at[B, C, dir, :, -1].set(semiring.one)
                
        start_idx = 0
        for k in range(1, N):
            f = np.arange(start_idx, N - k), np.arange(k+start_idx, N)
            ACL = chart[A, C, L, start_idx: N - k, :k]
            ACR = chart[A, C, R, start_idx: N - k, :k]
            BCL = chart[B, C, L, k+start_idx:, N - k :]
            BCR = chart[B, C, R, k+start_idx:, N - k :]
            x = semiring.dot(ACR, BCL)
            arcs_l = semiring.times(x, log_potentials[f[1], f[0]])
            chart = chart.at[A, I, L, start_idx:N - k, k].set(arcs_l)
            chart = chart.at[B, I, L, k+start_idx:N, N - k - 1].set(arcs_l)

            arcs_r = semiring.times(x, log_potentials[f[0], f[1]])
            chart = chart.at[A, I, R, start_idx:N - k, k].set(arcs_r)
            chart = chart.at[B, I, R, k+start_idx:N, N - k - 1].set(arcs_r)
            
            AIR = chart[A, I, R, start_idx: N - k, 1 : k + 1]
            BIL = chart[B, I, L, k+start_idx:, N - k - 1 : N - 1]
            new = semiring.dot(ACL, BIL)
            chart = chart.at[A, C, L, start_idx: N - k, k].set(new)
            chart = chart.at[B, C, L, k+start_idx:N, N - k - 1].set(new)

            new = semiring.dot(AIR, BCR)
            chart = chart.at[A, C, R, start_idx: N - k, k].set(new)
            chart = chart.at[B, C, R, k+start_idx:N, N - k - 1].set(new)

        return chart[A, C, R, 0, length]

    @staticmethod
    def _to_parts(sequence, extra, length):
        """
        Convert a sequence representation to arcs

        Parameters:
            sequence : N x 2 long tensor in [0, N] (indexing is +1)
        Returns:
            arcs :  N x N arc indicators
        """
        N, _ = sequence.shape
        labels = np.zeros((N + 1, N + 1, C))
        Ns = np.arange(1, N + 1)
        labels = labels.at[sequence[Ns - 1, 0], Ns, sequence[Ns - 1 , 1]].set(1)
        labels = np.where(np.arange(N+1).reshape(N+1, 1) >= length+1, labels, 0)
        labels = np.where(np.arange(N+1).reshape(1, N+1) >= length+1, labels, 0)
        return _unconvert(labels)

    @staticmethod
    def _from_parts(arcs):
        """
        Convert a arc representation to sequence

        Parameters:
            arcs : N x N x label arc indicators
        Returns:
            sequence : N x 2 long tensor in [0, N] (indexing is +1)
        """
        N, _, C = arcs.shape

        modifier = arcs.max(-1).argmax(-1)
        labels = arcs.max(-2).argmax(-1)
        
        return np.stack([modifier, labels]), np.array([N, C])

