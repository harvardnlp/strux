import jax.numpy as np
import jax

def matmul(cls, a, b):
    dims = 1
    act_on = -(dims + 1)
    a = a[..., np.newaxis]
    b = b[..., np.newaxis, :, :]
    c = cls.times(a, b)
    for d in range(act_on, -1, 1):
        c = cls.sum(np.swapaxes(c, -2, -1))
    return c


class Semiring:
    """
    Base semiring class.

    Based on description in:

    * Semiring parsing :cite:`goodman1999semiring`

    """

    @classmethod
    def matmul(cls, a, b):
        "Generalized tensordot. Classes should override."
        return matmul(cls, a, b)

    @classmethod
    def size(cls):
        "Additional *ssize* first dimension needed."
        return 1

    @classmethod
    def dot(cls, a, b):
        "Dot product along last dim."
        a = a[..., np.newaxis, :]
        b = b[..., np.newaxis]
        return cls.matmul(a, b).squeeze(-1).squeeze(-1)

    @classmethod
    def times(cls, *ls):
        "Multiply a list of tensors together"
        cur = ls[0]
        for l in ls[1:]:
            cur = cls.mul(cur, l)
        return cur

    @classmethod
    def convert(cls, potentials):
        "Convert to semiring by adding an extra first dimension."
        return potentials

    @classmethod
    def unconvert(cls, potentials):
        "Unconvert from semiring by removing extra first dimension."
        return potentials

    @staticmethod
    def zero_(xs):
        "Fill *ssize x ...* tensor with additive identity."
        raise NotImplementedError()

    @classmethod
    def zero_mask_(cls, xs, mask):
        "Fill *ssize x ...* tensor with additive identity."
        iu = np.mask_indices(mask[np.newaxis])
        jax.ops.index_update(xs, iu, cls.zero)

    @staticmethod
    def one_(xs):
        "Fill *ssize x ...* tensor with multiplicative identity."
        raise NotImplementedError()

    @staticmethod
    def sum(xs, dim=-1):
        "Sum over *dim* of tensor."
        raise NotImplementedError()

    @classmethod
    def plus(cls, a, b):
        return cls.sum(torch.stack([a, b], dim=-1))


class _Base(Semiring):
    zero = 0

    @staticmethod
    def mul(a, b):
        return torch.mul(a, b)

    @staticmethod
    def prod(a, dim=-1):
        return torch.prod(a, dim=dim)

    @staticmethod
    def zero_(xs):
        return xs.fill_(0)

    @staticmethod
    def one_(xs):
        return xs.fill_(1)


class _BaseLog(Semiring):
    zero = -1e9
    one = 0.0

    @staticmethod
    def sum(xs, dim=-1):
        return jax.scipy.special.logsumexp(xs, axis=dim)

    @staticmethod
    def mul(a, b):
        return a + b

    @staticmethod
    def zero_(xs):
        return np.full_like(xs, -1e5)

    @staticmethod
    def one_(xs):
        return np.full_like(xs, 0.0)

    @staticmethod
    def prod(a, dim=-1):
        return np.sum(a, axis=dim)

    # @classmethod
    # def matmul(cls, a, b):
    #     return super(cls).matmul(a, b)


class StdSemiring(_Base):
    """
    Implements the counting semiring (+, *, 0, 1).
    """

    @staticmethod
    def sum(xs, dim=-1):
        return torch.sum(xs, dim=dim)

    @classmethod
    def matmul(cls, a, b, dims=1):
        """
        Dot product along last dim.

        (Faster than calling sum and times.)
        """

        # if has_genbmm and isinstance(a, genbmm.BandedMatrix):
        #     return b.multiply(a.transpose())
        # else:
        return torch.matmul(a, b)


class LogSemiring(_BaseLog):
    """
    Implements the log-space semiring (logsumexp, +, -inf, 0).

    Gradients give marginals.
    """

    @classmethod
    def matmul(cls, a, b):
        # if has_genbmm and isinstance(a, genbmm.BandedMatrix):
        #     return b.multiply_log(a.transpose())
        # else:
        return _BaseLog.matmul(a, b)


class MaxSemiring(_BaseLog):
    """
    Implements the max semiring (max, +, -inf, 0).

    Gradients give argmax.
    """

    @classmethod
    def matmul(cls, a, b):
        # if has_genbmm and isinstance(a, genbmm.BandedMatrix):
        #     return b.multiply_max(a.transpose())
        # else:
        return matmul(cls, a, b)

    @staticmethod
    def sum(xs, dim=-1):
        return np.max(xs, axis=dim)

