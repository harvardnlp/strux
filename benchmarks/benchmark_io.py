import jax.numpy as np
import jax
import time


class Semiring:
    @classmethod
    def matmul(cls, a, b):
        "Generalized tensordot. Classes should override."
        a = a[..., np.newaxis]
        b = b[..., np.newaxis, :, :]
        c = cls.times(a, b)
        return cls.sum(np.swapaxes(c, -2, -1))

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

    @staticmethod
    def mul(a, b):
        return a + b

    @staticmethod
    def prod(a, dim=-1):
        return np.sum(a, axis=dim)


class LogSemiring(Semiring):
    zero = -1e9
    one = 0.0

    @staticmethod
    def sum(xs, dim=-1):
        return jax.scipy.special.logsumexp(xs, axis=dim)

    
class MaxSemiring(Semiring):
    zero = -1e9
    one = 0.0

    @staticmethod
    def sum(xs, dim=-1):
        return jax.max(xs, axis=dim)


    
    

A, B = 0, 1
def run(log_potentials, length, semiring="Log"):
    if semiring == "Log":
        semiring = LogSemiring
    else:
        semiring = MaxSemiring
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
    return chart[A, 0, length-1]

def main():
    BATCH = 4
    C = 1
    
    run1 = jax.jit(jax.vmap(run, (0, 0, None)), static_argnums=2)
    run2 = jax.jit(jax.grad(lambda *args: run1(*args).sum(0), 0), static_argnums=2)
    run3 = jax.jit(jax.grad(lambda *args: run2(*args).sum(0), 0), static_argnums=2)
    runs = [run1, run2] #  run3 is very slow

    for n in range(5, 25, 5):
        key = jax.random.PRNGKey(0)
        lengths = np.full((BATCH,), n)
        log_potentials = jax.random.normal(key, (BATCH, n, n, C))
        
        for i, fun in enumerate(runs):
        # Compile
            start = time.time()
            fun(log_potentials, lengths, "Log")
            print(i, n, "compile time", time.time() - start)

            start = time.time()
            for _ in range(50):
                x = fun(log_potentials, lengths, "Log").sum()
            print(i, n, x, (time.time() - start)  / 50.0)

if __name__ == "__main__":
    main()
    
