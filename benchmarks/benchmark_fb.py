import jax.numpy as np
import jax
import time


class Semiring:
    @classmethod
    def matmul(cls, a, b):
        a = a[..., np.newaxis]
        b = b[..., np.newaxis, :, :]
        c = cls.times(a, b)
        return cls.sum(np.swapaxes(c, -2, -1))

    @classmethod
    def dot(cls, a, b):
        a = a[..., np.newaxis, :]
        b = b[..., np.newaxis]
        return cls.matmul(a, b).squeeze(-1).squeeze(-1)

    @classmethod
    def times(cls, *ls):
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

def run(log_potentials, length, semiring="Log"):
    "Main code, associative forward-backward"
    if semiring == "Log":
        semiring = LogSemiring
    else:
        semiring = MaxSemiring

    N, C, C2 = log_potentials.shape
    assert C == C2, "Transition shape doesn't match"
    log_N = np.log2(N)
    assert log_N % 1 == 0.0

    extra = np.where(np.eye(C, C), semiring.one, semiring.zero)
    chart = np.where(np.arange(N).reshape(N, 1, 1) < length - 1, log_potentials, extra)    
    for _ in range(int(log_N)):
        chart = semiring.matmul(chart[1::2], chart[0::2])
    return semiring.sum(semiring.sum(chart[0]))


def main():
    BATCH = 32
    N = 128
    
    run1 = jax.jit(jax.vmap(run, (0, 0, None)), static_argnums=2)
    run2 = jax.jit(jax.grad(lambda *args: run1(*args).sum(0), 0), static_argnums=2)
    run3 = jax.jit(jax.grad(lambda *args: run2(*args).sum(0), 0), static_argnums=2)
    runs = [run1, run2] #  run3 is very slow
    
    for c in range(5, 250, 25):
        key = jax.random.PRNGKey(0)
        lengths = np.full((BATCH,), N+1)
        log_potentials = jax.random.normal(key, (BATCH, N, c, c))

        for i, fun in enumerate(runs):
            # Compile
            start = time.time()
            fun(log_potentials, lengths, "Log")
            print(i, c, "compile time", time.time() - start)

            start = time.time()
            for _ in range(50):
                x = fun(log_potentials, lengths, "Log").sum()
            print(i, c, x, (time.time() - start) / 50.0)
            
if __name__ == "__main__":
    main()
    
