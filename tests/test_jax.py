from strux import LogSemiring, LinearChain, MaxSemiring, CKY_CRF, DepTree, SemiMarkov
import strux
import torch_struct 
from .extensions import test_lookup
from hypothesis import given, settings
from hypothesis.strategies import integers, data, sampled_from
import jax
import jax.numpy as np
import torch
import numpy

@given(data())
@settings(max_examples=50, deadline=None)
def test_generic_a(data):
    model = data.draw(
        sampled_from(
            [SemiMarkov]
        )  
    )

    semiring, tsemiring = data.draw(sampled_from([(LogSemiring, torch_struct.LogSemiring),
                                                  (MaxSemiring, torch_struct.MaxSemiring)]))
    struct = model(semiring)
    
    vals, (batch, N) = test_lookup[model]._rand()
    vals_jax = struct.resize(np.array(vals.numpy()))
    Ns = np.array([N] * vals_jax.shape[0])
    alpha = struct.sum(vals_jax, Ns)
    count = test_lookup[model](tsemiring).enumerate(vals)[0]
    print('a')
    print(alpha)
    print('b')
    print(count)
    
    # # assert(False)
    assert alpha.shape[0] == batch
    assert count.shape[0] == batch
    assert alpha.shape == count.shape
    alpha = torch.tensor(numpy.array(alpha))
    assert torch.isclose(count[0], alpha[0])

@given(data())
@settings(max_examples=50, deadline=None)
def test_max(data):
    model = data.draw(
        sampled_from(
            [CKY_CRF, DepTree]
        )  
    )
    struct = model(MaxSemiring)
    vals, (batch, N) = test_lookup[model]._rand()
    vals_jax = struct.resize(np.array(vals.numpy()))
    Ns = np.array([N] * vals_jax.shape[0])
    score = struct.sum(vals_jax, Ns)
    marginals = struct.marginals(vals_jax, Ns)
    print(marginals)
    assert np.isclose(score, struct.score(vals_jax, marginals)).all()


@given(data(), integers(min_value=1, max_value=20))
@settings(max_examples=50, deadline=None)
def test_parts_from_marginals(data, seed):
    # todo: add CKY, DepTree too?
    model = data.draw(sampled_from([CKY_CRF, DepTree]))
    struct = model()
    torch.manual_seed(seed)
    vals, (batch, N) = test_lookup[model]._rand()
    vals_jax = struct.resize(np.array(vals.numpy()))
    Ns = np.array([N] * vals_jax.shape[0])

    edge = model(MaxSemiring).marginals(vals_jax, Ns)
    sequence, extra = struct.from_parts(edge)
    edge_ = struct.to_parts(sequence, extra, Ns)
    print(edge)
    print(sequence)
    print(edge_)
    assert (np.isclose(edge, edge_)).all(), edge - edge_

    sequence_, extra_ = struct.from_parts(edge_)
    assert (extra == extra_).all(), (extra, extra_)
    assert (np.isclose(sequence, sequence_)).all(), sequence - sequence_
