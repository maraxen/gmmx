import numpy as np
import pytest
from jax import numpy as jnp
from numpy.testing import assert_allclose

from gmmx.gmm import GaussianMixtureModelJax


@pytest.fixture
def gmm_jax():
    means = np.array([[-1, 0, 1], [-1, 0, 1]])

    covar_1 = np.array([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]])
    covar_2 = np.array([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]])
    covariances = np.array([covar_1, covar_2])
    weights = np.array([0.2, 0.8])

    gmm_jax = GaussianMixtureModelJax.from_squeezed(
        means=means,
        covariances=covariances,
        weights=weights,
    )
    return gmm_jax


def test_simple(gmm_jax):
    assert gmm_jax.n_features == 3
    assert gmm_jax.n_components == 2
    assert gmm_jax.n_parameters == 19


def test_against_sklearn(gmm_jax):
    x = np.array([
        [1, 2, 3],
        [1, 4, 2],
        [1, 0, 6],
        [4, 2, 4],
        [4, 4, 4],
        [4, 0, 2],
    ])

    gmm = gmm_jax.to_sklearn()
    result_ref = gmm._estimate_weighted_log_prob(X=x)
    result = gmm_jax.estimate_log_prob(x=jnp.asarray(x))

    assert_allclose(np.asarray(result), result_ref, rtol=1e-6)

    assert gmm_jax.n_parameters == gmm._n_parameters()
