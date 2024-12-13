import jax
import numpy as np
import pytest
from jax import numpy as jnp
from numpy.testing import assert_allclose

from gmmx import EMFitter, GaussianMixtureModelJax


@pytest.fixture
def gmm_jax():
    means = np.array([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])

    covar_1 = np.array([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]])
    covar_2 = np.array([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]])
    covariances = np.array([covar_1, covar_2])
    weights = np.array([0.2, 0.8])

    return GaussianMixtureModelJax.from_squeezed(
        means=means,
        covariances=covariances,
        weights=weights,
    )


@pytest.fixture
def gmm_jax_init():
    means = np.array([[-0.8, 0.0, 1.2], [-2.0, 0.0, 1.7]])

    covar_1 = np.array([[1.1, 0.5, 0.3], [0.3, 1, 0.5], [0.5, 0.5, 1]])
    covar_2 = np.array([[1, 0.5, 0.2], [0.5, 1.1, 0.5], [0.5, 0.5, 1]])
    covariances = np.array([covar_1, covar_2])
    weights = np.array([0.3, 0.7])

    return GaussianMixtureModelJax.from_squeezed(
        means=means,
        covariances=covariances,
        weights=weights,
    )


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
    result = gmm_jax.estimate_log_prob(x=jnp.asarray(x))[:, :, 0, 0]

    assert_allclose(np.asarray(result), result_ref, rtol=1e-6)

    assert gmm_jax.n_parameters == gmm._n_parameters()


def test_sample(gmm_jax):
    key = jax.random.PRNGKey(0)
    samples = gmm_jax.sample(key, 2)

    assert samples.shape == (2, 3)
    assert_allclose(samples[0, 0], -2.458194, rtol=1e-6)


def test_predict(gmm_jax):
    x = np.array([
        [1, 2, 3],
        [1, 4, 2],
        [1, 0, 6],
        [4, 2, 4],
        [4, 4, 4],
        [4, 0, 2],
    ])

    result = gmm_jax.predict(x=jnp.asarray(x))

    assert result.shape == (6, 1)
    assert_allclose(result[0], 1, rtol=1e-6)


def test_fit(gmm_jax, gmm_jax_init):
    x = gmm_jax.sample(jax.random.PRNGKey(0), 10_000)

    fitter = EMFitter(tol=1e-4)
    result = fitter.fit(x, gmm_jax_init)

    assert result.n_iter == 6
    assert_allclose(result.log_likelihood, -3.902018, rtol=1e-6)
    assert_allclose(result.log_likelihood_diff, 3.957748e-05, atol=fitter.tol)
    assert_allclose(result.gmm.weights_numpy, [0.586136, 0.413864], rtol=1e-6)
