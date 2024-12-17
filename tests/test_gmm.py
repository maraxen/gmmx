import jax
import numpy as np
import pytest
from jax import numpy as jnp
from numpy.testing import assert_allclose

from gmmx import EMFitter, GaussianMixtureModelJax


@pytest.fixture
def gmm_jax():
    means = np.array([[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]])

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
    means = np.array([[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]])

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


def test_create():
    gmm = GaussianMixtureModelJax.create(n_components=3, n_features=2)
    assert gmm.n_features == 2
    assert gmm.n_components == 3
    assert gmm.n_parameters == 17


def test_init_incorrect():
    with pytest.raises(ValueError):
        GaussianMixtureModelJax(
            means=jnp.zeros((2, 3)),
            covariances=jnp.zeros((2, 3, 3)),
            weights=jnp.zeros((2,)),
        )

    with pytest.raises(ValueError):
        GaussianMixtureModelJax(
            means=jnp.zeros((1, 2, 3, 1)),
            covariances=jnp.zeros((1, 2, 3, 3)),
            weights=jnp.zeros((1, 1, 4, 1)),
        )


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
    assert_allclose(samples[0, 0], -0.458194, rtol=1e-6)


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
    assert_allclose(result[0], 0, rtol=1e-6)


def test_fit(gmm_jax, gmm_jax_init):
    random_state = np.random.RandomState(827392)
    x, _ = gmm_jax.to_sklearn(random_state=random_state).sample(16_000)

    fitter = EMFitter(tol=1e-4)
    result = fitter.fit(x=x, gmm=gmm_jax_init)

    # The number of iterations is not deterministic across architectures
    assert int(result.n_iter) in [5, 6]
    assert_allclose(result.log_likelihood, -4.3686, rtol=2e-4)
    assert_allclose(result.log_likelihood_diff, 9.536743e-07, atol=fitter.tol)
    assert_allclose(result.gmm.weights_numpy, [0.2, 0.8], rtol=0.03)


def test_fit_against_sklearn(gmm_jax, gmm_jax_init):
    # Fitting is hard to test, especillay we cannot guarantee the fit converges to the same solution
    # However the "global" likelihood (summed accross all components) for a given feature vector
    # should be similar for both implementations
    random_state = np.random.RandomState(82792)
    x, _ = gmm_jax.to_sklearn(random_state=random_state).sample(16_000)

    tol = 1e-12
    fitter = EMFitter(tol=tol)
    result_jax = fitter.fit(x=x, gmm=gmm_jax_init)

    gmm_sklearn = gmm_jax_init.to_sklearn(
        warm_start=True, tol=tol, random_state=random_state
    )

    # This brings the sklearn model in the same state as the jax model
    gmm_sklearn.warm_start = True
    gmm_sklearn.converged_ = True
    gmm_sklearn.lower_bound_ = gmm_sklearn._estimate_log_prob(x).sum()
    gmm_sklearn.fit(x)

    assert_allclose(gmm_sklearn.weights_, [0.2, 0.8], rtol=0.03)
    assert_allclose(result_jax.gmm.weights_numpy, [0.2, 0.8], rtol=0.03)

    covar = np.array([
        [1.0, 0.5, 0.5],
        [0.5, 1.0, 0.5],
        [0.5, 0.5, 1.0],
    ])
    desired = np.array([covar, covar])

    assert_allclose(gmm_sklearn.covariances_, desired, rtol=0.1)
    assert_allclose(result_jax.gmm.covariances.values_numpy, desired, rtol=0.1)

    log_likelihood_jax = result_jax.gmm.estimate_log_prob(x[:10]).sum(axis=1)[:, 0, 0]
    log_likelihood_sklearn = gmm_sklearn._estimate_weighted_log_prob(x[:10]).sum(axis=1)

    # note this is agreement in log-likehood, not likelihood!
    assert_allclose(log_likelihood_jax, log_likelihood_sklearn, rtol=1e-2)
