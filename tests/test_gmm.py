import jax
import numpy as np
import pytest
from jax import numpy as jnp
from numpy.testing import assert_allclose

from gmmx import EMFitter, GaussianMixtureModelJax

TEST_COVARIANCES = {
    "full": np.array([
        [[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]],
        [[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]],
    ]),
    "diag": np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]),
}


@pytest.fixture(params=["full", "diag"])
def gmm_jax(request):
    means = np.array([[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]])

    covariances = TEST_COVARIANCES[request.param]
    weights = np.array([0.2, 0.8])

    return GaussianMixtureModelJax.from_squeezed(
        means=means,
        covariances=covariances,
        weights=weights,
        covariance_type=request.param,
    )


def test_simple(gmm_jax):
    assert gmm_jax.n_features == 3
    assert gmm_jax.n_components == 2

    expected = {"full": 19, "diag": 13}
    assert gmm_jax.n_parameters == expected[gmm_jax.covariances.type.value]


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
    result = gmm_jax.log_prob(x=jnp.asarray(x))[:, :, 0, 0]

    assert_allclose(np.asarray(result), result_ref, rtol=1e-6)

    assert gmm_jax.n_parameters == gmm._n_parameters()


@pytest.mark.parametrize(
    "method", ["aic", "bic", "predict", "predict_proba", "score", "score_samples"]
)
def test_against_sklearn_all(gmm_jax, method):
    gmm = gmm_jax.to_sklearn()
    x = np.array([
        [1, 2, 3],
        [1, 4, 2],
        [1, 0, 6],
        [4, 2, 4],
        [4, 4, 4],
        [4, 0, 2],
    ])
    result_sklearn = getattr(gmm, method)(x)
    result_jax = getattr(gmm_jax, method)(jnp.asarray(x))
    assert_allclose(np.squeeze(result_jax), result_sklearn, rtol=1e-5)


def test_sample(gmm_jax):
    key = jax.random.PRNGKey(0)
    samples = gmm_jax.sample(key, 2)

    assert samples.shape == (2, 3)

    expected = {"full": -0.458194, "diag": -1.525666}
    assert_allclose(samples[0, 0], expected[gmm_jax.covariances.type.value], rtol=1e-6)


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


def test_fit(gmm_jax):
    random_state = np.random.RandomState(827392)
    x, _ = gmm_jax.to_sklearn(random_state=random_state).sample(16_000)

    fitter = EMFitter(tol=1e-6)
    result = fitter.fit(x=x, gmm=gmm_jax)

    # The number of iterations is not deterministic across architectures
    covar_str = gmm_jax.covariances.type.value
    expected = {"full": [4, 7], "diag": [7]}
    assert int(result.n_iter) in expected[covar_str]

    expected = {"full": -4.3686, "diag": -5.422534}
    assert_allclose(result.log_likelihood, expected[covar_str], rtol=2e-4)

    expected = {"full": 9.536743e-07, "diag": 9.536743e-07}
    assert_allclose(result.log_likelihood_diff, expected[covar_str], atol=fitter.tol)

    assert_allclose(result.gmm.weights_numpy, [0.2, 0.8], rtol=0.05)


def test_fit_against_sklearn(gmm_jax):
    # Fitting is hard to test, especillay we cannot guarantee the fit converges to the same solution
    # However the "global" likelihood (summed accross all components) for a given feature vector
    # should be similar for both implementations
    random_state = np.random.RandomState(82792)
    x, _ = gmm_jax.to_sklearn(random_state=random_state).sample(16_000)

    tol = 1e-6
    fitter = EMFitter(tol=tol)
    result_jax = fitter.fit(x=x, gmm=gmm_jax)

    gmm_sklearn = gmm_jax.to_sklearn(tol=tol, random_state=random_state)

    # This brings the sklearn model in the same state as the jax model
    gmm_sklearn.fit(x)

    covar_str = gmm_jax.covariances.type.value

    expected = {"full": 9, "diag": 9}
    assert_allclose(gmm_sklearn.n_iter_, expected[covar_str])
    assert_allclose(gmm_sklearn.weights_, [0.2, 0.8], rtol=0.06)

    expected = {"full": [9], "diag": [8, 11]}
    assert result_jax.n_iter in expected[covar_str]
    assert_allclose(result_jax.gmm.weights_numpy, [0.2, 0.8], rtol=0.06)

    assert_allclose(gmm_sklearn.covariances_, TEST_COVARIANCES[covar_str], rtol=0.1)
    assert_allclose(
        result_jax.gmm.covariances.values_numpy, TEST_COVARIANCES[covar_str], rtol=0.1
    )

    log_likelihood_jax = result_jax.gmm.log_prob(x[:10]).sum(axis=1)[:, 0, 0]
    log_likelihood_sklearn = gmm_sklearn._estimate_weighted_log_prob(x[:10]).sum(axis=1)

    # note this is agreement in log-likehood, not likelihood!
    assert_allclose(log_likelihood_jax, log_likelihood_sklearn, rtol=1e-2)
