import numpy as np
from jax import numpy as jnp
from numpy.testing import assert_allclose
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import (
    _compute_precision_cholesky,
)

from gmmx.gmm import GaussianMixtureModelJax


def test_against_sklearn():
    means = np.array([[-1, 0, 1], [-1, 0, 1]])

    covar_1 = np.array([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]])
    covar_2 = np.array([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]])
    covariances = np.array([covar_1, covar_2])
    weights = np.array([0.2, 0.8])

    gmm_jax = GaussianMixtureModelJax.from_numpy(
        means=means,
        covariances=covariances,
        weights=weights,
    )

    gmm = GaussianMixture()
    gmm.weights_ = weights
    gmm.covariances_ = covariances
    gmm.means_ = means
    gmm.precisions_cholesky_ = _compute_precision_cholesky(covariances, "full")

    x = np.array([[1, 2, 3], [1, 4, 2], [1, 0, 6], [4, 2, 4], [4, 4, 4], [4, 0, 2]])

    result_ref = gmm._estimate_weighted_log_prob(X=x)
    result = gmm_jax.estimate_log_prob(x=jnp.asarray(x))

    assert_allclose(np.asarray(result), result_ref, rtol=1e-6)
