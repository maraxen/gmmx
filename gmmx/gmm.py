from dataclasses import dataclass
from enum import Enum

import jax
from jax import numpy as jnp
from jax import scipy as jsp

from gmmx.utils import register_dataclass_jax


class CovarianceType(str, Enum):
    """Convariance type"""

    full = "full"
    tied = "tied"
    diag = "diag"
    spherical = "spherical"


class AxisOrder(int, Enum):
    """Axis order"""

    n_components = 0
    n_features = 1
    n_features_covar = 2


@register_dataclass_jax(data_fields=["values"])
@dataclass
class FullCovariances:
    """Full covariance matrix"""

    values: jax.Array

    @classmethod
    def create(cls, n_components, n_features):
        """Create covariance matrix"""
        values = jnp.zeros((n_components, n_features, n_features))
        return cls(values=values)

    def log_prob(self, x):
        pass

    @property
    def n_components(self):
        """Number of components"""
        return self.values.shape[AxisOrder.n_components]

    @property
    def n_features(self):
        """Number of features"""
        return self.values.shape[AxisOrder.n_features]

    @property
    def log_det_cholesky(self):
        """Precision matrices pytorch"""
        reshaped = self.precisions_cholesky.reshape(self.n_components, -1)
        reshaped = reshaped[:, :: self.n_features + 1]
        return jnp.sum(jnp.log(reshaped), axis=1)

    @property
    def precisions_cholesky(self):
        """Compute precision matrices"""
        cov_chol = jsp.linalg.cholesky(self.values, lower=True)
        b = jnp.repeat(jnp.eye(self.n_features)[None], self.n_components, axis=0)
        precisions_chol = jsp.linalg.solve_triangular(cov_chol, b, lower=True)
        return precisions_chol.mT

    def dense(self):
        """Dense representation"""
        return self.values


COVARIANCE = {
    CovarianceType.full: FullCovariances,
}


@register_dataclass_jax(data_fields=["weights", "means", "covariances"])
@dataclass
class GaussianMixtureModelJax:
    """Gaussian Mixture Model

    Attributes
    ----------
    weights : jax.array
        Weights of each component.
    means : jax.array
        Mean of each component.
    covariances : jax.array
        Covariance of each component.
    """

    weights: jax.Array
    means: jax.Array
    covariances: FullCovariances

    @classmethod
    def create(cls, n_components, n_features, covariance_type="full"):
        """Create a GMM from configuration

        Parameters
        ----------
        n_components : int
            Number of components
        n_features : int
            Number of features
        covariance_type : str, optional
            Covariance type, by default "full"

        Returns
        -------
        gmm : GaussianMixtureModelJax
            Gaussian mixture model instance.
        """
        covariance_type = CovarianceType(covariance_type)

        weights = jnp.ones(n_components) / n_components
        means = jnp.zeros((n_components, n_features))
        covariances = COVARIANCE[covariance_type].create(n_components, n_features)
        return cls(weights=weights, means=means, covariances=covariances)

    @classmethod
    def from_numpy(cls, means, covariances, weights, covariance_type="full"):
        """Create a Jax GMM from numpy arrays

        Parameters
        ----------
        means : np.array
            Mean of each component.
        covariances : np.array
            Covariance of each component.
        weights : np.array
            Weights of each component.
        covariance_type : str, optional
            Covariance type, by default "full"

        Returns
        -------
        gmm : GaussianMixtureModelJax
            Gaussian mixture model instance.
        """
        covariance_type = CovarianceType(covariance_type)

        means = jnp.asarray(means)
        covariances = COVARIANCE[covariance_type](values=jnp.asarray(covariances))
        weights = jnp.asarray(weights)
        return cls(weights=weights, means=means, covariances=covariances)

    @classmethod
    def from_k_means(cls, x, n_components):
        """Init from k-means clustering

        Parameters
        ----------
        x : jax.array
            Feature vectors
        n_components : int
            Number of components

        Returns
        -------
        gmm : GaussianMixtureModelJax
            Gaussian mixture model instance.
        """
        raise NotImplementedError

    @property
    def n_features(self):
        """Number of features"""
        return self.covariances.n_features

    @property
    def n_components(self):
        """Number of components"""
        return self.covariances.n_components

    @property
    def log_weights(self):
        """Log weights (~jax.ndarray)"""
        return jnp.log(self.weights)

    @jax.jit
    def estimate_log_prob(self, x):
        """Compute log likelihood for given feature vector"""
        n_samples, n_features = x.shape

        log_prob = []

        # means_prec = jnp.matmul(self.means, self.covariances.precisions_cholesky)

        for mu, prec_chol in zip(self.means, self.covariances.precisions_cholesky):
            y = jnp.dot(x, prec_chol) - jnp.dot(mu, prec_chol)
            log_prob.append(jnp.sum(jnp.square(y), axis=1))

        log_prob = jnp.stack(log_prob, axis=1)
        # Since we are using the precision of the Cholesky decomposition,
        # `- 0.5 * log_det_precision` becomes `+ log_det_precision_chol`
        two_pi = jnp.array(2 * jnp.pi)
        return -0.5 * (n_features * jnp.log(two_pi) + log_prob) + self.covariances.log_det_cholesky + self.log_weights
