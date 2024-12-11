from dataclasses import dataclass
from enum import Enum

import jax
import numpy as np
from jax import numpy as jnp
from jax import scipy as jsp

from gmmx.utils import register_dataclass_jax


class CovarianceType(str, Enum):
    """Convariance type"""

    full = "full"
    tied = "tied"
    diag = "diag"
    spherical = "spherical"


class Axis(int, Enum):
    """Axis order"""

    batch = 0
    components = 1
    features = 2
    features_covar = 3


def expand_dims_if_needed(x, axis):
    """Expand dimensions if needed"""
    if x.ndim == len(Axis):
        return x

    return jnp.expand_dims(x, axis=axis)


@register_dataclass_jax(data_fields=["values"])
@dataclass
class FullCovariances:
    """Full covariance matrix

    Attributes
    ----------
    values : jax.array
        Covariance values.
    """

    values: jax.Array

    def __post_init__(self):
        self.values = expand_dims_if_needed(self.values, axis=Axis.batch)

    @property
    def values_numpy(self):
        """Covariance as numpy array"""
        return np.squeeze(np.asarray(self.values), axis=Axis.batch)

    @property
    def precisions_cholesky_numpy(self):
        """Compute precision matrices"""
        return np.squeeze(np.asarray(self.precisions_cholesky), axis=Axis.batch)

    @classmethod
    def create(cls, n_components, n_features):
        """Create covariance matrix"""
        values = jnp.zeros((n_components, n_features, n_features))
        return cls(values=values)

    def log_prob(self, x, means):
        """Compute log likelihood from the covariance for a given feature vector"""
        precisions_cholesky = self.precisions_cholesky

        y = jnp.matmul(x, precisions_cholesky) - jnp.matmul(means.mT, precisions_cholesky)
        return jnp.sum(jnp.square(y), axis=(Axis.features, Axis.features_covar), keepdims=True)

    @property
    def n_components(self):
        """Number of components"""
        return self.values.shape[Axis.components]

    @property
    def n_features(self):
        """Number of features"""
        return self.values.shape[Axis.features]

    @property
    def n_parameters(self):
        """Number of parameters"""
        return self.n_components * self.n_features * (self.n_features + 1) / 2.0

    @property
    def log_det_cholesky(self):
        """Precision matrices pytorch"""
        diag = jnp.diagonal(self.precisions_cholesky, axis1=Axis.features, axis2=Axis.features_covar)
        return jnp.expand_dims(
            jnp.sum(jnp.log(diag), axis=Axis.features, keepdims=True),
            axis=Axis.features_covar,
        )

    @property
    def precisions_cholesky(self):
        """Compute precision matrices"""
        cov_chol = jsp.linalg.cholesky(self.values, lower=True)

        identity = jnp.expand_dims(jnp.eye(self.n_features), axis=(Axis.batch, Axis.components))

        b = jnp.repeat(identity, self.n_components, axis=Axis.components)
        precisions_chol = jsp.linalg.solve_triangular(cov_chol, b, lower=True)
        return precisions_chol.mT


COVARIANCE = {
    CovarianceType.full: FullCovariances,
}

SKLEARN_COVARIANCE_TYPE = {FullCovariances: "full"}


@register_dataclass_jax(data_fields=["weights", "means", "covariances"])
@dataclass
class GaussianMixtureModelJax:
    """Gaussian Mixture Model

    Attributes
    ----------
    weights : jax.array
        Weights of each component. Expected shape is (1, n_components, 1, 1)
    means : jax.array
        Mean of each component. Expected shape is (1, n_components, n_features, 1)
    covariances : jax.array
        Covariance of each component. Expected shape is (1, n_components, n_features, n_features)
    """

    weights: jax.Array
    means: jax.Array
    covariances: FullCovariances

    def __post_init__(self):
        self.weights = expand_dims_if_needed(self.weights, axis=(Axis.batch, Axis.features, Axis.features_covar))
        self.means = expand_dims_if_needed(self.means, axis=(Axis.batch, Axis.features_covar))

    @property
    def weights_numpy(self):
        """Weights as numpy array"""
        return np.squeeze(
            np.asarray(self.weights),
            axis=(Axis.batch, Axis.features, Axis.features_covar),
        )

    @property
    def means_numpy(self):
        """Means as numpy array"""
        return np.squeeze(np.asarray(self.means), axis=(Axis.batch, Axis.features_covar))

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
    def n_parameters(self):
        """Number of parameters"""
        return int(self.n_components + self.n_components * self.n_features + self.covariances.n_parameters - 1)

    @property
    def log_weights(self):
        """Log weights (~jax.ndarray)"""
        return jnp.log(self.weights)

    def estimate_log_prob(self, x):
        """Compute log likelihood for given feature vector

        Parameters
        ----------
        x : jax.array
            Feature vectors

        Returns
        -------
        log_prob : jax.array
            Log likelihood
        """
        x = jnp.expand_dims(x, axis=(Axis.components, Axis.features))

        log_prob = self.covariances.log_prob(x, self.means)
        two_pi = jnp.array(2 * jnp.pi)

        value = (
            -0.5 * (self.n_features * jnp.log(two_pi) + log_prob) + self.covariances.log_det_cholesky + self.log_weights
        )
        return jnp.squeeze(value, axis=(Axis.features, Axis.features_covar))

    def to_sklearn(self):
        """Convert to sklearn GaussianMixture

        Returns
        -------
        gmm : GaussianMixture
            Gaussian mixture model instance.
        """
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=SKLEARN_COVARIANCE_TYPE[type(self.covariances)],
        )
        gmm.weights_ = self.weights_numpy
        gmm.covariances_ = self.covariances.values_numpy
        gmm.means_ = self.means_numpy
        gmm.precisions_cholesky_ = self.covariances.precisions_cholesky_numpy
        return gmm

    @jax.jit
    def predict(self, x):
        """Predict the component index for each sample

        Parameters
        ----------
        x : jax.array
            Feature vectors

        Returns
        -------
        predictions : jax.array
            Predicted component index
        """
        return jnp.argmax(self.estimate_log_prob(x), axis=Axis.components)
