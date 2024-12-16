"""Some notes on the implementation:

I have not tried to keep the implementation close to the sklearn implementation.
I have rather tried to realize my own best practices for code structure and
clarity. Here are some more detailed thoughts:

1. **Use dataclasses for the model representation**: this reduces the amount of
boilerplate code for initialization and in combination with the `register_dataclass_jax`
decorator it integrates seamleassly with JAX.

2. **Split up the different covariance types into different classes**: this avoids
the need for multiple blocks of if-else statements.

3. **Use a registry for the covariance types**:  This allows for easy extensibility
by the user.

3. **Remove Python loops**: I have not checked the reason why the sklearn implementation
still uses Python loops, but my guess is that it is simpler(?) and when there are
operations such as matmul and cholesky decomposition, the Python loop does not become
the bottleneck. In JAX, however, it is usually better to avoid Python loops and let
the JAX compiler take care of the optimization instead.

4. **Rely on same internal array dimension and axis order**:
Internally all(!) involved arrays (even 1d weights) are represented as 4d arrays
with the axes (batch, components, features, features_covar). This makes it much
easier to write array operations and rely on broadcasting. This minimizes the
amount of in-line reshaping and in-line extension of dimensions. If you think
about it, this is most likely the way how array programming was meant to be used
in first place. Yet, I have rarely seen this in practice, probably because people
struggle with the additional dimensions in the beginning. However once you get
used to it, it is much easier to write and understand the code! The only downside
is that the user has to face the additional "empty" dimensions when directy working
with the arrays. For convenience I have introduced properties, that return the arrays
with the empty dimensions removed. Another downside maybe that you have to use `keepdims=True`
more often, but there I would even argue that the default behavior in the array libraries
should change.

5. **"Poor-peoples" named axes**: The axis order convention is defined in the
code in the `Axis` enum, which maps the name to the integer dimension. Later I
can use, e.g. `Axis.batch` to refer to the batch axis in the code. This is the
simplest way to come close to named axes in any array library! So you can use
e.g. `jnp.sum(x, axes=Axis.components)` to sum over the components axis. I found
this to be a very powerful concept that improves the code clarity a lot, yet I
have not seen it often in other libraries. Of course there is `einops` but the
simple enum works just fine in many cases!

"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Any, Union

import jax
import numpy as np
from jax import numpy as jnp
from jax import scipy as jsp

from gmmx.utils import register_dataclass_jax

__all__ = ["FullCovariances", "GaussianMixtureModelJax"]


AnyArray = Union[np.ndarray, jax.Array]
Device = Union[str, jax.devices.Device, None]


class CovarianceType(str, Enum):
    """Convariance type"""

    full = "full"
    tied = "tied"
    diag = "diag"
    spherical = "spherical"


class Axis(int, Enum):
    """Internal axis order"""

    batch = 0
    components = 1
    features = 2
    features_covar = 3


def check_shape(array: jax.Array, expected: tuple[int | None, ...]) -> None:
    """Check shape of array"""
    if len(array.shape) != len(expected):
        message = f"Expected shape {expected}, got {array.shape}"
        raise ValueError(message)

    for n, m in zip(array.shape, expected):
        if m is not None and n != m:
            message = f"Expected shape {expected}, got {array.shape}"
            raise ValueError(message)


@register_dataclass_jax(data_fields=["values"])
@dataclass
class FullCovariances:
    """Full covariance matrix

    Attributes
    ----------
    values : jax.array
        Covariance values. Expected shape is (1, n_components, n_features, n_features)
    """

    values: jax.Array

    def __post_init__(self) -> None:
        check_shape(self.values, (1, None, None, None))

    @classmethod
    def from_squeezed(cls, values: AnyArray) -> FullCovariances:
        """Create a covariance matrix from squeezed array

        Parameters
        ----------
        values : jax.Array ot np.array
            Covariance values. Expected shape is (n_components, n_features, n_features)

        Returns
        -------
        covariances : FullCovariances
            Covariance matrix instance.
        """
        return cls(values=jnp.expand_dims(values, axis=Axis.batch))

    @property
    def values_numpy(self) -> np.ndarray:
        """Covariance as numpy array"""
        return np.squeeze(np.asarray(self.values), axis=Axis.batch)

    @property
    def precisions_cholesky_numpy(self) -> np.ndarray:
        """Compute precision matrices"""
        return np.squeeze(np.asarray(self.precisions_cholesky), axis=Axis.batch)

    @classmethod
    def create(
        cls, n_components: int, n_features: int, device: Device = None
    ) -> FullCovariances:
        """Create covariance matrix

        By default the covariance matrix is set to the identity matrix.

        Parameters
        ----------
        n_components : int
            Number of components
        n_features : int
            Number of features
        device : str, optional
            Device, by default None

        Returns
        -------
        covariances : FullCovariances
            Covariance matrix instance.
        """
        identity = jnp.expand_dims(
            jnp.eye(n_features), axis=(Axis.batch, Axis.components)
        )

        values = jnp.repeat(identity, n_components, axis=Axis.components)
        values = jax.device_put(values, device=device)
        return cls(values=values)

    def log_prob(self, x: jax.Array, means: jax.Array) -> jax.Array:
        """Compute log likelihood from the covariance for a given feature vector

        Parameters
        ----------
        x : jax.array
            Feature vectors
        means : jax.array
            Means of the components

        Returns
        -------
        log_prob : jax.array
            Log likelihood
        """
        precisions_cholesky = self.precisions_cholesky

        y = jnp.matmul(x.mT, precisions_cholesky) - jnp.matmul(
            means.mT, precisions_cholesky
        )
        return jnp.sum(
            jnp.square(y),
            axis=(Axis.features, Axis.features_covar),
            keepdims=True,
        )

    def update_parameters(
        self,
        x: jax.Array,
        means: jax.Array,
        resp: jax.Array,
        nk: jax.Array,
        reg_covar: float,
    ) -> FullCovariances:
        """Estimate updated covariance matrix from data

        Parameters
        ----------
        x : jax.array
            Feature vectors
        means : jax.array
            Means of the components
        resp : jax.array
            Responsibilities
        nk : jax.array
            Number of samples in each component
        reg_covar : float
            Regularization for the covariance matrix

        Returns
        -------
        covariances : FullCovariances
            Updated covariance matrix instance.
        """
        diff = x - means
        axes = (Axis.features_covar, Axis.components, Axis.features, Axis.batch)
        diff = jnp.transpose(diff, axes=axes)
        resp = jnp.transpose(resp, axes=axes)
        values = jnp.matmul(resp * diff, diff.mT) / nk
        idx = jnp.arange(self.n_features)
        values = values.at[:, :, idx, idx].add(reg_covar)
        return self.__class__(values=values)

    @property
    def n_components(self) -> int:
        """Number of components"""
        return self.values.shape[Axis.components]

    @property
    def n_features(self) -> int:
        """Number of features"""
        return self.values.shape[Axis.features]

    @property
    def n_parameters(self) -> int:
        """Number of parameters"""
        return int(self.n_components * self.n_features * (self.n_features + 1) / 2.0)

    @property
    def log_det_cholesky(self) -> jax.Array:
        """Log determinant of the cholesky decomposition"""
        diag = jnp.trace(
            jnp.log(self.precisions_cholesky),
            axis1=Axis.features,
            axis2=Axis.features_covar,
        )
        return jnp.expand_dims(diag, axis=(Axis.features, Axis.features_covar))

    @property
    def precisions_cholesky(self) -> jax.Array:
        """Compute precision matrices"""
        cov_chol = jsp.linalg.cholesky(self.values, lower=True)

        identity = jnp.expand_dims(
            jnp.eye(self.n_features), axis=(Axis.batch, Axis.components)
        )

        b = jnp.repeat(identity, self.n_components, axis=Axis.components)
        precisions_chol = jsp.linalg.solve_triangular(cov_chol, b, lower=True)
        return precisions_chol.mT


COVARIANCE: dict[CovarianceType, Any] = {
    CovarianceType.full: FullCovariances,
}

# keep this mapping separate, as names in sklearn might change
SKLEARN_COVARIANCE_TYPE: dict[Any, str] = {FullCovariances: "full"}


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

    def __post_init__(self) -> None:
        check_shape(self.weights, (1, None, 1, 1))
        check_shape(self.means, (1, None, None, 1))

    @property
    def weights_numpy(self) -> np.ndarray:
        """Weights as numpy array"""
        return np.squeeze(
            np.asarray(self.weights),
            axis=(Axis.batch, Axis.features, Axis.features_covar),
        )

    @property
    def means_numpy(self) -> np.ndarray:
        """Means as numpy array"""
        return np.squeeze(
            np.asarray(self.means), axis=(Axis.batch, Axis.features_covar)
        )

    @classmethod
    def create(
        cls,
        n_components: int,
        n_features: int,
        covariance_type: CovarianceType = CovarianceType.full,
        device: Device = None,
    ) -> GaussianMixtureModelJax:
        """Create a GMM from configuration

        Parameters
        ----------
        n_components : int
            Number of components
        n_features : int
            Number of features
        covariance_type : str, optional
            Covariance type, by default "full"
        device : str, optional
            Device, by default None

        Returns
        -------
        gmm : GaussianMixtureModelJax
            Gaussian mixture model instance.
        """
        covariance_type = CovarianceType(covariance_type)

        weights = jnp.ones((1, n_components, 1, 1)) / n_components
        means = jnp.zeros((1, n_components, n_features, 1))
        covariances = COVARIANCE[covariance_type].create(n_components, n_features)
        return cls(
            weights=jax.device_put(weights, device=device),
            means=jax.device_put(means, device=device),
            covariances=jax.device_put(covariances, device=device),
        )

    @classmethod
    def from_squeezed(
        cls,
        means: AnyArray,
        covariances: AnyArray,
        weights: AnyArray,
        covariance_type: CovarianceType = CovarianceType.full,
    ) -> GaussianMixtureModelJax:
        """Create a Jax GMM from squeezed arrays

        Parameters
        ----------
        means : jax.Array or np.array
            Mean of each component. Expected shape is (n_components, n_features)
        covariances : jax.Array or np.array
            Covariance of each component. Expected shape is (n_components, n_features, n_features)
        weights : jax.Array or np.array
            Weights of each component. Expected shape is (n_components,)
        covariance_type : str, optional
            Covariance type, by default "full"

        Returns
        -------
        gmm : GaussianMixtureModelJax
            Gaussian mixture model instance.
        """
        covariance_type = CovarianceType(covariance_type)

        means = jnp.expand_dims(means, axis=(Axis.batch, Axis.features_covar))
        weights = jnp.expand_dims(
            weights, axis=(Axis.batch, Axis.features, Axis.features_covar)
        )

        values = jnp.expand_dims(covariances, axis=Axis.batch)
        covariances = COVARIANCE[covariance_type](values=values)
        return cls(weights=weights, means=means, covariances=covariances)  # type: ignore [arg-type]

    def update_parameters(
        self, x: jax.Array, resp: jax.Array, reg_covar: float
    ) -> GaussianMixtureModelJax:
        """Update parameters

        Parameters
        ----------
        x : jax.array
            Feature vectors
        resp : jax.array
            Responsibilities
        reg_covar : float
            Regularization for the covariance matrix

        Returns
        -------
        gmm : GaussianMixtureModelJax
            Updated Gaussian mixture model
        """
        nk = jnp.sum(resp, axis=Axis.batch, keepdims=True)
        means = jnp.matmul(resp.T, x.T.mT).T / nk
        covariances = self.covariances.update_parameters(
            x=x, means=means, resp=resp, nk=nk, reg_covar=reg_covar
        )
        return self.__class__(
            weights=nk / nk.sum(), means=means, covariances=covariances
        )

    @classmethod
    def from_k_means(cls, x: jax.Array, n_components: int) -> None:
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
    def n_features(self) -> int:
        """Number of features"""
        return self.covariances.n_features

    @property
    def n_components(self) -> int:
        """Number of components"""
        return self.covariances.n_components

    @property
    def n_parameters(self) -> int:
        """Number of parameters"""
        return int(
            self.n_components
            + self.n_components * self.n_features
            + self.covariances.n_parameters
            - 1
        )

    @property
    def log_weights(self) -> jax.Array:
        """Log weights (~jax.ndarray)"""
        return jnp.log(self.weights)

    @jax.jit
    def estimate_log_prob(self, x: jax.Array) -> jax.Array:
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
        x = jnp.expand_dims(x, axis=(Axis.components, Axis.features_covar))
        log_prob = self.covariances.log_prob(x, self.means)
        two_pi = jnp.array(2 * jnp.pi)

        value = (
            -0.5 * (self.n_features * jnp.log(two_pi) + log_prob)
            + self.covariances.log_det_cholesky
            + self.log_weights
        )
        return value

    def to_sklearn(self, **kwargs: dict) -> Any:
        """Convert to sklearn GaussianMixture

        Parameters
        ----------
        **kwargs : dict
            Additional arguments passed to `~sklearn.mixture.GaussianMixture`

        Returns
        -------
        gmm : `~sklearn.mixture.GaussianMixture`
            Gaussian mixture model instance.
        """
        from sklearn.mixture import GaussianMixture  # type: ignore [import-untyped]

        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=SKLEARN_COVARIANCE_TYPE[type(self.covariances)],
            **kwargs,
        )
        gmm.weights_ = self.weights_numpy
        gmm.covariances_ = self.covariances.values_numpy
        gmm.means_ = self.means_numpy
        gmm.precisions_cholesky_ = self.covariances.precisions_cholesky_numpy
        return gmm

    @jax.jit
    def predict(self, x: jax.Array) -> jax.Array:
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
        log_prob = self.estimate_log_prob(x)
        predictions = jnp.argmax(log_prob, axis=Axis.components, keepdims=True)
        return jnp.squeeze(predictions, axis=(Axis.features, Axis.features_covar))

    @partial(jax.jit, static_argnames=["n_samples"])
    def sample(self, key: jax.Array, n_samples: int) -> jax.Array:
        """Sample from the model

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key
        n_samples : int
            Number of samples

        Returns
        -------
        samples : jax.array
            Samples
        """
        key, subkey = jax.random.split(key)

        selected = jax.random.categorical(
            key, self.log_weights.flatten(), shape=(n_samples,)
        )

        means = jnp.take(self.means, selected, axis=Axis.components)
        covar = jnp.take(self.covariances.values, selected, axis=Axis.components)

        samples = jax.random.multivariate_normal(
            subkey,
            jnp.squeeze(means, axis=(Axis.batch, Axis.features_covar)),
            jnp.squeeze(covar, axis=Axis.batch),
            shape=(n_samples,),
        )

        return samples
