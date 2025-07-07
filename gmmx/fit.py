from dataclasses import dataclass

import jax
from jax import numpy as jnp

from .gmm import Axis, GaussianMixtureModelJax
from .utils import register_dataclass_jax

__all__ = ["EMFitter", "EMFitterResult"]


@register_dataclass_jax(
    data_fields=[
        "x",
        "gmm",
        "n_iter",
        "log_likelihood",
        "log_likelihood_diff",
        "converged",
    ]
)
@dataclass
class EMFitterResult:
    """Expectation-Maximization Fitter Result

    Attributes
    ----------
    x : jax.array
        Feature vectors
    gmm : GaussianMixtureModelJax
        Gaussian mixture model instance.
    n_iter : int
        Number of iterations
    log_likelihood : jax.array
        Log-likelihood of the data
    log_likelihood_diff : jax.array
        Difference in log-likelihood with respect to the previous iteration
    converged : bool
        Whether the algorithm converged
    """

    x: jax.Array
    gmm: GaussianMixtureModelJax
    n_iter: int
    log_likelihood: jax.Array
    log_likelihood_diff: jax.Array
    converged: bool


@register_dataclass_jax(meta_fields=["max_iter", "tol", "reg_covar"])
@dataclass
class EMFitter:
    """Expectation-Maximization Fitter

    Attributes
    ----------
    max_iter : int
        Maximum number of iterations
    tol : float
        Tolerance
    reg_covar : float
        Regularization for covariance matrix
    """

    max_iter: int = 100
    tol: float = 1e-3
    reg_covar: float = 1e-6

    def e_step(
        self, x: jax.Array, gmm: GaussianMixtureModelJax
    ) -> tuple[jax.Array, jax.Array]:
        """Expectation step

        Parameters
        ----------
        x : jax.array
            Feature vectors
        gmm : GaussianMixtureModelJax
            Gaussian mixture model instance.

        Returns
        -------
        log_likelihood : jax.array
            Log-likelihood of the data
        """
        log_prob = gmm.log_prob(x)
        log_prob_norm = jax.scipy.special.logsumexp(
            log_prob, axis=Axis.components, keepdims=True
        )
        log_resp = log_prob - log_prob_norm
        return jnp.mean(log_prob_norm), log_resp

    def m_step(
        self, x: jax.Array, gmm: GaussianMixtureModelJax, log_resp: jax.Array
    ) -> GaussianMixtureModelJax:
        """Maximization step

        Parameters
        ----------
        x : jax.array
            Feature vectors
        gmm : GaussianMixtureModelJax
            Gaussian mixture model instance.
        log_resp : jax.array
            Logarithm of the responsibilities

        Returns
        -------
        gmm : GaussianMixtureModelJax
            Updated Gaussian mixture model instance.
        """
        x = jnp.expand_dims(x, axis=(Axis.components, Axis.features_covar))
        return gmm.from_responsibilities(
            x,
            jnp.exp(log_resp),
            reg_covar=self.reg_covar,
            covariance_type=gmm.covariances.type,
        )

    @jax.jit
    def fit(self, x: jax.Array, gmm: GaussianMixtureModelJax) -> EMFitterResult:
        """Fit the model to the data

        Parameters
        ----------
        x : jax.array
            Feature vectors
        gmm : GaussianMixtureModelJax
            Gaussian mixture model instance.

        Returns
        -------
        result : EMFitterResult
            Fitting result
        """

        def em_step(
            args: tuple[jax.Array, GaussianMixtureModelJax, int, jax.Array, jax.Array],
        ) -> tuple:
            """EM step function"""
            x, gmm, n_iter, log_likelihood_prev, _ = args
            log_likelihood, log_resp = self.e_step(x, gmm)
            gmm = self.m_step(x, gmm, log_resp)
            return (
                x,
                gmm,
                n_iter + 1,
                log_likelihood,
                jnp.abs(log_likelihood - log_likelihood_prev),
            )

        def em_cond(
            args: tuple[jax.Array, GaussianMixtureModelJax, int, jax.Array, jax.Array],
        ) -> jax.Array:
            """EM stop condition function"""
            _, _, n_iter, _, log_likelihood_diff = args
            return (n_iter < self.max_iter) & (log_likelihood_diff >= self.tol)

        result = jax.lax.while_loop(
            cond_fun=em_cond,
            body_fun=em_step,
            init_val=(x, gmm, 0, jnp.asarray(jnp.inf), jnp.array(jnp.inf)),
        )
        result = jax.block_until_ready(result)
        return EMFitterResult(*result, converged=result[2] < self.max_iter)  # type: ignore [misc]
