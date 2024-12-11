import timeit

import numpy as np
from jax import numpy as jnp

from gmmx import GaussianMixtureModelJax

random_state = np.random.RandomState(817237)

n_samples = 10_000
n_components = 128
n_features = 64

means = random_state.uniform(-10, 10, (n_components, n_features))

values = random_state.uniform(0, 1, (n_components, n_features, n_features))
covariances = np.matmul(values, values.mT)
weights = random_state.uniform(0, 1, n_components)
weights /= weights.sum()

gmm_jax = GaussianMixtureModelJax.from_squeezed(
    means=means,
    covariances=covariances,
    weights=weights,
)

gmm_sklearn = gmm_jax.to_sklearn()


x = random_state.uniform(-10, 10, (n_samples, n_features))
x_jax = jnp.asarray(x)

timer = timeit.Timer(
    "gmm_jax.predict(x=x_jax).block_until_ready()", globals=globals()
)
value = timer.timeit(10)

print(f"JAX: {value:.2f}")


timer = timeit.Timer("gmm_sklearn.predict(X=x)", globals=globals())
value = timer.timeit(10)
print(f"SKLEARN {value:.2f}")
