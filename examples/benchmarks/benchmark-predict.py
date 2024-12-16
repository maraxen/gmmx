import getpass
import importlib
import json
import logging
import platform
import sys
import timeit
from dataclasses import asdict, dataclass, field
from functools import partial
from itertools import product
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp
from jax.lib import xla_bridge

from gmmx import GaussianMixtureModelJax

INCLUDE_GPU = False

PATH = Path(__file__).parent
PATH_RESULTS = PATH / "results"
RANDOM_STATE = np.random.RandomState(817237)
N_AVERAGE = 10
DPI = 180


N_SAMPLES = 1024 * 2 ** np.arange(0, 11)
N_COMPONENTS = 2 ** np.arange(1, 8)
N_FEATURES = 2 ** np.arange(1, 8)

PATH_TEMPLATE = "{user}-{machine}-{system}-{cpu}-{device-platform}"


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


Array = list[float]


def get_provenance():
    """Compute provenance info about software and data used."""
    env = {
        "user": getpass.getuser(),
        "machine": platform.machine(),
        "system": platform.system(),
        "cpu": platform.processor(),
        "device-platform": xla_bridge.get_backend().platform,
    }

    software = {
        "python-executable": sys.executable,
        "python-version": platform.python_version(),
        "jax-version": str(importlib.import_module("jax").__version__),
        "numpy-version": str(importlib.import_module("numpy").__version__),
        "sklearn-version": str(importlib.import_module("sklearn").__version__),
    }

    return {
        "env": env,
        "software": software,
    }


@dataclass
class BenchmarkResult:
    """Benchmark result"""

    n_samples: Array
    n_components: Array
    n_features: Array
    time_sklearn: Array
    time_jax: Array
    time_jax_gpu: Optional[Array] = None
    provenance: dict = field(default_factory=get_provenance)

    def write_json(self, path):
        """Write the benchmark result to a JSON file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w") as f:
            json.dump(asdict(self), f)

    @classmethod
    def read(cls, path):
        """Read the benchmark result from a JSON file"""
        with Path(path).open("r") as f:
            data = json.load(f)
        return cls(**data)


def create_random_gmm(n_components, n_features, random_state=RANDOM_STATE, device=None):
    """Create a random Gaussian mixture model"""
    means = random_state.uniform(-10, 10, (n_components, n_features))

    values = random_state.uniform(0, 1, (n_components, n_features, n_features))
    covariances = np.matmul(values, values.mT)
    weights = random_state.uniform(0, 1, n_components)
    weights /= weights.sum()

    return GaussianMixtureModelJax.from_squeezed(
        means=jnp.device_put(means, device=device),
        covariances=jnp.device_put(covariances, device=device),
        weights=jnp.device_put(weights, device=device),
    )


def create_random_data(n_samples, n_features, random_state=RANDOM_STATE, device=None):
    """Create random data"""
    return random_state.uniform(-10, 10, (n_samples, n_features))


def get_meta_str(result, x_axis):
    """Get the metadata for the benchmark result"""
    if x_axis == "n_samples":
        meta = {"n_components": result.n_components, "n_features": result.n_features}

    elif x_axis == "n_components":
        meta = {"n_samples": result.n_samples, "n_features": result.n_features}

    elif x_axis == "n_features":
        meta = {"n_samples": result.n_samples, "n_components": result.n_components}
    else:
        message = f"Invalid x_axis: {x_axis}"
        raise ValueError(message)

    return ", ".join(f"{k}={v}" for k, v in meta.items())


def plot_result(result, x_axis, filename, title=""):
    """Plot the benchmark result"""
    log.info(f"Plotting {filename}")
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    x = getattr(result, x_axis)
    meta = get_meta_str(result, x_axis)

    color = "#F1C44D"
    ax.plot(x, result.time_sklearn, label=f"sklearn ({meta})", color=color)
    ax.scatter(x, result.time_sklearn, color=color)

    color = "#405087"
    ax.plot(x, result.time_jax, label=f"jax ({meta})", color=color)
    ax.scatter(x, result.time_jax, color=color)

    if result.time_jax_gpu:
        color = "#E58336"
        ax.plot(x, result.time_jax_gpu, label=f"jax-gpu ({meta})", color=color)
        ax.scatter(x, result.time_jax_gpu)

    ax.set_title(title)
    ax.set_xlabel(x_axis)
    ax.set_ylabel("Time (s)")
    ax.semilogx()
    ax.legend()

    log.info(f"Writing {filename}")
    plt.savefig(filename, dpi=DPI)


def measure_time_predict_sklearn(gmm, x):
    """Measure the time to predict the responsibilities"""
    func = partial(gmm.predict, X=x)
    timer = timeit.Timer(func)
    return timer.timeit(N_AVERAGE)


def measure_time_predict_jax(gmm, x):
    """Measure the time to predict the responsibilities"""

    def func():
        return gmm.predict(x=x).block_until_ready()

    timer = timeit.Timer(func)
    return timer.timeit(N_AVERAGE)


def measure_time_sklearn_vs_jax(n_components_grid, n_samples_grid, n_features_grid):
    """Measure the time to predict the responsibilities for sklearn and jax"""
    time_sklearn, time_jax, time_jax_gpu = [], [], []

    for n_component, n_samples, n_features in product(
        n_components_grid, n_samples_grid, n_features_grid
    ):
        log.info(
            f"Running n_components={n_component}, n_samples={n_samples}, n_features={n_features}"
        )
        gmm = create_random_gmm(n_component, n_features)
        x = create_random_data(n_samples, n_features)

        time_sklearn.append(measure_time_predict_sklearn(gmm.to_sklearn(), x))
        time_jax.append(measure_time_predict_jax(gmm, x))

        if INCLUDE_GPU:
            gmm_gpu = create_random_gmm(n_component, n_features, device="gpu")
            x_gpu = jnp.device_put(x, device="gpu")
            time_jax_gpu.append(measure_time_predict_jax(gmm_gpu, x_gpu))

    return BenchmarkResult(
        n_samples=n_samples_grid,
        n_components=n_components_grid,
        n_features=n_features_grid,
        time_sklearn=time_sklearn,
        time_jax=time_jax,
        time_jax_gpu=time_jax_gpu or None,
    )


def run_time_vs_n_components(filename):
    """Time vs n_components"""
    if not filename.exists():
        result = measure_time_sklearn_vs_jax(
            N_COMPONENTS.tolist(), n_samples_grid=[100_000], n_features_grid=[64]
        )
        filename = (
            PATH_RESULTS / PATH_TEMPLATE.format(**result.provenance["env"]) / filename
        )
        log.info(f"Writing {filename}")
        result.write_json(filename)

    result = BenchmarkResult.read(filename)
    plot_result(
        result,
        x_axis="n_components",
        filename=filename.with_suffix(".png"),
        title="Time vs Number of components",
    )


def run_time_vs_n_features(filename):
    """Time vs n_features"""
    if not filename.exists():
        result = measure_time_sklearn_vs_jax(
            n_components_grid=[128],
            n_samples_grid=[100_000],
            n_features_grid=N_FEATURES.tolist(),
        )
        result.write_json(filename)

    result = BenchmarkResult.read(filename)
    plot_result(
        result,
        x_axis="n_features",
        filename=filename.with_suffix(".png"),
        title="Time vs Number of features",
    )


def run_time_vs_n_samples(filename):
    """Time vs n_samples"""
    if not filename.exists():
        result = measure_time_sklearn_vs_jax(
            n_components_grid=[128],
            n_samples_grid=N_SAMPLES.tolist(),
            n_features_grid=[64],
        )
        result.write_json(filename)

    result = BenchmarkResult.read(filename)
    plot_result(
        result,
        x_axis="n_samples",
        filename=filename.with_suffix(".png"),
        title="Time vs Number of samples",
    )


if __name__ == "__main__":
    path = PATH_RESULTS / PATH_TEMPLATE.format(**get_provenance()["env"])
    run_time_vs_n_components(path / "time-vs-n-components-predict.json")
    run_time_vs_n_features(path / "time-vs-n-features-prefict.json")
    run_time_vs_n_samples(path / "time-vs-n-samples-predict.json")
