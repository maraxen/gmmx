import json
import logging
import timeit
from dataclasses import asdict, dataclass
from functools import partial
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gmmx import GaussianMixtureModelJax

PATH = Path(__file__).parent
RANDOM_STATE = np.random.RandomState(817237)
N_AVERAGE = 10
DPI = 180


N_SAMPLES = 1024 * 2 ** np.arange(0, 11)
N_COMPONENTS = 2 ** np.arange(1, 9)
N_FEATURES = 2 ** np.arange(1, 9)


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


Array = list[float]


@dataclass
class BenchmarkResult:
    """Benchmark result"""

    n_samples: Array
    n_components: Array
    n_features: Array
    time_sklearn: Array
    time_jax: Array

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


def create_random_gmm(n_components, n_features, random_state=RANDOM_STATE):
    """Create a random Gaussian mixture model"""
    means = random_state.uniform(-10, 10, (n_components, n_features))

    values = random_state.uniform(0, 1, (n_components, n_features, n_features))
    covariances = np.matmul(values, values.mT)
    weights = random_state.uniform(0, 1, n_components)
    weights /= weights.sum()

    return GaussianMixtureModelJax.from_squeezed(
        means=means,
        covariances=covariances,
        weights=weights,
    )


def create_random_data(n_samples, n_features, random_state=RANDOM_STATE):
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


def plot_result(result, x_axis, filename):
    """Plot the benchmark result"""
    log.info(f"Plotting {filename}")
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    x = getattr(result, x_axis)
    meta = get_meta_str(result, x_axis)

    ax.plot(x, result.time_sklearn, label=f"sklearn ({meta})")
    ax.plot(x, result.time_jax, label=f"jax ({meta})")
    ax.set_xlabel(x_axis)
    ax.set_ylabel("Time (s)")
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
    time_sklearn, time_jax = [], []

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

    return BenchmarkResult(
        n_samples=n_samples_grid,
        n_components=n_components_grid,
        n_features=n_features_grid,
        time_sklearn=time_sklearn,
        time_jax=time_jax,
    )


def run_time_vs_n_components(filename):
    """Time vs n_components"""
    if not filename.exists():
        result = measure_time_sklearn_vs_jax(
            N_COMPONENTS, n_samples_grid=[100_000], n_features_grid=[64]
        )
        log.info(f"Writing {filename}")
        result.write_json(filename)

    result = BenchmarkResult.read(filename)
    plot_result(result, x_axis="n_components", filename=filename.with_suffix(".png"))


def run_time_vs_n_features(filename):
    """Time vs n_features"""
    if not filename.exists():
        result = measure_time_sklearn_vs_jax(
            n_components_grid=[256],
            n_samples_grid=[100_000],
            n_features_grid=N_FEATURES,
        )
        result.write_json(filename)

    result = BenchmarkResult.read(filename)
    plot_result(result, x_axis="n_features", filename=filename.with_suffix(".png"))


def run_time_vs_n_samples(filename):
    """Time vs n_samples"""
    log.info("Running time vs n_samples")
    result = measure_time_sklearn_vs_jax(
        n_components_grid=[256], n_samples_grid=N_SAMPLES, n_features_grid=[64]
    )
    result.write_json(filename)
    plot_result(result, x_axis="n_samples", filename=filename.with_suffix(".png"))


if __name__ == "__main__":
    path = PATH / "results"
    run_time_vs_n_components(path / "time-vs-n-components-predict.json")
    run_time_vs_n_features(path / "time-vs-n-features-prefict.json")
    run_time_vs_n_samples(path / "time-vs-n-samples-predict.json")
