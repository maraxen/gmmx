# GMMX: Gaussian Mixture Models in Jax

[![Release](https://img.shields.io/github/v/release/adonath/gmmx)](https://img.shields.io/github/v/release/adonath/gmmx)
[![Build status](https://img.shields.io/github/actions/workflow/status/adonath/gmmx/main.yml?branch=main)](https://github.com/adonath/gmmx/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/adonath/gmmx/branch/main/graph/badge.svg)](https://codecov.io/gh/adonath/gmmx)
[![Commit activity](https://img.shields.io/github/commit-activity/m/adonath/gmmx)](https://img.shields.io/github/commit-activity/m/adonath/gmmx)
[![License](https://img.shields.io/github/license/adonath/gmmx)](https://img.shields.io/github/license/adonath/gmmx)

<p align="center">
<img width="50%" src="docs/_static/gmmx-logo.png" alt="GMMX Logo"/>
</p>

A minimal implementation of Gaussian Mixture Models in Jax

- **Github repository**: <https://github.com/adonath/gmmx/>
- **Documentation** <https://adonath.github.io/gmmx/>

## Installation

```bash
pip install gmmx
```

## Usage

```python
from gmmx import GaussianMixtureModelJax, EMFitter

# Create a Gaussian Mixture Model with 16 components and 32 features
gmm = GaussianMixtureModelJax.create(n_components=16, n_features=32)

n_samples = 10_000
x = gmm.sample(n_samples)

# Fit the model to the data
em_fitter = EMFitter()
gmm_fitted = em_fitter.fit(gmm, x)
```

## Benchmarks

Here are some results from the benchmarks in the `benchmarks` folder comparing against Scikit-Learn. The benchmarks were run on a 2021 MacBook Pro with an M1 Pro chip.

### Prediction Time

| Time vs. Number of Components                                                   | Time vs. Number of Samples                                                | Time vs. Number of Features                                                 |
| ------------------------------------------------------------------------------- | ------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| ![Time vs. Number of Components](docs/_static/time-vs-n-components-predict.png) | ![Time vs. Number of Samples](docs/_static/time-vs-n-samples-predict.png) | ![Time vs. Number of Features](docs/_static/time-vs-n-features-predict.png) |

### Training Time

| Time vs. Number of Components                                               | Time vs. Number of Samples                                            | Time vs. Number of Features                                             |
| --------------------------------------------------------------------------- | --------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| ![Time vs. Number of Components](docs/_static/time-vs-n-components-fit.png) | ![Time vs. Number of Samples](docs/_static/time-vs-n-samples-fit.png) | ![Time vs. Number of Features](docs/_static/time-vs-n-features-fit.png) |
