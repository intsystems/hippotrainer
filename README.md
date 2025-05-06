<div align="center">  
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="assets/logo-white.svg" width="200px">
      <source media="(prefers-color-scheme: light)" srcset="assets/logo.svg" width="200px">
      <img alt="HippoTrainer" src="assets/logo.svg" width="200px">
    </picture>
    <h1> HippoTrainer </h1>
    <p align="center"> Gradient-Based Hyperparameter Optimization for PyTorch 🦛 </p>
</div>

<p align="center">
    <a href="https://pytorch.org/">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white">
    </a>
    <a href="https://optuna.org/">
        <img alt="Inspired by Optuna" src="https://img.shields.io/badge/Inspired_by-Optuna-3366CC">
    </a>
</p>

<p align="center">
    <a href="https://intsystems.github.io/hippotrainer/">
        <img alt="Docs" src="https://github.com/intsystems/hippotrainer/actions/workflows/docs.yml/badge.svg">
    </a>
    <a href="https://github.com/intsystems/hippotrainer/tree/main/tests">
        <img alt="Tests" src="https://github.com/intsystems/hippotrainer/actions/workflows/tests.yml/badge.svg">
    </a>
    <a href="https://codecov.io/gh/intsystems/hippotrainer">
        <img alt="Coverage" src="https://codecov.io/gh/intsystems/hippotrainer/branch/main/graph/badge.svg">
    </a>
</p>

<p align="center">
    <a href="https://github.com/intsystems/hippotrainer/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/intsystems/hippotrainer">
    </a>
    <a href="https://github.com/intsystems/hippotrainer/graphs/contributors">
        <img alt="Contributors" src="https://img.shields.io/github/contributors/intsystems/hippotrainer">
    </a>
    <a href="https://github.com/intsystems/hippotrainer/issues">
        <img alt="Issues" src="https://img.shields.io/github/issues-closed/intsystems/hippotrainer">
    </a>
    <a href="https://github.com/intsystems/hippotrainer/pulls">
        <img alt="Pull Requests" src="https://img.shields.io/github/issues-pr-closed/intsystems/hippotrainer">
    </a>
</p>

<!-- start docs-index -->

**HippoTrainer** is a PyTorch-compatible library for gradient-based hyperparameter optimization, implementing cutting-edge algorithms that leverage automatic differentiation to efficiently tune hyperparameters.

<!-- end docs-index -->

## 📬 Assets

1. [Technical Meeting 1 - Presentation](https://github.com/intsystems/hippotrainer/blob/main/assets/presentation.pdf)
2. [Technical Meeting 2 - Jupyter Notebook](https://github.com/intsystems/hippotrainer/blob/main/notebooks/basic_code.ipynb)
3. [Technical Meeting 3 - Jupyter Notebook](https://github.com/intsystems/hippotrainer/blob/main/notebooks/demo.ipynb)
4. [Documentation](https://intsystems.github.io/hippotrainer/)
5. [Tests](https://github.com/intsystems/hippotrainer/tree/main/tests)
6. [Blog Post](https://kisnikser.github.io/projects/hippotrainer/)

## 🚀 Features
- **Algorithm Zoo**: T1-T2, Neumann, HOAG, DrMAD
- **PyTorch Native**: Direct integration with `torch.nn.Module`
- **Memory Efficient**: Checkpointing & implicit differentiation
- **Scalable**: From laptop to cluster with PyTorch backend

## 📜 Algorithms
- [x] **T1-T2** ([Paper](http://proceedings.mlr.press/v48/luketina16.pdf)): One-step unrolled optimization
- [x] **Neumann** ([Paper](http://proceedings.mlr.press/v108/lorraine20a/lorraine20a.pdf)): Leveraging Neumann series approximation for implicit differentiation
- [x] **HOAG** ([Paper](http://proceedings.mlr.press/v48/pedregosa16.pdf)): Implicit differentiation via conjugate gradient
- [x] **DrMAD** ([Paper](https://arxiv.org/abs/1601.00917)): Memory-efficient piecewise-linear backpropagation

## 🛠️ Installation

<!-- start installation -->

Our python library `hippotrainer` supports multiple ways to be installed.
You can choose the most suitable for your purposes.

We suggest to use installation from source (`pip install git+https://github.com/intsystems/hippotrainer`), if you want to get the latest package version.

### Install using pip

```bash
pip install hippotrainer
```

### Install from source

```bash
pip install git+https://github.com/intsystems/hippotrainer
```

### Install via Git clone

```bash
git clone https://github.com/intsystems/hippotrainer
cd hippotrainer
pip install -e .
```

<!-- end installation -->

## 🚀 Usage

<!-- start usage -->

You can use our library to tune almost all (see below) hyperparameters in your own code.
The `HyperOptimizer` interface is very similar to [`Optimizer`](https://pytorch.org/docs/2.6/optim.html#torch.optim.Optimizer) from `PyTorch`.

It supports key functionalities:
1. `step` to do an optimization step over parameters (or hyperparameters, see below)
2. `zero_grad` to zero out the parameters gradients (same as `optimizer.zero_grad()`)

We provide demo experiments with each implemented method in [this notebook](https://github.com/intsystems/hippotrainer/blob/main/notebooks/demo.ipynb).
They works as follows:
1. Get next batch from train dataloader
2. Forward and backward on calculated loss
3. `hyper_optimizer.step(loss)` do model parameters step and (if inner steps were accumulated) hyperparameters step (calculate hypergradients, do the optimization step, zeroes hypergradients)
4. `hyper_optimizer.zero_grad()` zeroes the model parameters gradients (same as `optimizer.zero_grad()`)

### `Optimizer` vs. `HyperOptimizer` method `step`

Gradient-based hyperparameters optimization involves hyper-optimization steps during the
model parameters optimization. Thus, we combine `Optimizer` method `step` with `inner_steps`,
defined by each method.

For example, `T1T2` do NOT use any inner steps, therefore optimization over parameters
and hyperparameters is done step by step. But `Neumann` method do some inner optimization steps
over model parameters before it do the hyperstep.

See more details [here](https://github.com/intsystems/hippotrainer/blob/60cbafd6614bf057e83268da6cebf04ae2e6d7e7/src/hippotrainer/hyper_optimizer.py#L121).

### Supported hyperparameters types

The `HyperOptimizer` logic is well-suited for almost all **CONTINUOUS** (required for gradient-based methods) hyperparameters types:
1. Model hyperparameters (e.g., gate coefficients)
2. Loss hyperparameters (e.g., L1/L2-regularization)

However, it currently does **not** support (or support, but actually was not sufficiently tested) **learning rate tuning**.
We plan to improve our functionality in future releases, stay tuned!

<!-- end usage -->

## 🤝 Contributors
- [Daniil Dorin](https://github.com/DorinDaniil) (Basic code writing, Final demo, Algorithms)
- [Igor Ignashin](https://github.com/ThunderstormXX) (Project wrapping, Documentation writing, Algorithms)
- [Nikita Kiselev](https://github.com/kisnikser) (Project planning, Blog post, Algorithms)
- [Andrey Veprikov](https://github.com/Vepricov) (Tests writing, Documentation writing, Algorithms)
- We welcome contributions!

## 📄 License
HippoTrainer is MIT licensed. See [LICENSE](https://github.com/intsystems/hippotrainer/blob/main/LICENSE) for details.