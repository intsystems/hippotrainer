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
- [ ] **HOAG** ([Paper](http://proceedings.mlr.press/v48/pedregosa16.pdf)): Implicit differentiation via conjugate gradient
- [ ] **DrMAD** ([Paper](https://arxiv.org/abs/1601.00917)): Memory-efficient piecewise-linear backpropagation

## 🤝 Contributors
- [Daniil Dorin](https://github.com/DorinDaniil) (Basic code writing, Final demo, Algorithms)
- [Igor Ignashin](https://github.com/ThunderstormXX) (Project wrapping, Documentation writing, Algorithms)
- [Nikita Kiselev](https://github.com/kisnikser) (Project planning, Blog post, Algorithms)
- [Andrey Veprikov](https://github.com/Vepricov) (Tests writing, Documentation writing, Algorithms)
- We welcome contributions!

## 📄 License
HippoTrainer is MIT licensed. See [LICENSE](https://github.com/intsystems/hippotrainer/blob/main/LICENSE) for details.

<!-- end docs-index -->