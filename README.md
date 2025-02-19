<div align="center">  
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="assets/logo-white.svg" width="200px">
      <source media="(prefers-color-scheme: light)" srcset="assets/logo.svg" width="200px">
      <img alt="HippoParams" src="assets/logo.svg" width="200px">
    </picture>
    <h1> HippoParams </h1>
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

<!--- <p align="center">
    <a href="https://github.com/intsystems/hippoparams/actions">
        <img alt="Tests" src="https://github.com/intsystems/hippoparams/actions/workflows/tests.yml/badge.svg">
    </a>
    <a href="https://codecov.io/gh/intsystems/hippoparams">
        <img alt="Coverage" src="https://codecov.io/gh/intsystems/hippoparams/branch/main/graph/badge.svg">
    </a>
    <a href="https://hippoparams.readthedocs.io">
        <img alt="Docs" src="https://github.com/intsystems/hippoparams/actions/workflows/docs.yml/badge.svg">
    </a>
</p> -->

<p align="center">
    <a href="https://github.com/intsystems/hippoparams/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/intsystems/hippoparams">
    </a>
    <a href="https://github.com/intsystems/hippoparams/graphs/contributors">
        <img alt="Contributors" src="https://img.shields.io/github/contributors/intsystems/hippoparams">
    </a>
    <a href="https://github.com/intsystems/hippoparams/issues">
        <img alt="Issues" src="https://img.shields.io/github/issues-closed/intsystems/hippoparams">
    </a>
    <a href="https://github.com/intsystems/hippoparams/pulls">
        <img alt="Pull Requests" src="https://img.shields.io/github/issues-pr-closed/intsystems/hippoparams">
    </a>
</p>

**HippoParams** is a PyTorch-compatible library for gradient-based hyperparameter optimization, implementing cutting-edge algorithms that leverage automatic differentiation to efficiently tune hyperparameters.

## 🚀 Features
- **Algorithm Zoo**: T1-T2, Billion Hyperparameters, HOAG, DrMAD
- **PyTorch Native**: Direct integration with `torch.nn.Module`
- **Memory Efficient**: Checkpointing & implicit differentiation
- **Scalable**: From laptop to cluster with PyTorch backend

## 📜 Algorithms
- [ ] **T1-T2** ([Paper](http://proceedings.mlr.press/v48/luketina16.pdf)): Unrolled optimization with explicit gradient computation
- [ ] **Billion Hyperparams** ([Paper](http://proceedings.mlr.press/v108/lorraine20a/lorraine20a.pdf)): Large-scale optimization with PyTorch fusion
- [ ] **HOAG** ([Paper](http://proceedings.mlr.press/v48/pedregosa16.pdf)): Implicit differentiation via conjugate gradient
- [ ] **DrMAD** ([Paper](https://arxiv.org/abs/1601.00917)): Memory-efficient piecewise-linear backprop

## 🤝 Contributors
- [Daniil Dorin](https://github.com/DorinDaniil) (Basic code writing, Final demo, Algorithms)
- [Igor Ignashin](https://github.com/ThunderstormXX) (Project wrapping, Documentation writing, Algorithms)
- [Nikita Kiselev](https://github.com/kisnikser) (Project planning, Blog post, Algorithms)
- [Andrey Veprikov](https://github.com/Vepricov) (Tests writing, Documentation writing, Algorithms)
- We welcome contributions!

## 📄 License
HippoParams is MIT licensed. See [LICENSE](LICENSE) for details.
