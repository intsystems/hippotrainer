# Documentation

Hyperparameter tuning is time-consuming and computationally expensive, often requiring extensive trial and error to find optimal configurations. There is a variety of hyperparameter optimization methods, such as Grid Search, Random Search, Bayesian Optimization, etc. In the case of continuous hyperparameters, the gradient-based methods arise.

We implemented four effective and popular methods in one package, leveraging the unified, simple and clean structure. Below we delve into the problem statement and methods description.

## Hyperparameter Optimization Problem

Given a vector of model parameters and a vector of hyperparameters, one aims to find optimal hyperparameters, solving the bi-level optimization problem:

```{math}
    \begin{aligned}
    &\boldsymbol{\lambda}^* = \arg\min_{\boldsymbol{\lambda}} \mathcal{L}_{\text{val}}(\mathbf{w}^*, \boldsymbol{\lambda}), \\
    \text{s.t. } &\mathbf{w}^* = \arg\min_{\mathbf{w}} \mathcal{L}_{\text{train}}(\mathbf{w}, \boldsymbol{\lambda})
    \end{aligned}
```

Often parameters are optimized with gradient descent, so **unrolled optimization** is typically used:

```{math}
    \mathbf{w}_{t+1} = \boldsymbol{\Phi}(\mathbf{w}_{t}, \boldsymbol{\lambda}), \quad t = 0, \ldots, T-1.
```

Typical way to optimize continuous hyperparameters is the **gradient-based optimization** that involves automatic differentiation through this unrolled optimization formula.

## Hypergradient Calculation

Chain rule gives us a hypergradient:

```{math}
    \underbrace{d_{\boldsymbol{\lambda}} \mathcal{L}_{\text{val}}(\mathbf{w}_T, \boldsymbol{\lambda})}_{\text{hypergradient}} = \underbrace{\nabla_{\boldsymbol{\lambda}} \mathcal{L}_{\text{val}}(\mathbf{w}_T, \boldsymbol{\lambda})}_{\text{hyperparam direct grad.}} + \underbrace{\nabla_{\mathbf{w}} \mathcal{L}_{\text{val}}(\mathbf{w}_T, \boldsymbol{\lambda})}_{\text{parameter direct grad.}} \times \underbrace{\frac{d\mathbf{w}_T}{d\boldsymbol{\lambda}}}_{\text{best-response Jacobian}}
```

- Here **best-response Jacobian** is hard to compute!

Typical Solution — Implicit Function Theorem:

```{math}
    \frac{d\mathbf{w}_T}{d\boldsymbol{\lambda}} = - \underbrace{\left[ \nabla^2_{\mathbf{w}} \mathcal{L}_{\text{train}}(\mathbf{w}_T, \boldsymbol{\lambda}) \right]^{-1}}_{\text{inverted training Hessian}} \times \underbrace{\nabla_{\mathbf{w}} \nabla_{\boldsymbol{\lambda}} \mathcal{L}_{\text{train}} (\mathbf{w}_T, \boldsymbol{\lambda})}_{\text{training mixed partials}}.
```

- Hessian **inversion** is a cornerstone of many algorithms.

The next section contains information about each of the methods presented in our library, as they can be generalized to solve the above problem in different ways.

## Methods

To exactly invert a Hessian, we require a cubed number of operations, which is intractable for modern NNs. 
There are many ways to approximate the inverse Hessian, e.g., using the Neumann series:

```{math}
  \left[ \nabla^2_{\mathbf{w}} \mathcal{L}_{\text{train}}(\mathbf{w}_T, \boldsymbol{\lambda}) \right]^{-1} = \lim_{i \to \infty} \sum_{j=0}^{i} \left[ \mathbf{I} - \nabla^2_{\mathbf{w}} \mathcal{L}_{\text{train}} (\mathbf{w}_T, \boldsymbol{\lambda}) \right]^j.
```

Using different numbers of terms in this series or considering another approaches of approximation, one can derive a list of methods. 
All of them are inherited from the base class `HyperOptimizer`.

```{toctree}
:titlesonly:

hyper_optimizer
```

The current available methods are the following.

```{toctree}
:titlesonly:

t1_t2
neumann
hoag
drmad
```

