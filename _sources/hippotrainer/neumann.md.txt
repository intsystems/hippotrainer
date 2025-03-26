# Neumann

This method uses a pre-determined number of terms in the Neumann series.
```{math}
  \left[ \nabla^2_{\mathbf{w}} \mathcal{L}_{\text{train}}(\mathbf{w}_T, \boldsymbol{\lambda}) \right]^{-1} \approx \sum_{j=0}^{i} \left[ \mathbf{I} - \nabla^2_{\mathbf{w}} \mathcal{L}_{\text{train}} (\mathbf{w}_T, \boldsymbol{\lambda}) \right]^j.
```

```{eval-rst}
.. automodule:: hippotrainer.neumann
   :members:
   :undoc-members:
```