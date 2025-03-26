# HOAG

This method solves the linear system using the Conjugate Gradient to invert the Hessian approximately. The following system is solved:
```{math}
\nabla^2_{\mathbf{w}} \mathcal{L}_{\text{train}}(\mathbf{w}_T, \boldsymbol{\lambda}) \cdot \mathbf{z} = \nabla_{\boldsymbol{\lambda}} \mathcal{L}_{\text{val}}(\mathbf{w}_T, \boldsymbol{\lambda}).
```

<!-- ```{eval-rst}
.. automodule:: hippotrainer.hoag
   :members:
   :undoc-members:
``` -->