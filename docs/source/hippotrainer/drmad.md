# DrMAD

This method is not straightforward. Instead of storing all intermediate weights, it approximates the training trajectory as a linear combination of the initial and final weights:
```{math}
  \mathbf{w}(\beta) = (1 - \beta) \mathbf{w}_0 + \beta \mathbf{w}_T, \quad 0 < \beta < 1.
```
Then it uses such approximation to perform the backward pass on the hyperparameters.

<!-- ```{eval-rst}
.. automodule:: hippotrainer.drmad
   :members:
   :undoc-members:
``` -->