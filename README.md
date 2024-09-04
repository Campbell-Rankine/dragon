# dragon
ML Utils library for publishing to PyPI

### Modules

  - search : Hyperparameter search classes and utilities. (Bayes Opt)
    a. Bayes opt continuous search.
    b. Gaussian Process Regressor
    c. Vizier Gaussian Process Bandit (No evolutionary Argmax)
  - backgrop : Gradient Accumulation (DL Training Module)
    a. Gradient Accumulation
  - tools : Model utils, Pruning, Logging, etc.
    a. Base Pruning
    b. Distinctiveness pruning 
    c. Tensor function window
    d. MADDPG Replay buffer
    e. OU Action noise for Policy Gradient based Agents
    f. Computer Vision CNN extension pytorch nn.Module's (from RGB, to RGB, Equalized2DConv, etc.)
  - utils : Utility functions (normally for internal use)