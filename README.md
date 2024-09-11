# dragon
ML Utils library for publishing to PyPI

### Modules

  - search : Hyperparameter search classes and utilities. (Bayes Opt)
    - Bayes opt continuous search.
    - Gaussian Process Regressor
    - Vizier Gaussian Process Bandit (No evolutionary Argmax)
  - backgrop : Gradient Accumulation (DL Training Module)
    - Gradient Accumulation
  - tools : Model utils, Pruning, Logging, etc.
    - Base Pruning
    - Distinctiveness pruning 
    - Tensor function window
    - MADDPG Replay buffer
    - OU Action noise for Policy Gradient based Agents
    - Computer Vision CNN extension pytorch nn.Module's (from RGB, to RGB, Equalized2DConv, etc.)
    - CUDA optimized Sobel Filter (from dragon import sobel_filter)
  - utils : Utility functions (normally for internal use)