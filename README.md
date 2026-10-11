# Optimization for Training Deep Neural Networks (PyTorch)

I implemented every algorithm from class — plain gradient descent, its batch/stochastic/mini-batch variants, AdaGrad, RMSProp, and Adam — from first principles in PyTorch, then checked each one against the built-in `torch.optim` version to make sure my math was right.

## Contents

| File | Description |
|---|---|
| `Optimizations.ipynb` | My completed lab notebook — all tasks and exercises, with code, output, and my written observations. |
| `Lab6_Optimization_for_Training_DNNs_PyTorch.docx` | The original lab tutorial handout I was given. |

## Requirements

- Python 3.9+
- PyTorch 2.x (CPU is sufficient)
- `matplotlib` (for the trajectory plot in Exercise 4)

```bash
pip install torch matplotlib
```

No dataset download is needed — every task uses synthetic data generated on the fly, and I fixed the random seed (`torch.manual_seed(0)`) everywhere so the results are reproducible.

## How to run

Open `Optimizations.ipynb` in Jupyter and run all cells top to bottom in a fresh kernel:

```bash
jupyter notebook Optimizations.ipynb
```

## What I did

**Main tasks**

1. **Gradient descent from scratch** — manual autograd update rule on `L(w) = 1.25w²`.
2. **Batch vs. stochastic vs. mini-batch GD** — update count vs. stability trade-off on a linear regression fit.
3. **Watching gradients vanish** — layer-by-layer gradient norms for an 8-layer stack, Sigmoid vs. ReLU.
4. **AdaGrad, RMSProp, and Adam from scratch** — hand-written update rules, cross-checked against `torch.optim`.
5. **Optimizer bake-off** — SGD, SGD+momentum, AdaGrad, RMSProp, and Adam compared on a non-linearly-separable ring classification task.
6. **Why AdaGrad stalls and RMSProp doesn't** — effective learning rate under a constant gradient, tracked over 500 steps.

**Lab exercises — my answers**

1. Found the divergence threshold for gradient descent as η increases (η = 0.1, 0.5, 1.0) and worked out why the quadratic loss lets you predict it in advance.
2. Re-ran Task 2 with a fixed number of parameter updates instead of a fixed number of epochs, to see whether mini-batch GD still wins.
3. Swapped Sigmoid for Tanh + `BatchNorm1d` on the vanishing-gradient stack and reported the first-layer gradient norm in all three settings.
4. Extended AdaGrad/RMSProp into 100-step loops and plotted their trajectories against plain GD.
5. Removed Adam's bias correction and quantified how training over the first 20 epochs changes.
6. Swept learning rates ({0.001, 0.01, 0.1, 0.5}) for every optimizer and checked whether the ranking changes after tuning.

## My key results

**Optimizer bake-off (Task 5, 200 epochs, ring classification):**

| Optimizer | State stored | Final accuracy | Behaviour |
|---|---|---|---|
| SGD | none | 0.978 | Slowest; still descending at epoch 200 |
| SGD + momentum | velocity | 0.997 | Large improvement for one extra hyperparameter |
| AdaGrad | sum of squared gradients | 0.993 | Fast early, then visibly slows down |
| RMSProp | moving average of squared gradients | 0.979 | Steady, sensitive to η |
| Adam | both moments, bias-corrected | 1.000 | Fastest, lowest final loss |

**After tuning learning rates (Exercise 6):** Adam still came out best (lr=0.1, loss=0.0038), followed by SGD+momentum, AdaGrad, RMSProp, then SGD — the ranking barely changed, though the gaps narrowed.

**Vanishing gradients (Task 3 / Exercise 3):** I saw the first-layer gradient norm go from ~0.0 with Sigmoid, to ~0.006 with Tanh, to ~0.41 with Tanh + BatchNorm — BatchNorm is what actually fixed it, not the activation swap alone.

## Before I submit

- [x] Every cell runs top to bottom in a fresh kernel.
- [x] The random seed is fixed throughout.
- [x] Each written answer names the specific number in my output that supports it.
