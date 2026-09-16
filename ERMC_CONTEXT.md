# Efficient Robust Mode Connectivity (ERMC) Implementation Context

This repository contains the reproduced codebase for the paper **"Deep Adversarial Defense Against Multilevel-Lp Attacks"** (arXiv:2407.09251) by Ren Wang, Yuxuan Li, and Alfred Hero.

## Goal
The goal of this project was to faithfully recreate the ERMC defense strategy and proper evaluation loop as described in the paper. Since the authors did not provide an explicit ERMC repository, the implementation was carefully reverse-engineered and cross-referenced using two related repositories by the same authors:
1. `wangren09/MCGR` (Prior version of the ERMC work containing the MSD attack, PGD-L1, and path evaluation logic)
2. `TIML-Group/MoCo-EA` (Newer work by the authors that utilizes the same Quadratic Bezier Curve formulations and Duchi's L1 projection algorithm)

## Implementation Details

### 1. Endpoint Training (`train_endpoints.py`)
To build the robust path, two endpoint models are required:
- **$\theta_1$ (AT-L∞ Model):** Instead of training from scratch, we download a strong, standardized baseline L∞ adversarially trained model (`Carmon2019Unlabeled` for WideResNet-28-10 on CIFAR-10) directly from **RobustBench**.
- **$\theta_2$ (AT-L1 Model):** We generate the second endpoint by fine-tuning the $\theta_1$ model against L1 perturbations (ε=12.0) for 10 epochs, matching the explicit instructions in Section 4.1 of the paper.

### 2. Quadratic Bezier Curve Path (`mode_connectivity.py`)
Mode connectivity is established using a Quadratic Bezier Curve (QBC) formulation. 
Instead of duplicating layer parameters as done in the author's original `CurveNet`, we use PyTorch's `torch.func.functional_call`. This approach is mathematically equivalent but much cleaner, as it allows us to interpolate state dictionaries dynamically and inject them into standard model architectures without modifying the base classes.

### 3. Multi-Steepest Descent (MSD) Attack (`attacks.py`)
- We implemented two variants of the MSD attack to strictly match the paper:
  - **Training (`msd_train_attack`):** At each iteration, the attack calculates only an L∞ step and an L1 step, and greedily selects the worst-case perturbation that maximizes the cross-entropy loss. This aligns perfectly with Algorithm 1, which strictly defines the inner loop maximization over $p \in \{1, \infty\}$.
  - **Evaluation (`msd_eval_attack`):** Includes L∞, L2, and L1 perturbations in the maximization loop to match the evaluation protocol described in Section 5 (where models are assessed against a 3-step MSD).
- The L1 projection utilizes the exact row-by-row Duchi's algorithm found in the author's reference code (`proj_simplex` / `proj_l1ball`).

### 4. ERMC Path Training (`train_ermc.py`)
Following Algorithm 1:
- We train a control point $\theta$ for 50 epochs.
- In each batch, a single $t \sim U(0,1)$ is sampled uniformly.
- The model at the interpolated state $\phi_\theta(t)$ is evaluated against an MSD attack.
- Gradients are backpropagated solely to the control point parameters.

### 5. Evaluation Loop (`evaluate.py`)
The evaluation strictly replicates the methodology from Section 5 (Table 1) of the paper, testing against multiple metrics:
- **SA (Standard Accuracy):** Clean test data performance.
- **PGD-L∞:** ε=8/255
- **PGD-L2:** ε=1.0
- **PGD-L1:** ε=12.0
- **MSD Attack:** Same bounds as above.
- **Union Accuracy:** A sample is considered correct *only* if it survives all three primary PGD attacks (L∞, L2, L1).
- **AutoAttack (AA):** Uses the `autoattack` library for a stronger, standardized evaluation.

**Modes of Evaluation:**
- `--mode path`: Scans $t \in [0, 1]$ (Fig. 1 in the paper) to evaluate the robust landscape.
- `--mode single`: Selects the single best model on the path (ERMC-1).
- `--mode ensemble`: Automatically finds a continuous path segment where accuracies meet defined thresholds (α_∞=37%, α_1=43%), selects `n` models uniformly, and ensembles them by averaging logits (ERMC-n).

## File Structure
- `requirements.txt`: Python dependencies (includes `robustbench` and `autoattack`).
- `dataset.py`: Helpers for automatic downloading and processing of CIFAR-10/CIFAR-100 without normalization bounds, ensuring perturbations are strictly in the [0, 1] pixel space.
- `attacks.py`: PGD-L∞, PGD-L2, PGD-L1 (sparse), and MSD attack implementations.
- `mode_connectivity.py`: Interpolation logic for the Bezier Curve and standardizing model calls.
- `train_endpoints.py`: Pipeline for preparing $\theta_1$ and $\theta_2$.
- `train_ermc.py`: Optimization script for the ERMC control point $\theta$.
- `evaluate.py`: Evaluation metrics to generate Table-1-style results.
- `run.sh`: Bash script to run the end-to-end pipeline automatically.
