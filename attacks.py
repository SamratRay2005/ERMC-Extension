"""
attacks.py — PGD-L∞, PGD-L2, PGD-L1, and Multi Steepest Descent (MSD).

Follows the evaluation protocol described in the EMRC paper (Section 5):
  • ε_∞ = 8/255,  ε_2 = 1,  ε_1 = 12
  • PGD with J = 10 inner steps during training
  • PGD-20 or PGD-50 during evaluation
  • MSD combines L∞ and L1 steepest descent steps per iteration.

The L1 projection uses Duchi et al.'s algorithm (same approach as the
author's MoCo-EA codebase: mocoea/attacks.py :: project_l1_ball).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────── projections ──────────────────────────────

def project_linf(delta, epsilon):
    """Project onto L-inf ball."""
    return torch.clamp(delta, -epsilon, epsilon)


def project_l2(delta, epsilon):
    """Project onto L2 ball (per-sample)."""
    flat = delta.view(delta.size(0), -1)
    norms = flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
    scale = torch.clamp(norms / epsilon, min=1.0)
    return (flat / scale).view_as(delta)


def project_l1(delta, epsilon):
    """
    Project each sample in *delta* onto the L1 ball of radius *epsilon*.
    Uses Duchi et al. (row-by-row) — matches the author's reference code.
    """
    original_shape = delta.shape
    x = delta.view(delta.size(0), -1)

    projected = []
    for i in range(x.size(0)):
        v = x[i]
        u = torch.abs(v)
        if u.sum() <= epsilon:
            projected.append(v)
            continue

        u_sorted, _ = torch.sort(u, descending=True)
        cumsum = torch.cumsum(u_sorted, dim=0)
        k = torch.arange(1, len(u_sorted) + 1, device=x.device, dtype=torch.float32)

        condition = u_sorted > (cumsum - epsilon) / k
        rho = len(condition) - torch.flip(condition, [0]).long().argmax().item()
        theta = (cumsum[rho - 1] - epsilon) / rho if rho > 0 else 0
        projected.append(torch.sign(v) * torch.clamp(torch.abs(v) - theta, min=0))

    return torch.stack(projected).view(original_shape)


def l1_steepest_step(delta, grad, alpha, k=50):
    """
    Take a sparse steepest-descent step for the L1 constraint.
    Upgraded to Top-k coordinates to prevent gradient masking.
    """
    batch_size = grad.size(0)
    flat_grad = grad.view(batch_size, -1)

    # Find the top-k absolute gradient values instead of just the single max.
    _, topk_indices = torch.topk(torch.abs(flat_grad), k, dim=1)

    # Create a sparse mask for all k coordinates.
    sparse_mask = torch.zeros_like(flat_grad).scatter_(1, topk_indices, 1.0)
    sparse_mask = sparse_mask.view_as(grad)

    # Distribute the L1 step size evenly across the k coordinates.
    distributed_alpha = alpha / k

    return delta + distributed_alpha * (sparse_mask * grad.sign())


# ────────────────────────────── PGD attacks ──────────────────────────────

def pgd_linf(model, x, y, epsilon=8/255, alpha=2/255, steps=20, random_start=True):
    """Standard PGD-L∞ attack."""
    model.eval()
    delta = torch.zeros_like(x)
    if random_start:
        delta.uniform_(-epsilon, epsilon)
        delta = torch.clamp(x + delta, 0, 1) - x

    for _ in range(steps):
        delta.requires_grad_(True)
        loss = F.cross_entropy(model(x + delta), y)
        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta = delta + alpha * grad.sign()
            delta = project_linf(delta, epsilon)
            delta = torch.clamp(x + delta, 0, 1) - x
        delta = delta.detach()
    return delta


def pgd_l2(model, x, y, epsilon=1.0, alpha=0.25, steps=20, random_start=True):
    """Standard PGD-L2 attack."""
    model.eval()
    delta = torch.zeros_like(x)
    if random_start:
        delta = torch.randn_like(x)
        flat = delta.view(delta.size(0), -1)
        norms = flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
        delta = delta / norms.view(delta.size(0), 1, 1, 1) * (
            torch.rand(delta.size(0), 1, 1, 1, device=x.device) * epsilon
        )
        delta = torch.clamp(x + delta, 0, 1) - x

    for _ in range(steps):
        delta.requires_grad_(True)
        loss = F.cross_entropy(model(x + delta), y)
        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            grad_norm = grad.view(grad.size(0), -1).norm(p=2, dim=1, keepdim=True).view(-1, 1, 1, 1).clamp(min=1e-12)
            delta = delta + alpha * grad / grad_norm
            delta = project_l2(delta, epsilon)
            delta = torch.clamp(x + delta, 0, 1) - x
        delta = delta.detach()
    return delta


def pgd_l1(model, x, y, epsilon=12.0, alpha=1.0, steps=20, random_start=True):
    """
    PGD-L1 attack.
    The steepest descent direction for L1 is sparse: sign(grad) at the
    coordinate(s) of maximum absolute gradient value.
    """
    model.eval()
    delta = torch.zeros_like(x)
    if random_start:
        # sparse random init, like author's code
        mask = torch.rand_like(x) < 0.1
        delta[mask] = torch.empty(mask.sum(), device=x.device).uniform_(
            -epsilon / 10, epsilon / 10
        )
        delta = project_l1(delta, epsilon)
        delta = torch.clamp(x + delta, 0, 1) - x

    for _ in range(steps):
        delta.requires_grad_(True)
        loss = F.cross_entropy(model(x + delta), y)
        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            # L1 steepest-descent: step in sign(grad) only at the max-abs coordinate
            delta = l1_steepest_step(delta, grad, alpha)
            delta = project_l1(delta, epsilon)
            delta = torch.clamp(x + delta, 0, 1) - x
        delta = delta.detach()
    return delta


# ───────────────────── Multi Steepest Descent (MSD) ─────────────────────

def msd_train_attack(model, x, y,
               epsilon_inf=8/255, epsilon_1=12.0,
               alpha_inf=2/255, alpha_1=1.0,
               steps=10):
    """
    Multi Steepest Descent (MSD) — Section 4.1 / Algorithm 1 inner loop.

    At each iteration, compute *both* an L∞ step and an L1 step, then
    pick whichever yields a higher cross-entropy loss (per sample).
    """
    model.eval()
    delta = torch.zeros_like(x)

    for _ in range(steps):
        delta.requires_grad_(True)
        loss = F.cross_entropy(model(x + delta), y)
        grad = torch.autograd.grad(loss, delta)[0]

        with torch.no_grad():
            # ── L∞ candidate ──
            d_inf = delta + alpha_inf * grad.sign()
            d_inf = project_linf(d_inf, epsilon_inf)
            d_inf = torch.clamp(x + d_inf, 0, 1) - x

            # ── L1 candidate ──
            d_l1 = l1_steepest_step(delta, grad, alpha_1)
            d_l1 = project_l1(d_l1, epsilon_1)
            d_l1 = torch.clamp(x + d_l1, 0, 1) - x

            # ── pick worst case (highest loss) per sample ──
            loss_inf = F.cross_entropy(model(x + d_inf), y, reduction='none')
            loss_l1 = F.cross_entropy(model(x + d_l1), y, reduction='none')

            cond = (loss_l1 > loss_inf).view(-1, 1, 1, 1).expand_as(delta)
            delta = torch.where(cond, d_l1, d_inf).detach()

    return delta


def msd_eval_attack(model, x, y,
                    epsilon_inf=8/255, epsilon_2=1.0, epsilon_1=12.0,
                    alpha_inf=2/255, alpha_2=0.25, alpha_1=1.0,
                    steps=10):
    """
    Multi Steepest Descent (MSD) for Evaluation — Section 5.

    At each iteration, compute an L∞, L2, and L1 step, then
    pick whichever yields the highest cross-entropy loss (per sample).
    """
    model.eval()
    delta = torch.zeros_like(x)

    for _ in range(steps):
        delta.requires_grad_(True)
        loss = F.cross_entropy(model(x + delta), y)
        grad = torch.autograd.grad(loss, delta)[0]

        with torch.no_grad():
            # ── L∞ candidate ──
            d_inf = delta + alpha_inf * grad.sign()
            d_inf = project_linf(d_inf, epsilon_inf)
            d_inf = torch.clamp(x + d_inf, 0, 1) - x

            # ── L2 candidate ──
            grad_norm = grad.view(grad.size(0), -1).norm(p=2, dim=1, keepdim=True).view(-1, 1, 1, 1).clamp(min=1e-12)
            d_l2 = delta + alpha_2 * grad / grad_norm
            d_l2 = project_l2(d_l2, epsilon_2)
            d_l2 = torch.clamp(x + d_l2, 0, 1) - x

            # ── L1 candidate ──
            d_l1 = l1_steepest_step(delta, grad, alpha_1)
            d_l1 = project_l1(d_l1, epsilon_1)
            d_l1 = torch.clamp(x + d_l1, 0, 1) - x

            # ── pick worst case (highest loss) per sample ──
            loss_inf = F.cross_entropy(model(x + d_inf), y, reduction='none')
            loss_l2 = F.cross_entropy(model(x + d_l2), y, reduction='none')
            loss_l1 = F.cross_entropy(model(x + d_l1), y, reduction='none')

            max_loss = torch.max(loss_inf, torch.max(loss_l2, loss_l1))
            
            cond_l1 = (loss_l1 == max_loss).view(-1, 1, 1, 1).expand_as(delta)
            cond_l2 = (loss_l2 == max_loss).view(-1, 1, 1, 1).expand_as(delta)
            
            delta = torch.where(cond_l1, d_l1, torch.where(cond_l2, d_l2, d_inf)).detach()

    return delta
