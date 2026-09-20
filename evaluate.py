"""
evaluate.py — Full evaluation as described in Section 5 / Table 1 of the paper.

Metrics (evaluated on CIFAR-10 test set):
    ① SA      — Standard (clean) accuracy
    ② PGD-L∞  — PGD attack under ε_∞ = 8/255
    ③ PGD-L₂  — PGD attack under ε₂  = 1
    ④ PGD-L₁  — PGD attack under ε₁  = 12
    ⑤ MSD     — Multi-Steepest-Descent attack
    ⑥ Union   — Sample-wise worst-case of PGD-L∞, PGD-L₂, PGD-L₁
    ⑦ AA-L∞   — AutoAttack under ε_∞ = 8/255  (optional, slow)
    ⑧ AA-L₂   — AutoAttack under ε₂  = 1       (optional, slow)
    ⑨ AA-L₁   — AutoAttack under ε₁  = 12      (optional, slow)

Modes:
    --mode path     : evaluate every t ∈ [0,1] along the Bézier path
    --mode single   : evaluate the best single model (ERMC-1)
    --mode ensemble : evaluate ERMC-n ensemble (n models along the path)
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from robustbench.utils import load_model
from dataset import get_cifar10_dataloaders
from attacks import pgd_linf, pgd_l2, pgd_l1, msd_eval_attack
from mode_connectivity import BezierPath, BezierWrapper


# ───────────────────────── helpers ──────────────────────────────────────

def clean_accuracy(model, loader, device):
    """Standard accuracy on clean test data."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += preds.eq(y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


def robust_accuracy(model, loader, device, attack_fn, **attack_kwargs):
    """Robust accuracy under a given PGD attack."""
    model.eval()
    correct, total = 0, 0
    for x, y in tqdm(loader, desc=f'  {attack_fn.__name__}', leave=False):
        x, y = x.to(device), y.to(device)
        delta = attack_fn(model, x, y, **attack_kwargs)
        adv_x = torch.clamp(x + delta, 0, 1)
        preds = model(adv_x).argmax(1)
        correct += preds.eq(y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def union_accuracy(model, loader, device,
                   eps_inf=8/255, eps_2=1.0, eps_1=12.0, steps=20):
    """
    Union accuracy: a sample is correct only if it survives ALL three
    basic PGD attacks (L∞, L₂, L₁).
    """
    model.eval()
    correct, total = 0, 0
    for x, y in tqdm(loader, desc='  Union', leave=False):
        x, y = x.to(device), y.to(device)

        d_inf = pgd_linf(model, x, y, epsilon=eps_inf, alpha=eps_inf/4, steps=steps)
        d_l2 = pgd_l2(model, x, y, epsilon=eps_2, alpha=eps_2/4, steps=steps)
        d_l1 = pgd_l1(model, x, y, epsilon=eps_1, alpha=1.0, steps=steps)

        p_inf = model(torch.clamp(x + d_inf, 0, 1)).argmax(1).eq(y)
        p_l2 = model(torch.clamp(x + d_l2, 0, 1)).argmax(1).eq(y)
        p_l1 = model(torch.clamp(x + d_l1, 0, 1)).argmax(1).eq(y)

        correct += (p_inf & p_l2 & p_l1).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def msd_accuracy(model, loader, device,
                 eps_inf=8/255, eps_2=1.0, eps_1=12.0, steps=20):
    """Robust accuracy under the MSD attack."""
    model.eval()
    correct, total = 0, 0
    for x, y in tqdm(loader, desc='  MSD', leave=False):
        x, y = x.to(device), y.to(device)
        delta = msd_eval_attack(model, x, y,
                           epsilon_inf=eps_inf, epsilon_2=eps_2, epsilon_1=eps_1,
                           alpha_inf=2/255, alpha_2=0.25, alpha_1=1.0, steps=steps)
        preds = model(torch.clamp(x + delta, 0, 1)).argmax(1)
        correct += preds.eq(y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def autoattack_accuracy(model, loader, device, norm, eps):
    """Run AutoAttack (Croce & Hein, 2020)."""
    try:
        from autoattack import AutoAttack
    except ImportError:
        print('  ⚠  autoattack not installed — skipping AA evaluation')
        return None

    # Collect full test set
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    x_test = torch.cat(xs).to(device)
    y_test = torch.cat(ys).to(device)

    adversary = AutoAttack(model, norm=norm, eps=eps, version='standard')
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=256)
    correct = model(x_adv).argmax(1).eq(y_test).sum().item()
    return 100.0 * correct / len(y_test)


# ───────────────────────── evaluation runners ──────────────────────────

def evaluate_model(model, loader, device, run_aa=False, label='Model'):
    """Full evaluation of a single model — prints Table-1-style results."""
    print(f'\n{"="*60}')
    print(f'  Evaluating: {label}')
    print(f'{"="*60}')

    sa = clean_accuracy(model, loader, device)
    print(f'  SA (clean)   : {sa:.2f}%')

    r_inf = robust_accuracy(model, loader, device, pgd_linf,
                            epsilon=8/255, alpha=2/255, steps=20)
    print(f'  PGD-L∞       : {r_inf:.2f}%')

    r_l2 = robust_accuracy(model, loader, device, pgd_l2,
                           epsilon=1.0, alpha=0.25, steps=20)
    print(f'  PGD-L₂       : {r_l2:.2f}%')

    r_l1 = robust_accuracy(model, loader, device, pgd_l1,
                           epsilon=12.0, alpha=1.0, steps=20)
    print(f'  PGD-L₁       : {r_l1:.2f}%')

    r_msd = msd_accuracy(model, loader, device, steps=20)
    print(f'  MSD          : {r_msd:.2f}%')

    r_union = union_accuracy(model, loader, device)
    print(f'  Union        : {r_union:.2f}%')

    results = dict(SA=sa, PGD_Linf=r_inf, PGD_L2=r_l2, PGD_L1=r_l1,
                   MSD=r_msd, Union=r_union)

    if run_aa:
        aa_inf = autoattack_accuracy(model, loader, device, 'Linf', 8/255)
        aa_l2 = autoattack_accuracy(model, loader, device, 'L2', 1.0)
        aa_l1 = autoattack_accuracy(model, loader, device, 'L1', 12.0)
        for name, val in [('AA-L∞', aa_inf), ('AA-L₂', aa_l2), ('AA-L₁', aa_l1)]:
            if val is not None:
                print(f'  {name:13s}: {val:.2f}%')
                results[name] = val

    return results


def evaluate_path(bezier, loader, device, num_points=21, run_aa=False):
    """Evaluate every point along the Bézier path (Fig. 1 of the paper)."""
    ts = np.linspace(0.0, 1.0, num_points)
    all_results = []

    for t_val in ts:
        model_t = BezierWrapper(bezier, t_val)
        model_t.eval()
        res = evaluate_model(model_t, loader, device,
                             run_aa=run_aa, label=f't = {t_val:.2f}')
        res['t'] = t_val
        all_results.append(res)

    # Print summary table
    print(f'\n{"="*80}')
    print('  Path Summary')
    print(f'{"="*80}')
    header = f'{"t":>5s} {"SA":>7s} {"PGD-L∞":>8s} {"PGD-L₂":>8s} {"PGD-L₁":>8s} {"MSD":>7s} {"Union":>7s}'
    print(header)
    print('-' * len(header))
    for r in all_results:
        print(f'{r["t"]:5.2f} {r["SA"]:7.2f} {r["PGD_Linf"]:8.2f} '
              f'{r["PGD_L2"]:8.2f} {r["PGD_L1"]:8.2f} '
              f'{r["MSD"]:7.2f} {r["Union"]:7.2f}')

    return all_results


def evaluate_ensemble(bezier, loader, device, n_models=5,
                      alpha_inf=37.0, alpha_1=43.0, num_probe=101,
                      run_aa=False):
    """
    ERMC-n ensemble selection (Section 4.2).

    1. Scan the path to find points where both RA_L∞ > α_∞ and RA_L₁ > α₁.
    2. Group valid points into contiguous segments and distribute n models
       across them in proportion to their lengths.
    3. Average their logits for prediction.
    """
    print('\n' + '='*60)
    print(f'  ERMC Ensemble Selection (n={n_models})')
    print('='*60)

    # ── quick scan of the path ──────────────────────────────────────────
    ts = np.linspace(0.0, 1.0, num_probe)
    ra_inf_vals = []
    ra_l1_vals = []

    print('  Scanning path for robust accuracy thresholds...')
    for t_val in tqdm(ts, desc='  Path scan'):
        model_t = BezierWrapper(bezier, t_val)
        model_t.eval()
        r_inf = robust_accuracy(model_t, loader, device, pgd_linf,
                                epsilon=8/255, alpha=2/255, steps=10)
        r_l1 = robust_accuracy(model_t, loader, device, pgd_l1,
                               epsilon=12.0, alpha=1.0, steps=10)
        ra_inf_vals.append(r_inf)
        ra_l1_vals.append(r_l1)

    ra_inf_vals = np.array(ra_inf_vals)
    ra_l1_vals = np.array(ra_l1_vals)

    # Find segment where both thresholds are met
    valid = (ra_inf_vals >= alpha_inf) & (ra_l1_vals >= alpha_1)
    valid_ts = ts[valid]

    if len(valid_ts) == 0:
        print('  ⚠  No segment meets both thresholds! Relaxing thresholds...')
        # Fallback: find t that maximises min(RA_L∞, RA_L₁)
        worst_case = np.minimum(ra_inf_vals, ra_l1_vals)
        best_t = ts[np.argmax(worst_case)]
        selected_ts = [best_t]
        print(f'  Using single best t = {best_t:.3f}')
    else:
        step = ts[1] - ts[0] if len(ts) > 1 else 0.0
        segments = []
        current_segment = [valid_ts[0]]
        for t_val in valid_ts[1:]:
            if np.isclose(t_val - current_segment[-1], step):
                current_segment.append(t_val)
            else:
                segments.append(np.array(current_segment))
                current_segment = [t_val]
        segments.append(np.array(current_segment))

        print(f'  Valid segments: {[(segment[0], segment[-1]) for segment in segments]}')
        if n_models == 1:
            # ERMC-1: pick the valid t that maximises min(RA_L∞, RA_L₁).
            worst_case = np.minimum(ra_inf_vals[valid], ra_l1_vals[valid])
            selected_ts = [valid_ts[np.argmax(worst_case)]]
        else:
            segment_lengths = np.array([
                max(segment[-1] - segment[0], step) for segment in segments
            ])
            raw_counts = n_models * segment_lengths / segment_lengths.sum()
            counts = np.floor(raw_counts).astype(int)
            if n_models >= len(segments):
                counts = np.maximum(counts, 1)

            while counts.sum() < n_models:
                remainders = raw_counts - counts
                counts[np.argmax(remainders)] += 1
            while counts.sum() > n_models:
                removable = np.where(counts > (1 if n_models >= len(segments) else 0))[0]
                counts[removable[np.argmin(raw_counts[removable] - counts[removable])]] -= 1

            selected_ts = []
            for segment, count in zip(segments, counts):
                if count == 1:
                    selected_ts.append(segment[len(segment) // 2])
                else:
                    selected_ts.extend(np.linspace(segment[0], segment[-1], count))
            selected_ts = np.asarray(selected_ts).tolist()

    print(f'  Selected t values: {[f"{t:.3f}" for t in selected_ts]}')

    # ── Evaluate ensemble ───────────────────────────────────────────────
    print('  Evaluating ensemble...')
    _, test_loader = get_cifar10_dataloaders(batch_size=128)

    # Build list of models
    models = []
    for t_val in selected_ts:
        m = BezierWrapper(bezier, t_val)
        m.eval()
        models.append(m)

    # Ensemble evaluation
    def ensemble_forward(x):
        """Average the logits from all models in the ensemble."""
        logits = torch.stack([m(x) for m in models], dim=0)
        return logits.mean(dim=0)

    # Create a wrapper for the ensemble
    class EnsembleModel(nn.Module):
        def __init__(self, models_list):
            super().__init__()
            self.models = models_list

        def forward(self, x):
            logits = torch.stack([m(x) for m in self.models], dim=0)
            return logits.mean(dim=0)

    ensemble = EnsembleModel(models)
    results = evaluate_model(ensemble, test_loader, device,
                             run_aa=run_aa,
                             label=f'ERMC-{len(selected_ts)}')
    return results


# ───────────────────────── main ────────────────────────────────────────

def parse_eval_args():
    p = argparse.ArgumentParser(description='ERMC Evaluation')
    p.add_argument('--data-dir', type=str, default='./data')
    p.add_argument('--save-dir', type=str, default='./models')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--mode', type=str, default='single',
                   choices=['path', 'single', 'ensemble'],
                   help='Evaluation mode')
    p.add_argument('--n-models', type=int, default=5,
                   help='Number of ensemble models (ERMC-n)')
    p.add_argument('--num-path-points', type=int, default=21,
                   help='Number of t values for path evaluation')
    p.add_argument('--run-aa', action='store_true',
                   help='Run AutoAttack (slow)')
    p.add_argument('--alpha-inf', type=float, default=37.0,
                   help='L∞ robustness threshold for ensemble selection (%%)')
    p.add_argument('--alpha-1', type=float, default=43.0,
                   help='L₁ robustness threshold for ensemble selection (%%)')
    return p.parse_args()


def main():
    args = parse_eval_args()
    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'Using device: {device}')

    _, test_loader = get_cifar10_dataloaders(
        batch_size=args.batch_size, data_dir=args.data_dir
    )

    # ── Reconstruct the Bézier path ─────────────────────────────────────
    print('Loading base model and endpoints...')
    base_model = load_model(
        model_name='Engstrom2019Robustness',
        dataset='cifar10',
        threat_model='Linf',
    )
    base_model = base_model.to(device)

    theta1_state = torch.load(
        os.path.join(args.save_dir, 'theta1_linf.pt'),
        map_location=device, weights_only=True,
    )
    theta2_state = torch.load(
        os.path.join(args.save_dir, 'theta2_l1.pt'),
        map_location=device, weights_only=True,
    )

    bezier = BezierPath(base_model, theta1_state, theta2_state, device)

    # Load the trained control point
    ctrl_state = torch.load(
        os.path.join(args.save_dir, 'theta_control.pt'),
        map_location=device, weights_only=True,
    )
    for k, v in ctrl_state.items():
        if k in bezier.theta:
            bezier.theta[k].data.copy_(v)
    print('Control point loaded.')

    # ── Run evaluation ──────────────────────────────────────────────────
    if args.mode == 'path':
        evaluate_path(bezier, test_loader, device,
                      num_points=args.num_path_points,
                      run_aa=args.run_aa)

    elif args.mode == 'single':
        evaluate_ensemble(bezier, test_loader, device,
                          n_models=1,
                          alpha_inf=args.alpha_inf,
                          alpha_1=args.alpha_1,
                          run_aa=args.run_aa)

    elif args.mode == 'ensemble':
        evaluate_ensemble(bezier, test_loader, device,
                          n_models=args.n_models,
                          alpha_inf=args.alpha_inf,
                          alpha_1=args.alpha_1,
                          run_aa=args.run_aa)


if __name__ == '__main__':
    main()
