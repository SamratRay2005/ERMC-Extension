"""
train_ermc.py — Algorithm 1: Efficient Robust Mode Connectivity.

Optimises the Bézier control point θ along the QBC path between θ₁ (AT-L∞)
and θ₂ (AT-L₁).  For each batch, t ~ U(0,1) is sampled, the model at ϕ_θ(t)
is attacked with MSD, and the control point θ is updated via SGD.

Hyperparameters from the paper:
    • 50 epochs for path optimisation
    • MSD inner loop: J = 10 steps, ε_∞ = 8/255, ε₁ = 12
    • SGD with lr = 0.01, momentum = 0.9, wd = 5e-4
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from robustbench.utils import load_model
from dataset import get_cifar10_dataloaders
from attacks import msd_train_attack
from mode_connectivity import BezierPath, BezierWrapper


def parse_args():
    p = argparse.ArgumentParser(description='ERMC Path Training')
    p.add_argument('--data-dir', type=str, default='./data')
    p.add_argument('--save-dir', type=str, default='./models')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--epsilon-inf', type=float, default=8/255)
    p.add_argument('--epsilon-1', type=float, default=12.0)
    p.add_argument('--alpha-inf', type=float, default=2/255)
    p.add_argument('--alpha-1', type=float, default=1.0)
    p.add_argument('--msd-steps', type=int, default=10)
    p.add_argument('--resume', type=str, default=None,
                   help='Resume ERMC training from an epoch checkpoint')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'Using device: {device}')

    train_loader, _ = get_cifar10_dataloaders(
        batch_size=args.batch_size, data_dir=args.data_dir
    )

    # ── Load base model architecture + endpoint state dicts ─────────────
    print('Loading base model architecture from RobustBench...')
    base_model = load_model(
        model_name='Madry2018',
        dataset='cifar10',
        threat_model='Linf',
    )
    base_model = base_model.to(device)

    theta1_path = os.path.join(args.save_dir, 'theta1_linf.pt')
    theta2_path = os.path.join(args.save_dir, 'theta2_l1.pt')
    print(f'Loading θ₁ from {theta1_path}')
    print(f'Loading θ₂ from {theta2_path}')

    theta1_state = torch.load(theta1_path, map_location=device, weights_only=True)
    theta2_state = torch.load(theta2_path, map_location=device, weights_only=True)

    # ── Create Bézier path ──────────────────────────────────────────────
    bezier = BezierPath(base_model, theta1_state, theta2_state, device)

    # Optimise only the control point parameters
    optimizer = optim.SGD(
        bezier.control_parameters(), lr=args.lr,
        momentum=0.9, weight_decay=5e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    start_epoch = 0

    if args.resume:
        print(f'\nResuming ERMC training from {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        with torch.no_grad():
            for key, value in checkpoint['control_state'].items():
                bezier.theta[key].copy_(value.to(device))
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch']
        print(f'Resuming at epoch {start_epoch + 1}/{args.epochs}')

    # ── Training loop (Algorithm 1) ─────────────────────────────────────
    print(f'\nStarting ERMC optimisation ({args.epochs} epochs)...')
    checkpoint_path = os.path.join(args.save_dir, 'ermc_checkpoint.pt')
    for epoch in range(start_epoch, args.epochs):
        total_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f'ERMC Epoch {epoch+1}/{args.epochs}')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            # Sample t ~ U(0, 1) per batch
            t = torch.rand(1).item()

            # Wrap the Bezier path at this t for the attack function
            t_model = BezierWrapper(bezier, t)
            t_model.eval()

            # MSD attack at ϕ_θ(t)
            delta = msd_train_attack(
                t_model, inputs, targets,
                epsilon_inf=args.epsilon_inf, epsilon_1=args.epsilon_1,
                alpha_inf=args.alpha_inf, alpha_1=args.alpha_1,
                steps=args.msd_steps,
            )
            adv_inputs = torch.clamp(inputs + delta, 0, 1)

            # Restore training behavior for the control-point update.
            t_model.train()

            # Forward through the path at t and update θ
            optimizer.zero_grad()
            outputs = bezier.forward(adv_inputs, t)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = outputs.max(1)
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()

            pbar.set_postfix(
                loss=f'{total_loss/(pbar.n+1):.3f}',
                acc=f'{100.*correct/total:.1f}%',
                t=f'{t:.2f}',
            )

        scheduler.step()

        torch.save({
            'epoch': epoch + 1,
            'control_state': {
                key: value.detach().clone() for key, value in bezier.theta.items()
            },
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
        }, checkpoint_path)
        print(f'  Checkpoint saved -> {checkpoint_path}')

    # ── Save the control-point parameters ───────────────────────────────
    ctrl_path = os.path.join(args.save_dir, 'theta_control.pt')
    torch.save(
        {k: v.data for k, v in bezier.theta.items()},
        ctrl_path,
    )
    print(f'\nControl point θ saved → {ctrl_path}')
    print('ERMC training complete.')


if __name__ == '__main__':
    main()
