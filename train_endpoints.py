"""
train_endpoints.py — Generate the two endpoint models for ERMC.

Step 1:  θ₁ = AT-L∞  → downloaded from RobustBench (Carmon2019_unlabeled,
         WideResNet-28-10, CIFAR-10, ε_∞ = 8/255).

Step 2:  θ₂ = fine-tune θ₁ with AT-L₁ for 10 epochs (Section 4.1).
         "the number of fine-tuning epochs is set to 10"
"""

import os
import copy
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from robustbench.utils import load_model
from dataset import get_cifar10_dataloaders
from attacks import pgd_linf, pgd_l1


def parse_args():
    p = argparse.ArgumentParser(description='ERMC Endpoint Generation')
    p.add_argument('--data-dir', type=str, default='./data')
    p.add_argument('--save-dir', type=str, default='./models')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--finetune-epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--epsilon-1', type=float, default=12.0)
    p.add_argument('--alpha-1', type=float, default=1.0)
    p.add_argument('--pgd-steps', type=int, default=10,
                   help='J=10 inner PGD steps during AT (paper)')
    p.add_argument('--resume', type=str, default=None,
                   help='Resume fine-tuning from an epoch checkpoint')
    return p.parse_args()


def robust_accuracy(model, loader, device, attack_fn, **attack_kwargs):
    """Measure accuracy after one configured attack over a data loader."""
    model.eval()
    correct, total = 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        delta = attack_fn(model, inputs, targets, **attack_kwargs)
        predictions = model(torch.clamp(inputs + delta, 0, 1)).argmax(1)
        correct += predictions.eq(targets).sum().item()
        total += targets.size(0)
    return 100.0 * correct / total


def main():
    args = parse_args()
    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'Using device: {device}')

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Step 1: Load pretrained AT-L∞ model (θ₁) ────────────────────────
    print('Downloading AT-L∞ model from RobustBench (Carmon2019_unlabeled)...')
    model = load_model(
        model_name='Carmon2019Unlabeled',
        dataset='cifar10',
        threat_model='Linf',
    )
    model = model.to(device)

    theta1_path = os.path.join(args.save_dir, 'theta1_linf.pt')
    torch.save(model.state_dict(), theta1_path)
    print(f'θ₁ (AT-L∞) saved → {theta1_path}')

    # ── Step 2: Fine-tune with AT-L₁ for 10 epochs (θ₂) ────────────────
    print(f'\nFine-tuning with AT-L₁ for {args.finetune_epochs} epochs...')
    train_loader, test_loader = get_cifar10_dataloaders(
        batch_size=args.batch_size, data_dir=args.data_dir
    )

    print('\nRaw model robust accuracy before L1 fine-tuning:')
    raw_linf = robust_accuracy(
        model, test_loader, device, pgd_linf,
        epsilon=8/255, alpha=2/255, steps=20,
    )
    raw_l1 = robust_accuracy(
        model, test_loader, device, pgd_l1,
        epsilon=args.epsilon_1, alpha=args.alpha_1,
        steps=20, random_start=True,
    )
    print(f'  PGD-Linf: {raw_linf:.2f}%')
    print(f'  PGD-L1  : {raw_l1:.2f}%')

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.finetune_epochs
    )
    criterion = nn.CrossEntropyLoss()
    start_epoch = 0

    if args.resume:
        print(f'\nResuming endpoint fine-tuning from {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch']
        print(f'Resuming at epoch {start_epoch + 1}/{args.finetune_epochs}')

    checkpoint_path = os.path.join(args.save_dir, 'endpoint_checkpoint.pt')
    for epoch in range(start_epoch, args.finetune_epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader,
                    desc=f'AT-L₁ Epoch {epoch+1}/{args.finetune_epochs}')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            # generate L₁ adversarial examples
            delta = pgd_l1(
                model, inputs, targets,
                epsilon=args.epsilon_1, alpha=args.alpha_1,
                steps=args.pgd_steps, random_start=True,
            )
            adv_inputs = torch.clamp(inputs + delta, 0, 1)

            model.train()
            optimizer.zero_grad()
            outputs = model(adv_inputs)
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
            )

        scheduler.step()

        torch.save({
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
        }, checkpoint_path)
        print(f'  Checkpoint saved -> {checkpoint_path}')

    theta2_path = os.path.join(args.save_dir, 'theta2_l1.pt')
    torch.save(model.state_dict(), theta2_path)
    print(f'θ₂ (AT-L₁ fine-tuned) saved → {theta2_path}')
    print('\nEndpoint generation complete.')


if __name__ == '__main__':
    main()
