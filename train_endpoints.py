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
from attacks import pgd_l1


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
    return p.parse_args()


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
    train_loader, _ = get_cifar10_dataloaders(
        batch_size=args.batch_size, data_dir=args.data_dir
    )

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.finetune_epochs
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.finetune_epochs):
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

    theta2_path = os.path.join(args.save_dir, 'theta2_l1.pt')
    torch.save(model.state_dict(), theta2_path)
    print(f'θ₂ (AT-L₁ fine-tuned) saved → {theta2_path}')
    print('\nEndpoint generation complete.')


if __name__ == '__main__':
    main()
