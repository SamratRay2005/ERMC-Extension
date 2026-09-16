#!/bin/bash
set -e

echo "============================================"
echo "  ERMC — Reproducing arXiv:2407.09251"
echo "============================================"

echo ""
echo "Step 0: Installing requirements..."
pip install -r requirements.txt

echo ""
echo "Step 1: Downloading dataset + AT-L∞ model, then fine-tuning for AT-L₁..."
python train_endpoints.py --data-dir ./data --save-dir ./models

echo ""
echo "Step 2: Training ERMC path (50 epochs)..."
python train_ermc.py --data-dir ./data --save-dir ./models --epochs 50

echo ""
echo "Step 3a: Evaluating path (Fig. 1 — accuracy vs t)..."
python evaluate.py --data-dir ./data --save-dir ./models --mode path --num-path-points 21

echo ""
echo "Step 3b: Evaluating ERMC-1 (Table 1 — single best model)..."
python evaluate.py --data-dir ./data --save-dir ./models --mode single

echo ""
echo "Step 3c: Evaluating ERMC-5 (Table 1 — 5-model ensemble)..."
python evaluate.py --data-dir ./data --save-dir ./models --mode ensemble --n-models 5

echo ""
echo "Done!"
