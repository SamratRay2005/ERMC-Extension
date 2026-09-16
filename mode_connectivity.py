"""
mode_connectivity.py — Quadratic Bézier Curve (QBC) path interpolation.

From the paper (Eq. 5):
    ϕ_θ(t) = (1−t)² θ₁ + 2t(1−t) θ + t² θ₂

θ₁ and θ₂ are frozen endpoints; θ (the control point) is optimised.
We store only θ as nn.Parameters and compute ϕ_θ(t) on-the-fly via
`torch.func.functional_call` so the base model architecture is unchanged.
"""

import copy
import torch
import torch.nn as nn
from torch.func import functional_call


class BezierPath:
    """
    Manages the QBC interpolation between two endpoint state dicts.

    Usage:
        bp = BezierPath(base_model, theta1_state, theta2_state, device)
        bp.forward(x, t=0.5)          # inference at a specific t
        bp.control_parameters()       # parameters to feed the optimiser
    """

    def __init__(self, base_model, theta1_state, theta2_state, device):
        self.base_model = copy.deepcopy(base_model).to(device)
        self.device = device

        # Frozen endpoints
        self.theta1 = {k: v.detach().clone().to(device) for k, v in theta1_state.items()}
        self.theta2 = {k: v.detach().clone().to(device) for k, v in theta2_state.items()}

        # Trainable control point — initialised to midpoint of endpoints
        self.theta = {}
        self._param_keys = []
        self._buffer_keys = []

        for k, v in theta1_state.items():
            v1 = self.theta1[k]
            v2 = self.theta2[k]
            if v1.is_floating_point() and v1.requires_grad is not False:
                # Trainable parameter
                self.theta[k] = nn.Parameter(((v1 + v2) / 2).clone())
                self._param_keys.append(k)
            else:
                # Buffer (e.g. num_batches_tracked) — just linearly interp
                self._buffer_keys.append(k)

    # ── public API ──────────────────────────────────────────────────────

    def control_parameters(self):
        """Return only the trainable control-point parameters."""
        return list(self.theta.values())

    def interpolate(self, t):
        """Return a full state dict for the model evaluated at path position t."""
        state = {}
        for k in self._param_keys:
            p1 = self.theta1[k]
            p2 = self.theta2[k]
            pc = self.theta[k]
            state[k] = (1 - t) ** 2 * p1 + 2 * t * (1 - t) * pc + t ** 2 * p2

        for k in self._buffer_keys:
            b1 = self.theta1[k]
            b2 = self.theta2[k]
            if b1.is_floating_point():
                state[k] = (1 - t) * b1 + t * b2
            else:
                state[k] = b1  # integer buffer (num_batches_tracked)
        return state

    def forward(self, x, t):
        """Forward pass through the model at path position t."""
        state = self.interpolate(t)
        return functional_call(self.base_model, state, (x,))

    def load_state_to_model(self, t):
        """
        Load the interpolated weights into `self.base_model` in-place so
        it can be used directly with external attacks that call model(x).
        Returns the model.
        """
        state = self.interpolate(t)
        self.base_model.load_state_dict(state, strict=False)
        return self.base_model


class BezierWrapper(nn.Module):
    """
    Thin nn.Module wrapper so that external attack functions that call
    model(x) can use a BezierPath at a fixed t.
    """

    def __init__(self, bezier_path, t):
        super().__init__()
        self.bp = bezier_path
        self.t = t

    def forward(self, x):
        return self.bp.forward(x, self.t)
