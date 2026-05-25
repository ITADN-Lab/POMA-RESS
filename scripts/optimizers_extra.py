"""Extra optimizers (MUON, SOAP, Adan) for C-MAPSS / SECOM fair-baseline experiments.

This module provides a single `build_extra_optimizer(name, params, lr, weight_decay)` entry.
Returns (optimizer, scheduler) where scheduler is None unless the optimizer needs a special one.
"""
import torch
import torch.optim as optim
from torch.optim import lr_scheduler


def build_extra_optimizer(name, params, lr, weight_decay, total_steps=None, ns_interval=100, ns_steps=1, ns_max_dim=256):
    """Returns (optimizer, scheduler_or_None) for optimizers not in the original train script."""
    name = name.upper()
    if name == 'MUON':
        from heavyball import Muon
        opt = Muon(list(params), lr=lr, weight_decay=weight_decay)
        return opt, None
    elif name == 'SOAP':
        from heavyball import SOAP
        opt = SOAP(list(params), lr=lr, weight_decay=weight_decay)
        return opt, None
    elif name in ('ADAN', 'ADAN-PYTORCH'):
        from adan_pytorch import Adan
        opt = Adan(list(params), lr=lr, weight_decay=weight_decay,
                   betas=(0.98, 0.92, 0.99))
        return opt, None
    elif name == 'PERIODICMUON':
        # periodic gradient-orthogonalization baseline
        from PeriodicMuon import PeriodicMuon
        opt = PeriodicMuon(list(params), lr=lr, weight_decay=weight_decay,
                           ns_interval=ns_interval, ns_steps=ns_steps,
                           ns_max_dim=ns_max_dim)
        return opt, None
    else:
        raise ValueError(f"Unknown optimizer: {name}")
