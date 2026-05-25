"""Mechanism ablation: 5 variants isolating what makes NS work.

1. LAKTJU_NS  — momentum NS (existing, the proposed method)
2. GradNS     — gradient NS at same interval (PeriodicMuon baseline)
3. NormOnly   — norm-preserving rescale only, no orthogonalization
4. RandRot    — random orthogonal rotation at same interval (regularization control)
5. NoNS       — plain AdamW (negative control)

Each variant is a drop-in AdamW subclass with the same outer API.
"""
import torch
from torch.optim import AdamW
import math


@torch.no_grad()
def _ns_ortho_bf16(G, ns_steps=1):
    """Newton-Schulz quintic orthogonalization in bfloat16."""
    m, n = G.shape
    transposed = m > n
    if transposed:
        G = G.T
        m, n = n, m
    norm = G.norm()
    if norm < 1e-12:
        return G.T if transposed else G
    X = G / norm
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(ns_steps):
        A = X @ X.T
        X = a * X + b * (A @ X) + c * (A @ (A @ X))
    return X.T if transposed else X


def _apply_to_buffer(optimizer, buffer_key, transform_fn):
    """Apply transform_fn to the specified buffer of all 2D params."""
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None or p.ndim < 2:
                continue
            state = optimizer.state.get(p, {})
            buf = state.get(buffer_key)
            if buf is None:
                continue
            shape = buf.shape
            rows, cols = shape[0], buf.numel() // shape[0]
            mdim = min(rows, cols)
            if not (1 <= mdim <= 256):
                continue
            G = buf.reshape(rows, cols).to(torch.bfloat16)
            U = transform_fn(G)
            u_norm = U.norm()
            if u_norm > 1e-12:
                buf.copy_(U.to(buf.dtype).reshape(shape).mul_(buf.norm() / u_norm))


# ──────────────────────────────
# Variant 1: LAKTJU_NS  (already in optimizer/) — imported for reference
# ──────────────────────────────
class LAKTJU_NS_REF(AdamW):
    """Reference copy of LAKTJU_NS for ablation (self-contained)."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, interval=100, steps=1, max_dim=256):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.interval = interval
        self.steps = steps
        self.max_dim = max_dim
        self._counter = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        self._counter += 1
        if self._counter % self.interval == 0:
            def do_ns(G):
                return _ns_ortho_bf16(G, ns_steps=self.steps)
            _apply_to_buffer(self, 'exp_avg', do_ns)
        return loss


# ──────────────────────────────
# Variant 2: GradNS (PeriodicMuon)
# ──────────────────────────────
class GradNS(AdamW):
    """PeriodicMuon: orthogonalize GRADIENT before AdamW step."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, interval=100, steps=1, max_dim=256):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.interval = interval
        self.steps = steps
        self.max_dim = max_dim
        self._counter = 0

    @torch.no_grad()
    def step(self, closure=None):
        self._counter += 1
        if self._counter % self.interval == 0:
            def do_ns(G):
                return _ns_ortho_bf16(G, ns_steps=self.steps)
            # Transform gradients BEFORE AdamW
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None or p.ndim < 2: continue
                    shape = p.grad.shape
                    rows, cols = shape[0], p.grad.numel()//shape[0]
                    mdim = min(rows, cols)
                    if not (1 <= mdim <= self.max_dim): continue
                    G = p.grad.reshape(rows, cols).to(torch.bfloat16)
                    U = do_ns(G)
                    u_norm = U.norm()
                    if u_norm > 1e-12:
                        p.grad = U.to(p.grad.dtype).reshape(shape).mul_(p.grad.norm()/u_norm)
        return super().step(closure)


# ──────────────────────────────
# Variant 3: NormOnly (rescale without rotation)
# ──────────────────────────────
class NormOnly(AdamW):
    """Periodically rescale momentum to unit Frobenius norm, no rotation."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, interval=100, steps=1, max_dim=256):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.interval = interval
        self.steps = steps
        self.max_dim = max_dim
        self._counter = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        self._counter += 1
        if self._counter % self.interval == 0:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None or p.ndim < 2: continue
                    state = self.state.get(p, {})
                    buf = state.get('exp_avg')
                    if buf is None: continue
                    old_norm = buf.norm()
                    if old_norm > 1e-12:
                        buf.div_(old_norm)  # unit norm (discard spectrum entirely)
        return loss


# ──────────────────────────────
# Variant 4: RandomRotation
# ──────────────────────────────
class RandRot(AdamW):
    """Periodically apply a random orthogonal rotation to momentum buffer."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, interval=100, max_dim=256):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.interval = interval
        self.max_dim = max_dim
        self._counter = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        self._counter += 1
        if self._counter % self.interval == 0:
            # Apply random orthogonal rotation per 2D param
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None or p.ndim < 2: continue
                    state = self.state.get(p, {})
                    buf = state.get('exp_avg')
                    if buf is None: continue
                    shape = buf.shape
                    rows, cols = shape[0], buf.numel() // shape[0]
                    mdim = min(rows, cols)
                    if not (1 <= mdim <= self.max_dim): continue
                    M = buf.reshape(rows, cols).float()
                    # QR of random Gaussian → random orthogonal
                    R = torch.randn(cols, cols, device=M.device)
                    Q, _ = torch.linalg.qr(R.float())
                    M_rot = M @ Q
                    buf.copy_(M_rot.reshape(shape).mul_(buf.norm() / (M_rot.norm() + 1e-12)))
        return loss


# Registry
ABLATION_VARIANTS = {
    'LAKTJU_NS': LAKTJU_NS_REF,
    'GradNS': GradNS,
    'NormOnly': NormOnly,
    'RandRot': RandRot,
    'AdamW': lambda params, **kw: AdamW(params, lr=kw.get('lr',1e-3),
                                         betas=kw.get('betas',(0.9,0.999)),
                                         weight_decay=kw.get('weight_decay',0)),
}
