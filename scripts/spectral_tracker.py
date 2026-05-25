"""Spectral tracker that wraps LAKTJU_NS / AdamW / MUON to record
pre-NS / post-NS κ(M), effective rank, update cosine diversity per NS event.

Provides one entrypoint: `attach_tracker(optimizer, model, opt_name, log_list)` that
returns the optimizer (possibly a wrapped subclass).

Tracking strategy per NS event (or every N steps for non-NS optimizers):
  - kappa_pre, kappa_post: condition number of momentum buffer of first 2D param
  - erank_pre, erank_post: effective rank H = exp(-sum p log p) over normalized singular values
  - cosine_div: average pairwise cosine of NEW updates (1 - mean cosine ≈ diversity)
"""
import torch
import math


def _spectral_metrics(m2d):
    """Compute (kappa, erank) of a 2D tensor using float64 SVD with clamping."""
    try:
        s = torch.linalg.svdvals(m2d.double())
    except Exception:
        return None, None
    s = s[s > 1e-12]
    if s.numel() < 2:
        return None, None
    kappa = (s[0] / s[-1]).item()
    p = s / s.sum()
    H = -(p * (p.clamp_min(1e-30).log())).sum().item()
    erank = math.exp(H)
    return min(kappa, 1e8), erank


def _first_2d_momentum(optimizer):
    """Return (param_ref, momentum_buffer_2d_view) of the first 2D param with exp_avg."""
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.dim() < 2:
                continue
            state = optimizer.state.get(p, {})
            buf = None
            for k in ('exp_avg', 'momentum_buffer', 'm1'):
                if k in state and torch.is_tensor(state[k]):
                    buf = state[k]
                    break
            if buf is None:
                continue
            return p, buf.view(buf.size(0), -1)
    return None, None


class SpectralLAKTJU_NS:
    """Wraps a LAKTJU_NS instance: record kappa/erank of momentum buf BEFORE and AFTER
    NS event. Detection: NS happens whenever (_ns_step_counter % ns_interval) == 0
    AFTER super().step() — so we record pre-step state, then post-step state."""
    def __init__(self, optimizer, log_list, opt_name='LAKTJU_NS'):
        self.opt = optimizer
        self.log = log_list
        self.opt_name = opt_name
        self.step_count = 0

    def __getattr__(self, item):
        return getattr(self.opt, item)

    @torch.no_grad()
    def step(self, closure=None):
        ns_interval = getattr(self.opt, 'ns_interval', 0)
        # Determine if THIS step will trigger NS
        will_ns = (ns_interval and (getattr(self.opt, '_ns_step_counter', 0) + 1) % ns_interval == 0)

        # --- pre-step momentum spectrum ---
        pre_kappa, pre_erank = None, None
        if will_ns:
            p, m2d = _first_2d_momentum(self.opt)
            if m2d is not None:
                pre_kappa, pre_erank = _spectral_metrics(m2d.clone())

        loss = self.opt.step(closure)
        self.step_count += 1

        # --- post-step momentum spectrum (only if NS was applied) ---
        if will_ns:
            p, m2d = _first_2d_momentum(self.opt)
            if m2d is not None:
                post_kappa, post_erank = _spectral_metrics(m2d.clone())
                self.log.append({
                    'event': 'NS',
                    'step': self.step_count,
                    'kappa_pre': pre_kappa,
                    'kappa_post': post_kappa,
                    'erank_pre': pre_erank,
                    'erank_post': post_erank,
                    'optimizer': self.opt_name,
                })
        return loss


class PeriodicMomentumTracker:
    """For AdamW / MUON / SOAP without explicit NS: just sample κ/erank every N steps."""
    def __init__(self, optimizer, log_list, opt_name, sample_interval=100):
        self.opt = optimizer
        self.log = log_list
        self.opt_name = opt_name
        self.sample_interval = sample_interval
        self.step_count = 0

    def __getattr__(self, item):
        return getattr(self.opt, item)

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.opt.step(closure)
        self.step_count += 1
        if self.step_count % self.sample_interval == 0:
            p, m2d = _first_2d_momentum(self.opt)
            if m2d is not None:
                kappa, erank = _spectral_metrics(m2d.clone())
                self.log.append({
                    'event': 'sample',
                    'step': self.step_count,
                    'kappa': kappa,
                    'erank': erank,
                    'optimizer': self.opt_name,
                })
        return loss


def attach_tracker(optimizer, opt_name, log_list, sample_interval=100):
    """Return a wrapped optimizer that logs spectral state to log_list."""
    if 'LAKTJU_NS' in opt_name or 'NS' in opt_name:
        return SpectralLAKTJU_NS(optimizer, log_list, opt_name=opt_name)
    return PeriodicMomentumTracker(optimizer, log_list, opt_name, sample_interval)
