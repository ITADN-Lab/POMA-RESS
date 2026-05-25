"""Per-LAYER spectral tracker (B2): logs kappa/erank pre+post of EVERY 2D
momentum buffer at each NS event, not just the first layer. Answers the
reviewers' cross-layer-consistency question.
"""
import torch
import math


def _spectral_metrics(m2d):
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
    return min(kappa, 1e8), math.exp(H)


def _all_2d_momenta(optimizer):
    """Every 2D param that has a momentum buffer -> (idx, shape, m2d view)."""
    out, idx = [], 0
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
            out.append((idx, tuple(buf.shape), buf.view(buf.size(0), -1)))
            idx += 1
    return out


class SpectralPerLayer:
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
        will_ns = (ns_interval and
                   (getattr(self.opt, '_ns_step_counter', 0) + 1) % ns_interval == 0)
        pre = {}
        if will_ns:
            for i, shp, m in _all_2d_momenta(self.opt):
                k, e = _spectral_metrics(m.clone())
                pre[i] = (shp, k, e)
        loss = self.opt.step(closure)
        self.step_count += 1
        if will_ns:
            layers = []
            for i, shp, m in _all_2d_momenta(self.opt):
                kpost, epost = _spectral_metrics(m.clone())
                spre = pre.get(i, (shp, None, None))
                layers.append({'layer': i, 'shape': list(shp),
                               'kappa_pre': spre[1], 'kappa_post': kpost,
                               'erank_pre': spre[2], 'erank_post': epost})
            self.log.append({'event': 'NS', 'step': self.step_count,
                             'optimizer': self.opt_name, 'n_layers': len(layers),
                             'layers': layers})
        return loss


def attach_tracker_perlayer(optimizer, opt_name, log_list, sample_interval=100):
    return SpectralPerLayer(optimizer, log_list, opt_name=opt_name)
