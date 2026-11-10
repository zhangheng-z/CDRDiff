import torch
from torch import nn
from torch.nn import functional as F

from cldm.model import load_state_dict
from hfnm import HFNM


class SevenChannelHintConv(nn.Conv2d):
    """Apply separate convolutions to the reflectance and HFNM channels."""

    def forward(self, hint):
        original = F.conv2d(hint[:, :6].contiguous(), self.weight[:, :6].contiguous(),
                            self.bias, self.stride, self.padding, self.dilation)
        hfnm = F.conv2d(hint[:, 6:].contiguous(), self.weight[:, 6:].contiguous(),
                        None, self.stride, self.padding, self.dilation)
        return original + hfnm


def attach_hfnm(prior, checkpoint):
    weights = load_state_dict(checkpoint, location='cpu')
    prefix = 'decoupling_head.'
    head = {key[len(prefix):]: value for key, value in weights.items() if key.startswith(prefix)}
    with torch.random.fork_rng(devices=[]):
        hfnm = HFNM().eval()
    hfnm.load_state_dict(head or weights)
    cached_reflectance = cached_hint = None

    def append_hfnm(module, inputs, original_hint):
        nonlocal cached_reflectance, cached_hint
        if module.R is not cached_reflectance:
            rgb = original_hint[:, :3].cpu().float()[:, [2, 1, 0]]
            n = hfnm(rgb).to(original_hint)
            if not torch.isfinite(n).all():
                raise ValueError('HFNM produced non-finite values')
            cached_hint = torch.cat((original_hint, n), dim=1)
            cached_reflectance = module.R
            print(f'ControlNet hint {tuple(cached_hint.shape)}; HFNM range: '
                  f'[{n.min().item():.6f}, {n.max().item():.6f}]')
        return cached_hint

    prior.register_forward_hook(append_hfnm)
