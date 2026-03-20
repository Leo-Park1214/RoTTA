import torch
import torch.nn as nn
from .base_adapter import BaseAdapter
from .base_adapter import self_entropy
from ..utils.bn_layers import RobustBN1d, RobustBN2d
from ..utils.bn_layers_signpow import RobustBN1d as RobustBN1d_sp
from ..utils.bn_layers_signpow import RobustBN2d as RobustBN2d_sp
from ..utils.utils import set_named_submodule, get_named_submodule


class PoTTA(BaseAdapter):
    def __init__(self, cfg, model, optimizer, scalar=False):
        super(PoTTA, self).__init__(cfg, model, optimizer, scalar)

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        model.train()

        outputs = model(batch_data)
        loss = self_entropy(outputs).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return outputs

    def configure_model(self, model: nn.Module):
        model.requires_grad_(False)
        normlayer_names = []

        for name, sub_module in model.named_modules():
            if isinstance(sub_module, nn.BatchNorm1d) or isinstance(sub_module, nn.BatchNorm2d):
                normlayer_names.append(name)

        for name in normlayer_names:
            bn_layer = get_named_submodule(model, name)

            if isinstance(bn_layer, nn.BatchNorm1d):
                NewBN = RobustBN1d_sp if self.scalar else RobustBN1d
            elif isinstance(bn_layer, nn.BatchNorm2d):
                NewBN = RobustBN2d_sp if self.scalar else RobustBN2d
            else:
                raise RuntimeError()

            momentum_bn = NewBN(
                bn_layer,
                self.cfg.ADAPTER.RoTTA.ALPHA
            )

            if not self.scalar:
                momentum_bn.requires_grad_(True)

            set_named_submodule(model, name, momentum_bn)

        return model