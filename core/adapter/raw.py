import os
import torch
import torch.nn as nn
import wandb

from .base_adapter import BaseAdapter
from .base_adapter import self_entropy
from ..utils.bn_layers import RobustBN1d, RobustBN2d
from ..utils.bn_layers_signpow import RobustBN1d as RobustBN1d_sp
from ..utils.bn_layers_signpow import RobustBN2d as RobustBN2d_sp
from ..utils.utils import set_named_submodule, get_named_submodule


class RAW(BaseAdapter):
    def __init__(self, cfg, model, optimizer, scalar=False):
        super(RAW, self).__init__(cfg, model, optimizer, scalar)
        self.adapt_step = 0

        if wandb.run is None:
            api_key = os.environ.get("WANDB_API_KEY", None)
            if api_key is not None:
                try:
                    wandb.login(key=api_key)
                except Exception:
                    pass

            wandb.init(
                project="potta-tta",
                name=f"{cfg.CORRUPTION.DATASET}_{cfg.ADAPTER.NAME}_{cfg.CORRUPTION.TYPE}_{cfg.CORRUPTION.SEVERITY}",
                config={
                    "dataset": cfg.CORRUPTION.DATASET,
                    "adapter": cfg.ADAPTER.NAME,
                    "corruption_type": cfg.CORRUPTION.TYPE,
                    "severity": cfg.CORRUPTION.SEVERITY,
                    "steps": cfg.OPTIM.STEPS,
                    "seed": cfg.SEED,
                    "alpha": cfg.ADAPTER.RoTTA.ALPHA,
                    "scalar": scalar,
                },
                reinit=False,
            )

    @torch.no_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        model.eval()   # 학습 안 함, 추론만
        outputs = model(batch_data)
        return outputs

    def _log_param_update_to_wandb(self, model, old_params, loss=None):
        pass

    def configure_model(self, model: nn.Module):
        return model