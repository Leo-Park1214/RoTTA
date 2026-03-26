import torch
import torch.nn as nn

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
        self.ds_name = cfg.CORRUPTION.DATASET

        # RoTTA 방식처럼 내부에서 wandb.init/log 하지 않고
        # 바깥에서 꺼내서 wandb.log 하도록 로그 버퍼만 유지
        self.pending_logs = []

    def pop_wandb_logs(self):
        logs = self.pending_logs[:]
        self.pending_logs.clear()
        return logs

    @torch.no_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        model.eval()   # 학습 안 함, 추론만
        outputs = model(batch_data)

        # 필요하면 RAW에서도 로깅용 값 적재
        ent = self_entropy(outputs).mean()

        log_dict = {
            "tta_step": self.adapt_step,
            "loss/raw_entropy": ent.item(),
        }
        self.pending_logs.append(log_dict)

        self.adapt_step += 1
        return outputs

    def _log_param_update_to_wandb(self, model, old_params, loss=None):
        # RoTTA처럼 즉시 wandb.log() 하지 않고 pending_logs에 저장
        group_stats = {
            "all": {"delta_sum": 0.0, "numel": 0},
        }

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in old_params:
                continue

            delta = (p.detach() - old_params[name]).abs()
            group_stats["all"]["delta_sum"] += delta.sum().item()
            group_stats["all"]["numel"] += delta.numel()

        log_dict = {
            "tta_step": self.adapt_step,
            "loss/raw": loss.item() if loss is not None else 0.0,
            "param_delta/all_mean_abs": (
                group_stats["all"]["delta_sum"] / group_stats["all"]["numel"]
                if group_stats["all"]["numel"] > 0 else 0.0
            ),
            "param_count/all_numel": group_stats["all"]["numel"],
        }

        self.pending_logs.append(log_dict)

    def configure_model(self, model: nn.Module):
        return model