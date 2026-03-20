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


class PoTTA(BaseAdapter):
    def __init__(self, cfg, model, optimizer, scalar=False):
        # BaseAdapter.__init__ 안에서 configure_model()이 호출되므로 먼저 생성
        self._bn_hooks = []
        self._bn_global_stats = {
            "in_range_count": 0,
            "total_count": 0,
        }
        self.adapt_step = 0

        super(PoTTA, self).__init__(cfg, model, optimizer, scalar)

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

    def _reset_bn_global_stats(self):
        self._bn_global_stats["in_range_count"] = 0
        self._bn_global_stats["total_count"] = 0

    def _make_bn_output_hook(self):
        def hook(module, inputs, output):
            with torch.no_grad():
                if isinstance(output, (tuple, list)):
                    return

                out = output.detach()
                total = out.numel()
                if total == 0:
                    return

                in_range_count = ((out >= -1.0) & (out <= 1.0)).sum().item()

                self._bn_global_stats["in_range_count"] += in_range_count
                self._bn_global_stats["total_count"] += total

        return hook

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        model.train()

        # 현재 배치 기준으로 다시 집계
        self._reset_bn_global_stats()

        outputs = model(batch_data)
        loss = self_entropy(outputs).mean()

        optimizer.zero_grad()
        loss.backward()

        old_params = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                old_params[name] = p.detach().clone()

        optimizer.step()

        self.adapt_step += 1
        self._log_param_update_to_wandb(model, old_params, loss)

        return outputs

    def _log_param_update_to_wandb(self, model, old_params, loss=None):
        bn_delta_sum = 0.0
        bn_numel = 0

        signpow_delta_sum = 0.0
        signpow_numel = 0

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in old_params:
                continue

            delta = (p.detach() - old_params[name]).abs()

            if "sign_pow" in name:
                signpow_delta_sum += delta.sum().item()
                signpow_numel += delta.numel()
            else:
                bn_delta_sum += delta.sum().item()
                bn_numel += delta.numel()

        global_in_range_count = self._bn_global_stats["in_range_count"]
        global_total_count = self._bn_global_stats["total_count"]
        global_ratio = (
            global_in_range_count / global_total_count
            if global_total_count > 0 else 0.0
        )

        log_dict = {
            "tta_step": self.adapt_step,
            "loss/self_entropy": loss.item() if loss is not None else 0.0,
            "param_delta/bn_mean_abs": bn_delta_sum / bn_numel if bn_numel > 0 else 0.0,
            "param_delta/signpow_mean_abs": signpow_delta_sum / signpow_numel if signpow_numel > 0 else 0.0,
            "param_count/bn_numel": bn_numel,
            "param_count/signpow_numel": signpow_numel,

            # 전체 BN 출력 기준
            "bn_output/in_range_count_global": global_in_range_count,
            "bn_output/total_count_global": global_total_count,
            "bn_output/in_range_ratio_global": global_ratio,
        }

        if wandb.run is not None:
            wandb.log(log_dict, step=self.adapt_step)

    def configure_model(self, model: nn.Module):
        model.requires_grad_(False)
        normlayer_names = []

        for name, sub_module in model.named_modules():
            if isinstance(sub_module, nn.BatchNorm1d) or isinstance(sub_module, nn.BatchNorm2d):
                normlayer_names.append(name)

        if not hasattr(self, "_bn_hooks"):
            self._bn_hooks = []
        if not hasattr(self, "_bn_global_stats"):
            self._bn_global_stats = {
                "in_range_count": 0,
                "total_count": 0,
            }

        for h in self._bn_hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._bn_hooks = []

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

            new_bn_layer = get_named_submodule(model, name)
            hook_handle = new_bn_layer.register_forward_hook(
                self._make_bn_output_hook()
            )
            self._bn_hooks.append(hook_handle)

        return model