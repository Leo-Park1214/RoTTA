import torch
import torch.nn as nn

from ..utils import memory
from .base_adapter import BaseAdapter
from copy import deepcopy
from .base_adapter import softmax_entropy
from ..utils.bn_layers import RobustBN1d, RobustBN2d
from ..utils.bn_layers_signpow import RobustBN1d as RobustBN1d_sp
from ..utils.bn_layers_signpow import RobustBN2d as RobustBN2d_sp
from ..utils.utils import set_named_submodule, get_named_submodule
from ..utils.custom_transforms import get_tta_transforms


class RoTTA(BaseAdapter):
    def __init__(self, cfg, model, optimizer, scalar=False):
        self._bn_hooks = []
        self._bn_global_stats = {
            "in_range_count": 0,
            "total_count": 0,
        }
        self.adapt_step = 0
        self.ds_name = cfg.CORRUPTION.DATASET
        self.pending_logs = []   # 바깥(testTimeAdaptation)에서 wandb.log 할 값 저장

        super(RoTTA, self).__init__(cfg, model, optimizer, scalar)

        self.mem = memory.CSTU(
            capacity=self.cfg.ADAPTER.RoTTA.MEMORY_SIZE,
            num_class=cfg.CORRUPTION.NUM_CLASS,
            lambda_t=cfg.ADAPTER.RoTTA.LAMBDA_T,
            lambda_u=cfg.ADAPTER.RoTTA.LAMBDA_U
        )
        self.model_ema = self.build_ema(self.model)
        self.transform = get_tta_transforms(cfg)
        self.nu = cfg.ADAPTER.RoTTA.NU
        self.update_frequency = cfg.ADAPTER.RoTTA.UPDATE_FREQUENCY
        self.current_instance = 0

    def pop_wandb_logs(self):
        logs = self.pending_logs[:]
        self.pending_logs.clear()
        return logs

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

    def _build_param_update_log(self, model, old_params, loss=None):
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
            "loss/rotta_sup": loss.item() if loss is not None else 0.0,
            "param_delta/bn_mean_abs": bn_delta_sum / bn_numel if bn_numel > 0 else 0.0,
            "param_delta/signpow_mean_abs": signpow_delta_sum / signpow_numel if signpow_numel > 0 else 0.0,
            "param_count/bn_numel": bn_numel,
            "param_count/signpow_numel": signpow_numel,
            "bn_output/in_range_count_global": global_in_range_count,
            "bn_output/total_count_global": global_total_count,
            "bn_output/in_range_ratio_global": global_ratio,
        }
        return log_dict

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        with torch.no_grad():
            model.eval()
            self.model_ema.eval()
            ema_out = self.model_ema(batch_data)
            predict = torch.softmax(ema_out, dim=1)
            pseudo_label = torch.argmax(predict, dim=1)
            entropy = torch.sum(- predict * torch.log(predict + 1e-6), dim=1)

        for i, data in enumerate(batch_data):
            p_l = pseudo_label[i].item()
            uncertainty = entropy[i].item()
            current_instance = (data, p_l, uncertainty)
            self.mem.add_instance(current_instance)
            self.current_instance += 1

            if self.current_instance % self.update_frequency == 0:
                self.update_model(model, optimizer)

        return ema_out

    def update_model(self, model, optimizer):
        model.train()
        self.model_ema.train()

        self._reset_bn_global_stats()

        sup_data, ages = self.mem.get_memory()
        l_sup = None

        if len(sup_data) > 0:
            sup_data = torch.stack(sup_data)
            strong_sup_aug = self.transform(sup_data)

            ema_sup_out = self.model_ema(sup_data)
            stu_sup_out = model(strong_sup_aug)

            instance_weight = timeliness_reweighting(ages)
            l_sup = (softmax_entropy(stu_sup_out, ema_sup_out) * instance_weight).mean()

        l = l_sup
        if l is not None:
            optimizer.zero_grad()
            l.backward()

            old_params = {}
            for name, p in model.named_parameters():
                if p.requires_grad:
                    old_params[name] = p.detach().clone()

            optimizer.step()

            self.adapt_step += 1

            log_dict = self._build_param_update_log(model, old_params, l)
            self.pending_logs.append(log_dict)

        self.update_ema_variables(self.model_ema, self.model, self.nu)

    @staticmethod
    def update_ema_variables(ema_model, model, nu):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = (1 - nu) * ema_param[:].data[:] + nu * param[:].data[:]
        return ema_model

    def configure_model(self, model: nn.Module):
        model.requires_grad_(False)
        normlayer_names = []
        convlayer_names = []

        print(f"{len(list(model.named_modules()))} modules in total. Searching for BN layers...")
        for name, sub_module in model.named_modules():
            if isinstance(sub_module, nn.BatchNorm1d) or isinstance(sub_module, nn.BatchNorm2d):
                normlayer_names.append(name)
            elif isinstance(sub_module, nn.Conv2d):
                convlayer_names.append(name)

        print(f"Found {len(convlayer_names)} Conv layers. Freezing their parameters...")
        print(f"Found {len(normlayer_names)} BN layers. Replacing with RobustBN...")

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
            #if not self.scalar:
            momentum_bn.requires_grad_(True)
            set_named_submodule(model, name, momentum_bn)

            new_bn_layer = get_named_submodule(model, name)
            hook_handle = new_bn_layer.register_forward_hook(
                self._make_bn_output_hook()
            )
            self._bn_hooks.append(hook_handle)

        return model


def timeliness_reweighting(ages):
    if isinstance(ages, list):
        ages = torch.tensor(ages).float().cuda()
    return torch.exp(-ages) / (1 + torch.exp(-ages))