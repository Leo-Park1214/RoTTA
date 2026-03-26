import logging
import os
import torch
import argparse
import wandb

from core.configs import cfg
from core.utils import *
from core.model import build_model
from core.data import build_loader
from core.optim import build_optimizer
from core.adapter import build_adapter
from tqdm import tqdm
from setproctitle import setproctitle


def testTimeAdaptation(cfg, name):
    logger = logging.getLogger("TTA.test_time")

    # model, optimizer
    model = build_model(cfg)
    optimizer = build_optimizer(cfg)
    tta_adapter = build_adapter(cfg)

    if ("sp" in cfg.ADAPTER.NAME) or ("potta" == cfg.ADAPTER.NAME):
        tta_model = tta_adapter(cfg, model, optimizer, scalar=True)
    else:
        tta_model = tta_adapter(cfg, model, optimizer)

    tta_model.cuda()

    # wandb init을 여기서 수행
    if wandb.run is None:
        api_key = os.environ.get("WANDB_API_KEY", None)
        if api_key is not None:
            try:
                wandb.login(key=api_key)
            except Exception:
                pass

        wandb.init(
            project="rotta-tta",
            name=name,
            config={
                "dataset": cfg.CORRUPTION.DATASET,
                "adapter": cfg.ADAPTER.NAME,
                "corruption_type": cfg.CORRUPTION.TYPE,
                "severity": cfg.CORRUPTION.SEVERITY,
                "steps": cfg.OPTIM.STEPS,
                "seed": cfg.SEED,
                "alpha": getattr(cfg.ADAPTER.RoTTA, "ALPHA", None) if hasattr(cfg.ADAPTER, "RoTTA") else None,
                "nu": getattr(cfg.ADAPTER.RoTTA, "NU", None) if hasattr(cfg.ADAPTER, "RoTTA") else None,
                "update_frequency": getattr(cfg.ADAPTER.RoTTA, "UPDATE_FREQUENCY", None) if hasattr(cfg.ADAPTER, "RoTTA") else None,
                "memory_size": getattr(cfg.ADAPTER.RoTTA, "MEMORY_SIZE", None) if hasattr(cfg.ADAPTER, "RoTTA") else None,
                "lambda_t": getattr(cfg.ADAPTER.RoTTA, "LAMBDA_T", None) if hasattr(cfg.ADAPTER, "RoTTA") else None,
                "lambda_u": getattr(cfg.ADAPTER.RoTTA, "LAMBDA_U", None) if hasattr(cfg.ADAPTER, "RoTTA") else None,
                "scalar": (("sp" in cfg.ADAPTER.NAME) or ("potta" == cfg.ADAPTER.NAME)),
            },
            reinit=False,
        )

    loader, processor = build_loader(
        cfg,
        cfg.CORRUPTION.DATASET,
        cfg.CORRUPTION.TYPE,
        cfg.CORRUPTION.SEVERITY
    )

    tbar = tqdm(loader)
    for batch_id, data_package in enumerate(tbar):
        data, label, domain = data_package["image"], data_package["label"], data_package["domain"]

        if len(label) == 1:
            continue  # ignore the final single point

        data, label = data.cuda(), label.cuda()
        output = tta_model(data)

        predict = torch.argmax(output, dim=1)
        accurate = (predict == label)
        processor.process(accurate, domain)

        # adapter 내부에서 쌓아둔 wandb 로그를 바깥에서 기록
        # 새로 기록되는 key 예시:
        # - param_delta/conv_mean_abs
        # - param_delta/conv_signpow_mean_abs
        # - param_delta/other_mean_abs
        if hasattr(tta_model, "pop_wandb_logs"):
            pending_logs = tta_model.pop_wandb_logs()
            current_acc = processor.cumulative_acc()

            for log_dict in pending_logs:
                log_dict["acc"] = current_acc
                log_dict["batch_id"] = batch_id
                if wandb.run is not None:
                    wandb.log(log_dict, step=log_dict["tta_step"])

        if batch_id % 10 == 0:
            if hasattr(tta_model, "mem"):
                tbar.set_postfix(
                    acc=processor.cumulative_acc(),
                    bank=tta_model.mem.get_occupancy()
                )
            else:
                tbar.set_postfix(acc=processor.cumulative_acc())

    processor.calculate()

    final_acc = processor.cumulative_acc()
    if wandb.run is not None:
        wandb.log({
            "final_acc": final_acc,
            "final_results": processor.info()
        })

    logger.info(f"All Results\n{processor.info()}")

    if wandb.run is not None:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser("Pytorch Implementation for Test Time Adaptation!")
    parser.add_argument(
        '-acfg',
        '--adapter-config-file',
        metavar="FILE",
        default="",
        help="path to adapter config file",
        type=str)
    parser.add_argument(
        '-dcfg',
        '--dataset-config-file',
        metavar="FILE",
        default="",
        help="path to dataset config file",
        type=str)
    parser.add_argument(
        '-ocfg',
        '--order-config-file',
        metavar="FILE",
        default="",
        help="path to order config file",
        type=str)
    parser.add_argument("--name", type=str, default="", help="name for wandb logging")
    parser.add_argument(
        'opts',
        help='modify the configuration by command line',
        nargs=argparse.REMAINDER,
        default=None)
    

    args = parser.parse_args()

    if len(args.opts) > 0:
        args.opts[-1] = args.opts[-1].strip('\r\n')

    torch.backends.cudnn.benchmark = True

    cfg.merge_from_file(args.adapter_config_file)
    cfg.merge_from_file(args.dataset_config_file)
    if not args.order_config_file == "":
        cfg.merge_from_file(args.order_config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    ds = cfg.CORRUPTION.DATASET
    adapter = cfg.ADAPTER.NAME
    setproctitle(f"TTA:{ds:>8s}:{adapter:<10s}")

    if cfg.OUTPUT_DIR:
        mkdir(cfg.OUTPUT_DIR)

    logger = setup_logger('TTA', cfg.OUTPUT_DIR, 0, filename=cfg.LOG_DEST)
    logger.info(args)

    logger.info(
        f"Loaded configuration file: \n"
        f"\tadapter: {args.adapter_config_file}\n"
        f"\tdataset: {args.dataset_config_file}\n"
        f"\torder: {args.order_config_file}"
    )
    logger.info("Running with config:\n{}".format(cfg))

    set_random_seed(cfg.SEED)

    testTimeAdaptation(cfg, name=args.name)


if __name__ == "__main__":
    main()
