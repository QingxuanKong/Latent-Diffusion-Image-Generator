import os
import sys
import argparse
import numpy as np
import ruamel.yaml as yaml
import torch
import wandb
import logging
from logging import getLogger as get_logger
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import lpips

from models import UNet, VAE, ClassEmbedder
from schedulers import DDPMScheduler, DDIMScheduler
from pipelines import DDPMPipeline
from utils import (
    seed_everything,
    init_distributed_device,
    is_primary,
    AverageMeter,
    str2bool,
    save_checkpoint,
    load_checkpoint,
    evaluate_fid_is_lpips,
    build_val_loader,
)

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model.")

    # config file
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ddpm.yaml",
        help="config file used to specify parameters",
    )

    # data
    parser.add_argument(
        "--dataset", type=str, default="imagenet100", choices=["imagenet100", "cifar10"]
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/imagenet100_128x128/train",
        help="data folder",
    )
    parser.add_argument(
        "--val_data_dir",
        type=str,
        default="data/imagenet100_128x128/validation",
        help="Path to validation data folder (for ImageFolder-based datasets)",
    )
    parser.add_argument(
        "--subset",
        type=float,
        default=1.0,
        help="Fraction of validation set to use (e.g., 0.1 for 10%)",
    )
    parser.add_argument("--image_size", type=int, default=128, help="image size")
    parser.add_argument(
        "--num_classes", type=int, default=100, help="number of classes in dataset"
    )

    # training
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--batch_size", type=int, default=4, help="per gpu batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="batch size")
    parser.add_argument("--num_epochs", type=int, default=10)

    # optimizer
    parser.add_argument("--warmup_epochs", type=int, default=0, help="warmup epochs")
    parser.add_argument("--scheduler", type=str, default="cosine", help="scheduler")
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="gradient clip")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="none",
        choices=["fp16", "bf16", "fp32", "none"],
        help="mixed precision",
    )

    # log
    parser.add_argument("--DEBUG", type=str2bool, default=False, help="debug_mode")
    parser.add_argument(
        "--wandb_username", type=str, default=None, help="wandb_username"
    )
    parser.add_argument("--wandb_key", type=str, default=None, help="wandb_key")
    parser.add_argument("--project_name", type=str, default=None, help="project_name")
    parser.add_argument("--run_name", type=str, default=None, help="run_name")
    parser.add_argument(
        "--output_dir", type=str, default="experiments", help="output folder"
    )

    # resume
    parser.add_argument(
        "--resume",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--wandb_resume_id",
        type=str,
        default="none",
    )
    parser.add_argument(
        "--resume_checkpoint_path",
        type=str,
        default="none",
    )

    # unet
    parser.add_argument(
        "--unet_in_size", type=int, default=128, help="unet input image size"
    )
    parser.add_argument(
        "--unet_in_ch", type=int, default=3, help="unet input channel size"
    )
    parser.add_argument("--unet_ch", type=int, default=128, help="unet channel size")
    parser.add_argument(
        "--unet_ch_mult",
        type=int,
        default=[1, 2, 2, 2],
        nargs="+",
        help="unet channel multiplier",
    )
    parser.add_argument(
        "--unet_attn",
        type=int,
        default=[1, 2, 3],
        nargs="+",
        help="unet attantion stage index",
    )
    parser.add_argument(
        "--unet_num_res_blocks", type=int, default=2, help="unet number of res blocks"
    )
    parser.add_argument("--unet_dropout", type=float, default=0.0, help="unet dropout")
    parser.add_argument(
        "--use_adagn_resblock", type=str2bool, default=False, help="adagn resnet"
    )
    parser.add_argument(
        "--use_transformer_bottleneck",
        type=str2bool,
        default=False,
        help="transformer bottleneck",
    )
    parser.add_argument(
        "--transformer_depth", type=int, default=1, help="transformer depth"
    )
    parser.add_argument(
        "--transformer_num_heads", type=int, default=8, help="transformer num heads"
    )

    # ddpm
    parser.add_argument(
        "--num_train_timesteps", type=int, default=1000, help="ddpm training timesteps"
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=200, help="ddpm inference timesteps"
    )
    parser.add_argument(
        "--beta_start", type=float, default=0.0002, help="ddpm beta start"
    )
    parser.add_argument("--beta_end", type=float, default=0.02, help="ddpm beta end")
    parser.add_argument(
        "--beta_schedule", type=str, default="linear", help="ddpm beta schedule"
    )
    parser.add_argument(
        "--variance_type", type=str, default="fixed_small", help="ddpm variance type"
    )
    parser.add_argument(
        "--prediction_type", type=str, default="epsilon", help="ddpm epsilon type"
    )
    parser.add_argument(
        "--clip_sample",
        type=str2bool,
        default=True,
        help="whether to clip sample at each step of reverse process",
    )
    parser.add_argument(
        "--clip_sample_range", type=float, default=1.0, help="clip sample range"
    )

    # vae
    parser.add_argument(
        "--latent_ddpm", type=str2bool, default=False, help="use vqvae for latent ddpm"
    )
    parser.add_argument(
        "--vae_ckpt", type=str, default="pretrained/model.ckpt", help="pretrained vae ckpt"
    )
    parser.add_argument(
        "--freeze_vae_epoch",
        type=int,
        default=10,
        help="free vae params after this epoch",
    )
    parser.add_argument(
        "--kl_beta",
        type=float,
        default=0.001,
        help="kl divergence beta to add to vae loss",
    )
    parser.add_argument(
        "--lpips_lambda",
        type=float,
        default=1.0,
        help="lpips lambda to add to vae loss",
    )
    parser.add_argument(
        "--vae_lambda", type=float, default=0.05, help="vae lambda to add to ddpm loss"
    )

    # cfg
    parser.add_argument(
        "--use_cfg",
        type=str2bool,
        default=False,
        help="use cfg for conditional (latent) ddpm",
    )
    parser.add_argument(
        "--cfg_guidance_scale", type=float, default=2.0, help="cfg for inference"
    )
    parser.add_argument(
        "--cond_drop_rate",
        type=float,
        default=0.1,
        help="use cfg for conditional (latent) ddpm",
    )

    # ddim sampler for inference
    parser.add_argument(
        "--use_ddim",
        type=str2bool,
        default=False,
        help="use ddim sampler for inference",
    )

    # checkpoint path for inference
    parser.add_argument(
        "--ckpt", type=str, default=None, help="checkpoint path for inference"
    )
    parser.add_argument(
        "--keep_last_n", type=int, default=10, help="number of checkpoints kept"
    )
    parser.add_argument(
        "--keep_last_model", type=str2bool, default=True, help="keep best model"
    )
    parser.add_argument(
        "--keep_best_model", type=str2bool, default=True, help="keep best model"
    )

    # evaluation for inference in training
    parser.add_argument(
        "--eval_during_train",
        type=str2bool,
        default=False,
        help="Whether to run evaluation during training (e.g., on 500 images)",
    )
    parser.add_argument(
        "--eval_every_n_epoch",
        type=int,
        default=5,
        help="Evaluate every n epochs during training",
    )
    parser.add_argument(
        "--eval_samples",
        type=int,
        default=500,
        help="the number of samples to generate for evaluation",
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=50, help="batch size for evaluation"
    )
    parser.add_argument(
        "--eval_classes", type=int, default=50, help="batch size for evaluation"
    )

    # inference
    parser.add_argument(
        "--inference_samples",
        type=int,
        default=5000,
        help="Total number of images generated in inference.py",
    )

    # distributed training settings (used in DDP or multi-GPU)
    parser.add_argument(
        "--distributed", action="store_true", help="Use DistributedDataParallel"
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Number of total processes (GPUs) for distributed training",
    )
    parser.add_argument(
        "--rank", type=int, default=0, help="Rank of the current process"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="Local rank of the process (used by torch.distributed.launch)",
    )

    # first parse of command-line args to check for config file
    args = parser.parse_args()

    # If a config file is specified, load it and set defaults
    if args.config is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            file_yaml = yaml.YAML()
            config_args = file_yaml.load(f)
            print(f"[INFO] Loaded config file: {config_args}")
            parser.set_defaults(**config_args)

    # re-parse command-line args to overwrite with any command-line inputs
    args = parser.parse_args()
    return args


def main():

    # parse arguments
    args = parse_args()

    for k, v in vars(args).items():
        print(f"[INFO] {k}: {v} (type: {type(v)})")

    # seed everything
    seed_everything(args.seed)

    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # setup distributed initialize and device
    device = init_distributed_device(args)
    if args.distributed:
        logger.info(
            "Training in distributed mode with multiple processes, 1 device per process."
            f"Process {args.rank}, total {args.world_size}, device {args.device}."
        )
    else:
        logger.info(f"Training with a single process on 1 device ({args.device}).")
    assert args.rank >= 0

    # -------------------------------------------
    # ------------------dataset------------------
    # -------------------------------------------
    # setup dataset
    logger.info("Creating dataset")
    # TODO: use transform to normalize your images to [-1, 1]
    # TODO: you can also use horizontal flip
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    # TOOD: use image folder for your train dataset
    train_dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    subset_size = int(len(train_dataset) * args.subset)
    train_dataset = torch.utils.data.Subset(train_dataset, range(subset_size))

    # TODO: setup dataloader
    sampler = None
    if args.distributed:
        # TODO: distributed sampler
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True
        )
    # TODO: shuffle
    shuffle = False if sampler else True
    # TODO dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=True,
        persistent_workers=True,
    )

    # calculate total batch_size
    total_batch_size = args.batch_size * args.world_size
    args.total_batch_size = total_batch_size

    # -------------------------------------------
    # ---------------set up folder---------------
    # -------------------------------------------
    # setup experiment folder
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.run_name is None:
        args.run_name = f"exp-{len(os.listdir(args.output_dir))}"
    else:
        args.run_name = f"exp-{len(os.listdir(args.output_dir))}-{args.run_name}"
    output_dir = os.path.join(args.output_dir, args.run_name)
    save_dir = os.path.join(output_dir, "checkpoints")
    if is_primary(args):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)

    # -------------------------------------------
    # ---------set up model and scheduler--------
    # -------------------------------------------
    # setup model
    logger.info("Creating model")
    # unet
    unet = UNet(
        input_size=args.unet_in_size,
        input_ch=args.unet_in_ch,
        T=args.num_train_timesteps,
        ch=args.unet_ch,
        ch_mult=args.unet_ch_mult,
        attn=args.unet_attn,
        num_res_blocks=args.unet_num_res_blocks,
        dropout=args.unet_dropout,
        conditional=args.use_cfg,
        c_dim=args.unet_ch,
        # --- New arguments for AdaGN ResNet ---
        use_adagn_resblock=args.use_adagn_resblock,
        # --- New arguments for Transformer Bottleneck ---
        use_transformer_bottleneck=args.use_transformer_bottleneck,
        transformer_depth=args.transformer_depth,
        transformer_num_heads=args.transformer_num_heads,
        # --------------------------------------------
    )
    # preint number of parameters
    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params / 10 ** 6:.2f}M")

    # TODO: ddpm scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        num_inference_steps=args.num_inference_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        variance_type=args.variance_type,
        prediction_type=args.prediction_type,
        clip_sample=args.clip_sample,
        clip_sample_range=args.clip_sample_range,
    )

    # TODO: setup ddim
    if args.use_ddim:
        scheduler_wo_ddp = DDIMScheduler(
            num_train_timesteps=args.num_train_timesteps,
            num_inference_steps=args.num_inference_steps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
            prediction_type=args.prediction_type,
            clip_sample=args.clip_sample,
            clip_sample_range=args.clip_sample_range,
        )
    else:
        scheduler_wo_ddp = scheduler
    scheduler_wo_ddp = scheduler_wo_ddp.to(device)

    # NOTE: this is for latent DDPM
    vae = None
    if args.latent_ddpm:
        vae = VAE(**args.vae_config)
        # NOTE: do not change this
        vae.init_from_ckpt(args.vae_ckpt)
        # Set VAE to trainable
        for param in vae.parameters():
            param.requires_grad = True
        vae.train()

    lpips_fn = lpips.LPIPS(net="vgg").to(device)
    lpips_fn.eval()

    # Note: this is for cfg
    class_embedder = None
    if args.use_cfg:
        # TODO:
        class_embedder = ClassEmbedder(
            embed_dim=args.unet_ch,
            n_classes=args.num_classes,
            cond_drop_rate=args.cond_drop_rate,
        )

    # send to device
    unet = unet.to(device)
    scheduler = scheduler.to(device)
    if vae:
        vae = vae.to(device)
    if class_embedder:
        class_embedder = class_embedder.to(device)

    # -------------------------------------------
    # -----set up optimizer and scheduler--------
    # -------------------------------------------
    # TODO: setup optimizer
    vae_params = list(vae.parameters()) if vae else []
    unet_params = list(unet.parameters())

    # if vae:
    #     optimizer = torch.optim.AdamW(
    #         [
    #             {"params": unet_params, "lr": args.learning_rate},
    #             {"params": vae_params, "lr": args.learning_rate * 0.5},
    #         ],
    #         weight_decay=args.weight_decay,
    #     )
    # else:
    #     optimizer = torch.optim.AdamW(
    #         [
    #             {"params": unet_params, "lr": args.learning_rate},
    #         ],
    #         weight_decay=args.weight_decay,
    #     )

    optimizer = torch.optim.AdamW(
        [
            {"params": unet_params, "lr": args.learning_rate},
        ],
        weight_decay=args.weight_decay,
    )

    print("[DEBUG] Current optimizer.param_groups:")
    for i, group in enumerate(optimizer.param_groups):
        print(f"  Group {i}: {len(group['params'])} params")

    # max train steps
    num_update_steps_per_epoch = len(train_loader)
    args.max_train_steps = args.num_epochs * num_update_steps_per_epoch

    # TODO: setup scheduler
    warmup_steps = args.warmup_epochs * num_update_steps_per_epoch
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.max_epochs * num_update_steps_per_epoch - warmup_steps,
        eta_min=1e-9,
    )
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],  # when to switch to cosine
        last_epoch=-1,
    )

    # todo: check this
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision in ["fp16", "bf16"])

    # -------------------------------------------
    # -----set up distributed training-----------
    # -------------------------------------------
    #  setup distributed training
    if args.distributed:
        unet = torch.nn.parallel.DistributedDataParallel(
            unet,
            device_ids=[args.device],
            output_device=args.device,
            find_unused_parameters=False,
        )
        unet_wo_ddp = unet.module
        if class_embedder:
            class_embedder = torch.nn.parallel.DistributedDataParallel(
                class_embedder,
                device_ids=[args.device],
                output_device=args.device,
                find_unused_parameters=False,
            )
            class_embedder_wo_ddp = class_embedder.module
    else:
        unet_wo_ddp = unet
        class_embedder_wo_ddp = class_embedder
    vae_wo_ddp = vae

    # -------------------------------------------
    # ------set up evaluation training-----------
    # -------------------------------------------
    # TODO: setup evaluation pipeline
    # NOTE: this pipeline is not differentiable and only for evaluatin
    pipeline = DDPMPipeline(
        unet=unet_wo_ddp,
        scheduler=scheduler_wo_ddp,
        vae=vae_wo_ddp,
        class_embedder=class_embedder_wo_ddp,
    )

    # -------------------------------------------
    # ----------------dump config----------------
    # -------------------------------------------
    # dump config file
    if is_primary(args):
        experiment_config = vars(args)
        with open(os.path.join(output_dir, "config.yaml"), "w", encoding="utf-8") as f:
            # Use the round_trip_dump method to preserve the order and style
            file_yaml = yaml.YAML()
            file_yaml.dump(experiment_config, f)

    # -------------------------------------------
    # --------------------log--------------------
    # -------------------------------------------
    # start tracker
    if is_primary(args) and not args.DEBUG:
        wandb_kwargs = {
            "project": args.project_name,
            "name": args.run_name,
            "config": vars(args),
            "settings": wandb.Settings(api_key=args.wandb_key),
        }

        # If resuming, use the same run id
        if args.resume and hasattr(args, "wandb_resume_id") and args.wandb_resume_id:
            wandb_kwargs["id"] = args.wandb_resume_id
            wandb_kwargs["resume"] = "must"
            print(f"[INFO] Resuming WandB run: {args.wandb_resume_id}")
        else:
            wandb_kwargs["resume"] = None  # fresh run

        wandb_logger = wandb.init(**wandb_kwargs)

    # ----------------------------------------------------
    # -------- Resume training from checkpoint if needed --------
    # ----------------------------------------------------
    if args.resume:
        if args.resume_checkpoint_path != "none":
            checkpoint_path = args.resume_checkpoint_path
        else:
            checkpoint_path = (
                f"{args.wandb_username}/{args.project_name}/{args.wandb_resume_id}"
            )
        checkpoint = load_checkpoint(
            unet_wo_ddp,
            scheduler_wo_ddp,
            vae=vae_wo_ddp,
            class_embedder=class_embedder_wo_ddp,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            checkpoint_path=checkpoint_path,
        )

        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        else:
            start_epoch = 0

        if "best_fid" in checkpoint:
            args.best_fid = checkpoint["best_fid"]
        else:
            args.best_fid = float("inf")

        if "best_is" in checkpoint:
            args.best_is = checkpoint["best_is"]
        else:
            args.best_is = float("-inf")

        if "best_lpips" in checkpoint:
            args.best_lpips = checkpoint["best_lpips"]
        else:
            args.best_lpips = float("inf")

    else:
        start_epoch = 0
        args.best_fid = float("inf")
        args.best_is = float("-inf")
        args.best_lpips = float("inf")

    # Start training
    if is_primary(args):
        logger.info("***** Training arguments *****")
        logger.info(args)
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(
            f"  Total optimization steps per epoch {num_update_steps_per_epoch}"
        )
        logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # -------------------------------------------
    # ----------------experiment-----------------
    # -------------------------------------------
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not is_primary(args))

    # unfreeze vae
    if (start_epoch + 1) < args.freeze_vae_epoch:
        vae_frozen = False
        print(f"[INFO] VAE is trainable")
    else:
        vae_frozen = True
        for param in vae.parameters():
            param.requires_grad = False
        vae.eval()
        vae_frozen = True
        print(f"[INFO] VAE is frozen")

    for epoch in range(start_epoch, args.num_epochs):

        # -------------------------------------------
        # -------------------train-------------------
        # -------------------------------------------
        # set epoch for distributed sampler, this is for distribution training
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        args.epoch = epoch
        if is_primary(args):
            logger.info(f"Epoch {epoch+1}/{args.num_epochs}")

        loss_m = AverageMeter()

        # TODO: set unet and scheduler to train
        unet.train()
        scheduler.train()

        # freeze vae
        if vae is not None:
            if (epoch + 1) == args.freeze_vae_epoch and not vae_frozen:
                for param in vae.parameters():
                    param.requires_grad = False
                vae.eval()
                vae_frozen = True

                new_param_groups = []
                for group in optimizer.param_groups:
                    new_params = [p for p in group["params"] if p.requires_grad]
                    if len(new_params) > 0:
                        new_group = {k: v for k, v in group.items()}
                        new_group["params"] = new_params
                        new_param_groups.append(new_group)
                optimizer.param_groups = new_param_groups
                optimizer.state = {
                    k: v
                    for k, v in optimizer.state.items()
                    if any(
                        k is p
                        for group in optimizer.param_groups
                        for p in group["params"]
                    )
                }

            if vae_frozen:
                logger.info(f"[INFO] VAE is frozen at epoch {epoch + 1}.")
            else:
                logger.info(f"[INFO] VAE is trainable at epoch {epoch + 1}.")

        # TODO: finish this
        for step, (images, labels) in enumerate(train_loader):

            # initialize the save model flags to False
            if_best_fid, if_best_is, if_best_lpips = False, False, False

            # record batch size
            batch_size = images.size(0)

            # TODO: send to device
            images = images.to(device)
            labels = labels.to(device)

            # NOTE: this is for latent DDPM
            if vae is not None:
                real_images = images.clone()
                # use vae to encode images as latents
                with torch.no_grad():
                    with torch.cuda.amp.autocast(
                        enabled=args.mixed_precision in ["fp16", "bf16"],
                    ):
                        images = vae.encode(images).sample()
                # NOTE: do not change this line, this is to ensure the latent has unit std
                images = images * 0.1845  # 0.1845

            # TODO: zero grad optimizer
            optimizer.zero_grad()

            # NOTE: this is for CFG
            if class_embedder is not None:
                # TODO: use class embedder to get class embeddings
                class_emb = class_embedder(labels)
            else:
                # NOTE: if not cfg, set class_emb to None
                class_emb = None

            # TODO: sample noise
            noise = torch.randn_like(images)

            # TODO: sample timestep t
            timesteps = torch.randint(
                0, scheduler.num_train_timesteps, (batch_size,), device=images.device
            )

            # TODO: add noise to images using scheduler
            noisy_images = scheduler.add_noise(images, noise, timesteps)

            # TODO: model prediction
            with torch.cuda.amp.autocast(
                enabled=args.mixed_precision in ["fp16", "bf16"],
            ):
                model_pred = unet(noisy_images, timesteps, class_emb)

                if args.prediction_type == "epsilon":
                    target = noise

                # TODO: calculate loss
                diffusion_loss = F.mse_loss(model_pred, target)

                # VAE losses (reconstruction + KL)
                if vae is not None and any(
                    param.requires_grad for param in vae.parameters()
                ):
                    posterior = vae.encode(real_images)
                    z = posterior.sample()
                    recon_images = vae.decode(z)

                    recon_loss = F.mse_loss(recon_images, real_images)
                    kl_loss = posterior.kl().mean()
                    perceptual_loss = lpips_fn(recon_images, real_images).mean()

                    # calculate loss
                    vae_loss = (
                        recon_loss
                        + args.kl_beta * kl_loss
                        + args.lpips_lambda * perceptual_loss
                    )
                    vae_lambda = (
                        min(1.0, (epoch + 1) / (args.freeze_vae_epoch + 1))
                        * args.vae_lambda
                    )
                    # combine loss
                    loss = diffusion_loss + vae_lambda * vae_loss

                else:
                    loss = diffusion_loss

            # record loss
            loss_m.update(loss.item())

            # backward and step
            # todo : check gradient_accumulation_steps
            if scaler:
                scaler.scale(loss).backward()

                # TODO: grad clip
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), args.grad_clip)

                # TODO: step your optimizer
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()

                # TODO: grad clip
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), args.grad_clip)

                optimizer.step()

            # learning rate scheduler step
            lr_scheduler.step()

            progress_bar.update(1)

            # logger
            if step % 100 == 0 and is_primary(args):
                logger.info(
                    f"Epoch {epoch+1}/{args.num_epochs}, Step {step}/{num_update_steps_per_epoch}, Loss {loss.item()} ({loss_m.avg}), LR {optimizer.param_groups[0]['lr']}"
                )

                if is_primary(args) and not args.DEBUG:
                    wandb_log_dict = {
                        "train/loss": loss_m.avg,
                        "train/diffusion_loss": diffusion_loss.item(),
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                    }
                    if vae is not None and any(
                        param.requires_grad for param in vae.parameters()
                    ):
                        wandb_log_dict.update(
                            {
                                "train/vae_loss": vae_loss.item(),
                                "train/vae_lambda": vae_lambda,
                                "train/recon_loss": recon_loss.item(),
                                "train/kl_loss": kl_loss.item(),
                                "train/perceptual_loss": perceptual_loss.item(),
                            }
                        )

                    wandb_logger.log(wandb_log_dict)

                if (
                    args.DEBUG
                    and vae is not None
                    and any(param.requires_grad for param in vae.parameters())
                ):
                    vae_grads = [
                        param.grad.abs().mean()
                        for param in vae.parameters()
                        if param.grad is not None
                    ]
                    if len(vae_grads) > 0:
                        mean_grad = torch.stack(vae_grads).mean().item()
                        print(
                            f"[Epoch {epoch} Step {step}] VAE Mean Grad: {mean_grad:.6f}"
                        )
                    else:
                        print(f"[Epoch {epoch} Step {step}] VAE: No grads")

        print("[DEBUG] Current optimizer.param_groups:")
        for i, group in enumerate(optimizer.param_groups):
            print(f"  Group {i}: {len(group['params'])} params")

        # -------------------------------------------
        # ----------------validation-----------------
        # -------------------------------------------
        # send unet to evaluation mode
        unet.eval()
        generator = torch.Generator(device=device)
        generator.manual_seed(epoch + args.seed)

        # NOTE: this is for CFG
        if args.use_cfg:
            # random sample 4 classes
            classes = torch.randint(
                0, args.num_classes, (args.batch_size,), device=device
            )
            # TODO: fill pipeline
            gen_images = pipeline(
                batch_size=args.batch_size,
                num_inference_steps=args.num_inference_steps,
                classes=classes,
                guidance_scale=args.cfg_guidance_scale,
                generator=generator,
                device=device,
            )
        else:
            # TODO: fill pipeline
            # ERROR
            gen_images = pipeline(
                batch_size=args.batch_size,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                device=device,
            )

        # create a blank canvas for the grid
        grid_image = Image.new("RGB", (4 * args.image_size, 1 * args.image_size))
        # paste images into the grid
        for i, image in enumerate(gen_images[:4]):
            x = (i % 4) * args.image_size
            y = 0
            grid_image.paste(image, (x, y))

        # Send to wandb
        if is_primary(args) and not args.DEBUG:
            wandb_logger.log({"gen_images": wandb.Image(grid_image)})

        # -------------------------------------------
        # -----------FID / IS Evaluation(Optional)-----------
        # -------------------------------------------
        if (
            args.eval_during_train
            and (epoch + 1) % args.eval_every_n_epoch == 0
            and is_primary(args)
        ):  # Execute evaluation every 10 epochs
            # if (epoch + 1) % args.eval_every_n_epoch == 0 and is_primary(args):  # Execute evaluation every 10 epochs
            logger.info(f"[Epoch {epoch+1}] Running FID/IS Evaluation...")

            # Generate 500 images（unconditional or conditional）
            all_images = []
            eval_samples = args.eval_samples
            eval_classes = args.eval_classes
            steps = eval_samples // eval_classes
            logger.info(
                f"Generating {eval_samples} images for {eval_classes} classes in {steps} steps..."
            )

            for _ in range(steps):
                if args.use_cfg:
                    classes = torch.randint(
                        0, args.num_classes, (eval_classes,), device=device
                    )
                    batch = pipeline(
                        batch_size=eval_classes,
                        num_inference_steps=args.num_inference_steps,
                        classes=classes,
                        guidance_scale=args.cfg_guidance_scale,
                        generator=generator,
                        device=device,
                    )
                else:
                    batch = pipeline(
                        batch_size=eval_classes,
                        num_inference_steps=args.num_inference_steps,
                        generator=generator,
                        device=device,
                    )
                all_images.extend(batch)

            # convert to tensor
            from torchvision import transforms

            to_tensor = transforms.ToTensor()
            all_images = [to_tensor(img) for img in all_images]
            all_images = torch.stack(all_images)

            val_loader = build_val_loader(
                dataset_name=args.dataset,
                val_data_dir=args.val_data_dir,
                data_dir=args.data_dir,
                image_size=args.image_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                subset_ratio=args.subset,
                distributed=args.distributed,
                world_size=args.world_size,
                rank=args.rank,
            )

            # Call evaluation function
            fid_val, is_mean, is_std, lpips_value = evaluate_fid_is_lpips(
                generated_images=all_images,
                val_loader=val_loader,
                device=device,
                total_images=eval_samples,
                logger=logger,
            )

            # compare and save best FID model
            if fid_val < args.best_fid:
                args.best_fid = fid_val
                if_best_fid = True

            # compare and save best IS model
            if is_mean > args.best_is:
                args.best_is = is_mean
                if_best_is = True

            if lpips_value < args.best_lpips:
                args.best_lpips = lpips_value
                if_best_lpips = True

            # log to console
            logger.info(
                f"Epoch {epoch+1}/{args.num_epochs}, FID: {fid_val}, IS: {is_mean} ± {is_std}, Best FID: {args.best_fid}, Best IS: {args.best_is}, Best LPIPS: {args.best_lpips}"
            )

            # log to wandb
            if is_primary(args) and not args.DEBUG:
                print("[INFO] Save fid/is to wandb")
                wandb_logger.log(
                    {
                        "eval/fid": fid_val,
                        "eval/is_mean": is_mean,
                        "eval/is_std": is_std,
                        "eval/lpips": lpips_value,
                        "eval/best_fid": args.best_fid,
                        "eval/best_is": args.best_is,
                        "eval/best_lpips": args.best_lpips,
                    },
                )

        # save checkpoint
        save_checkpoint(
            unet_wo_ddp,
            scheduler_wo_ddp,
            vae_wo_ddp,
            class_embedder,
            optimizer,
            lr_scheduler,
            epoch,
            save_dir=save_dir,
            keep_last_n=args.keep_last_n,
            keep_last_model=(args.keep_last_model if (epoch + 1) % 10 == 0 else False),
            keep_best_model=args.keep_best_model,
            if_best_fid=if_best_fid,
            if_best_is=if_best_is,
            if_best_lpips=if_best_lpips,
            best_fid=float("inf") if not args.best_fid else args.best_fid,
            best_is=float("-inf") if not args.best_is else args.best_is,
            best_lpips=float("inf") if not args.best_lpips else args.best_lpips,
        )

    if is_primary(args) and not args.DEBUG:
        wandb_logger.finish()
        logger.info("WandB run finished.")

    logger.info("Training finished.")


if __name__ == "__main__":
    # args = parse_args()
    main()
