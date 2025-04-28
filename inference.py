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

from models import UNet, VAE, ClassEmbedder
from schedulers import DDPMScheduler, DDIMScheduler
from pipelines import DDPMPipeline
from utils import (
    seed_everything,
    load_checkpoint,
    is_primary,
    evaluate_fid_is_lpips,
    build_val_loader,
)

from train import parse_args

logger = get_logger(__name__)


def main():
    # parse arguments
    args = parse_args()

    # start tracker
    if is_primary(args) and not args.DEBUG:
        wandb.login(key=args.wandb_key)
        wandb_logger = wandb.init(
            project=args.project_name, name=args.run_name, config=vars(args)
        )

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # seed everything
    seed_everything(args.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

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
    # print number of parameters
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

    # vae
    vae = None
    if args.latent_ddpm:
        vae = VAE()
        vae.init_from_ckpt("pretrained/model.ckpt")
        vae.eval()
    # cfg
    class_embedder = None
    if args.use_cfg:
        # TODO: class embeder
        class_embedder = ClassEmbedder(
            embed_dim=args.unet_ch,
            n_classes=args.num_classes,
            cond_drop_rate=args.cond_drop_rate,
        )
        # class_embedder = ClassEmbedder(None)

    # send to device
    unet = unet.to(device)
    scheduler = scheduler.to(device)
    if vae:
        vae = vae.to(device)
    if class_embedder:
        class_embedder = class_embedder.to(device)

    # scheduler
    if args.use_ddim:
        scheduler = DDIMScheduler(
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
        scheduler = scheduler

    # # scheduler
    # if args.use_ddim:
    #     scheduler_class = DDIMScheduler
    # else:
    #     scheduler_class = DDPMScheduler
    # # TOOD: scheduler
    # scheduler = scheduler_class(None)

    # load checkpoint
    load_checkpoint(
        unet,
        scheduler,
        vae=vae,
        class_embedder=class_embedder,
        checkpoint_path=args.ckpt,
    )

    # TODO: pipeline
    pipeline = DDPMPipeline(
        unet=unet, scheduler=scheduler, vae=vae, class_embedder=class_embedder
    )

    logger.info("***** Running Inference *****")

    # TODO: we run inference to generation 5000 images
    # TODO: with cfg, we generate 50 images per class
    all_images = []
    if args.use_cfg:
        # generate images per class
        total_images = args.inference_samples
        batch_size = total_images // args.num_classes
        for i in tqdm(range(args.num_classes)):
            # for i in tqdm(range(1)):
            logger.info(f"Generating {batch_size} images for class {i}")
            classes = torch.full((batch_size,), i, dtype=torch.long, device=device)
            gen_images = pipeline(
                batch_size=batch_size,
                num_inference_steps=args.num_inference_steps,
                classes=classes,
                guidance_scale=args.cfg_guidance_scale,
                generator=generator,
                device=device,
            )
            all_images.append(gen_images)
    else:
        # generate 5000 images
        total_images = args.inference_samples
        remaining = total_images
        batch_size = args.batch_size

        while remaining > 0:
            curr_batch_size = min(batch_size, remaining)
            gen_images = pipeline(
                batch_size=curr_batch_size,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                device=device,
            )
            # why need to rescale
            all_images.extend(gen_images)
            remaining -= curr_batch_size

    from torchvision import transforms

    to_tensor = transforms.ToTensor()
    # flatten list if necessary
    if isinstance(all_images[0], list):  # means use_cfg == True
        all_images = [img for batch in all_images for img in batch]

    # now convert to tensor
    all_images = [to_tensor(img) for img in all_images]
    all_images = torch.stack(all_images, dim=0)

    # sample from all_images
    for i in range(50):
        start_idx = i * 4
        end_idx = start_idx + 4
        sample_images = all_images[start_idx:end_idx]
        grid_image = Image.new("RGB", (4 * args.image_size, 1 * args.image_size))
        for i, image in enumerate(sample_images):
            x = (i % 4) * args.image_size
            y = 0
            pil_img = transforms.ToPILImage()(image.cpu())
            grid_image.paste(pil_img, (x, y))

        if is_primary(args) and not args.DEBUG:
            wandb_logger.log({"infer_images": wandb.Image(grid_image)})

    """
    # TODO: load validation images as reference batch
    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    if args.dataset == "imagenet100":
        val_dataset = datasets.ImageFolder(args.val_data_dir, transform=transform)
    elif args.dataset == "cifar10":
        val_dataset = datasets.CIFAR10(
            args.data_dir, train=False, transform=transform, download=True
        )

    subset_size = int(len(val_dataset) * args.subset)
    indices = np.random.choice(len(val_dataset), subset_size, replace=False)
    val_dataset = torch.utils.data.Subset(val_dataset, indices)

    sampler = None
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False,  # No shuffling for inference
        )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
    )
    
    real_images = []
    for batch, _ in tqdm(val_loader, desc="Loading validation images"):
        real_images.append(batch)
        if (
            len(real_images) * batch_size >= total_images
        ):  # Use same number of real images as generated
            break

    real_images = torch.cat(real_images, dim=0)[:5000]  # should i load all then sample?

    # TODO: using torchmetrics for evaluation, check the documents of torchmetrics
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    from torch.utils.data import DataLoader, TensorDataset

    # Set up metrics
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    is_score = InceptionScore(normalize=True).to(device)

    # Ensure image format is uint8 in [0, 255] and move to device
    if all_images.dtype != torch.uint8:
        all_images = (all_images * 255).clamp(0, 255).to(torch.uint8)
    if real_images.dtype != torch.uint8:
        real_images = (real_images * 0.5 + 0.5) * 255
        #real_images = (real_images * 255)
        real_images = real_images.clamp(0, 255).to(torch.uint8)

    # Dataloaders (batching avoids OOM)
    batch_size = 50
    real_loader = DataLoader(real_images, batch_size=batch_size)
    fake_loader = DataLoader(all_images, batch_size=batch_size)

    # Update FID: real
    for batch in tqdm(real_loader, desc="Updating FID with real images"):
        fid.update(batch.to(device), real=True)

    # Update FID + IS: fake
    for batch in tqdm(fake_loader, desc="Updating FID and IS with generated images"):
        batch = batch.to(device)
        fid.update(batch, real=False)
        is_score.update(batch)

    # Compute results
    fid_value = fid.compute()
    is_mean, is_std = is_score.compute()

    logger.info(f"FID: {fid_value.item():.2f}")
    logger.info(f"Inception Score: {is_mean.item():.2f} ± {is_std.item():.2f}")
    """
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

    fid_val, is_mean, is_std, lpips_value = evaluate_fid_is_lpips(
        generated_images=all_images,  # [N, C, H, W] float in [-1, 1]
        val_loader=val_loader,  # already built earlier
        device=device,
        total_images=len(all_images),  # usually 5000
        batch_size=50,
        logger=logger,  # so it prints via logger
    )


if __name__ == "__main__":
    main()
