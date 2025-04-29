import torch
import os
import wandb
import shutil


def load_checkpoint(
    unet,
    scheduler,
    vae=None,
    class_embedder=None,
    optimizer=None,
    lr_scheduler=None,
    checkpoint_path="checkpoints/checkpoint.pth",
):

    print("[INFO] Loading checkpoint")
    if "experiments/" not in checkpoint_path:
        artifact = wandb.run.use_artifact(
            f"{checkpoint_path}-last_model:latest", type="model"
        )
        artifact_dir = artifact.download()
        checkpoint_path = os.path.join(artifact_dir, "last_model.pth")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        shutil.rmtree(artifact_dir)
    else:
        checkpoint = torch.load(checkpoint_path, weights_only=False)

    if os.path.exists(checkpoint_path):
        print(f"[INFO] Resumed from checkpoint: {checkpoint_path}")
    else:
        print(
            f"[WARN] Resume flag is True but no checkpoint found at: {checkpoint_path}"
        )

    print("[INFO] Loading unet")
    unet.load_state_dict(checkpoint["unet_state_dict"])

    print("[INFO] Loading scheduler")
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if vae is not None and "vae_state_dict" in checkpoint:
        print("[INFO] Loading vae")
        vae.load_state_dict(checkpoint["vae_state_dict"])

    if class_embedder is not None and "class_embedder_state_dict" in checkpoint:
        print("[INFO] Loading class embedder")
        class_embedder.load_state_dict(checkpoint["class_embedder_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        print("[INFO] Loading optimizer")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if lr_scheduler is not None and "lr_scheduler_state_dict" in checkpoint:
        print("[INFO] Loading lr scheduler")
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

    return checkpoint


def save_checkpoint(
    unet,
    scheduler,
    vae=None,
    class_embedder=None,
    optimizer=None,
    lr_scheduler=None,
    epoch=None,
    save_dir="checkpoints",
    keep_last_n=10,
    keep_last_model=True,
    keep_best_model=True,
    if_best_fid=False,
    if_best_is=False,
    if_best_lpips=False,
    best_fid=None,
    best_is=None,
    best_lpips=None,
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define checkpoint file name
    checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth")

    checkpoint = {
        "unet_state_dict": unet.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }

    if vae is not None:
        checkpoint["vae_state_dict"] = vae.state_dict()

    if class_embedder is not None:
        checkpoint["class_embedder_state_dict"] = class_embedder.state_dict()

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if lr_scheduler is not None:
        checkpoint["lr_scheduler_state_dict"] = lr_scheduler.state_dict()

    if epoch is not None:
        checkpoint["epoch"] = epoch

    if best_fid is not None:
        checkpoint["best_fid"] = best_fid

    if best_is is not None:
        checkpoint["best_is"] = best_is

    if best_lpips is not None:
        checkpoint["best_lpips"] = best_lpips

    # Manage checkpoint history
    print(
        f"[INFO] Managing checkpoint history (only keep {keep_last_n} last checkpoints)"
    )
    manage_checkpoints(save_dir, keep_last_n=keep_last_n)

    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"[INFO] Checkpoint saved at {checkpoint_path}")

    # Save the last_model，and upload to WandB
    if keep_last_model:
        if wandb.run:
            artifact = wandb.Artifact(f"{wandb.run.id}-last_model", type="model")
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact, aliases=["latest"])
            print("[INFO] Uploaded last model to WandB with alias 'latest'")

    # Save the best fid model and upload to WandB
    if keep_best_model and if_best_fid:
        if wandb.run:
            artifact = wandb.Artifact(f"{wandb.run.id}-best_fid_model", type="model")
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact, aliases=["best-fid"])
            print("[INFO] Uploaded best FID model to WandB with alias 'best-fid'")

    if keep_best_model and if_best_is:
        if wandb.run:
            artifact = wandb.Artifact(f"{wandb.run.id}-best_is_model", type="model")
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact, aliases=["best-is"])
            print("[INFO] Uploaded best IS model to WandB with alias 'best-is'")

    if keep_best_model and if_best_lpips:
        if wandb.run:
            artifact = wandb.Artifact(f"{wandb.run.id}-best_lpips_model", type="model")
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact, aliases=["best-lpips"])
            print("[INFO] Uploaded best LPIPS model to WandB with alias 'best-lpips'")


def manage_checkpoints(save_dir, keep_last_n=10):
    # List all checkpoint files in the save directory
    checkpoints = [f for f in os.listdir(save_dir) if f.startswith("checkpoint_epoch_")]
    checkpoints.sort(
        key=lambda f: int(f.split("_")[-1].split(".")[0])
    )  # Sort by epoch number

    # If more than `keep_last_n` checkpoints exist, remove the oldest ones
    print(f"Found {len(checkpoints)} checkpoints in {save_dir}.")
    if keep_last_n == 1:
        for checkpoint_file in checkpoints:
            checkpoint_path = os.path.join(save_dir, checkpoint_file)
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                print(f"Removed old checkpoint: {checkpoint_path}")
    elif len(checkpoints) > keep_last_n - 1:
        for checkpoint_file in checkpoints[: -keep_last_n + 1]:
            checkpoint_path = os.path.join(save_dir, checkpoint_file)
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                print(f"Removed old checkpoint: {checkpoint_path}")
