import torch
import os
import wandb


def load_checkpoint(
    unet,
    scheduler,
    vae=None,
    class_embedder=None,
    optimizer=None,
    checkpoint_path="checkpoints/checkpoint.pth",
):

    print("loading checkpoint")
    checkpoint = torch.load(checkpoint_path)

    print("loading unet")
    unet.load_state_dict(checkpoint["unet_state_dict"])
    print("loading scheduler")
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if vae is not None and "vae_state_dict" in checkpoint:
        print("loading vae")
        vae.load_state_dict(checkpoint["vae_state_dict"])

    if class_embedder is not None and "class_embedder_state_dict" in checkpoint:
        print("loading class_embedder")
        class_embedder.load_state_dict(checkpoint["class_embedder_state_dict"])


def save_checkpoint(
    unet,
    scheduler,
    vae=None,
    class_embedder=None,
    optimizer=None,
    epoch=None,
    save_dir="checkpoints",
    keep_last_n=10,
    keep_last_model=True,
    keep_best_model=True,
    if_best_fid=False,
    if_best_is=False,
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

    if epoch is not None:
        checkpoint["epoch"] = epoch

    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

    # Save the last_model，and upload to WandB
    if keep_last_model:
        last_model_path = os.path.join(save_dir, "last_model.pth")
        torch.save(checkpoint, last_model_path)
        print(f"Last model saved at {last_model_path}")

        if wandb.run:
            artifact = wandb.Artifact("last_model", type="model")
            artifact.add_file(last_model_path)
            wandb.log_artifact(artifact, aliases=["latest"])
            print("Uploaded last model to WandB with alias 'latest'")

    # Manage checkpoint history
    print(f"Managing checkpoint history (only keep {keep_last_n} last checkpoints)")
    manage_checkpoints(save_dir, keep_last_n=keep_last_n)

    # Save the best fid model and upload to WandB
    if keep_best_model and if_best_fid:
        best_fid_model_path = os.path.join(save_dir, "best_fid_model.pth")
        torch.save(checkpoint, best_fid_model_path)
        print(f"Best FID model saved at {best_fid_model_path}")

        if wandb.run:
            artifact = wandb.Artifact("best_fid_model", type="model")
            artifact.add_file(best_fid_model_path)
            wandb.log_artifact(artifact, aliases=["best-fid"])
            print("Uploaded best FID model to WandB with alias 'best-fid'")

    if keep_best_model and if_best_is:
        best_is_model_path = os.path.join(save_dir, "best_is_model.pth")
        torch.save(checkpoint, best_is_model_path)
        print(f"Best IS model saved at {best_is_model_path}")

        if wandb.run:
            artifact = wandb.Artifact("best_is_model", type="model")
            artifact.add_file(best_is_model_path)
            wandb.log_artifact(artifact, aliases=["best-is"])
            print("Uploaded best IS model to WandB with alias 'best-is'")


def manage_checkpoints(save_dir, keep_last_n=10):
    # List all checkpoint files in the save directory
    checkpoints = [f for f in os.listdir(save_dir) if f.startswith("checkpoint_epoch_")]
    checkpoints.sort(
        key=lambda f: int(f.split("_")[-1].split(".")[0])
    )  # Sort by epoch number

    # If more than `keep_last_n` checkpoints exist, remove the oldest ones
    print(f"Found {len(checkpoints)} checkpoints in {save_dir}.")
    if len(checkpoints) > keep_last_n:
        for checkpoint_file in checkpoints[:-keep_last_n]:
            checkpoint_path = os.path.join(save_dir, checkpoint_file)
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                print(f"Removed old checkpoint: {checkpoint_path}")
