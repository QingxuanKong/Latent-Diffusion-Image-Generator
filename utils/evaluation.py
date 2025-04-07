from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from tqdm import tqdm

@torch.no_grad()
def evaluate_fid_is(
    generated_images,             # tensor of shape [N, C, H, W]
    val_loader,                   # dataloader yielding real images
    device,
    total_images=5000,
    batch_size=50,
    logger=None
):
    """
    Evaluate FID and Inception Score on given generated images.
    Assumes val_loader yields normalized real images in [-1, 1].
    """
    # Ensure type and range for generated images
    if generated_images.dtype != torch.uint8:
        generated_images = (generated_images * 255).clamp(0, 255).to(torch.uint8)

    # Load real images (already normalized in val_loader)
    real_images = []
    for batch, _ in tqdm(val_loader, desc="Loading real images for FID"):
        real_images.append(batch)
        if len(real_images) * batch.size(0) >= total_images:
            break
    real_images = torch.cat(real_images, dim=0)[:total_images]
    real_images = (real_images * 0.5 + 0.5) * 255
    real_images = real_images.clamp(0, 255).to(torch.uint8)

    # Create metrics
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    is_score = InceptionScore(normalize=True).to(device)

    # Dataloaders for batching
    fake_loader = DataLoader(generated_images, batch_size=batch_size)
    real_loader = DataLoader(real_images, batch_size=batch_size)

    for batch in tqdm(real_loader, desc="Updating FID with real"):
        fid.update(batch.to(device), real=True)
    for batch in tqdm(fake_loader, desc="Updating FID and IS with fake"):
        batch = batch.to(device)
        fid.update(batch, real=False)
        is_score.update(batch)

    fid_value = fid.compute().item()
    is_mean, is_std = is_score.compute()

    if logger:
        logger.info(f"[Eval] FID: {fid_value:.2f}")
        logger.info(f"[Eval] IS: {is_mean:.2f} ± {is_std:.2f}")
    else:
        print(f"[Eval] FID: {fid_value:.2f}")
        print(f"[Eval] IS: {is_mean:.2f} ± {is_std:.2f}")

    return fid_value, is_mean, is_std
