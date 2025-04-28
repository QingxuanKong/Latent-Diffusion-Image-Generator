from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from tqdm import tqdm

@torch.no_grad()
def evaluate_fid_is_lpips(
    generated_images,  # tensor of shape [N, C, H, W]
    val_loader,  # dataloader yielding real images
    device,
    total_images=5000,
    batch_size=50,
    logger=None,
):
    """
    Evaluate FID, Inception Score, and LPIPS on given generated images.
    Assumes val_loader yields normalized real images in [-1, 1].
    """
    # Ensure type and range for generated images
    if generated_images.dtype != torch.uint8:
        generated_images = (generated_images * 255).clamp(0, 255).to(torch.uint8)

    # Load real images (already normalized in val_loader)
    real_images = []
    for batch, _ in tqdm(val_loader, desc="Loading real images for FID/LPIPS"):
        real_images.append(batch)
        if len(real_images) * batch.size(0) >= total_images:
            break
    real_images = torch.cat(real_images, dim=0)[:total_images]
    real_images = (real_images * 0.5 + 0.5) * 255
    real_images = real_images.clamp(0, 255).to(torch.uint8)

    # Create metrics
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    is_score = InceptionScore(normalize=True).to(device)
    lpips = LPIPS(net_type='vgg').to(device)  # Using VGG backbone (common choice)

    # Dataloaders for batching
    fake_loader = DataLoader(generated_images, batch_size=batch_size)
    real_loader = DataLoader(real_images, batch_size=batch_size)

    # Update FID and LPIPS
    lpips_scores = []
    real_iter = iter(real_loader)
    for fake_batch in tqdm(fake_loader, desc="Updating metrics with fake"):
        fake_batch = fake_batch.to(device)

        # FID
        fid.update(fake_batch, real=False)

        # IS
        is_score.update(fake_batch)

        # LPIPS (pair fake-real images)
        try:
            real_batch = next(real_iter).to(device)
        except StopIteration:
            break  # end
        lpips_value = lpips(fake_batch.float() / 255.0, real_batch.float() / 255.0)
        lpips_scores.append(lpips_value)

    # Update FID real part
    for real_batch in tqdm(real_loader, desc="Updating FID with real"):
        fid.update(real_batch.to(device), real=True)

    # Compute
    fid_value = fid.compute().item()
    is_mean, is_std = is_score.compute()
    is_mean = is_mean.item()
    is_std = is_std.item()
    lpips_value = torch.stack(lpips_scores, dim=0).mean().item()

    # Logging
    if logger:
        logger.info(f"[Eval] FID: {fid_value:.2f}")
        logger.info(f"[Eval] IS: {is_mean:.2f} ± {is_std:.2f}")
        logger.info(f"[Eval] LPIPS: {lpips_value:.4f}")
    else:
        print(f"[Eval] FID: {fid_value:.2f}")
        print(f"[Eval] IS: {is_mean:.2f} ± {is_std:.2f}")
        print(f"[Eval] LPIPS: {lpips_value:.4f}")

    return fid_value, is_mean, is_std, lpips_value
