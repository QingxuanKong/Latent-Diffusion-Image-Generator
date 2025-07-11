from typing import List, Optional, Tuple, Union
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from utils import randn_tensor


class DDPMPipeline:
    def __init__(self, unet, scheduler, vae=None, class_embedder=None):
        self.unet = unet
        self.scheduler = scheduler

        # NOTE: this is for latent DDPM
        self.vae = None
        if vae is not None:
            self.vae = vae

        # NOTE: this is for CFG
        if class_embedder is not None:
            self.class_embedder = class_embedder

    def numpy_to_pil(self, images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[
                None, ...
            ]  # [height, width, channels] -> [batch_size, height, width, channels]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [
                Image.fromarray(image.squeeze(), mode="L") for image in images
            ]  # [batch_size, height, width, channels] -> [batch_size, height, width]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        num_inference_steps: int = 1000,
        classes: Optional[Union[int, List[int]]] = None,
        guidance_scale: Optional[float] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        device=None,
    ):
        image_shape = (
            batch_size,
            self.unet.input_ch,
            self.unet.input_size,
            self.unet.input_size,
        )
        if device is None:
            device = next(self.unet.parameters()).device

        # NOTE: this is for CFG
        if classes is not None or guidance_scale is not None:
            assert hasattr(self, "class_embedder"), "class_embedder is not defined"

        if classes is not None:
            # convert classes to tensor
            if isinstance(classes, int):
                classes = [classes] * batch_size
            elif isinstance(classes, list):
                assert (
                    len(classes) == batch_size
                ), "Length of classes must be equal to batch_size"
                classes = torch.tensor(classes, device=device)

            # TODO: get uncond classes
            uncond_classes = torch.full_like(classes, self.class_embedder.uncond_id)
            # TODO: get class embeddings from classes
            class_embeds = self.class_embedder(classes)
            # TODO: get uncon class embeddings
            uncond_embeds = self.class_embedder(uncond_classes)

        # TODO: starts with random noise
        image = randn_tensor(
            image_shape, generator=generator, device=device, dtype=torch.float32
        )
        # print(
        #     f"DEBUG: Initial noise stats - min: {image.min().item():.4f}, max: {image.max().item():.4f}, mean: {image.mean().item():.4f}"
        # )

        # TODO: set step values using set_timesteps of scheduler
        self.scheduler.set_timesteps(num_inference_steps, device)

        # TODO: inverse diffusion process with for loop
        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # print(f"\nDEBUG: Step {t.item()+1}/{len(self.scheduler.timesteps)}")

            # NOTE: this is for CFG
            if guidance_scale is not None and guidance_scale != 1.0:
                # TODO: implement cfg
                model_input = torch.cat([image] * 2, dim=0)
                c = torch.cat([uncond_embeds, class_embeds], dim=0)
            else:
                model_input = image
                # NOTE: leave c as None if you are not using CFG
                c = class_embeds if classes is not None else None
            # print(
            #     f"DEBUG: Initial noise stats - min: {model_input.min().item():.4f}, max: {model_input.max().item():.4f}, mean: {model_input.mean().item():.4f}"
            # )

            # TODO: 1. predict noise model_output
            model_output = self.unet(model_input, t, c)  # what is c
            # print(
            #     f"DEBUG: Model output shape: {model_output.shape}, min: {model_output.min().item():.4f}, max: {model_output.max().item():.4f}"
            # )

            if guidance_scale is not None and guidance_scale != 1.0:
                # TODO: implement cfg
                uncond_model_output, cond_model_output = model_output.chunk(2)
                model_output = uncond_model_output + guidance_scale * (
                    cond_model_output - uncond_model_output
                )

            # TODO: 2. compute previous image: x_t -> x_t-1 (less noisy) using scheduler
            prev_image = self.scheduler.step(model_output, t, image, generator)
            # print(
            #     f"DEBUG: After step - min: {prev_image.min().item():.4f}, max: {prev_image.max().item():.4f}, diff: {torch.abs(prev_image - image).mean().item():.6f}"
            # )
            image = prev_image

        # NOTE: this is for latent DDPM
        # TODO: use VQVAE to get final image
        if self.vae is not None:
            # NOTE: remember to rescale your images
            image = 1 / 0.1845 * image
            # Decode the image
            image = self.vae.decode(image)

        # TODO: return final image, re-scale to [0, 1]
        image = (image + 1) / 2
        image = torch.clamp(image, 0, 1)
        # print(
        #     f"DEBUG: Final image stats - min: {image.min().item():.4f}, max: {image.max().item():.4f}, mean: {image.mean().item():.4f}"
        # )

        # convert to PIL images
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = self.numpy_to_pil(image)

        return image
