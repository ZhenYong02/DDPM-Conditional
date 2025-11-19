"""
Unconditional Image Generation Script for DDPM Model.

This script generates images using an unconditional DDPM model (no class conditioning).

Usage:
    python generate_unconditioned.py --checkpoint checkpoint/cifar10.pth \\
        --output_dir result_for_eval/uncond --num_images 200

    python generate_unconditioned.py --checkpoint checkpoint/cifar10.pth \\
        --output_dir result_for_eval/uncond --num_images 100 --device cpu
"""

import argparse
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def load_model_and_sampler(checkpoint_path, device):
    """
    Load unconditional model and sampler.

    Args:
        checkpoint_path: Path to checkpoint file
        device: torch.device

    Returns:
        model, sampler, config
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    cp = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Import unconditional model and sampler
    from model_uncond.UNet import UNet
    from utils_uncond.engine import DDPMSampler

    model = UNet(**cp["config"]["Model"])
    model.load_state_dict(cp["model"])
    model.to(device)
    model.eval()

    sampler = DDPMSampler(model, **cp["config"]["Trainer"]).to(device)

    return model, sampler, cp["config"]


@torch.no_grad()
def generate_single_image(sampler, in_channels, image_size, device):
    """
    Generate a single unconditional image.

    Args:
        sampler: DDPM sampler
        in_channels: Number of input channels
        image_size: Image size tuple (H, W)
        device: torch.device

    Returns:
        Generated image tensor
    """
    # Generate Gaussian noise for a single image
    z_t = torch.randn((1, in_channels, *image_size), device=device)

    # Generate image (unconditional - no class labels)
    x = sampler(z_t, only_return_x_0=True, interval=50)

    return x


def save_single_image(tensor, path):
    """
    Save a single image tensor to file.

    Args:
        tensor: Image tensor with shape (1, C, H, W) in range [-1, 1]
        path: Output file path
    """
    # Normalize from [-1, 1] to [0, 1]
    tensor = tensor * 0.5 + 0.5
    # Clamp to [0, 1] and convert to [0, 255]
    tensor = (tensor.squeeze(0)
              .mul(255)
              .clamp_(0, 255)
              .permute(1, 2, 0)
              .to("cpu", torch.uint8)
              .numpy())
    # Save using PIL
    im = Image.fromarray(tensor)
    im.save(path)


def main(args):
    """Main generation function."""
    device = torch.device(args.device)

    # Print configuration
    print("=" * 80)
    print("DDPM Unconditional Image Generation")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of images: {args.num_images}")
    print(f"Device: {args.device}")
    print("=" * 80)
    print()

    # Load model and sampler
    model, sampler, config = load_model_and_sampler(args.checkpoint, device)

    # Get model configuration
    in_channels = config["Model"]["in_channels"]
    image_size = config["Dataset"]["image_size"]

    print(f"Model loaded successfully!")
    print(f"  Image size: {image_size}")
    print(f"  Channels: {in_channels}")
    print(f"  Mode: Unconditional (no class labels)")
    print()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate images
    print("Starting image generation...")
    print()

    with tqdm(total=args.num_images, desc="Generating images") as pbar:
        for img_idx in range(1, args.num_images + 1):
            # Generate image
            image = generate_single_image(sampler, in_channels, image_size, device)

            # Save image with naming convention: image<index>.png
            filename = f"image{img_idx}.png"
            save_path = output_path / filename
            save_single_image(image, save_path)

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({"current": img_idx})

    print()
    print(f"✓ Generation complete! {args.num_images} images saved to {output_path}/")
    print()

    # Verify generated images
    total_generated = len(list(output_path.glob("image*.png")))
    print(f"Total images generated: {total_generated}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate unconditional images using DDPM model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 200 images
  python generate_unconditioned.py --checkpoint checkpoint/cifar10.pth \\
      --output_dir result_for_eval/uncond --num_images 200

  # Generate 100 images on CPU
  python generate_unconditioned.py --checkpoint checkpoint/cifar10.pth \\
      --output_dir result_for_eval/uncond --num_images 100 --device cpu
        """
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint file"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for generated images"
    )

    # Optional arguments
    parser.add_argument(
        "--num_images",
        type=int,
        default=200,
        help="Total number of images to generate (default: 200)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use: cuda or cpu (default: cuda)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.num_images <= 0:
        parser.error("num_images must be > 0")

    main(args)
