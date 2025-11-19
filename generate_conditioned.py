"""
Unified Evaluation Image Generation Script for DDPM Models.

This script supports three conditioning methods:
1. time: Conditioning embedded at timestep (UNet_emb_at_time)
2. input: Conditioning concatenated at input channels (UNet_emb_at_input)
3. cfg: Classifier-Free Guidance (UNet_emb_at_time + CFG sampler)

Usage:
    # Time-based conditioning
    python generate_conditioned.py --method time --checkpoint checkpoint/cifar10_cond.pth \\
        --output_dir cond_time --images_per_class 20

    # Input-based conditioning
    python generate_conditioned.py --method input --checkpoint checkpoint/cifar10_cond_at_input.pth \\
        --output_dir cond_input --images_per_class 20

    # CFG with guidance scale
    python generate_conditioned.py --method cfg --checkpoint checkpoint/cifar10_cfg.pth \\
        --output_dir cfg_eval --images_per_class 20 --guidance_scale 3.0
"""

import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from tqdm import tqdm


# CIFAR-10 class names
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def load_model_and_sampler(checkpoint_path, method, device):
    """
    Load model and sampler based on conditioning method.

    Args:
        checkpoint_path: Path to checkpoint file
        method: Conditioning method ('time', 'input', or 'cfg')
        device: torch.device

    Returns:
        model, sampler, config
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    cp = torch.load(checkpoint_path, map_location=device)

    if method == "input":
        # Use UNet_emb_at_input for input channel conditioning
        from model.UNet_emb_at_input import UNet
        from utils.engine import DDPMSampler

        model = UNet(**cp["config"]["Model"])
        model.load_state_dict(cp["model"])
        model.to(device)
        model.eval()

        sampler = DDPMSampler(model, **cp["config"]["Trainer"]).to(device)

    elif method == "time":
        # Use UNet_emb_at_time for timestep conditioning
        from model.UNet_emb_at_time import UNet
        from utils.engine import DDPMSampler

        model = UNet(**cp["config"]["Model"])
        model.load_state_dict(cp["model"])
        model.to(device)
        model.eval()

        sampler = DDPMSampler(model, **cp["config"]["Trainer"]).to(device)

    elif method == "cfg":
        # Use UNet_emb_at_time with CFG sampler
        from model.UNet_emb_at_time import UNet
        from utils.engine_cfg import DDPMSampler_CFG

        model = UNet(**cp["config"]["Model"])
        model.load_state_dict(cp["model"])
        model.to(device)
        model.eval()

        sampler = DDPMSampler_CFG(model, **cp["config"]["Trainer"]).to(device)

    else:
        raise ValueError(f"Unknown method: {method}. Must be 'time', 'input', or 'cfg'")

    return model, sampler, cp["config"]


@torch.no_grad()
def generate_single_image(sampler, class_idx, in_channels, image_size, num_classes,
                          device, method, guidance_scale=None):
    """
    Generate a single image for the given class.

    Args:
        sampler: DDPM sampler (DDPMSampler or DDPMSampler_CFG)
        class_idx: Class index (0-9 for CIFAR-10)
        in_channels: Number of input channels
        image_size: Image size tuple (H, W)
        num_classes: Number of classes
        device: torch.device
        method: Conditioning method ('time', 'input', or 'cfg')
        guidance_scale: CFG guidance scale (only for method='cfg')

    Returns:
        Generated image tensor
    """
    # Generate Gaussian noise for a single image
    z_t = torch.randn((1, in_channels, *image_size), device=device)

    # Create class label (conditional)
    labels = torch.tensor([class_idx], device=device, dtype=torch.long)
    y_cond = F.one_hot(labels, num_classes=num_classes).float()

    # Generate image based on method
    if method == "cfg":
        # CFG generation with guidance scale
        x = sampler(z_t, y_cond, guidance_scale=guidance_scale,
                    only_return_x_0=True, interval=50)
    else:
        # Standard conditional generation (time or input)
        x = sampler(z_t, y_cond, only_return_x_0=True, interval=50)

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
    print("DDPM Evaluation Image Generation")
    print("=" * 80)
    print(f"Method: {args.method}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {args.output_dir}")
    print(f"Images per class: {args.images_per_class}")
    print(f"Total images: {args.images_per_class * len(CIFAR10_CLASSES)}")
    if args.method == "cfg":
        print(f"CFG guidance scale: {args.guidance_scale}")
    print(f"Device: {args.device}")
    print("=" * 80)
    print()

    # Load model and sampler
    model, sampler, config = load_model_and_sampler(
        args.checkpoint, args.method, device
    )

    # Get model configuration
    in_channels = config["Model"]["in_channels"]
    image_size = config["Dataset"]["image_size"]
    num_classes = config["Model"].get("num_classes", len(CIFAR10_CLASSES))

    print(f"Model loaded successfully!")
    print(f"  Image size: {image_size}")
    print(f"  Channels: {in_channels}")
    print(f"  Number of classes: {num_classes}")
    print()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate images
    print("Starting image generation...")
    print()

    total_images = len(CIFAR10_CLASSES) * args.images_per_class
    with tqdm(total=total_images, desc="Generating images") as pbar:
        for class_idx, class_name in enumerate(CIFAR10_CLASSES):
            for img_idx in range(1, args.images_per_class + 1):
                # Generate image
                image = generate_single_image(
                    sampler, class_idx, in_channels, image_size, num_classes,
                    device, args.method, args.guidance_scale
                )

                # Save image with naming convention: <class_name><index>
                filename = f"{class_name}{img_idx}.png"
                save_path = output_path / filename
                save_single_image(image, save_path)

                # Update progress bar
                pbar.update(1)
                postfix = {"class": class_name, "img": f"{img_idx}/{args.images_per_class}"}
                if args.method == "cfg":
                    postfix["cfg_scale"] = args.guidance_scale
                pbar.set_postfix(postfix)

    print()
    print(f"✓ Generation complete! {total_images} images saved to {output_path}/")
    print()

    # Verify generated images
    print("Verification:")
    for class_name in CIFAR10_CLASSES:
        class_images = list(output_path.glob(f"{class_name}*.png"))
        print(f"  {class_name}: {len(class_images)} images")

    total_generated = len(list(output_path.glob("*.png")))
    print(f"\nTotal images generated: {total_generated}")
    if args.method == "cfg":
        print(f"CFG guidance scale used: {args.guidance_scale}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate evaluation images for DDPM models with different conditioning methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Time-based conditioning
  python generate_conditioned.py --method time --checkpoint checkpoint/cifar10_cond.pth \\
      --output_dir result_and_script_for_eval/cond_time --images_per_class 20

  # Input-based conditioning
  python generate_conditioned.py --method input --checkpoint checkpoint/cifar10_cond_at_input.pth \\
      --output_dir result_and_script_for_eval/cond_input --images_per_class 20

  # CFG with guidance scale 3.0
  python generate_conditioned.py --method cfg --checkpoint checkpoint/cifar10_cfg.pth \\
      --output_dir result_and_script_for_eval/cfg_w3.0 --images_per_class 20 --guidance_scale 3.0

  # CFG with strong guidance
  python generate_conditioned.py --method cfg --checkpoint checkpoint/cifar10_cfg.pth \\
      --output_dir result_and_script_for_eval/cfg_w7.0 --images_per_class 20 --guidance_scale 7.0
        """
    )

    # Required arguments
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["time", "input", "cfg"],
        help="Conditioning method: 'time' (timestep embedding), 'input' (input channel), 'cfg' (classifier-free guidance)"
    )

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
        "--images_per_class",
        type=int,
        default=20,
        help="Number of images to generate per class (default: 20)"
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.0,
        help="CFG guidance scale (only for --method cfg). 0.0=uncond, 1.0=cond, 3.0-7.0=typical (default: 3.0)"
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
    if args.method == "cfg" and args.guidance_scale < 0.0:
        parser.error("guidance_scale must be >= 0.0")

    if args.method != "cfg" and args.guidance_scale != 3.0:
        print(f"Warning: --guidance_scale is only used with --method cfg, ignoring value {args.guidance_scale}")

    main(args)
