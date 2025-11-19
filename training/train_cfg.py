"""
Classifier-Free Guidance (CFG) Training Script.

This script trains a DDPM model with CFG support by:
1. Using the standard UNet architecture (UNet_emb_at_time)
2. Applying conditional dropout during training via train_one_epoch_cfg
3. Enabling the model to learn both conditional and unconditional generation

The trained model can then be used with DDPMSampler_CFG for guided generation.
"""

from dataset import create_dataset
from model.UNet_emb_at_time import UNet
from utils.engine_cfg import GaussianDiffusionTrainer
from utils.tools_cfg import train_one_epoch_cfg, load_yaml
import torch
from utils.callbacks import ModelCheckpoint


def format_paths_with_run_name(config):
    """
    Format paths in config that contain {run_name} placeholder.
    """
    run_name = config.get("run_name")
    if not run_name:
        return config

    def _format(value):
        if isinstance(value, dict):
            for k, v in value.items():
                value[k] = _format(v)
            return value
        if isinstance(value, (list, tuple)):
            formatted = [_format(v) for v in value]
            return type(value)(formatted)
        if isinstance(value, str) and "{run_name" in value:
            return value.format(run_name=run_name)
        return value

    _format(config)
    return config


def train_cfg(config):
    """
    Main CFG training function.

    Parameters:
        config: Configuration dictionary loaded from config_cfg.yml
    """
    config = format_paths_with_run_name(config)
    consume = config["consume"]
    if consume:
        cp = torch.load(config["consume_path"])
        config = format_paths_with_run_name(cp["config"])

    print("=" * 80)
    print("CFG Training Configuration:")
    print("=" * 80)
    print(f"Run name: {config.get('run_name', 'N/A')}")
    print(f"CFG dropout probability: {config['cfg_dropout_prob']}")
    print(f"Default guidance scale: {config['guidance_scale']} (for generation, not used in training)")
    print(f"Model: UNet_emb_at_time")
    print(f"Dataset: {config['Dataset']['dataset']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Device: {config['device']}")
    print("=" * 80)
    print()

    device = torch.device(config["device"])
    loader = create_dataset(**config["Dataset"])
    start_epoch = 1

    # Initialize model (using UNet_emb_at_time architecture)
    model = UNet(**config["Model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)

    # Use standard Gaussian diffusion trainer
    trainer = GaussianDiffusionTrainer(model, **config["Trainer"]).to(device)

    model_checkpoint = ModelCheckpoint(**config["Callback"])

    # Resume from checkpoint if needed
    if consume:
        model.load_state_dict(cp["model"])
        optimizer.load_state_dict(cp["optimizer"])
        model_checkpoint.load_state_dict(cp["model_checkpoint"])
        start_epoch = cp["start_epoch"] + 1
        print(f"Resumed training from epoch {start_epoch}")

    num_classes = config["Model"].get("num_classes")

    # Get CFG-specific parameters from config (required)
    if "cfg_dropout_prob" not in config:
        raise ValueError(
            "cfg_dropout_prob not found in config! "
            "Please add 'cfg_dropout_prob' to config_cfg.yml (e.g., cfg_dropout_prob: 0.1)"
        )
    cfg_dropout_prob = config["cfg_dropout_prob"]

    print(f"Starting CFG training with dropout probability: {cfg_dropout_prob:.2%}")
    print(f"This means {cfg_dropout_prob:.2%} of samples will use null embedding (all zeros)")
    print()

    # Training loop with CFG dropout
    for epoch in range(start_epoch, config["epochs"] + 1):
        # Use CFG training function with conditional dropout
        loss = train_one_epoch_cfg(
            trainer=trainer,
            loader=loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            num_classes=num_classes,
            cfg_dropout_prob=cfg_dropout_prob
        )

        # Save checkpoint
        model_checkpoint.step(
            loss,
            model=model.state_dict(),
            config=config,
            optimizer=optimizer.state_dict(),
            start_epoch=epoch,
            model_checkpoint=model_checkpoint.state_dict()
        )

    print()
    print("=" * 80)
    print("CFG Training Complete!")
    print("=" * 80)
    print(f"Model saved to: {config['Callback']['filepath']}")
    print(f"To generate images with CFG, use: python generate_cfg.py")
    print("=" * 80)


if __name__ == "__main__":
    config = load_yaml("config_cfg.yml", encoding="utf-8")
    train_cfg(config)
