from dataset import create_dataset
from model.UNet_emb_at_time import UNet as UNet_time
from model.UNet_emb_at_input import UNet as UNet_input
from utils.engine import GaussianDiffusionTrainer
from utils.tools import train_one_epoch, load_yaml
import torch
from utils.callbacks import ModelCheckpoint


def format_paths_with_run_name(config):
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


def train(config):
    config = format_paths_with_run_name(config)
    consume = config["consume"]
    if consume:
        cp = torch.load(config["consume_path"])
        config = format_paths_with_run_name(cp["config"])

    # Get conditioning strategy from config
    conditioning_strategy = config.get("conditioning_strategy", "time")
    if conditioning_strategy not in ["time", "input"]:
        raise ValueError(
            f"Invalid conditioning_strategy: {conditioning_strategy}. "
            "Must be 'time' or 'input'."
        )

    print("=" * 80)
    print("Training Configuration:")
    print("=" * 80)
    print(f"Run name: {config.get('run_name', 'N/A')}")
    print(f"Conditioning strategy: {conditioning_strategy}")
    if conditioning_strategy == "time":
        print("  Using UNet_emb_at_time (class embedding at timestep level)")
    else:
        print("  Using UNet_emb_at_input (class embedding at input channel level)")
    print(f"Dataset: {config['Dataset']['dataset']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Device: {config['device']}")
    print("=" * 80)
    print()

    device = torch.device(config["device"])
    loader = create_dataset(**config["Dataset"])
    start_epoch = 1

    # Select model based on conditioning strategy
    if conditioning_strategy == "time":
        model = UNet_time(**config["Model"]).to(device)
    else:  # input
        model = UNet_input(**config["Model"]).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    trainer = GaussianDiffusionTrainer(model, **config["Trainer"]).to(device)

    model_checkpoint = ModelCheckpoint(**config["Callback"])

    if consume:
        model.load_state_dict(cp["model"])
        optimizer.load_state_dict(cp["optimizer"])
        model_checkpoint.load_state_dict(cp["model_checkpoint"])
        start_epoch = cp["start_epoch"] + 1

    num_classes = config["Model"].get("num_classes")

    for epoch in range(start_epoch, config["epochs"] + 1):
        loss = train_one_epoch(trainer, loader, optimizer, device, epoch, num_classes)
        model_checkpoint.step(loss, model=model.state_dict(), config=config,
                              optimizer=optimizer.state_dict(), start_epoch=epoch,
                              model_checkpoint=model_checkpoint.state_dict())


if __name__ == "__main__":
    config = load_yaml("config.yml", encoding="utf-8")
    train(config)
