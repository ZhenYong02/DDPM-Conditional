from utils.engine import DDPMSampler
from model.UNet import UNet
import torch
import torch.nn.functional as F
from utils.tools import save_sample_image, save_image
from argparse import ArgumentParser


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


def parse_option():
    parser = ArgumentParser()
    parser.add_argument("-cp", "--checkpoint_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")

    # generator param
    parser.add_argument("-bs", "--batch_size", type=int, default=16)

    # sampler param
    parser.add_argument("--result_only", default=False, action="store_true")
    parser.add_argument("--interval", type=int, default=50)

    # save image param
    parser.add_argument("--nrow", type=int, default=4)
    parser.add_argument("--show", default=False, action="store_true")
    parser.add_argument("-sp", "--image_save_path", type=str, default=None)
    parser.add_argument("--to_grayscale", default=False, action="store_true")

    parser.add_argument("--class_name", type=str, choices=CIFAR10_CLASSES, required=True,
                        help="CIFAR-10 class name to condition generation on")

    args = parser.parse_args()
    return args


@torch.no_grad()
def generate(args):
    device = torch.device(args.device)

    cp = torch.load(args.checkpoint_path)
    # load trained model
    model = UNet(**cp["config"]["Model"])
    model.load_state_dict(cp["model"])
    model.to(device)
    model = model.eval()

    sampler = DDPMSampler(model, **cp["config"]["Trainer"]).to(device)

    # generate Gaussian noise
    z_t = torch.randn((args.batch_size, cp["config"]["Model"]["in_channels"],
                       *cp["config"]["Dataset"]["image_size"]), device=device)

    num_classes = cp["config"]["Model"].get("num_classes", len(CIFAR10_CLASSES))
    class_idx = CIFAR10_CLASSES.index(args.class_name)
    if class_idx >= num_classes:
        raise ValueError(
            f"Checkpoint was trained with num_classes={num_classes}, but class '{args.class_name}' index {class_idx}"
            " is out of range."
        )
    labels = torch.full((args.batch_size,), class_idx, device=device, dtype=torch.long)
    y_onehot = F.one_hot(labels, num_classes=num_classes).float()

    x = sampler(z_t, y_onehot, only_return_x_0=args.result_only, interval=args.interval)

    if args.result_only:
        save_image(x, nrow=args.nrow, show=args.show, path=args.image_save_path, to_grayscale=args.to_grayscale)
    else:
        save_sample_image(x, show=args.show, path=args.image_save_path, to_grayscale=args.to_grayscale)


if __name__ == "__main__":
    args = parse_option()
    generate(args)
