import torch
import os
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 200
import torchvision.transforms.functional as functional
import torch.nn.functional as F
from glob import glob
from models import build_model
from util.misc import nested_tensor_from_tensor_list
from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument(
    "--threshold",
    type=float,
    default=0.3,
    help="threshold for line predictions.",
)

parser.add_argument(
    "--exp_folder",
    type=str,
    default="res50_stage1",
    help="experiment folder.",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="synthetic",
    help="val dataset folder.",
)
parser.add_argument(
    "--ckpt",
    type=str,
    default="checkpoints/checkpoint0204.pth",
    help="Checkpoint path",
)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = functional.normalize(image, mean=self.mean, std=self.std)
        return image


class ToTensor(object):
    def __call__(self, img):
        return functional.to_tensor(img)


def resize(image, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            original_image_size = float(min((w, h)))
            original_image_size = float(max((w, h)))
            if original_image_size / original_image_size * size > max_size:
                size = int(round(max_size * original_image_size / original_image_size))
        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = functional.resize(image, size)

    return rescaled_image


class Resize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img):
        size = self.sizes
        return resize(img, size, self.max_size)


def load_model(path):
    # obtain checkpoints
    checkpoint = torch.load(path, map_location="cpu")

    # load model
    args = checkpoint["args"]
    model, _, postprocessors = build_model(args)
    model.load_state_dict(checkpoint["model"])
    return model


def get_lines(outputs, image_size, args):
    print(outputs)
    out_logits, out_line = outputs["pred_logits"], outputs["pred_lines"]
    prob = F.softmax(out_logits, -1)
    scores, labels = prob[..., :-1].max(-1)
    img_h, img_w = image_size.unbind(0)

    scale_fct = torch.unsqueeze(torch.stack([img_w, img_h, img_w, img_h], dim=0), dim=0)
    lines = out_line * scale_fct[:, None, :]
    lines = lines.view(1000, 2, 2)
    lines = lines.flip([-1])  # this is yxyx format
    scores = scores.detach().numpy()
    print("#########", np.max(scores), "###########")
    keep = scores >= args.threshold
    keep = keep.squeeze()

    final_lines = lines[keep]
    if len(final_lines) != 0:
        print(f"detected {len(final_lines)}")
        final_lines = final_lines.reshape(final_lines.shape[0], -1)

    return final_lines


def main(args):
    root = Path(
        "/home/kallelis/PrimitiveExtraction/PrimitiveExtraction/Detection/LETR/"
    )
    experiment_dir = root / f"exp/{args.exp_folder}"
    dataset_dir = f"data/{args.dataset}_processed/val"
    if args.dataset == "wireframe":
        dataset_dir = dataset_dir + "2017"

    input_dir = root / dataset_dir
    output_dir = experiment_dir / "prediction_plots"
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = experiment_dir / args.ckpt
    print("loading model")
    model = load_model(checkpoint_path)
    print(model)
    print("successfully loaded model")
    model.eval()

    for image_path in tqdm(glob(str(input_dir / "*.png"))):
        image_name = os.path.basename(image_path)
        raw_image = plt.imread(image_path)
        raw_image = raw_image[:, :, :3]
        h, w = raw_image.shape[0], raw_image.shape[1]
        image_size = torch.as_tensor([int(h), int(w)])

        test_size = 1100
        normalize = Compose(
            [
                ToTensor(),
                Normalize([0.538, 0.494, 0.453], [0.257, 0.263, 0.273]),
                Resize([test_size]),
            ]
        )

        img = normalize(raw_image)
        inputs = nested_tensor_from_tensor_list([img])
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            print("########", len(outputs), "#########")
            outputs, origin_indices = outputs

        final_lines = get_lines(outputs, image_size, args)

        fig = plt.figure()
        plt.imshow(raw_image)
        for line in final_lines:
            y1, x1, y2, x2 = line.detach().numpy()  # this is yxyx
            p1 = (x1, y1)
            p2 = (x2, y2)
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=1, color="red", zorder=1)
        plt.axis("off")
        plt.savefig(output_dir / image_name, dpi=300, bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
