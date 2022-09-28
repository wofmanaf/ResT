import math

import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.ticker import FormatStrFormatter
from matplotlib.pyplot import MultipleLocator
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from models.fft_rest_v2 import restv2_tiny
from tqdm import tqdm

config = {"font.family": "sans-serif", "font.size": 16, "mathtext.fontset": 'stix',
          'font.sans-serif': ['Times New Roman'], }
rcParams.update(config)


# Sampler for pytorch loader. Given range r loader will only
# return dataset[r] instead of whole dataset.
class RangeSampler(Sampler):
    def __init__(self, r):
        self.r = r

    def __iter__(self):
        return iter(self.r)

    def __len__(self):
        return len(self.r)


def fourier(x):  # 2D Fourier transform
    f = torch.fft.fft2(x)
    f = f.abs() + 1e-6
    f = f.log()
    return f


def shift(x):  # shift Fourier transformed feature map
    b, c, h, w = x.shape
    return torch.roll(x, shifts=(int(h / 2), int(w / 2)), dims=(2, 3))


def make_segments(x, y):  # make segment for `plot_segment`
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def plot_segment(fig, ax, xs, ys, cmap_name="plasma", marker="o"):  # plot with cmap segments
    z = np.linspace(0.0, 1.0, len(ys))
    z = np.asarray(z)

    cmap = cm.get_cmap(cmap_name)
    norm = plt.Normalize(0.0, 1.0)
    segments = make_segments(xs, ys)
    lc = LineCollection(segments, array=z, cmap=cmap_name, norm=norm,
                        linewidth=2.5, alpha=1.0)
    ax.add_collection(lc)

    colors = [cmap(x) for x in xs]
    sc = ax.scatter(xs, ys, color=colors, marker=marker, zorder=100)
    fig.colorbar(sc, ticks=[0, 0.1, 1.])


def plot(latents, name=''):
    # latents: list of hidden feature maps in the latent space
    fig, ax = plt.subplots(1, 1, figsize=(3.6, 4), dpi=300)
    fig.set_tight_layout(True)

    # Fourier transform feature maps
    fourier_latents = []
    for latent in latents:
        if len(latent.shape) == 3:  # for vit
            b, n, c = latent.shape
            h, w = int(math.sqrt(n)), int(math.sqrt(n))
            latent = latent.permute(0, 2, 1).reshape(b, c, h, w)
        elif len(latent.shape) == 4:  # for cnn
            b, c, h, w = latent.shape
        else:
            raise Exception("shape: %s" % str(latent.shape))
        latent = fourier(latent)
        latent = shift(latent).mean(dim=(0, 1))
        latent = latent.diag()[int(h / 2):]  # only use the half-diagonal components
        latent = latent - latent[0]  # visualize 'relative' log amplitudes (i.e., low_freq - high_freq)
        fourier_latents.append(latent)
    # plot fourier transformed relative log amplitudes
    for i, latent in enumerate(reversed(fourier_latents)):
        freq = np.linspace(0, 1, len(latent))
        ax.plot(freq, latent, color=cm.plasma_r(i / len(fourier_latents)))

    ax.set_xlim(left=0, right=1)
    x_major_locator = MultipleLocator(0.5)
    y_major_locator = MultipleLocator(2.0)

    ax.set_xlabel("Frequency")
    ax.set_ylabel(r"$\Delta$ Log amplitude")

    ax.xaxis.set_major_locator(x_major_locator)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1fπ'))

    ax.yaxis.set_major_locator(y_major_locator)
    plt.savefig(name)
    # plt.show()


def main():
    model = restv2_tiny().cuda()
    checkpoint = torch.load("output_dir/restv2_tiny_224.pth", map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.eval()

    val_dir = '/data/ilsvrc2012/val/'
    batch_size = 1
    num_workers = 4
    batch = 0

    sample_range = range(10000 * batch, 10000 * (batch + 1))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_loader = DataLoader(
        datasets.ImageFolder(val_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=RangeSampler(sample_range)
    )

    latents_attn = []
    latents_up = []
    latents_com = []

    for i, (samples, targets) in enumerate(tqdm(val_loader, total=len(val_loader), desc='Loading Images')):
        samples = samples.cuda()
        with torch.no_grad():
            feats = model(samples)

        for j in range(11):
            if i == 0:
                latents_attn.append(feats[j]["attn"].cpu())
                latents_up.append(feats[j]["up"].cpu())
                latents_com.append(feats[j]["com"].cpu())
            else:
                latents_attn[j] = torch.cat([latents_attn[j], feats[j]["attn"].cpu()], dim=0)
                latents_up[j] = torch.cat([latents_up[j], feats[j]["up"].cpu()], dim=0)
                latents_com[j] = torch.cat([latents_com[j], feats[j]["com"].cpu()], dim=0)
    plot(latents_attn, name="attn.png")
    plot(latents_up, name="up.png")
    plot(latents_com, name="combine.png")


if __name__ == '__main__':
    main()
