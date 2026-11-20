import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from models import Discriminator, Encoder, Generator, reparameterize


ROOT = Path(__file__).resolve().parents[3]
NOISE_STD = 0.5
KL_WEIGHT = 0.001
ADVERSARIAL_WEIGHT = 0.02
LOSS_NAMES = ("reconstruction_bce", "kl", "generator_adversarial", "discriminator")


def make_loader(train, batch_size):
    dataset = MNIST(ROOT / "data", train=train, download=True)
    images = dataset.data.unsqueeze(1).float().div_(255)
    return DataLoader(images, batch_size=batch_size, shuffle=train)


def add_noise(clean, noise_std, random=None):
    noise = torch.randn(clean.shape, device=clean.device, generator=random)
    return (clean + noise_std * noise).clamp_(0, 1)


def adversarial_loss(logits, real):
    targets = torch.ones_like(logits) if real else torch.zeros_like(logits)
    return F.binary_cross_entropy_with_logits(logits, targets)


def train(encoder, generator, loader, args, device):
    discriminator = Discriminator().to(device)
    vae_optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(generator.parameters()), lr=0.001
    )
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

    with (args.output / "history.csv").open("w", encoding="utf-8", newline="") as file:
        history = csv.writer(file)
        history.writerow(("epoch", *LOSS_NAMES))

        for epoch in range(1, args.epochs + 1):
            totals = torch.zeros(len(LOSS_NAMES), device=device)
            for clean in loader:
                clean = clean.to(device)
                noisy = add_noise(clean, NOISE_STD)
                mean, log_variance = encoder(noisy)
                reconstruction_logits = generator(reparameterize(mean, log_variance))
                prior_logits = generator(torch.randn_like(mean))
                fake = torch.cat((reconstruction_logits, prior_logits)).sigmoid()

                # Фаза дискриминатора. Градиент не идёт в энкодер и генератор.
                discriminator_optimizer.zero_grad(set_to_none=True)
                real_loss = adversarial_loss(discriminator(clean), real=True)
                fake_loss = adversarial_loss(discriminator(fake.detach()), real=False)
                discriminator_loss = (real_loss + fake_loss) / 2
                discriminator_loss.backward()
                discriminator_optimizer.step()

                # Фаза VAE-GAN. Обе группы изображений учатся обманывать дискриминатор.
                vae_optimizer.zero_grad(set_to_none=True)
                discriminator.requires_grad_(False)
                reconstruction = F.binary_cross_entropy_with_logits(
                    reconstruction_logits, clean
                )
                kl = -0.5 * (1 + log_variance - mean.square() - log_variance.exp()).sum(1).mean()
                adversarial = adversarial_loss(discriminator(fake), real=True)
                loss = reconstruction + KL_WEIGHT * kl + ADVERSARIAL_WEIGHT * adversarial
                loss.backward()
                vae_optimizer.step()
                discriminator.requires_grad_(True)

                batch_losses = (reconstruction, kl, adversarial, discriminator_loss)
                totals += torch.stack(batch_losses).detach() * len(clean)

            averages = (totals / len(loader.dataset)).tolist()
            history.writerow((epoch, *averages))
            file.flush()
            summary = ", ".join(f"{name}={value:.4f}" for name, value in zip(LOSS_NAMES, averages))
            print(f"Эпоха {epoch}/{args.epochs}: {summary}", flush=True)

    torch.save(
        {
            "encoder": encoder.state_dict(),
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "noise_std": NOISE_STD,
            "seed": args.seed,
        },
        args.output / "checkpoint.pt",
    )


@torch.inference_mode()
def evaluate(encoder, generator, loader, device, noise_std, seed):
    encoder.eval()
    generator.eval()
    noise_random = torch.Generator(device=device).manual_seed(seed + 1)
    noisy_error = restored_error = pixels = 0
    examples = None

    for clean in loader:
        clean = clean.to(device)
        noisy = add_noise(clean, noise_std, noise_random)
        mean, _ = encoder(noisy)
        restored = generator(mean).sigmoid()
        noisy_error += F.mse_loss(noisy, clean, reduction="sum").item()
        restored_error += F.mse_loss(restored, clean, reduction="sum").item()
        pixels += clean.numel()
        if examples is None:
            examples = tuple(batch[:10].cpu() for batch in (clean, noisy, restored))

    def scores(error):
        mse = error / pixels
        return {"mse": mse, "psnr_db": -10 * math.log10(max(mse, 1e-12))}

    metrics = {
        "test_images": len(loader.dataset),
        "noise_std": noise_std,
        "noisy": scores(noisy_error),
        "restored": scores(restored_error),
    }
    return metrics, examples


def save_examples(examples, path):
    count = len(examples[0])
    fig, axes = plt.subplots(3, count, figsize=(1.5 + count, 4), squeeze=False, layout="constrained")
    labels = ("Чистые", "Зашумлённые", "Восстановленные")
    for row, (label, batch) in enumerate(zip(labels, examples)):
        axes[row, 0].set_ylabel(label, rotation=0, ha="right", va="center")
        for axis, image in zip(axes[row], batch):
            axis.imshow(image.squeeze(0), cmap="gray", vmin=0, vmax=1)
            axis.set(xticks=[], yticks=[])
            for spine in axis.spines.values():
                spine.set_visible(False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Восстановление зашумлённых изображений MNIST с помощью VAE-GAN.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "pytorch_runs")
    parser.add_argument("--checkpoint", type=Path, help="Оценить сохранённую модель без обучения")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.seed < 0:
        parser.error("Число эпох, размер батча и число потоков должны быть положительными, а seed — неотрицательным.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA недоступна в этом окружении.")

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Устройство: {device}", flush=True)

    encoder, generator = Encoder().to(device), Generator().to(device)
    if args.checkpoint is None:
        train(encoder, generator, make_loader(train=True, batch_size=args.batch_size), args, device)
        noise_std, seed = NOISE_STD, args.seed
    else:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        encoder.load_state_dict(checkpoint["encoder"])
        generator.load_state_dict(checkpoint["generator"])
        noise_std, seed = checkpoint["noise_std"], checkpoint["seed"]

    test = make_loader(train=False, batch_size=args.batch_size)
    metrics, examples = evaluate(encoder, generator, test, device, noise_std, seed)
    save_examples(examples, args.output / "reconstructions.png")
    report = json.dumps(metrics, indent=2)
    (args.output / "metrics.json").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
