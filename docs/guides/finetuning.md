# Fine-tuning Custom Models

Fine-tune Real-ESRGAN on domain-specific data (e.g., 8mm home movie grain,
specific camera noise, DVD compression artifacts) for better results than
the generic pretrained weights.

## Requirements

```bash
pip install basicsr tensorboard
```

## 1. Prepare a dataset

Your dataset needs paired low-resolution/high-resolution images or a set of
high-quality source frames from which degraded versions will be synthesized.

```
data/
├── train/
│   └── HR/          # High-resolution source images (512×512 or larger)
│       ├── 001.png
│       ├── 002.png
│       └── ...
└── val/
    └── HR/
        └── ...
```

Extract frames from high-quality source material:

```bash
# Extract every 10th frame as PNG
ffmpeg -i source_hq.mp4 -vf "select=not(mod(n\,10))" -vsync 0 data/train/HR/%06d.png
```

## 2. Configure the training run

Create `options/train_realesrgan_x4plus_custom.yml`:

```yaml
name: train_realesrgan_x4plus_custom
scale: 4
num_gpu: 1
manual_seed: 0

# Dataset
datasets:
  train:
    name: custom_train
    type: RealESRGANDataset
    dataroot_gt: data/train/HR
    io_backend:
      type: disk
    gt_size: 256
    use_hflip: true
    use_rot: true
    # Degradation pipeline — customize these for your source material
    blur_kernel_size: 21
    kernel_list: [iso, aniso, generalized_iso, generalized_aniso, plateau_iso, plateau_aniso, sinc]
    kernel_prob: [0.45, 0.25, 0.12, 0.03, 0.12, 0.03, 0.01]
    sinc_prob: 0.1
    blur_sigma: [0.2, 3]
    noise_range: [1, 30]
    poisson_scale_range: [0.05, 3]
    jpeg_range: [30, 95]
    # Second-order degradation
    second_blur_prob: 0.8

  val:
    name: custom_val
    type: PairedImageDataset
    dataroot_lq: data/val/LR
    dataroot_gt: data/val/HR
    io_backend:
      type: disk

# Network
network_g:
  type: RRDBNet
  num_in_ch: 3
  num_out_ch: 3
  num_feat: 64
  num_block: 23
  num_grow_ch: 32
  scale: 4

# Pretrained weights (resume from official checkpoint)
path:
  pretrain_network_g: models/real_esrgan/RealESRGAN_x4plus.pth
  strict_load_g: true
  resume_state: ~

# Training
train:
  ema_decay: 0.999
  optim_g:
    type: Adam
    lr: !!float 1e-4
    weight_decay: 0
    betas: [0.9, 0.99]
  scheduler:
    type: MultiStepLR
    milestones: [200000]
    gamma: 0.5
  total_iter: 400000
  warmup_iter: -1

  # Pixel loss
  pixel_opt:
    type: L1Loss
    loss_weight: 1.0
    reduction: mean
  # Perceptual loss
  perceptual_opt:
    type: PerceptualLoss
    layer_weights:
      conv1_2: 0.1
      conv2_2: 0.1
      conv3_4: 1
      conv4_4: 1
      conv5_4: 1
    vgg_type: vgg19
    use_input_norm: true
    perceptual_weight: !!float 1.0
    style_weight: 0
    criterion: l1

# Logging
logger:
  print_freq: 100
  save_checkpoint_freq: !!float 5e3
  use_tb_logger: true
  wandb:
    project: ~

val:
  val_freq: !!float 5e3
  save_img: false
  metrics:
    psnr:
      type: calculate_psnr
      crop_border: 4
      test_y_channel: false
```

## 3. Train

```bash
cd /path/to/BasicSR
python basicsr/train.py -opt options/train_realesrgan_x4plus_custom.yml
```

Monitor with TensorBoard:
```bash
tensorboard --logdir experiments/
```

## 4. Use the fine-tuned weights in RestoraX

Copy your checkpoint to the models directory:

```bash
cp experiments/train_realesrgan_x4plus_custom/models/net_g_400000.pth \
   models/real_esrgan/RealESRGAN_x4plus.pth
```

The restorer auto-loads from this path on next use.

Or create a new restorer subclass pointing to your custom weights:

```python
# my_plugin/my_domain_esrgan.py
from restorax.restorers.super_resolution.real_esrgan import RealESRGANx4Restorer
from pathlib import Path

class MyDomainESRGAN(RealESRGANx4Restorer):
    @property
    def name(self) -> str:
        return "my_domain_esrgan"

    def _resolve_weight_path(self) -> Path:
        return Path("models/my_domain/my_esrgan_weights.pth")
```

## Tips for specific content types

| Content | Recommended degradation changes |
|---|---|
| 8mm film | Increase blur sigma, add film grain noise, reduce JPEG compression |
| VHS tape | Add chroma noise, reduce saturation, increase noise range |
| DVD compression | Increase JPEG range (lower values = heavier), add blocking artifacts |
| Surveillance camera | Simulate low-light noise, motion blur, low resolution |
| Old TV broadcast | Add interlacing artifacts, NTSC composite noise |
