# RestoraX

**Modern AI video restoration platform for old films, series, and archival footage.**

RestoraX combines 12 state-of-the-art deep learning models into a single modular pipeline:
upscale, colorize, restore faces, remove scratches, deinterlace, stabilize, boost frame rate,
and convert to HDR — all orchestrated by a REST API, web UI, and CLI.

## Quick links

- [Installation](guides/installation.md)
- [Quick Start](guides/quickstart.md)
- [Pipeline Presets](presets/overview.md)
- [Writing a Plugin](guides/plugins.md)

## Feature matrix

| Category | Models | Scale |
|---|---|---|
| Super-Resolution | Real-ESRGAN, BasicVSR++, VRT, Upscale-A-Video | 2×–8× |
| Colorization | DDColor | 1× |
| Face Restoration | CodeFormer, GFPGAN | 1× |
| Frame Interpolation | RIFE v4.22 | 2× FPS |
| Scratch & Dust | ProPainter | 1× |
| Deinterlacing | AI + YADIF | 1× |
| Stabilization | Optical flow | 1× |
| SDR → HDR | HDRTVDM | 1× |
