# Local AI Model Files

AI Beautify is designed to run high-quality generative beauty edits locally. The app can launch without these models and will fall back to OpenCV/MediaPipe effects, but AI 高质量 mode needs model files in this directory.

Recommended layout:

```text
models/
  face_parsing/
    <face parsing / semantic segmentation model files>
  inpainting/
    <local inpainting model files>
  face_restoration_gan/
    <GFPGAN / CodeFormer style restoration model files>
  diffusion_inpaint/
    <Diffusers-compatible Stable Diffusion inpainting pipeline>
```

Model responsibilities:

- `face_parsing/`: more accurate skin, lips, eyes, brows, hair, and teeth masks than landmark-only masks.
- `inpainting/`: local repair for acne marks, freckles, wrinkles, stains, and dark-circle patches.
- `face_restoration_gan/`: GAN-based face/skin texture restoration after heavy repair or smoothing.
- `diffusion_inpaint/`: low-strength local diffusion inpainting/img2img for style beauty and advanced repairs.

Implementation notes:

- Keep model use local; no cloud API is required.
- Preserve identity by using strict masks and low diffusion strength.
- Track model source, license, checksum, and expected filename here when adding real model binaries.
- Large model binaries should be handled with Git LFS or release assets rather than normal Git blobs.
