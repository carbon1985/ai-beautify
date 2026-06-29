import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class ModelManager:
    """Local AI model facade with graceful OpenCV fallbacks.

    The app is designed to ship model files under models/, but the code must remain
    usable while those large binaries are absent during development. Heavy backends
    are imported lazily so the desktop app can still launch without torch/diffusers.
    """

    def __init__(self, models_dir=None):
        self.models_dir = Path(models_dir or Path(__file__).resolve().parent / "models")
        self.face_parsing_dir = self.models_dir / "face_parsing"
        self.inpainting_dir = self.models_dir / "inpainting"
        self.gan_dir = self.models_dir / "face_restoration_gan"
        self.diffusion_dir = self.models_dir / "diffusion_inpaint"
        self._device = None
        self._diffusion_pipe = None

    def status(self):
        return {
            "models_dir": str(self.models_dir),
            "device": self.device(),
            "face_parsing_available": self.face_parsing_dir.exists() and any(self.face_parsing_dir.iterdir()),
            "inpainting_available": self.inpainting_dir.exists() and any(self.inpainting_dir.iterdir()),
            "gan_available": self.gan_dir.exists() and any(self.gan_dir.iterdir()),
            "diffusion_available": self.diffusion_dir.exists() and any(self.diffusion_dir.iterdir()),
            "torch_available": self._can_import("torch"),
            "diffusers_available": self._can_import("diffusers"),
            "fallback": "OpenCV inpainting and CV effects are used when local AI models are missing.",
        }

    def device(self):
        if self._device is not None:
            return self._device
        try:
            import torch
            if torch.cuda.is_available():
                self._device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        except Exception:
            self._device = "cpu"
        return self._device

    def segment_face(self, image):
        """Placeholder for a bundled face-parsing model.

        Returns None until a model is placed under models/face_parsing. The caller
        combines this with MediaPipe landmark masks, so None is a safe fallback.
        """
        return None

    def inpaint(self, image, mask, prompt=None, strength=0.5, use_ai=False):
        if mask is None or float(np.max(mask)) <= 0:
            return image.copy()

        if use_ai:
            result = self.diffusion_edit(
                image,
                mask,
                prompt or "natural realistic skin texture, preserve identity",
                "distorted face, changed identity, artifacts, blurry skin",
                min(0.35, max(0.05, strength)),
                preset=None,
            )
            if result is not None:
                return result

        mask_u8 = self._mask_u8(mask)
        radius = max(3, int(3 + 8 * strength))
        return cv2.inpaint(image, mask_u8, radius, cv2.INPAINT_TELEA)

    def restore_face(self, image, strength=0.35, use_ai=False):
        """GAN restoration hook.

        If a GFPGAN/CodeFormer integration is added later, this method is where it
        should run. For now it performs a subtle detail-preserving blend so the
        pipeline and UI already expose a GAN stage without breaking when models are
        absent.
        """
        if not use_ai or strength <= 0:
            return image.copy()
        smooth = cv2.bilateralFilter(image, 7, 45, 45)
        detail = cv2.addWeighted(image, 1.4, cv2.GaussianBlur(image, (0, 0), 1.2), -0.4, 0)
        restored = cv2.addWeighted(smooth, 0.65, detail, 0.35, 0)
        return cv2.addWeighted(image, 1.0 - strength * 0.35, restored, strength * 0.35, 0)

    def diffusion_edit(self, image, mask, prompt, negative_prompt, strength=0.25, preset=None):
        if not self.diffusion_dir.exists() or not any(self.diffusion_dir.iterdir()):
            return None
        if self.device() == "cpu" and strength > 0.2:
            return None
        try:
            pipe = self._load_diffusion_pipeline()
            if pipe is None:
                return None
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            pil_mask = Image.fromarray(self._mask_u8(mask))
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=pil_image,
                mask_image=pil_mask,
                strength=float(np.clip(strength, 0.05, 0.45)),
                guidance_scale=5.5,
                num_inference_steps=18,
            ).images[0]
            return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        except Exception:
            return None

    def _load_diffusion_pipeline(self):
        if self._diffusion_pipe is not None:
            return self._diffusion_pipe
        try:
            import torch
            from diffusers import StableDiffusionInpaintPipeline

            dtype = torch.float16 if self.device() in ("cuda", "mps") else torch.float32
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                str(self.diffusion_dir),
                torch_dtype=dtype,
                local_files_only=True,
                safety_checker=None,
                requires_safety_checker=False,
            )
            pipe = pipe.to(self.device())
            self._diffusion_pipe = pipe
            return self._diffusion_pipe
        except Exception:
            self._diffusion_pipe = None
            return None

    def _mask_u8(self, mask):
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if mask.dtype != np.uint8:
            mask = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
        return mask

    def _can_import(self, module_name):
        try:
            __import__(module_name)
            return True
        except Exception:
            return False
