import cv2
import mediapipe as mp
import numpy as np

from ai_models import ModelManager
from beauty_config import get_preset, normalize_params
from face_regions import FACE_OVAL, FaceRegionBuilder


class ImageProcessor:
    FACE_OVAL = FACE_OVAL

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        self.ai_models = ModelManager()
        self.original_image = None
        self.landmarks = None
        self.landmark_preview = None
        self.height = 0
        self.width = 0
        self.masks = {}
        self.skin_mask = None

        self.analysis_cache = None
        self.stage_cache = {}
        self.base_map_x = None
        self.base_map_y = None

    def load_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return False

        self.original_image = img
        self.landmark_preview = None
        self.height, self.width = img.shape[:2]

        self.base_map_x, self.base_map_y = np.meshgrid(np.arange(self.width), np.arange(self.height))
        self.base_map_x = self.base_map_x.astype(np.float32)
        self.base_map_y = self.base_map_y.astype(np.float32)

        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            self.landmarks = results.multi_face_landmarks[0]
            self._generate_masks()
        else:
            self.landmarks = None
            self.masks = {}
            self.skin_mask = None

        self.analysis_cache = {
            "landmarks": self.landmarks,
            "masks": self.masks,
            "model_segmentation": self.ai_models.segment_face(img),
        }
        self.stage_cache = {}
        return True

    def _get_pt(self, idx):
        pt = self.landmarks.landmark[idx]
        return (int(pt.x * self.width), int(pt.y * self.height))

    def _get_expanded_face_points(self, indices):
        if self.landmarks is None:
            return np.empty((0, 2), dtype=np.int32)
        return FaceRegionBuilder(self.landmarks, self.width, self.height).expanded_points(indices)

    def _generate_masks(self):
        builder = FaceRegionBuilder(self.landmarks, self.width, self.height)
        self.masks = builder.build(self.original_image)
        self.skin_mask = self.masks.get("skin")

    def get_landmark_preview(self):
        if self.original_image is None:
            return None

        if self.landmark_preview is not None:
            return self.landmark_preview

        preview = self.original_image.copy()
        if self.landmarks is None:
            self.landmark_preview = preview
            return self.landmark_preview

        line_thickness = max(1, round(min(self.width, self.height) / 700))
        point_radius = max(1, round(min(self.width, self.height) / 450))

        for start_idx, end_idx in self.mp_face_mesh.FACEMESH_TESSELATION:
            cv2.line(
                preview,
                self._get_pt(start_idx),
                self._get_pt(end_idx),
                (255, 190, 40),
                line_thickness,
                cv2.LINE_AA
            )

        for index in range(len(self.landmarks.landmark)):
            cv2.circle(
                preview,
                self._get_pt(index),
                point_radius,
                (40, 230, 255),
                -1,
                cv2.LINE_AA
            )

        processing_contour = self._get_expanded_face_points(self.FACE_OVAL).reshape((-1, 1, 2))
        cv2.polylines(
            preview,
            [processing_contour],
            True,
            (80, 255, 120),
            max(2, line_thickness + 1),
            cv2.LINE_AA
        )

        for key, color in [("under_eye", (255, 120, 80)), ("cheeks", (90, 90, 255)), ("teeth", (255, 255, 180))]:
            mask = self.masks.get(key)
            if mask is not None and np.max(mask) > 0:
                overlay = np.zeros_like(preview)
                overlay[:] = color
                preview = self._blend(preview, overlay, mask, 0.18)

        self.landmark_preview = preview
        return self.landmark_preview

    def process(self, params):
        if self.original_image is None:
            return None

        params = normalize_params(params)
        preset = get_preset(params.get("style_preset", "无"))
        use_ai = bool(params.get("_ai_mode", False))

        skin = self._cached_stage(
            "skin",
            self.original_image,
            params,
            [
                "skin_smooth", "pore_blur", "noise_reduce", "blemish_remove", "freckle_fade",
                "wrinkle_soften", "dark_circle_remove", "shine_remove", "skin_whiten",
                "skin_tone_even", "skin_texture_restore", "_ai_mode",
            ],
            lambda img: self._apply_skin_effects(img, params, use_ai),
        )
        makeup = self._cached_stage(
            "makeup",
            skin,
            params,
            ["blush", "lip_color", "eye_brighten", "brow_enhance", "style_preset", "_ai_mode"],
            lambda img: self._apply_makeup_effects(img, params, preset, use_ai),
        )
        teeth = self._cached_stage(
            "teeth",
            makeup,
            params,
            ["teeth_whiten", "teeth_stain_remove", "teeth_straighten", "_ai_mode"],
            lambda img: self._apply_teeth_effects(img, params, use_ai),
        )
        lighting = self._cached_stage(
            "lighting",
            teeth,
            params,
            ["fill_light", "face_brighten", "shadow_enhance", "rim_light", "film_light"],
            lambda img: self._apply_lighting_effects(img, params),
        )
        styled = self._cached_stage(
            "style",
            lighting,
            params,
            ["style_preset", "style_strength", "style_warmth", "style_contrast", "style_saturation", "_ai_mode"],
            lambda img: self._apply_style_effects(img, params, preset, use_ai),
        )
        return self._apply_warping(styled, params)

    def _cached_stage(self, name, source, params, keys, renderer):
        key = (self._image_signature(source), self._param_key(params, keys))
        cached = self.stage_cache.get(name)
        if cached and cached["key"] == key:
            return cached["image"].copy()
        image = renderer(source)
        self.stage_cache[name] = {"key": key, "image": image.copy()}
        return image

    def _image_signature(self, image):
        return (
            image.shape,
            int(np.sum(image, dtype=np.uint64) % 1000000007),
            int(np.mean(image)),
        )

    def _param_key(self, params, keys):
        key = []
        for name in keys:
            value = params.get(name)
            if isinstance(value, float):
                value = round(value, 3)
            key.append((name, value))
        return tuple(key)

    def _mask(self, name):
        return self.masks.get(name)

    def _blend(self, base, adjusted, mask, strength=1.0):
        if mask is None or np.max(mask) <= 0 or strength <= 0:
            return base.copy()
        alpha = np.clip(mask.astype(np.float32) * float(strength), 0.0, 1.0)
        return np.clip(base.astype(np.float32) * (1 - alpha) + adjusted.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

    def _float_strength(self, params, key):
        return float(np.clip(params.get(key, 0.0), 0.0, 1.0))

    def _signed_strength(self, params, key):
        return float(np.clip(params.get(key, 0.0), -1.0, 1.0))

    def _apply_skin_effects(self, img, params, use_ai):
        if self.skin_mask is None:
            return img.copy()

        result = img.copy()
        skin = self.skin_mask
        smooth = self._float_strength(params, "skin_smooth")
        pore = self._float_strength(params, "pore_blur")
        noise = self._float_strength(params, "noise_reduce")
        total_smooth = np.clip(smooth * 0.75 + pore * 0.55 + noise * 0.45, 0, 1)

        if total_smooth > 0:
            strength = float(total_smooth ** 0.58)
            min_size = min(self.width, self.height)
            diameter = max(9, min(45, int(9 + min_size * 0.035 * strength) | 1))
            sigma_color = 55 + 185 * strength
            sigma_space = 45 + 155 * strength
            source = result.astype(np.uint8)
            edge_base = cv2.bilateralFilter(source, diameter, sigma_color, sigma_space)
            soft_base = cv2.GaussianBlur(edge_base, (max(5, int(5 + 14 * strength) | 1),) * 2, 0)
            detail = result.astype(np.float32) - soft_base.astype(np.float32)
            keep = max(0.08, 1.0 - 0.82 * strength)
            smoothed = soft_base.astype(np.float32) + detail * keep
            result = self._blend(result, np.clip(smoothed, 0, 255).astype(np.uint8), skin, 0.55 + 0.35 * strength)

        for key, mode in [("blemish_remove", "blemish"), ("freckle_fade", "freckle")]:
            amount = self._float_strength(params, key)
            if amount > 0:
                repair_mask = self._detect_spots(result, amount, mode)
                result = self._repair_masked_region(
                    result,
                    repair_mask,
                    amount,
                    use_ai,
                    "natural clean skin texture, remove small blemishes, preserve identity",
                )

        wrinkle = self._float_strength(params, "wrinkle_soften")
        if wrinkle > 0:
            wrinkle_mask = self._detect_wrinkles(result, wrinkle)
            lifted = self._adjust_lab(result, l_delta=16 * wrinkle)
            result = self._blend(result, lifted, wrinkle_mask, 0.35 + 0.35 * wrinkle)
            if use_ai and wrinkle > 0.25:
                result = self._repair_masked_region(
                    result,
                    wrinkle_mask,
                    wrinkle,
                    use_ai,
                    "natural smooth skin texture, softened fine wrinkles, preserve face identity",
                )

        dark_circle = self._float_strength(params, "dark_circle_remove")
        if dark_circle > 0:
            result = self._remove_dark_circles(result, dark_circle, use_ai)

        shine = self._float_strength(params, "shine_remove")
        if shine > 0:
            result = self._remove_shine(result, shine, use_ai)

        whiten = self._float_strength(params, "skin_whiten")
        tone_even = self._float_strength(params, "skin_tone_even")
        if whiten > 0 or tone_even > 0:
            adjusted = self._adjust_lab(result, l_delta=28 * whiten + 8 * tone_even, b_shift=-10 * whiten)
            if tone_even > 0:
                blurred = cv2.bilateralFilter(adjusted, 13, 55, 55)
                adjusted = cv2.addWeighted(adjusted, 1 - 0.35 * tone_even, blurred, 0.35 * tone_even, 0)
            result = self._blend(result, adjusted, skin, 0.25 + 0.55 * max(whiten, tone_even))

        restore = self._float_strength(params, "skin_texture_restore")
        if restore > 0:
            restored = self.ai_models.restore_face(result, restore, use_ai=use_ai)
            result = self._blend(result, restored, skin, restore)

        return result

    def _apply_makeup_effects(self, img, params, preset, use_ai):
        result = img.copy()
        blush = self._float_strength(params, "blush")
        if blush > 0:
            color = self._blush_color_for_preset(preset.get("id"))
            overlay = np.zeros_like(result)
            overlay[:] = color
            result = self._blend(result, overlay, self._mask("cheeks"), 0.28 * blush)
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + 0.16 * blush), 0, 255)
            result = self._blend(result, cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), self._mask("cheeks"), 0.35 * blush)

        lip = self._float_strength(params, "lip_color")
        if lip > 0:
            overlay = np.zeros_like(result)
            overlay[:] = (80, 45, 180)
            result = self._blend(result, overlay, self._mask("lips"), 0.22 * lip)

        eye = self._float_strength(params, "eye_brighten")
        if eye > 0:
            bright = self._adjust_lab(result, l_delta=18 * eye, b_shift=-4 * eye)
            result = self._blend(result, bright, self._mask("eyes"), 0.35 * eye)

        brow = self._float_strength(params, "brow_enhance")
        if brow > 0:
            darker = self._adjust_lab(result, l_delta=-20 * brow)
            result = self._blend(result, darker, self._mask("brows"), 0.45 * brow)

        return result

    def _apply_teeth_effects(self, img, params, use_ai):
        teeth_mask = self._mask("teeth")
        if teeth_mask is None or np.max(teeth_mask) <= 0:
            return img.copy()

        result = img.copy()
        whiten = self._float_strength(params, "teeth_whiten")
        if whiten > 0:
            adjusted = self._adjust_lab(result, l_delta=45 * whiten, b_shift=-26 * whiten)
            result = self._blend(result, adjusted, teeth_mask, 0.65 * whiten)

        stain = self._float_strength(params, "teeth_stain_remove")
        if stain > 0:
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            stain_mask = ((teeth_mask[:, :, 0] > 0.15) & (lab[:, :, 2] > 142)).astype(np.float32)
            stain_mask = np.expand_dims(cv2.GaussianBlur((stain_mask * 255).astype(np.uint8), (9, 9), 0) / 255.0, axis=2)
            result = self._repair_masked_region(result, stain_mask, stain, use_ai, "clean natural white teeth, remove stains, preserve mouth shape")
            adjusted = self._adjust_lab(result, l_delta=20 * stain, b_shift=-22 * stain)
            result = self._blend(result, adjusted, stain_mask, 0.5 * stain)

        straighten = self._float_strength(params, "teeth_straighten")
        if straighten > 0:
            smoothed = cv2.bilateralFilter(result, 7, 45, 45)
            result = self._blend(result, smoothed, teeth_mask, 0.35 * straighten)
            if use_ai and straighten > 0.3:
                ai_result = self.ai_models.diffusion_edit(
                    result,
                    teeth_mask,
                    "natural even clean teeth, preserve mouth shape, realistic smile",
                    "extra teeth, distorted teeth, changed lips, changed face, artifacts",
                    min(0.28, 0.12 + 0.18 * straighten),
                    preset=None,
                )
                if ai_result is not None:
                    result = self._blend(result, ai_result, teeth_mask, 0.7)

        return result

    def _apply_lighting_effects(self, img, params):
        result = img.copy()
        face = self._mask("face")
        if face is None:
            face = np.ones((self.height, self.width, 1), dtype=np.float32)

        fill = self._float_strength(params, "fill_light")
        brighten = self._float_strength(params, "face_brighten")
        if fill > 0 or brighten > 0:
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
            l = lab[:, :, 0]
            shadow_weight = np.clip((170 - l) / 170, 0, 1)
            lab[:, :, 0] = np.clip(l + shadow_weight * 42 * fill + 24 * brighten, 0, 255)
            adjusted = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
            result = self._blend(result, adjusted, face, max(fill, brighten))

        shadow = self._float_strength(params, "shadow_enhance")
        if shadow > 0:
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
            blurred = cv2.GaussianBlur(lab[:, :, 0], (0, 0), 9)
            detail = lab[:, :, 0] - blurred
            lab[:, :, 0] = np.clip(lab[:, :, 0] + detail * 0.45 * shadow, 0, 255)
            adjusted = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
            contour = np.maximum.reduce([
                self._mask("nose") if self._mask("nose") is not None else face * 0,
                self._mask("left_cheek") if self._mask("left_cheek") is not None else face * 0,
                self._mask("right_cheek") if self._mask("right_cheek") is not None else face * 0,
            ])
            result = self._blend(result, adjusted, np.maximum(contour, face * 0.35), 0.75 * shadow)

        rim = self._float_strength(params, "rim_light")
        if rim > 0:
            overlay = np.zeros_like(result)
            overlay[:] = (235, 225, 205)
            result = self._blend(result, overlay, self._mask("rim"), 0.35 * rim)

        film = self._float_strength(params, "film_light")
        if film > 0:
            adjusted = result.astype(np.float32)
            adjusted[:, :, 0] *= 1 + 0.04 * film
            adjusted[:, :, 1] *= 1 + 0.02 * film
            adjusted[:, :, 2] *= 1 + 0.08 * film
            adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
            result = cv2.addWeighted(result, 1 - 0.35 * film, adjusted, 0.35 * film, 0)

        return result

    def _apply_style_effects(self, img, params, preset, use_ai):
        result = img.copy()
        warmth = self._signed_strength(params, "style_warmth") - 0.5
        contrast = self._signed_strength(params, "style_contrast")
        saturation = self._signed_strength(params, "style_saturation")
        style_strength = self._float_strength(params, "style_strength")

        if abs(warmth) > 0.01:
            adjusted = result.astype(np.float32)
            adjusted[:, :, 2] *= 1 + 0.18 * warmth
            adjusted[:, :, 0] *= 1 - 0.18 * warmth
            result = np.clip(adjusted, 0, 255).astype(np.uint8)

        if abs(contrast) > 0.01:
            alpha = 1 + 0.55 * contrast
            beta = -18 * contrast
            result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)

        if abs(saturation) > 0.01:
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + 0.65 * saturation), 0, 255)
            result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if use_ai and style_strength > 0:
            mask = self._mask("face")
            ai_result = self.ai_models.diffusion_edit(
                result,
                mask,
                preset.get("prompt"),
                preset.get("negative_prompt"),
                min(0.35, 0.08 + style_strength * 0.35),
                preset=preset,
            )
            if ai_result is not None:
                result = self._blend(result, ai_result, mask, 0.75)

        return result

    def _detect_spots(self, img, amount, mode):
        skin = self.skin_mask[:, :, 0] if self.skin_mask is not None else np.ones((self.height, self.width), dtype=np.float32)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        blur = cv2.GaussianBlur(lab, (0, 0), 5)
        diff = np.abs(lab - blur)
        if mode == "freckle":
            score = np.clip((blur[:, :, 0] - lab[:, :, 0]) / 28, 0, 1) * np.clip((lab[:, :, 2] - 128) / 28, 0, 1)
        else:
            score = np.clip(diff[:, :, 1] / 18, 0, 1) + np.clip((blur[:, :, 0] - lab[:, :, 0]) / 35, 0, 1)
        mask = ((score > (0.45 - 0.2 * amount)) & (skin > 0.35)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        return np.expand_dims((mask / 255.0).astype(np.float32), axis=2)

    def _detect_wrinkles(self, img, amount):
        skin = self.skin_mask[:, :, 0] if self.skin_mask is not None else np.ones((self.height, self.width), dtype=np.float32)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 3))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        mask = ((blackhat > (10 - 4 * amount)) & (skin > 0.35)).astype(np.uint8) * 255
        mask = cv2.GaussianBlur(mask, (9, 9), 0)
        return np.expand_dims((mask / 255.0).astype(np.float32), axis=2)

    def _repair_masked_region(self, img, mask, amount, use_ai, prompt):
        if mask is None or np.max(mask) <= 0:
            return img.copy()
        hard_mask = (mask[:, :, 0] > 0.18).astype(np.float32)
        if np.count_nonzero(hard_mask) < 12:
            return img.copy()
        repaired = self.ai_models.inpaint(img, np.expand_dims(hard_mask, axis=2), prompt=prompt, strength=amount, use_ai=use_ai)
        return self._blend(img, repaired, mask, 0.35 + 0.45 * amount)

    def _remove_dark_circles(self, img, amount, use_ai):
        mask = self._mask("under_eye")
        if mask is None or np.max(mask) <= 0:
            return img.copy()
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 0] = np.clip(lab[:, :, 0] + 24 * amount, 0, 255)
        lab[:, :, 1] = lab[:, :, 1] * (1 - 0.08 * amount) + 128 * (0.08 * amount)
        lab[:, :, 2] = lab[:, :, 2] * (1 - 0.10 * amount) + 134 * (0.10 * amount)
        adjusted = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        adjusted = cv2.bilateralFilter(adjusted, 7, 35, 35)
        result = self._blend(img, adjusted, mask, 0.55 * amount)
        if use_ai and amount > 0.35:
            result = self._repair_masked_region(result, mask, amount, use_ai, "natural under-eye skin, remove dark circles, preserve eye shape and identity")
        return result

    def _remove_shine(self, img, amount, use_ai):
        skin = self.skin_mask if self.skin_mask is not None else None
        if skin is None:
            return img.copy()
        t_zone = self._mask("t_zone")
        area = np.maximum(skin * 0.6, t_zone if t_zone is not None else skin)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        highlight = ((area[:, :, 0] > 0.2) & (hsv[:, :, 2] > 175) & (hsv[:, :, 1] < 95) & (lab[:, :, 0] > 160)).astype(np.uint8) * 255
        highlight = cv2.dilate(highlight, np.ones((5, 5), np.uint8), iterations=1)
        highlight = cv2.GaussianBlur(highlight, (21, 21), 0)
        mask = np.expand_dims((highlight / 255.0).astype(np.float32), axis=2)
        if np.max(mask) <= 0:
            return img.copy()
        base = cv2.bilateralFilter(img, 21, 75, 75)
        lab_base = cv2.cvtColor(base, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_img[:, :, 0] = lab_img[:, :, 0] * (1 - 0.55 * amount) + lab_base[:, :, 0] * (0.55 * amount)
        adjusted = cv2.cvtColor(np.clip(lab_img, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        result = self._blend(img, adjusted, mask, 0.75 * amount)
        if use_ai and amount > 0.4:
            result = self._repair_masked_region(result, mask, amount, use_ai, "natural matte skin texture, remove oily shine, preserve identity")
        return result

    def _adjust_lab(self, img, l_delta=0, a_shift=0, b_shift=0):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 0] = np.clip(lab[:, :, 0] + l_delta, 0, 255)
        lab[:, :, 1] = np.clip(lab[:, :, 1] + a_shift, 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + b_shift, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _blush_color_for_preset(self, preset_id):
        colors = {
            "korean": (150, 115, 245),
            "japanese": (120, 100, 230),
            "western": (95, 105, 190),
            "high_fashion": (120, 80, 190),
            "hongkong": (105, 95, 205),
        }
        return colors.get(preset_id, (125, 105, 225))

    def _warp_bloat(self, map_x, map_y, center, radius, strength):
        dx = map_x - center[0]
        dy = map_y - center[1]
        dist = np.sqrt(dx**2 + dy**2)
        mask = dist < radius
        factor = 1.0 - strength * ((1.0 - dist[mask] / radius) ** 2)
        map_x[mask] = center[0] + dx[mask] * factor
        map_y[mask] = center[1] + dy[mask] * factor

    def _warp_translate(self, map_x, map_y, pt1, pt2, radius, strength):
        dx = map_x - pt1[0]
        dy = map_y - pt1[1]
        dist = np.sqrt(dx**2 + dy**2)
        mask = dist < radius
        vec = np.array([pt2[0] - pt1[0], pt2[1] - pt1[1]])
        weight = (1.0 - dist[mask] / radius) ** 2
        map_x[mask] -= vec[0] * weight * strength
        map_y[mask] -= vec[1] * weight * strength

    def _apply_warping(self, img, params):
        if self.landmarks is None:
            return img.copy()

        shape_keys = [
            "eye_enlarge", "slim_face", "nose_slim", "mouth_smile", "chin", "forehead",
            "cheekbones", "jaw_angle", "apple_cheeks", "mouth_size", "nose_lift",
            "eye_spacing", "eyebrow_lift",
        ]
        if all(abs(float(params.get(key, 0.0))) < 0.001 for key in shape_keys):
            return img.copy()

        map_x = self.base_map_x.copy()
        map_y = self.base_map_y.copy()

        left_face = self._get_pt(234)
        right_face = self._get_pt(454)
        face_width = np.linalg.norm(np.array(left_face) - np.array(right_face))
        nose_tip = self._get_pt(4)
        chin_pt = self._get_pt(152)

        eye_enlarge = self._float_strength(params, "eye_enlarge")
        if eye_enlarge > 0:
            radius = face_width * 0.15
            strength = eye_enlarge * 0.3
            self._warp_bloat(map_x, map_y, self._get_pt(468), radius, strength)
            self._warp_bloat(map_x, map_y, self._get_pt(473), radius, strength)

        slim_face = self._float_strength(params, "slim_face")
        jaw_angle = self._float_strength(params, "jaw_angle")
        if slim_face > 0 or jaw_angle > 0:
            jaw_points_left = [132, 58, 172, 136, 150, 149, 176, 148]
            jaw_points_right = [361, 288, 397, 365, 379, 378, 400, 377]
            radius = face_width * (0.19 + 0.05 * jaw_angle)
            strength = slim_face * 0.18 + jaw_angle * 0.12
            for pt_idx in jaw_points_left + jaw_points_right:
                self._warp_translate(map_x, map_y, self._get_pt(pt_idx), nose_tip, radius, strength)

        cheekbones = self._float_strength(params, "cheekbones")
        if cheekbones > 0:
            self._warp_translate(map_x, map_y, self._get_pt(234), nose_tip, face_width * 0.18, cheekbones * 0.13)
            self._warp_translate(map_x, map_y, self._get_pt(454), nose_tip, face_width * 0.18, cheekbones * 0.13)

        apple = self._float_strength(params, "apple_cheeks")
        if apple > 0:
            self._warp_bloat(map_x, map_y, self._get_pt(205), face_width * 0.12, apple * 0.16)
            self._warp_bloat(map_x, map_y, self._get_pt(425), face_width * 0.12, apple * 0.16)

        nose_slim = self._float_strength(params, "nose_slim")
        if nose_slim > 0:
            radius = face_width * 0.1
            strength = nose_slim * 0.2
            self._warp_translate(map_x, map_y, self._get_pt(129), nose_tip, radius, strength)
            self._warp_translate(map_x, map_y, self._get_pt(358), nose_tip, radius, strength)

        nose_lift = self._float_strength(params, "nose_lift")
        if nose_lift > 0:
            target = (nose_tip[0], nose_tip[1] - face_width * 0.06)
            self._warp_translate(map_x, map_y, nose_tip, target, face_width * 0.12, nose_lift * 0.16)

        mouth_smile = self._float_strength(params, "mouth_smile")
        if mouth_smile > 0:
            left_corner = self._get_pt(61)
            right_corner = self._get_pt(291)

            def get_smile_target(corner, is_left):
                dy = chin_pt[1] - corner[1]
                dx = corner[0] - chin_pt[0] if is_left else chin_pt[0] - corner[0]
                target_x = corner[0] + (1 if is_left else -1) * (dx * 0.2)
                target_y = corner[1] - (dy * 0.2)
                return (target_x, target_y)

            radius = face_width * 0.1
            strength = mouth_smile * 0.3
            self._warp_translate(map_x, map_y, left_corner, get_smile_target(left_corner, True), radius, strength)
            self._warp_translate(map_x, map_y, right_corner, get_smile_target(right_corner, False), radius, strength)

        mouth_size = self._float_strength(params, "mouth_size")
        if mouth_size > 0:
            mouth_center = self._get_pt(13)
            self._warp_bloat(map_x, map_y, mouth_center, face_width * 0.13, mouth_size * 0.12)

        chin = self._float_strength(params, "chin")
        if chin > 0:
            target = (chin_pt[0], chin_pt[1] + face_width * 0.08)
            self._warp_translate(map_x, map_y, chin_pt, target, face_width * 0.16, chin * 0.18)

        forehead = self._float_strength(params, "forehead")
        if forehead > 0:
            forehead_pt = self._get_pt(10)
            target = (forehead_pt[0], forehead_pt[1] - face_width * 0.06)
            self._warp_translate(map_x, map_y, forehead_pt, target, face_width * 0.16, forehead * 0.12)

        eye_spacing = self._float_strength(params, "eye_spacing")
        if eye_spacing > 0:
            left_eye = self._get_pt(468)
            right_eye = self._get_pt(473)
            center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
            self._warp_translate(map_x, map_y, left_eye, center, face_width * 0.12, eye_spacing * 0.08)
            self._warp_translate(map_x, map_y, right_eye, center, face_width * 0.12, eye_spacing * 0.08)

        brow = self._float_strength(params, "eyebrow_lift")
        if brow > 0:
            for idx in [65, 295, 105, 334]:
                pt = self._get_pt(idx)
                self._warp_translate(map_x, map_y, pt, (pt[0], pt[1] - face_width * 0.05), face_width * 0.11, brow * 0.12)

        warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return warped
