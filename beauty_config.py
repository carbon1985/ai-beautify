from copy import deepcopy

AI_MODE_PARAM = "_ai_mode"
STYLE_PRESET_PARAM = "style_preset"

FEATURE_GROUPS = [
    {
        "id": "skin",
        "name": "肤质 Skin",
        "features": [
            {"id": "skin_smooth", "name": "磨皮", "default": 0.0, "mode": "cv"},
            {"id": "pore_blur", "name": "去毛孔", "default": 0.0, "mode": "cv"},
            {"id": "noise_reduce", "name": "去噪点", "default": 0.0, "mode": "cv"},
            {"id": "blemish_remove", "name": "去痘印", "default": 0.0, "mode": "inpaint"},
            {"id": "freckle_fade", "name": "去雀斑", "default": 0.0, "mode": "inpaint"},
            {"id": "wrinkle_soften", "name": "去皱纹", "default": 0.0, "mode": "inpaint"},
            {"id": "dark_circle_remove", "name": "去黑眼圈", "default": 0.0, "mode": "hybrid"},
            {"id": "shine_remove", "name": "去油光", "default": 0.0, "mode": "hybrid"},
            {"id": "skin_whiten", "name": "美白", "default": 0.0, "mode": "cv"},
            {"id": "skin_tone_even", "name": "肤色均匀", "default": 0.0, "mode": "cv"},
            {"id": "skin_texture_restore", "name": "AI 肌理恢复", "default": 0.0, "mode": "gan"},
        ],
    },
    {
        "id": "makeup",
        "name": "彩妆 Makeup",
        "features": [
            {"id": "blush", "name": "红润/腮红", "default": 0.0, "mode": "hybrid"},
            {"id": "lip_color", "name": "唇色增强", "default": 0.0, "mode": "cv"},
            {"id": "eye_brighten", "name": "眼神提亮", "default": 0.0, "mode": "cv"},
            {"id": "brow_enhance", "name": "眉毛增强", "default": 0.0, "mode": "cv"},
        ],
    },
    {
        "id": "teeth",
        "name": "牙齿 Teeth",
        "features": [
            {"id": "teeth_whiten", "name": "牙齿美白", "default": 0.0, "mode": "cv"},
            {"id": "teeth_stain_remove", "name": "去牙渍", "default": 0.0, "mode": "inpaint"},
            {"id": "teeth_straighten", "name": "牙齿整齐", "default": 0.0, "mode": "diffusion"},
        ],
    },
    {
        "id": "lighting",
        "name": "光影 Lighting",
        "features": [
            {"id": "fill_light", "name": "补光", "default": 0.0, "mode": "cv"},
            {"id": "face_brighten", "name": "面部提亮", "default": 0.0, "mode": "cv"},
            {"id": "shadow_enhance", "name": "阴影增强", "default": 0.0, "mode": "cv"},
            {"id": "rim_light", "name": "轮廓光", "default": 0.0, "mode": "cv"},
            {"id": "film_light", "name": "胶片光感", "default": 0.0, "mode": "cv"},
        ],
    },
    {
        "id": "shape",
        "name": "五官 Shape",
        "features": [
            {"id": "slim_face", "name": "瘦脸", "default": 0.0, "mode": "cv"},
            {"id": "chin", "name": "下巴", "default": 0.0, "mode": "cv"},
            {"id": "forehead", "name": "额头", "default": 0.0, "mode": "cv"},
            {"id": "cheekbones", "name": "颧骨", "default": 0.0, "mode": "cv"},
            {"id": "jaw_angle", "name": "下颌角", "default": 0.0, "mode": "cv"},
            {"id": "apple_cheeks", "name": "苹果肌", "default": 0.0, "mode": "cv"},
            {"id": "mouth_size", "name": "嘴型大小", "default": 0.0, "mode": "cv"},
            {"id": "mouth_smile", "name": "微笑嘴角", "default": 0.0, "mode": "cv"},
            {"id": "nose_slim", "name": "瘦鼻", "default": 0.0, "mode": "cv"},
            {"id": "nose_lift", "name": "鼻尖/鼻梁", "default": 0.0, "mode": "cv"},
            {"id": "eye_enlarge", "name": "大眼", "default": 0.0, "mode": "cv"},
            {"id": "eye_spacing", "name": "眼距", "default": 0.0, "mode": "cv"},
            {"id": "eyebrow_lift", "name": "眉形上扬", "default": 0.0, "mode": "cv"},
        ],
    },
    {
        "id": "style",
        "name": "风格 Style",
        "features": [
            {"id": "style_strength", "name": "AI 风格强度", "default": 0.0, "mode": "diffusion"},
            {"id": "style_warmth", "name": "冷暖色调", "default": 0.5, "mode": "cv"},
            {"id": "style_contrast", "name": "风格对比", "default": 0.0, "mode": "cv"},
            {"id": "style_saturation", "name": "风格饱和", "default": 0.0, "mode": "cv"},
        ],
    },
]

LEGACY_PARAM_ALIASES = {
    "smooth": "skin_smooth",
    "whiten": "skin_whiten",
    "enlarge_eyes": "eye_enlarge",
    "slim_nose": "nose_slim",
    "lip_shape": "mouth_smile",
}

DEFAULT_NEGATIVE_PROMPT = (
    "identity drift, different person, distorted eyes, distorted teeth, extra teeth, "
    "asymmetry, over-smoothed skin, plastic skin, waxy skin, blurry face, artifacts"
)

STYLE_PRESETS = {
    "无": {
        "id": "none",
        "name": "无",
        "params": {},
        "prompt": "natural realistic portrait, preserve identity, realistic skin texture",
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
    },
    "韩系": {
        "id": "korean",
        "name": "韩系",
        "params": {
            "skin_smooth": 0.38,
            "skin_whiten": 0.32,
            "skin_tone_even": 0.25,
            "blush": 0.22,
            "eye_enlarge": 0.18,
            "slim_face": 0.18,
            "face_brighten": 0.25,
            "style_strength": 0.22,
            "style_warmth": 0.56,
            "style_contrast": 0.08,
            "style_saturation": 0.06,
        },
        "prompt": "clean korean beauty portrait, bright soft skin, subtle pink blush, delicate makeup, preserve identity",
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
    },
    "日系": {
        "id": "japanese",
        "name": "日系",
        "params": {
            "skin_smooth": 0.25,
            "skin_tone_even": 0.18,
            "blush": 0.26,
            "face_brighten": 0.18,
            "film_light": 0.18,
            "style_strength": 0.18,
            "style_warmth": 0.6,
            "style_contrast": 0.02,
            "style_saturation": 0.04,
        },
        "prompt": "natural japanese beauty portrait, soft film tone, gentle rosy cheeks, realistic skin texture, preserve identity",
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
    },
    "港风": {
        "id": "hongkong",
        "name": "港风",
        "params": {
            "skin_smooth": 0.18,
            "shadow_enhance": 0.28,
            "rim_light": 0.16,
            "style_strength": 0.2,
            "style_warmth": 0.68,
            "style_contrast": 0.24,
            "style_saturation": 0.08,
        },
        "prompt": "hong kong cinematic beauty portrait, warm contrast, defined facial lighting, elegant natural makeup, preserve identity",
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
    },
    "欧美": {
        "id": "western",
        "name": "欧美",
        "params": {
            "skin_smooth": 0.18,
            "shadow_enhance": 0.35,
            "cheekbones": 0.16,
            "jaw_angle": 0.12,
            "nose_slim": 0.1,
            "style_strength": 0.22,
            "style_warmth": 0.52,
            "style_contrast": 0.22,
            "style_saturation": 0.04,
        },
        "prompt": "western editorial beauty portrait, sculpted contour, defined features, realistic detailed skin, preserve identity",
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
    },
    "胶片": {
        "id": "film",
        "name": "胶片",
        "params": {
            "skin_smooth": 0.14,
            "film_light": 0.35,
            "face_brighten": 0.08,
            "style_strength": 0.2,
            "style_warmth": 0.62,
            "style_contrast": 0.08,
            "style_saturation": -0.12,
        },
        "prompt": "film portrait, soft grain, muted colors, natural skin texture, cinematic but realistic, preserve identity",
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
    },
    "原生感": {
        "id": "natural",
        "name": "原生感",
        "params": {
            "skin_smooth": 0.12,
            "pore_blur": 0.08,
            "skin_tone_even": 0.1,
            "dark_circle_remove": 0.12,
            "shine_remove": 0.12,
            "style_strength": 0.08,
            "style_warmth": 0.5,
            "style_contrast": 0.0,
            "style_saturation": 0.0,
        },
        "prompt": "natural untouched beauty portrait, realistic skin texture, minimal makeup, preserve identity exactly",
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
    },
    "高级脸": {
        "id": "high_fashion",
        "name": "高级脸",
        "params": {
            "skin_smooth": 0.22,
            "skin_texture_restore": 0.22,
            "shadow_enhance": 0.32,
            "rim_light": 0.22,
            "cheekbones": 0.18,
            "jaw_angle": 0.14,
            "eye_enlarge": 0.08,
            "style_strength": 0.28,
            "style_warmth": 0.46,
            "style_contrast": 0.28,
            "style_saturation": -0.03,
        },
        "prompt": "high fashion editorial beauty portrait, sculpted light, refined makeup, sharp elegant facial contour, preserve identity",
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
    },
}


def default_params():
    params = {}
    for group in FEATURE_GROUPS:
        for feature in group["features"]:
            params[feature["id"]] = feature["default"]
    params[AI_MODE_PARAM] = False
    params[STYLE_PRESET_PARAM] = "无"
    return params


def normalize_params(params):
    normalized = default_params()
    normalized.update(params or {})
    for old_key, new_key in LEGACY_PARAM_ALIASES.items():
        if old_key in params and new_key not in params:
            normalized[new_key] = params[old_key]
    return normalized


def get_preset(name):
    return deepcopy(STYLE_PRESETS.get(name) or STYLE_PRESETS["无"])


def preset_names():
    return list(STYLE_PRESETS.keys())
