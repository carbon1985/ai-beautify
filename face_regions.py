import cv2
import numpy as np


FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]


class FaceRegionBuilder:
    def __init__(self, landmarks, width, height):
        self.landmarks = landmarks
        self.width = width
        self.height = height

    def get_pt(self, idx):
        pt = self.landmarks.landmark[idx]
        return (int(pt.x * self.width), int(pt.y * self.height))

    def expanded_points(self, indices, x_scale=1.12, upper_y_scale=1.22, lower_y_scale=1.10):
        pts = np.array([self.get_pt(i) for i in indices], dtype=np.float32)
        center = pts.mean(axis=0)
        expanded = pts.copy()
        for i, pt in enumerate(pts):
            vector = pt - center
            y_scale = upper_y_scale if pt[1] < center[1] else lower_y_scale
            expanded[i] = center + np.array([vector[0] * x_scale, vector[1] * y_scale], dtype=np.float32)
        expanded[:, 0] = np.clip(expanded[:, 0], 0, self.width - 1)
        expanded[:, 1] = np.clip(expanded[:, 1], 0, self.height - 1)
        return expanded.astype(np.int32)

    def polygon_mask(self, indices, feather=17, expand=False):
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        pts = self.expanded_points(indices) if expand else np.array([self.get_pt(i) for i in indices], dtype=np.int32)
        cv2.fillPoly(mask, [pts.reshape((-1, 1, 2))], 255)
        return self.feather(mask, feather)

    def ellipse_mask(self, center, axes, angle=0, feather=31):
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.ellipse(mask, tuple(np.int32(center)), tuple(np.int32(axes)), angle, 0, 360, 255, -1)
        return self.feather(mask, feather)

    def feather(self, mask, feather=31):
        if feather > 1:
            feather = max(3, int(feather) | 1)
            mask = cv2.GaussianBlur(mask, (feather, feather), 0)
        return np.expand_dims((mask / 255.0).astype(np.float32), axis=2)

    def skin_color_mask(self, image, seed_mask):
        seed = seed_mask[:, :, 0] > 0.65
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        samples = ycrcb[seed]
        color_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        if len(samples) > 0:
            median_cr = np.median(samples[:, 1])
            median_cb = np.median(samples[:, 2])
            cr_diff = ycrcb[:, :, 1] - median_cr
            cb_diff = ycrcb[:, :, 2] - median_cb
            color_distance = np.sqrt(cr_diff * cr_diff + cb_diff * cb_diff)
            color_mask = (np.clip(1.0 - color_distance / 34.0, 0.0, 1.0) * 255).astype(np.uint8)
        return self.feather(color_mask, 21)

    def build(self, image=None):
        min_size = min(self.width, self.height)
        face_mask = self.polygon_mask(FACE_OVAL, feather=max(51, int(min_size * 0.06) | 1), expand=True)
        left_eye = self.polygon_mask(LEFT_EYE, feather=13)
        right_eye = self.polygon_mask(RIGHT_EYE, feather=13)
        lips = self.polygon_mask(LIPS_OUTER, feather=17)
        left_brow = self.polygon_mask(LEFT_EYEBROW, feather=13)
        right_brow = self.polygon_mask(RIGHT_EYEBROW, feather=13)

        skin_mask = face_mask.copy()
        exclude = np.maximum.reduce([left_eye, right_eye, lips, left_brow, right_brow])
        skin_mask = np.clip(skin_mask * (1.0 - exclude), 0.0, 1.0)
        if image is not None:
            skin_mask = np.minimum(skin_mask, np.maximum(self.skin_color_mask(image, skin_mask), skin_mask * 0.6))

        chin = np.array(self.get_pt(152), dtype=np.float32)
        left_jaw = np.array(self.get_pt(172), dtype=np.float32)
        right_jaw = np.array(self.get_pt(397), dtype=np.float32)
        left_face = np.array(self.get_pt(234), dtype=np.float32)
        right_face = np.array(self.get_pt(454), dtype=np.float32)
        face_width = np.linalg.norm(left_face - right_face)

        neck_bottom_y = min(self.height - 1, chin[1] + face_width * 0.55)
        neck_width = face_width * 0.36
        neck = np.zeros((self.height, self.width), dtype=np.uint8)
        neck_pts = np.array([
            left_jaw,
            right_jaw,
            [chin[0] + neck_width, neck_bottom_y],
            [chin[0] - neck_width, neck_bottom_y],
        ], dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(neck, [neck_pts], 255)
        neck = self.feather(neck, max(51, int(min_size * 0.05) | 1))
        skin_mask = np.maximum(skin_mask, neck * 0.75)

        left_eye_center = np.array(self.get_pt(468), dtype=np.float32)
        right_eye_center = np.array(self.get_pt(473), dtype=np.float32)
        under_left = self.ellipse_mask(left_eye_center + [0, face_width * 0.07], (face_width * 0.13, face_width * 0.055), -8, 25)
        under_right = self.ellipse_mask(right_eye_center + [0, face_width * 0.07], (face_width * 0.13, face_width * 0.055), 8, 25)
        under_eye = np.maximum(under_left, under_right) * skin_mask

        nose_tip = np.array(self.get_pt(4), dtype=np.float32)
        forehead = np.array(self.get_pt(10), dtype=np.float32)
        nose_mask = self.ellipse_mask(nose_tip + [0, -face_width * 0.03], (face_width * 0.105, face_width * 0.24), 0, 31) * skin_mask
        forehead_mask = self.ellipse_mask(forehead + [0, face_width * 0.12], (face_width * 0.22, face_width * 0.12), 0, 31) * skin_mask
        t_zone = np.clip(np.maximum(nose_mask, forehead_mask), 0, 1)

        left_cheek_center = (np.array(self.get_pt(205), dtype=np.float32) + np.array(self.get_pt(50), dtype=np.float32)) / 2
        right_cheek_center = (np.array(self.get_pt(425), dtype=np.float32) + np.array(self.get_pt(280), dtype=np.float32)) / 2
        left_cheek = self.ellipse_mask(left_cheek_center, (face_width * 0.16, face_width * 0.10), -15, 45) * skin_mask
        right_cheek = self.ellipse_mask(right_cheek_center, (face_width * 0.16, face_width * 0.10), 15, 45) * skin_mask
        cheeks = np.maximum(left_cheek, right_cheek)

        mouth_inner = self.polygon_mask(LIPS_INNER, feather=9)
        teeth = mouth_inner.copy()
        if image is not None:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            candidate = ((mouth_inner[:, :, 0] > 0.2) & (hsv[:, :, 1] < 95) & (lab[:, :, 0] > 95)).astype(np.uint8) * 255
            kernel = np.ones((3, 3), np.uint8)
            candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
            candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
            teeth = self.feather(candidate, 9)

        face_uint = (face_mask[:, :, 0] * 255).astype(np.uint8)
        eroded = cv2.erode(face_uint, np.ones((max(5, int(face_width * 0.035)),) * 2, np.uint8), iterations=1)
        contour_edge = self.feather(cv2.subtract(face_uint, eroded), 31)

        return {
            "face": face_mask,
            "skin": np.clip(skin_mask, 0, 1),
            "neck": neck,
            "left_eye": left_eye,
            "right_eye": right_eye,
            "eyes": np.maximum(left_eye, right_eye),
            "under_eye": under_eye,
            "lips": lips,
            "mouth_inner": mouth_inner,
            "teeth": teeth,
            "left_brow": left_brow,
            "right_brow": right_brow,
            "brows": np.maximum(left_brow, right_brow),
            "nose": nose_mask,
            "forehead": forehead_mask,
            "t_zone": t_zone,
            "left_cheek": left_cheek,
            "right_cheek": right_cheek,
            "cheeks": cheeks,
            "rim": contour_edge,
        }
