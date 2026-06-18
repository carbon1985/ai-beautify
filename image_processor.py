import cv2
import mediapipe as mp
import numpy as np

class ImageProcessor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        self.original_image = None
        self.landmarks = None
        self.height = 0
        self.width = 0
        
        self.skin_mask = None
        
        self.cached_filtered = None
        self.last_filter_params = None
        
        self.base_map_x = None
        self.base_map_y = None
        
    def load_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return False
            
        self.original_image = img
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
            self.skin_mask = None
            
        self.cached_filtered = None
        self.last_filter_params = None
        return True
        
    def _get_pt(self, idx):
        pt = self.landmarks.landmark[idx]
        return (int(pt.x * self.width), int(pt.y * self.height))
        
    def _generate_masks(self):
        face_oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        left_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        lips = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
        left_eyebrow = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
        right_eyebrow = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
        
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        def draw_poly(indices, color):
            pts = np.array([self._get_pt(i) for i in indices], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], color)
            
        draw_poly(face_oval, 255)
        draw_poly(left_eye, 0)
        draw_poly(right_eye, 0)
        draw_poly(lips, 0)
        draw_poly(left_eyebrow, 0)
        draw_poly(right_eyebrow, 0)
        
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        self.skin_mask = (mask / 255.0).astype(np.float32)
        self.skin_mask = np.expand_dims(self.skin_mask, axis=2)

    def _apply_filtering(self, img, params):
        if self.skin_mask is None:
            return img.copy()
            
        smooth = params.get("smooth", 0.0)
        whiten = params.get("whiten", 0.0)
        
        if smooth == 0.0 and whiten == 0.0:
            return img.copy()
            
        filtered = img.astype(np.float32)
        
        if smooth > 0.0:
            d = 15
            sigmaColor = 75 * smooth
            sigmaSpace = 75 * smooth
            smoothed = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)
            filtered = img * (1 - self.skin_mask) + smoothed * self.skin_mask
            
        if whiten > 0.0:
            hsv = cv2.cvtColor(filtered.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = hsv[:, :, 1] * (1 - 0.2 * whiten)
            hsv[:, :, 2] = hsv[:, :, 2] + (255 - hsv[:, :, 2]) * (0.3 * whiten)
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            whitened = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)
            filtered = filtered * (1 - self.skin_mask) + whitened * self.skin_mask
            
        return np.clip(filtered, 0, 255).astype(np.uint8)

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
            
        enlarge_eyes = params.get("enlarge_eyes", 0.0)
        slim_face = params.get("slim_face", 0.0)
        slim_nose = params.get("slim_nose", 0.0)
        lip_shape = params.get("lip_shape", 0.0)
        
        if enlarge_eyes == 0 and slim_face == 0 and slim_nose == 0 and lip_shape == 0:
            return img.copy()
            
        map_x = self.base_map_x.copy()
        map_y = self.base_map_y.copy()
        
        left_face = self._get_pt(234)
        right_face = self._get_pt(454)
        face_width = np.linalg.norm(np.array(left_face) - np.array(right_face))
        
        if enlarge_eyes > 0:
            left_eye_center = self._get_pt(468)
            right_eye_center = self._get_pt(473)
            radius = face_width * 0.15
            strength = enlarge_eyes * 0.3
            self._warp_bloat(map_x, map_y, left_eye_center, radius, strength)
            self._warp_bloat(map_x, map_y, right_eye_center, radius, strength)
            
        if slim_face > 0:
            jaw_points_left = [132, 58, 172, 136, 150, 149, 176, 148]
            jaw_points_right = [361, 288, 397, 365, 379, 378, 400, 377]
            nose_tip = self._get_pt(4)
            radius = face_width * 0.2
            strength = slim_face * 0.2
            
            for pt_idx in jaw_points_left:
                pt = self._get_pt(pt_idx)
                self._warp_translate(map_x, map_y, pt, nose_tip, radius, strength)
            for pt_idx in jaw_points_right:
                pt = self._get_pt(pt_idx)
                self._warp_translate(map_x, map_y, pt, nose_tip, radius, strength)
                
        if slim_nose > 0:
            left_alar = self._get_pt(129)
            right_alar = self._get_pt(358)
            nose_center = self._get_pt(4)
            radius = face_width * 0.1
            strength = slim_nose * 0.2
            self._warp_translate(map_x, map_y, left_alar, nose_center, radius, strength)
            self._warp_translate(map_x, map_y, right_alar, nose_center, radius, strength)
            
        if lip_shape > 0:
            left_corner = self._get_pt(61)
            right_corner = self._get_pt(291)
            chin = self._get_pt(152)
            
            def get_smile_target(corner, is_left):
                dy = chin[1] - corner[1]
                dx = corner[0] - chin[0] if is_left else chin[0] - corner[0]
                target_x = corner[0] + (1 if is_left else -1) * (dx * 0.2)
                target_y = corner[1] - (dy * 0.2)
                return (target_x, target_y)
                
            left_target = get_smile_target(left_corner, True)
            right_target = get_smile_target(right_corner, False)
            radius = face_width * 0.1
            strength = lip_shape * 0.3
            self._warp_translate(map_x, map_y, left_corner, left_target, radius, strength)
            self._warp_translate(map_x, map_y, right_corner, right_target, radius, strength)

        warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return warped

    def process(self, params):
        if self.original_image is None:
            return None
            
        if self.last_filter_params is None or \
           self.last_filter_params.get("smooth") != params.get("smooth") or \
           self.last_filter_params.get("whiten") != params.get("whiten"):
            
            self.cached_filtered = self._apply_filtering(self.original_image, params)
            self.last_filter_params = {
                "smooth": params.get("smooth", 0.0),
                "whiten": params.get("whiten", 0.0)
            }
            
        final_img = self._apply_warping(self.cached_filtered, params)
        return final_img
