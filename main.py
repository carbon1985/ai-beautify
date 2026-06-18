import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import cv2
import numpy as np
from image_processor import ImageProcessor

class AIBeautifyApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("AI Beautify")
        self.geometry("1000x700")
        self.minsize(800, 600)
        
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.processor = ImageProcessor()
        
        self.global_params = {
            "smooth": 0.0,
            "whiten": 0.0,
            "enlarge_eyes": 0.0,
            "slim_face": 0.0,
            "slim_nose": 0.0,
            "lip_shape": 0.0
        }
        self.temp_params_snapshot = {}
        self.active_feature_id = None
        
        self.features = [
            {"id": "smooth", "name": "磨皮/祛痘 (Smooth & Acne)"},
            {"id": "whiten", "name": "美白 (Whitening)"},
            {"id": "enlarge_eyes", "name": "大眼 (Enlarge Eyes)"},
            {"id": "slim_face", "name": "瘦脸 (Slim Face)"},
            {"id": "slim_nose", "name": "瘦鼻 (Slim Nose)"},
            {"id": "lip_shape", "name": "唇形 (Lip Shape)"},
        ]
        
        self.setup_ui()
        
        self.bind("<Configure>", self.on_resize)
        self.resize_after_id = None
        
    def setup_ui(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Top Bar
        self.top_frame = ctk.CTkFrame(self)
        self.top_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        
        self.title_label = ctk.CTkLabel(self.top_frame, text="AI Beautify", font=("Roboto", 24, "bold"))
        self.title_label.pack(side="left", padx=20, pady=10)
        
        self.btn_export = ctk.CTkButton(self.top_frame, text="Export Image", command=self.export_image)
        self.btn_export.pack(side="right", padx=10, pady=10)
        
        self.btn_import = ctk.CTkButton(self.top_frame, text="Import Image", command=self.import_image)
        self.btn_import.pack(side="right", padx=10, pady=10)
        
        # Image Area
        self.images_frame = ctk.CTkFrame(self)
        self.images_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        self.images_frame.grid_columnconfigure((0, 1), weight=1)
        self.images_frame.grid_rowconfigure(0, weight=1)
        
        self.lbl_img_left = ctk.CTkLabel(self.images_frame, text="Original Image", fg_color="gray15", corner_radius=10)
        self.lbl_img_left.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.lbl_img_right = ctk.CTkLabel(self.images_frame, text="Edited Image", fg_color="gray15", corner_radius=10)
        self.lbl_img_right.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        # Bottom Panel
        self.bottom_frame = ctk.CTkFrame(self, height=150)
        self.bottom_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=(10, 20))
        self.bottom_frame.grid_columnconfigure(0, weight=1)
        self.bottom_frame.grid_propagate(False)
        
        # Feature List View
        self.feature_list_frame = ctk.CTkFrame(self.bottom_frame, fg_color="transparent")
        self.feature_list_frame.place(relx=0.5, rely=0.5, anchor="center")
        
        for idx, f in enumerate(self.features):
            btn = ctk.CTkButton(self.feature_list_frame, text=f["name"], width=150, height=40,
                                command=lambda f_id=f["id"], f_name=f["name"]: self.open_feature_panel(f_id, f_name))
            btn.grid(row=idx//3, column=idx%3, padx=10, pady=10)
            
        # Feature Edit View
        self.feature_edit_frame = ctk.CTkFrame(self.bottom_frame, fg_color="transparent")
        
        self.feature_name_label = ctk.CTkLabel(self.feature_edit_frame, text="", font=("Roboto", 16, "bold"))
        self.feature_name_label.pack(pady=(10, 5))
        
        self.slider = ctk.CTkSlider(self.feature_edit_frame, from_=0, to=1.0, width=400, command=self.on_slider_change)
        self.slider.pack(pady=10, padx=20, fill="x")
        
        self.btn_frame = ctk.CTkFrame(self.feature_edit_frame, fg_color="transparent")
        self.btn_frame.pack(pady=5)
        
        self.btn_reset = ctk.CTkButton(self.btn_frame, text="Reset", command=self.on_reset_click)
        self.btn_reset.pack(side="left", padx=10)
        
        self.btn_exit = ctk.CTkButton(self.btn_frame, text="Exit", command=self.on_exit_click)
        self.btn_exit.pack(side="left", padx=10)

    def import_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_path:
            if self.processor.load_image(file_path):
                self.update_images(update_left=True, update_right=True)
            else:
                print("Failed to load image.")

    def export_image(self):
        if self.processor.original_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
            if file_path:
                res_img = self.processor.process(self.global_params)
                cv2.imwrite(file_path, res_img)

    def _cv2_to_ctk_image(self, img):
        if img is None:
            return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        frame_width = self.images_frame.winfo_width() // 2 - 40
        frame_height = self.images_frame.winfo_height() - 40
        
        if frame_width <= 10 or frame_height <= 10:
            frame_width, frame_height = 400, 400 
            
        img_ratio = pil_img.width / pil_img.height
        frame_ratio = frame_width / frame_height
        
        if img_ratio > frame_ratio:
            new_width = frame_width
            new_height = int(new_width / img_ratio)
        else:
            new_height = frame_height
            new_width = int(new_height * img_ratio)
            
        return ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(new_width, new_height))

    def update_images(self, update_left=False, update_right=True):
        if self.processor.original_image is None:
            return
            
        if update_left:
            ctk_img_left = self._cv2_to_ctk_image(self.processor.original_image)
            self.lbl_img_left.configure(image=ctk_img_left, text="")
            
        if update_right:
            res_img = self.processor.process(self.global_params)
            ctk_img_right = self._cv2_to_ctk_image(res_img)
            self.lbl_img_right.configure(image=ctk_img_right, text="")

    def open_feature_panel(self, f_id, f_name):
        self.active_feature_id = f_id
        self.temp_params_snapshot = self.global_params.copy()
        
        self.feature_name_label.configure(text=f_name)
        self.slider.set(self.global_params[f_id])
        
        self.feature_list_frame.place_forget()
        self.feature_edit_frame.place(relx=0.5, rely=0.5, anchor="center")

    def on_slider_change(self, value):
        if self.active_feature_id:
            self.global_params[self.active_feature_id] = float(value)
            self.update_images(update_left=False, update_right=True)

    def on_reset_click(self):
        if self.active_feature_id:
            val = self.temp_params_snapshot[self.active_feature_id]
            self.global_params[self.active_feature_id] = val
            self.slider.set(val)
            self.update_images(update_left=False, update_right=True)

    def on_exit_click(self):
        self.active_feature_id = None
        self.feature_edit_frame.place_forget()
        self.feature_list_frame.place(relx=0.5, rely=0.5, anchor="center")
        
    def on_resize(self, event):
        if event.widget == self:
            if self.resize_after_id:
                self.after_cancel(self.resize_after_id)
            self.resize_after_id = self.after(200, self._debounced_resize)
            
    def _debounced_resize(self):
        if self.processor.original_image is not None:
            self.update_images(update_left=True, update_right=True)

if __name__ == "__main__":
    app = AIBeautifyApp()
    app.mainloop()
