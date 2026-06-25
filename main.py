import customtkinter as ctk
from tkinter import filedialog, messagebox
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import cv2
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
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.processed_image = None
        self.loading_dialog = None
        self.loading_progress = None
        self.is_closing = False
        self.show_landmarks = ctk.BooleanVar(value=False)
        
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
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.resize_after_id = None
        
    def setup_ui(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Top Bar
        self.top_frame = ctk.CTkFrame(self)
        self.top_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        
        self.title_label = ctk.CTkLabel(self.top_frame, text="AI Beautify", font=("Roboto", 24, "bold"))
        self.title_label.pack(side="left", padx=20, pady=10)

        self.landmark_switch = ctk.CTkSwitch(
            self.top_frame,
            text="显示网格关键点",
            variable=self.show_landmarks,
            command=self.on_landmark_toggle
        )
        self.landmark_switch.pack(side="left", padx=10, pady=10)
        
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
        self.slider.bind("<ButtonRelease-1>", self.on_slider_release)
        
        self.btn_frame = ctk.CTkFrame(self.feature_edit_frame, fg_color="transparent")
        self.btn_frame.pack(pady=5)
        
        self.btn_reset = ctk.CTkButton(self.btn_frame, text="Reset", command=self.on_reset_click)
        self.btn_reset.pack(side="left", padx=10)
        
        self.btn_exit = ctk.CTkButton(self.btn_frame, text="Exit", command=self.on_exit_click)
        self.btn_exit.pack(side="left", padx=10)

    def import_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_path:
            params = self.global_params.copy()

            def load_and_process():
                if not self.processor.load_image(file_path):
                    raise ValueError("无法读取所选图片。")
                return self.processor.process(params)

            def show_imported_images(processed_image):
                self.processed_image = processed_image
                self._show_input_image()
                self._show_image(self.lbl_img_right, processed_image)

            self._submit_background(load_and_process, show_imported_images, "正在分析并处理图片…")

    def export_image(self):
        if self.processor.original_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
            if file_path:
                params = self.global_params.copy()

                def process_and_save():
                    result = self.processor.process(params)
                    if not cv2.imwrite(file_path, result):
                        raise IOError("图片导出失败。")
                    return result

                def export_finished(result):
                    self.processed_image = result
                    messagebox.showinfo("导出完成", "图片已成功导出。", parent=self)

                self._submit_background(process_and_save, export_finished, "正在导出图片…")

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

    def _show_image(self, label, image):
        ctk_image = self._cv2_to_ctk_image(image)
        label.configure(image=ctk_image, text="")

    def _show_input_image(self):
        if self.processor.original_image is None:
            return

        image = (
            self.processor.get_landmark_preview()
            if self.show_landmarks.get()
            else self.processor.original_image
        )
        self._show_image(self.lbl_img_left, image)

    def _refresh_displayed_images(self):
        if self.processor.original_image is not None:
            self._show_input_image()
        if self.processed_image is not None:
            self._show_image(self.lbl_img_right, self.processed_image)

    def _process_current_params(self):
        if self.processor.original_image is None:
            return

        params = self.global_params.copy()

        def process():
            return self.processor.process(params)

        def show_result(result):
            self.processed_image = result
            self._show_image(self.lbl_img_right, result)

        self._submit_background(process, show_result, "正在应用美化效果…")

    def _submit_background(self, task, on_success, loading_text):
        self._show_loading(loading_text)
        future = self.executor.submit(task)
        self.after(50, lambda: self._poll_future(future, on_success))

    def _poll_future(self, future, on_success):
        if self.is_closing:
            return
        if not future.done():
            self.after(50, lambda: self._poll_future(future, on_success))
            return

        self._hide_loading()
        try:
            result = future.result()
        except Exception as exc:
            messagebox.showerror("处理失败", str(exc), parent=self)
            return
        on_success(result)

    def _show_loading(self, text):
        if self.loading_dialog is not None:
            self.loading_dialog.destroy()

        dialog = ctk.CTkToplevel(self)
        dialog.title("处理中")
        dialog.geometry("320x150")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.protocol("WM_DELETE_WINDOW", lambda: None)

        ctk.CTkLabel(dialog, text=text, font=("Roboto", 16, "bold")).pack(pady=(28, 16))
        progress = ctk.CTkProgressBar(dialog, width=240, mode="indeterminate")
        progress.pack()
        progress.start()

        dialog.update_idletasks()
        x = self.winfo_rootx() + (self.winfo_width() - dialog.winfo_width()) // 2
        y = self.winfo_rooty() + (self.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")
        dialog.grab_set()
        dialog.focus_force()

        self.loading_dialog = dialog
        self.loading_progress = progress

    def _hide_loading(self):
        if self.loading_progress is not None:
            self.loading_progress.stop()
            self.loading_progress = None
        if self.loading_dialog is not None:
            self.loading_dialog.grab_release()
            self.loading_dialog.destroy()
            self.loading_dialog = None

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

    def on_slider_release(self, _event):
        self._process_current_params()

    def on_landmark_toggle(self):
        self._show_input_image()

    def on_reset_click(self):
        if self.active_feature_id:
            val = self.temp_params_snapshot[self.active_feature_id]
            self.global_params[self.active_feature_id] = val
            self.slider.set(val)
            self._process_current_params()

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
        if self.loading_dialog is None and self.processor.original_image is not None:
            self._refresh_displayed_images()

    def on_close(self):
        self.is_closing = True
        self._hide_loading()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.destroy()

if __name__ == "__main__":
    app = AIBeautifyApp()
    app.mainloop()
