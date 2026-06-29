import customtkinter as ctk
from tkinter import filedialog, messagebox
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import cv2

from beauty_config import AI_MODE_PARAM, FEATURE_GROUPS, STYLE_PRESET_PARAM, default_params, get_preset, preset_names
from image_processor import ImageProcessor


SIGNED_SLIDERS = {"style_contrast", "style_saturation"}


class AIBeautifyApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Beautify")
        self.geometry("1180x780")
        self.minsize(960, 680)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.processor = ImageProcessor()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.processed_image = None
        self.loading_dialog = None
        self.loading_progress = None
        self.is_closing = False
        self.show_landmarks = ctk.BooleanVar(value=False)
        self.ai_mode = ctk.BooleanVar(value=False)
        self.active_group_id = FEATURE_GROUPS[0]["id"]

        self.global_params = default_params()
        self.slider_widgets = {}
        self.value_labels = {}
        self.group_buttons = {}

        self.setup_ui()

        self.bind("<Configure>", self.on_resize)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.resize_after_id = None

    def setup_ui(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.top_frame = ctk.CTkFrame(self)
        self.top_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))

        self.title_label = ctk.CTkLabel(self.top_frame, text="AI Beautify", font=("Roboto", 24, "bold"))
        self.title_label.pack(side="left", padx=(20, 14), pady=10)

        self.landmark_switch = ctk.CTkSwitch(
            self.top_frame,
            text="显示网格关键点",
            variable=self.show_landmarks,
            command=self.on_landmark_toggle
        )
        self.landmark_switch.pack(side="left", padx=8, pady=10)

        self.ai_switch = ctk.CTkSwitch(
            self.top_frame,
            text="AI 高质量",
            variable=self.ai_mode,
            command=self.on_ai_mode_toggle
        )
        self.ai_switch.pack(side="left", padx=8, pady=10)

        self.preset_menu = ctk.CTkOptionMenu(
            self.top_frame,
            values=preset_names(),
            command=self.on_preset_selected,
            width=120,
        )
        self.preset_menu.set(self.global_params[STYLE_PRESET_PARAM])
        self.preset_menu.pack(side="left", padx=8, pady=10)

        self.btn_reset_all = ctk.CTkButton(self.top_frame, text="Reset All", width=90, command=self.reset_all)
        self.btn_reset_all.pack(side="left", padx=8, pady=10)

        self.btn_export = ctk.CTkButton(self.top_frame, text="Export Image", command=self.export_image)
        self.btn_export.pack(side="right", padx=10, pady=10)

        self.btn_import = ctk.CTkButton(self.top_frame, text="Import Image", command=self.import_image)
        self.btn_import.pack(side="right", padx=10, pady=10)

        self.model_status_label = ctk.CTkLabel(
            self.top_frame,
            text=self._model_status_text(),
            font=("Roboto", 11),
            text_color="gray70",
        )
        self.model_status_label.pack(side="right", padx=10, pady=10)

        self.images_frame = ctk.CTkFrame(self)
        self.images_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        self.images_frame.grid_columnconfigure((0, 1), weight=1)
        self.images_frame.grid_rowconfigure(0, weight=1)

        self.lbl_img_left = ctk.CTkLabel(self.images_frame, text="Original Image", fg_color="gray15", corner_radius=10)
        self.lbl_img_left.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.lbl_img_right = ctk.CTkLabel(self.images_frame, text="Edited Image", fg_color="gray15", corner_radius=10)
        self.lbl_img_right.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.bottom_frame = ctk.CTkFrame(self, height=230)
        self.bottom_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=(10, 20))
        self.bottom_frame.grid_columnconfigure(1, weight=1)
        self.bottom_frame.grid_rowconfigure(0, weight=1)
        self.bottom_frame.grid_propagate(False)

        self.group_frame = ctk.CTkFrame(self.bottom_frame, width=180)
        self.group_frame.grid(row=0, column=0, sticky="nsw", padx=(12, 8), pady=12)
        self.group_frame.grid_columnconfigure(0, weight=1)
        self.group_frame.grid_propagate(False)

        for idx, group in enumerate(FEATURE_GROUPS):
            btn = ctk.CTkButton(
                self.group_frame,
                text=group["name"],
                width=156,
                height=28,
                command=lambda group_id=group["id"]: self.select_group(group_id),
            )
            btn.grid(row=idx, column=0, padx=10, pady=4, sticky="ew")
            self.group_buttons[group["id"]] = btn

        self.controls_frame = ctk.CTkScrollableFrame(self.bottom_frame, fg_color="transparent")
        self.controls_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 12), pady=12)
        self.controls_frame.grid_columnconfigure((0, 1), weight=1)

        self._build_sliders()
        self.select_group(self.active_group_id)

    def _build_sliders(self):
        row = 0
        for group in FEATURE_GROUPS:
            for feature in group["features"]:
                container = ctk.CTkFrame(self.controls_frame, fg_color="gray17", corner_radius=8)
                container.grid(row=row // 2, column=row % 2, sticky="ew", padx=8, pady=6)
                container.grid_columnconfigure(1, weight=1)
                container.group_id = group["id"]

                label = ctk.CTkLabel(container, text=feature["name"], width=110, anchor="w")
                label.grid(row=0, column=0, padx=(10, 6), pady=8, sticky="w")

                from_, to = (-1.0, 1.0) if feature["id"] in SIGNED_SLIDERS else (0.0, 1.0)
                slider = ctk.CTkSlider(
                    container,
                    from_=from_,
                    to=to,
                    command=lambda value, key=feature["id"]: self.on_slider_change(key, value),
                )
                slider.set(self.global_params.get(feature["id"], feature["default"]))
                slider.grid(row=0, column=1, padx=8, pady=8, sticky="ew")
                slider.bind("<ButtonRelease-1>", self.on_slider_release)

                value_label = ctk.CTkLabel(container, text=self._format_value(feature["id"]), width=46)
                value_label.grid(row=0, column=2, padx=(4, 10), pady=8)

                reset_btn = ctk.CTkButton(
                    container,
                    text="↺",
                    width=30,
                    command=lambda key=feature["id"], default=feature["default"]: self.reset_param(key, default),
                )
                reset_btn.grid(row=0, column=3, padx=(0, 10), pady=8)

                self.slider_widgets[feature["id"]] = (slider, container)
                self.value_labels[feature["id"]] = value_label
                row += 1

    def select_group(self, group_id):
        self.active_group_id = group_id
        for gid, btn in self.group_buttons.items():
            btn.configure(fg_color=("#1f6aa5" if gid == group_id else "#3a7ebf"))
        for _, container in self.slider_widgets.values():
            if container.group_id == group_id:
                container.grid()
            else:
                container.grid_remove()
        self.after_idle(self._reset_controls_scroll)

    def _reset_controls_scroll(self):
        self.controls_frame._parent_canvas.yview_moveto(0)

    def import_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_path:
            params = self._params_for_processing()

            def load_and_process():
                if not self.processor.load_image(file_path):
                    raise ValueError("无法读取所选图片。")
                return self.processor.process(params)

            def show_imported_images(processed_image):
                self.processed_image = processed_image
                self._show_input_image()
                self._show_image(self.lbl_img_right, processed_image)
                self.model_status_label.configure(text=self._model_status_text())

            self._submit_background(load_and_process, show_imported_images, "正在分析并处理图片…")

    def export_image(self):
        if self.processor.original_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
            if file_path:
                params = self._params_for_processing()

                def process_and_save():
                    result = self.processor.process(params)
                    if not cv2.imwrite(file_path, result):
                        raise IOError("图片导出失败。")
                    return result

                def export_finished(result):
                    self.processed_image = result
                    messagebox.showinfo("导出完成", "图片已成功导出。", parent=self)

                self._submit_background(process_and_save, export_finished, "正在导出图片…")

    def _params_for_processing(self):
        params = self.global_params.copy()
        params[AI_MODE_PARAM] = self.ai_mode.get()
        return params

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

        params = self._params_for_processing()

        def process():
            return self.processor.process(params)

        def show_result(result):
            self.processed_image = result
            self._show_image(self.lbl_img_right, result)

        text = "正在应用 AI 高质量美化…" if self.ai_mode.get() else "正在应用美化效果…"
        self._submit_background(process, show_result, text)

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
        dialog.geometry("360x150")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.protocol("WM_DELETE_WINDOW", lambda: None)

        ctk.CTkLabel(dialog, text=text, font=("Roboto", 16, "bold")).pack(pady=(28, 16))
        progress = ctk.CTkProgressBar(dialog, width=260, mode="indeterminate")
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

    def on_slider_change(self, key, value):
        self.global_params[key] = float(value)
        if key in self.value_labels:
            self.value_labels[key].configure(text=self._format_value(key))

    def on_slider_release(self, _event):
        self._process_current_params()

    def on_landmark_toggle(self):
        self._show_input_image()

    def on_ai_mode_toggle(self):
        self.global_params[AI_MODE_PARAM] = self.ai_mode.get()
        self.model_status_label.configure(text=self._model_status_text())
        self._process_current_params()

    def on_preset_selected(self, name):
        preset = get_preset(name)
        self.global_params[STYLE_PRESET_PARAM] = name
        for key, value in preset.get("params", {}).items():
            self.global_params[key] = value
        self._refresh_sliders()
        self._process_current_params()

    def reset_param(self, key, default):
        self.global_params[key] = default
        self._refresh_sliders()
        self._process_current_params()

    def reset_all(self):
        self.global_params = default_params()
        self.ai_mode.set(False)
        self.preset_menu.set(self.global_params[STYLE_PRESET_PARAM])
        self._refresh_sliders()
        self._process_current_params()

    def _refresh_sliders(self):
        for key, (slider, _) in self.slider_widgets.items():
            slider.set(self.global_params.get(key, 0.0))
            self.value_labels[key].configure(text=self._format_value(key))

    def _format_value(self, key):
        value = self.global_params.get(key, 0.0)
        if key in SIGNED_SLIDERS:
            return f"{value:+.2f}"
        return f"{value:.2f}"

    def _model_status_text(self):
        status = self.processor.ai_models.status()
        available = []
        if status["inpainting_available"]:
            available.append("Inpaint")
        if status["gan_available"]:
            available.append("GAN")
        if status["diffusion_available"]:
            available.append("Diffusion")
        if not available:
            return f"AI模型: 未安装 · {status['device'].upper()} · 使用CV fallback"
        return f"AI模型: {', '.join(available)} · {status['device'].upper()}"

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
