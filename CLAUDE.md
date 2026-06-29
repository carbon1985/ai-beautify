# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Beautify is a small desktop image beautification app. The UI is built with CustomTkinter, image I/O and processing use OpenCV/Pillow/NumPy, and face landmarks come from MediaPipe Face Mesh. There is no README, test suite, or packaging metadata beyond `environment.yml` at the time this file was created.

## Development Commands

Set up the conda environment from the pinned dependency file:

```bash
conda env create -f environment.yml
conda activate ai-beautify
```

Update an existing environment after dependency changes:

```bash
conda env update -f environment.yml --prune
```

Run the desktop app:

```bash
python main.py
```

Quick syntax check for all Python files:

```bash
python -m py_compile main.py image_processor.py
```

There are currently no configured lint, format, or test commands, and no single-test command to document. If tests are added later, update this file with the canonical full-suite and single-test invocations.

## Architecture

- `main.py` owns the desktop application lifecycle. `AIBeautifyApp` subclasses `ctk.CTk`, builds the import/export controls, the two image preview panes, the feature selector, and the per-feature slider panel.
- `main.py` keeps UI state in `global_params`, keyed by processor parameter names: `smooth`, `whiten`, `enlarge_eyes`, `slim_face`, `slim_nose`, and `lip_shape`. The feature list in `AIBeautifyApp.features` must stay in sync with the keys consumed by `ImageProcessor.process()`.
- Image processing is run through a single-worker `ThreadPoolExecutor` in `AIBeautifyApp._submit_background()` so the Tk event loop stays responsive. Worker callbacks are polled with `after()`, and UI updates happen back on the main thread in `_poll_future()`.
- `image_processor.py` contains all computer-vision logic in `ImageProcessor`. `load_image()` reads the BGR OpenCV image, initializes remap grids, runs MediaPipe Face Mesh once, and precomputes masks/landmark data for later slider changes.
- `ImageProcessor.process(params)` is the main processing pipeline. It caches the expensive skin filtering result for the current `smooth`/`whiten` values, then applies geometric warps for facial shape parameters on top of that cached image.
- Skin operations are mask-based. `_generate_masks()` builds an expanded face/neck skin mask, excludes eyes/lips/eyebrows, and blends with a YCrCb color mask; `_apply_filtering()` uses that mask for smoothing and whitening.
- Shape operations are remap-based. `_apply_warping()` starts from `base_map_x`/`base_map_y`, applies local bloat/translate deformations around MediaPipe landmark points, then calls `cv2.remap()`.
- The left preview can show either the original image or a landmark/tessellation overlay from `get_landmark_preview()`. That preview is cached until a new image is loaded.

## Implementation Notes

- OpenCV images are BGR throughout `image_processor.py` and when saving with `cv2.imwrite()`. Convert to RGB only for Pillow/CustomTkinter display in `AIBeautifyApp._cv2_to_ctk_image()`.
- If no face is detected, filtering and warping safely fall back to copies of the original image because `landmarks` and `skin_mask` are `None`.
- Adding a new beautification control usually requires three coordinated changes: add a default key to `global_params`, add a feature entry in `AIBeautifyApp.features`, and consume the key in `ImageProcessor.process()` via filtering or warping.
- Keep long-running image operations off the Tk main thread. Follow the existing `_submit_background(task, on_success, loading_text)` pattern for import, export, or processing work.
