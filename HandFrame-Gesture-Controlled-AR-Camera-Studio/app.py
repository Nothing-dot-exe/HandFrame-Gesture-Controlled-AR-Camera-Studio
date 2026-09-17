"""
app.py
------
Main application loop and event coordinator for HandFrame AI Studio.

Integrates:
  - Video Capture: Auto-detects 1080p webcams, secondary cameras, or phone streams.
  - Hardware Acceleration: Enables NVIDIA GPU OpenCL 3.0 CUDA pipeline.
  - Hand Tracking: Google MediaPipe Tasks HandLandmarker with 7 custom skeleton styles.
  - Spatial AR Framing: Jitter-free perspective warping pinned between fingertips.
  - 32 Artistic Filter Presets: Real-time OpenCV offline filters and async cloud AI.
  - 5 Unique AR Frame Styles: 'cyber', 'neon', 'classic', 'film', 'minimal' (toggle with 'F').
  - 7 Hand Skeleton Styles: 'neon', 'hologram', 'fire', 'matrix', 'minimal', 'rainbow', 'stealth' (toggle with 'H').
  - Glowing Hand Borders: Soft perimeter convex hull shield and interactive pinch halos.
  - Snapshot Capture: Full-resolution photo saving to saved_scans/ (press 'S').
"""

import os
import datetime
import argparse
import time
import subprocess
import sys

# ==============================================================================
# 📦 AUTO-DEPENDENCY CHECK
# ==============================================================================

def _ensure_packages():
    """
    Verifies that required core packages are installed in the Python environment.
    Automatically installs any missing packages silently.
    """
    required_core = ["opencv-python", "mediapipe", "numpy", "Pillow", "fal-client"]

    import importlib

    def _missing(pkgs):
        name_map = {
            "opencv-python": "cv2",
            "Pillow": "PIL",
            "fal-client": "fal_client",
        }
        missing = []
        for pkg in pkgs:
            mod_name = name_map.get(pkg, pkg)
            try:
                importlib.import_module(mod_name)
            except ImportError:
                missing.append(pkg)
        return missing

    missing_core = _missing(required_core)
    if missing_core:
        print(f"[setup] Installing missing packages: {missing_core} ...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", *missing_core],
            check=True,
        )


_ensure_packages()

import cv2
import numpy as np

from hand_tracker import HandTracker
from perspective import FloatingPlane, QuadSmoother, quad_to_capture_rect
from inference import AsyncFluxEngine
from styles import get_style, style_count
import ui_overlay


# ==============================================================================
# ⚙️ COMMAND-LINE ARGUMENT PARSER
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="HandFrame AI — Gesture-Controlled AR Camera Studio")
    p.add_argument("--camera", default="0",
                   help="Webcam index (0, 1) or Phone Wi-Fi stream URL (e.g. http://192.168.1.5:8080/video)")
    p.add_argument("--width", type=int, default=1920, help="Webcam capture width (default 1920 for 1080p Full HD)")
    p.add_argument("--height", type=int, default=1080, help="Webcam capture height (default 1080)")
    p.add_argument("--fal_key", type=str, default=None,
                   help="Optional fal.ai API key for FLUX.2 cloud inference. Omit to use offline filters.")
    p.add_argument("--strength", type=float, default=0.65, help="Denoising strength for cloud AI [0.0 - 1.0]")
    p.add_argument("--steps", type=int, default=8, help="Inference steps for cloud AI")
    p.add_argument("--capture_size", type=int, default=640,
                   help="Square resolution of the region sent to cloud AI")
    p.add_argument("--live_size", type=int, default=380,
                   help="Square resolution of the region re-stylized every frame for the live AR effect")
    p.add_argument("--hold_threshold", type=float, default=0.6,
                   help="Seconds a pinch must be held to trigger full cloud AI generation")
    p.add_argument("--cycle_cooldown", type=float, default=0.35,
                   help="Minimum seconds between quick-pinch style cycles")
    p.add_argument("--mirror", action="store_true", default=True,
                   help="Mirror the webcam feed horizontally for natural selfie view")
    return p.parse_args()


# ==============================================================================
# ⏱️ ROLLING FPS METER
# ==============================================================================

class FPSMeter:
    """Calculates smoothed frame-rate using exponential moving average."""

    def __init__(self, smoothing=0.9):
        self._t_prev = time.time()
        self._fps = 0.0
        self._smoothing = smoothing

    def tick(self):
        t = time.time()
        dt = max(t - self._t_prev, 1e-6)
        inst_fps = 1.0 / dt
        self._fps = self._smoothing * self._fps + (1 - self._smoothing) * inst_fps
        self._t_prev = t
        return self._fps


# ==============================================================================
# 🎛️ APPLICATION STATE MANAGER
# ==============================================================================

class AppState:
    """
    Tracks state machine modes, active styles, gestures, and AR frame selections.

    Modes:
      - IDLE: No hand-frame gesture detected; raw camera feed shown.
      - LIVE: Hand-frame gesture active; plane is stylized every frame with fast OpenCV filter.
      - GENERATING: User held pinch >= 0.6s; waiting for cloud AI result while spinner plays.
      - AI_RESULT: High-resolution cloud AI image is pinned to the moving hand quad.
    """

    IDLE = "idle"
    LIVE = "live"
    GENERATING = "generating"
    AI_RESULT = "ai_result"

    FRAME_OPTIONS = ["auto", "cyber", "neon", "classic", "film", "minimal"]
    SKELETON_OPTIONS = ["auto", "neon", "hologram", "fire", "matrix", "minimal", "rainbow", "stealth"]

    def __init__(self):
        self.mode = self.IDLE
        self.style_idx = 0
        self.frame_opt_idx = 0       # 0="auto" (matches style preset), 1..5=manual override
        self.skeleton_opt_idx = 0    # 0="auto" (matches style preset), 1..7=manual override
        self.pending_job_id = None
        self.ai_result_image = None

        # Gesture tracking and hold-time edge detection
        self.pinch_active = False
        self.pinch_start_time = 0.0
        self.last_style_cycle_time = 0.0
        self.held_pinch_fired = False

        # Visual snapshot shutter flash counter
        self.flash_frames = 0


def _resolve_skeleton_type(style_name, opt_idx):
    """Resolves skeleton style. When in 'auto' mode, pairs with active filter."""
    selected = AppState.SKELETON_OPTIONS[opt_idx]
    if selected != "auto":
        return selected, selected

    name_low = style_name.lower()
    if "matrix" in name_low:
        return "matrix", "matrix (auto)"
    elif "inferno" in name_low or "lava" in name_low:
        return "fire", "fire (auto)"
    elif any(k in name_low for k in ["terminator", "predator", "hud", "blueprint", "tron"]):
        return "hologram", "hologram (auto)"
    elif any(k in name_low for k in ["pencil", "charcoal", "sin city", "minimal"]):
        return "minimal", "minimal (auto)"
    elif any(k in name_low for k in ["psychedelic", "pop art", "rainbow"]):
        return "rainbow", "rainbow (auto)"
    return "neon", "neon (auto)"


# ==============================================================================
# 🚀 MAIN APPLICATION LOOP
# ==============================================================================

def main():
    args = parse_args()

    # --------------------------------------------------------------------------
    # 📷 CAMERA DISCOVERY & INITIALIZATION
    # --------------------------------------------------------------------------
    cam_arg = str(args.camera).strip()
    if cam_arg in ("0", "auto"):
        test_cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if test_cap.isOpened():
            print("[camera] Camera 1 (1080p HD phone/webcam) detected! Using Camera 1.")
            cam_source = 1
            test_cap.release()
        else:
            cam_source = 0
    else:
        try:
            cam_source = int(cam_arg)
        except ValueError:
            cam_source = cam_arg

    def _open_camera(src):
        """Opens video capture device with high-bitrate MJPG and Full HD config."""
        if isinstance(src, int):
            c = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            if not c.isOpened():
                c = cv2.VideoCapture(src)
            c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            c.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            c.set(cv2.CAP_PROP_FPS, 60)
            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return c
        else:
            print(f"[camera] Connecting to mobile stream URL: {src} ...")
            c = cv2.VideoCapture(src)
            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return c

    # --------------------------------------------------------------------------
    # ⚡ GPU ACCELERATION (NVIDIA OpenCL 3.0 CUDA)
    # --------------------------------------------------------------------------
    gpu_badge = "CPU"
    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
        dev = cv2.ocl.Device.getDefault()
        gpu_badge = dev.name()
        print(f"[GPU] Hardware Acceleration ACTIVE: {dev.name()} ({dev.vendorName()})")

    cap = _open_camera(cam_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {cam_source}. Please verify camera connection.")

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[camera] Active Resolution: {actual_w}x{actual_h} on camera {cam_source}")

    # Initialize subsystems
    tracker = HandTracker(max_hands=2)
    plane = FloatingPlane()
    quad_smoother = QuadSmoother(min_cutoff=1.0, beta=0.35)

    engine = AsyncFluxEngine(
        fal_key=args.fal_key,
        strength=args.strength,
        num_inference_steps=args.steps,
    )
    engine.start()

    state = AppState()
    fps_meter = FPSMeter()

    window_name = "HandFrame AI — Gesture-Controlled AR Camera Studio"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, actual_w, actual_h)

    # --------------------------------------------------------------------------
    # 🔄 MAIN INTERACTIVE FRAME LOOP
    # --------------------------------------------------------------------------
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Mirror image for intuitive selfie interaction
            if args.mirror:
                frame = cv2.flip(frame, 1)

            now = time.time()

            # Process MediaPipe hand detection and update keypoints
            tracker.process(frame, t=now)

            # Resolve active style, theme color, frame type, and skeleton style
            style = get_style(state.style_idx)
            theme_color = style.get("theme_color", (0, 220, 255))

            # Resolve AR Frame Style
            selected_frame_opt = AppState.FRAME_OPTIONS[state.frame_opt_idx]
            if selected_frame_opt == "auto":
                resolved_frame_type = style.get("frame_type", "cyber")
                frame_display_name = f"{resolved_frame_type} (auto)"
            else:
                resolved_frame_type = selected_frame_opt
                frame_display_name = resolved_frame_type

            # Resolve Hand Skeleton Style
            resolved_skel_type, skel_display_name = _resolve_skeleton_type(
                style["name"], state.skeleton_opt_idx
            )

            # Draw holographic hand skeletons and glowing perimeter borders
            tracker.draw_landmarks(frame, style=resolved_skel_type, theme_color=theme_color, show_border=True)

            # Extract 4-corner perspective quad anchored between both hands
            quad = tracker.get_frame_gesture()

            status_text = "Raise both hands to anchor the floating AI style frame"

            # ------------------------------------------------------------------
            # 🖐️ HAND-FRAME GESTURE ACTIVE
            # ------------------------------------------------------------------
            if quad is not None:
                # Apply One-Euro temporal filter to eliminate corner jitter
                smoothed_quad = quad_smoother.smooth(quad, t=now)

                if state.mode == AppState.IDLE:
                    state.mode = AppState.LIVE

                # Check thumb-index pinch gesture
                pinched_now = tracker.get_pinch()

                # Pinch Started
                if pinched_now and not state.pinch_active:
                    state.pinch_active = True
                    state.pinch_start_time = now
                    state.held_pinch_fired = False

                # Pinch Held
                elif pinched_now and state.pinch_active:
                    held_for = now - state.pinch_start_time
                    # If held >= threshold (0.6s), submit high-res crop to cloud AI
                    if held_for >= args.hold_threshold and not state.held_pinch_fired \
                            and state.mode != AppState.GENERATING:
                        capture = quad_to_capture_rect(
                            frame, smoothed_quad,
                            out_size=(args.capture_size, args.capture_size),
                        )
                        job_id = engine.submit(capture, style)
                        if job_id is not None:
                            state.pending_job_id = job_id
                            state.mode = AppState.GENERATING
                        state.held_pinch_fired = True

                # Pinch Released (Quick Pinch = Cycle Filter)
                elif not pinched_now and state.pinch_active:
                    held_for = now - state.pinch_start_time
                    can_cycle = (now - state.last_style_cycle_time) > args.cycle_cooldown
                    if not state.held_pinch_fired and held_for < args.hold_threshold and can_cycle:
                        state.style_idx = (state.style_idx + 1) % style_count()
                        state.last_style_cycle_time = now
                        if state.mode == AppState.AI_RESULT:
                            state.mode = AppState.LIVE
                    state.pinch_active = False

                # --------------------------------------------------------------
                # 📡 POLL ASYNC CLOUD AI RESULTS
                # --------------------------------------------------------------
                result = engine.poll_result()
                if result is not None and result.job_id == state.pending_job_id:
                    if result.error is None:
                        state.ai_result_image = result.image_bgr
                        state.mode = AppState.AI_RESULT
                        status_text = f"AI styled in {result.elapsed:.2f}s"
                    else:
                        state.mode = AppState.LIVE
                        status_text = f"Generation notice: {result.error}"
                    state.pending_job_id = None

                # --------------------------------------------------------------
                # 🪟 RENDER AR FLOATING PLANE CONTENT
                # --------------------------------------------------------------
                if state.mode == AppState.LIVE:
                    live_crop = quad_to_capture_rect(
                        frame, smoothed_quad, out_size=(args.live_size, args.live_size)
                    )
                    stylized = AsyncFluxEngine._run_fallback_filter(live_crop, style)
                    plane.set_image(stylized)
                    plane.render(
                        frame, smoothed_quad,
                        frame_type=resolved_frame_type,
                        frame_color=theme_color,
                    )
                    status_text = ("Quick pinch = Next filter  |  "
                                   f"Hold pinch {args.hold_threshold:.1f}s = AI render  |  [F] Frame  |  [H] Skeleton")

                elif state.mode == AppState.GENERATING:
                    live_crop = quad_to_capture_rect(
                        frame, smoothed_quad, out_size=(args.live_size, args.live_size)
                    )
                    stylized = AsyncFluxEngine._run_fallback_filter(live_crop, style)
                    plane.set_image(stylized)
                    plane.render(
                        frame, smoothed_quad, opacity=0.55,
                        frame_type=resolved_frame_type,
                        frame_color=theme_color,
                    )
                    ui_overlay.draw_quad_loading_overlay(frame, smoothed_quad,
                                                          label="Generating AI Style...")
                    status_text = "Running cloud FLUX.2 inference..."

                elif state.mode == AppState.AI_RESULT:
                    plane.set_image(state.ai_result_image)
                    plane.render(
                        frame, smoothed_quad,
                        frame_type=resolved_frame_type,
                        frame_color=theme_color,
                    )
                    status_text = "Quick pinch for next filter  |  Hold pinch to re-render"

            else:
                if state.mode != AppState.GENERATING:
                    state.mode = AppState.IDLE
                state.pinch_active = False

            # ------------------------------------------------------------------
            # 📸 SHUTTER FLASH VISUAL FEEDBACK
            # ------------------------------------------------------------------
            if state.flash_frames > 0:
                flash_overlay = np.full_like(frame, 255)
                alpha = state.flash_frames * 0.25
                cv2.addWeighted(flash_overlay, alpha, frame, 1.0 - alpha, 0, dst=frame)
                state.flash_frames -= 1

            # ------------------------------------------------------------------
            # 📊 DRAW HUD & INSTRUCTIONS
            # ------------------------------------------------------------------
            fps = fps_meter.tick()
            style_name = style["name"]
            ui_overlay.draw_hud(
                frame, state.style_idx, style_count(), style_name,
                frame_display_name, skel_display_name, status_text, fps,
                gpu_info=gpu_badge, theme_color=theme_color,
            )
            ui_overlay.draw_instructions(frame)

            # Display frame in OpenCV window
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            # ------------------------------------------------------------------
            # ⌨️ KEYBOARD SHORTCUT ROUTING
            # ------------------------------------------------------------------
            if key in (ord('q'), ord('Q'), 27):
                break

            elif key in (ord('s'), ord('S')):
                # Save high-resolution snapshot photo
                out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_scans")
                os.makedirs(out_dir, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(out_dir, f"handframe_{timestamp}.png")
                cv2.imwrite(save_path, frame)
                print(f"[SNAPSHOT] Saved image to: {save_path}")
                state.flash_frames = 3  # Trigger camera shutter flash
                status_text = f"Snapshot saved: handframe_{timestamp}.png"

            elif key in (ord('f'), ord('F')):
                # Cycle through AR floating frame designs
                state.frame_opt_idx = (state.frame_opt_idx + 1) % len(AppState.FRAME_OPTIONS)
                print(f"[frame] Active frame mode: {AppState.FRAME_OPTIONS[state.frame_opt_idx]}")

            elif key in (ord('h'), ord('H')):
                # Cycle through hand skeleton styles
                state.skeleton_opt_idx = (state.skeleton_opt_idx + 1) % len(AppState.SKELETON_OPTIONS)
                print(f"[skeleton] Active hand skeleton mode: {AppState.SKELETON_OPTIONS[state.skeleton_opt_idx]}")

            elif key == ord(']'):
                # Cycle to next artistic filter
                state.style_idx = (state.style_idx + 1) % style_count()
                if state.mode == AppState.AI_RESULT:
                    state.mode = AppState.LIVE

            elif key == ord('['):
                # Cycle to previous artistic filter
                state.style_idx = (state.style_idx - 1) % style_count()
                if state.mode == AppState.AI_RESULT:
                    state.mode = AppState.LIVE

            elif key == ord('r'):
                # Reset AR frame
                state.mode = AppState.LIVE if quad is not None else AppState.IDLE
                state.ai_result_image = None

            elif key in (ord('c'), ord('C')):
                # Toggle camera source dynamically
                if isinstance(cam_source, int):
                    new_src = 0 if cam_source == 1 else 1
                    new_cap = _open_camera(new_src)
                    if new_cap.isOpened():
                        cap.release()
                        cap = new_cap
                        cam_source = new_src
                        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        cv2.resizeWindow(window_name, actual_w, actual_h)
                        print(f"[camera] Switched to Camera {cam_source} ({actual_w}x{actual_h})")

    finally:
        engine.stop()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()