"""
ui_overlay.py
-------------
Provides glassmorphism HUD telemetry, animated loading spinners,
status messaging, and interactive keybinding guides for HandFrame AI.

Features:
  - Top Glassmorphism HUD: Active filter counter [X/32], style title,
    active AR frame design, hand skeleton mode, live FPS meter, and GPU hardware telemetry.
  - Dynamic Theme Accent: HUD accent bar and tags glow in the active style's theme color.
  - Animated Loading Spinner: Time-based smooth rotating arc for async operations.
  - Bottom Gesture & Keyboard Guide: Instructions for hand pinches, [F] frame cycling,
    and [H] skeleton cycling.
"""

import time
import cv2
import numpy as np


# ==============================================================================
# ⏳ ANIMATED SPINNER & LOADING OVERLAYS
# ==============================================================================

def draw_loading_spinner(frame, center, radius=28, color=(255, 255, 255), thickness=3):
    """
    Renders an animated rotating arc spinner at the given (x, y) coordinates.
    The start angle rotates continuously based on high-precision system time.
    """
    t = time.time()
    # Rotation speed: 320 degrees per second
    start_angle = (t * 320) % 360
    sweep = 110  # Length of the visible glowing arc

    # Rotating bright arc
    cv2.ellipse(
        frame, center, (radius, radius), 0,
        start_angle, start_angle + sweep, color, thickness, cv2.LINE_AA
    )
    # Faint 360-degree background guide ring for visual polish
    faint_color = (int(color[0] * 0.3), int(color[1] * 0.3), int(color[2] * 0.3))
    cv2.circle(frame, center, radius, faint_color, 1, cv2.LINE_AA)
    return frame


def draw_quad_loading_overlay(frame, quad_pts, label="Generating AI style..."):
    """
    Dims the interior of the floating quad with a dark transparent tint
    and centers an animated spinner with status text during async AI jobs.
    """
    quad_pts = quad_pts.astype(np.int32)
    overlay = frame.copy()
    # Darken region inside the hand-frame
    cv2.fillConvexPoly(overlay, quad_pts, (15, 15, 15))
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, dst=frame)

    # Compute polygon centroid
    cx = int(np.mean(quad_pts[:, 0]))
    cy = int(np.mean(quad_pts[:, 1]))

    # Draw centered rotating spinner
    draw_loading_spinner(frame, (cx, cy), radius=32, color=(0, 230, 255), thickness=3)

    # Center-aligned text label below spinner
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.putText(
        frame, label, (cx - text_size[0] // 2, cy + 56),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
    )
    return frame


# ==============================================================================
# 📊 TOP GLASSMORPHISM HUD TELEMETRY BANNER
# ==============================================================================

def draw_hud(frame, style_idx, total_styles, style_name, frame_name,
             skeleton_name, status_text, fps, gpu_info="NVIDIA RTX 2050",
             theme_color=(0, 220, 255)):
    """
    Draws the top glassmorphism HUD bar across the screen width.
    
    Includes:
      - Colored theme accent bar on the left.
      - Filter index counter: e.g. [12/32] and style title.
      - Active AR frame style indicator (e.g. Frame: CYBER).
      - Active Hand Skeleton mode indicator (e.g. Hands: HOLOGRAM).
      - Status text and gesture tips.
      - GPU acceleration badge and real-time FPS readout.
    """
    h, w = frame.shape[:2]
    banner_h = 68

    # Semi-transparent dark background bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (12, 14, 18), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, dst=frame)

    # Vertical glowing theme color accent line on the far left
    cv2.rectangle(frame, (0, 0), (6, banner_h), theme_color, -1)

    # ---------------- Left Column: Style & Frame -------------------------------
    # Filter Index and Name (e.g., "Filter [7/32]: Anime")
    filter_label = f"Filter [{style_idx + 1}/{total_styles}]: {style_name}"
    cv2.putText(frame, filter_label, (16, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

    # Active AR Frame & Hand Skeleton Badges
    badges_str = f"Frame: {frame_name.upper()}  |  Hands: {skeleton_name.upper()}"
    cv2.putText(frame, badges_str, (w - 740, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, theme_color, 2, cv2.LINE_AA)

    # Lower line: Dynamic status message / gesture prompt
    cv2.putText(frame, status_text, (16, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 220, 255), 1, cv2.LINE_AA)

    # ---------------- Right Column: Hardware Telemetry ------------------------
    # GPU Acceleration Tag
    cv2.putText(frame, f"[{gpu_info}]", (w - 380, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 140), 2, cv2.LINE_AA)

    # Live FPS Counter (colored green for smooth performance)
    fps_color = (160, 255, 160) if fps >= 25 else (80, 140, 255)
    cv2.putText(frame, f"{fps:.1f} FPS", (w - 95, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, fps_color, 2, cv2.LINE_AA)

    return frame


# ==============================================================================
# 🎮 BOTTOM INTERACTIVE SHORTCUTS & GESTURE GUIDE
# ==============================================================================

def draw_instructions(frame):
    """
    Renders keyboard hotkeys and hand gesture instructions at the bottom of the screen.
    """
    h, w = frame.shape[:2]
    lines = [
        "Raise both hands to anchor AI frame  |  Quick pinch = Next style  |  Hold pinch = AI render",
        "[S] Save Photo  |  [F] Frame Style  |  [H] Hand Skeleton  |  [ ] Filter  |  [C] Camera  |  [Q] Quit",
    ]
    y = h - 46
    for line in lines:
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    return frame