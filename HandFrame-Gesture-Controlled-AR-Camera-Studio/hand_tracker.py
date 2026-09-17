"""
hand_tracker.py
---------------
Real-time 21-keypoint 3D hand tracking, spatial gesture recognition,
and advanced holographic hand skeleton rendering.

Features:
  - Dual Backend: MediaPipe Tasks API (modern 0.10+) with auto-download of
    hand_landmarker.task, with automatic fallback to classic mp.solutions.
  - Landmark Smoothing: One-Euro temporal filtering per keypoint to eliminate jitter.
  - Spatial Floating Frame Gesture: Deterministic left-to-right hand anchoring
    that prevents corner flipping, jitter, and inverted rotations.
  - Normalized Pinch Detection: Invariant to camera distance.
  - 7 Distinct Hand Skeleton Styles:
      1. 'neon': Multi-layered blooming glow lines with glowing fingertip halos.
      2. 'hologram': Jarvis/Iron Man HUD with palm arc-reactor reticle and knuckle rings.
      3. 'fire': Solar flame gradient (Crimson wrist -> Molten orange -> Yellow tips).
      4. 'matrix': Segmented phosphor green dashed bones with digital square nodes.
      5. 'minimal': Ultra-clean fine silver wireframe with precision crosshair '+' nodes.
      6. 'rainbow': 5-color spectrum mapping per finger branch.
      7. 'stealth': Subtle translucent dotted wireframe for minimal visual obstruction.
  - Glowing Hand Borders: Soft perimeter convex hull shield around each hand.
  - Interactive Pinch Proximity: Glowing connection ring between thumb and index.
"""

import math
import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp

# ==============================================================================
# 🖐️ MEDIAPIPE 21-LANDMARK INDICES
# ==============================================================================

WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

# Structural finger chains connecting wrist to fingertips
FINGER_CHAINS = {
    "thumb": [(0, 1), (1, 2), (2, 3), (3, 4)],
    "index": [(0, 5), (5, 6), (6, 7), (7, 8)],
    "middle": [(0, 9), (9, 10), (10, 11), (11, 12)],
    "ring": [(0, 13), (13, 14), (14, 15), (15, 16)],
    "pinky": [(0, 17), (17, 18), (18, 19), (19, 20)],
}
PALM_BASE_CHAIN = [(5, 9), (9, 13), (13, 17)]


# ==============================================================================
# 📦 PER-HAND STATE CONTAINER
# ==============================================================================

class HandState:
    """Stores smoothed 2D/3D pixel landmarks and presence for a single hand."""

    def __init__(self, label):
        self.label = label          # "Left" or "Right" slot
        self.landmarks_px = None    # (21, 2) numpy array of smoothed pixel coords
        self.present = False


# ==============================================================================
# 🤖 MODEL WEIGHTS DOWNLOADER
# ==============================================================================

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")


def _ensure_task_model():
    """Ensures Google MediaPipe Tasks float16 model file exists locally."""
    if not os.path.exists(MODEL_PATH):
        print("[hand_tracker] Downloading hand_landmarker.task model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[hand_tracker] Download complete.")
    return MODEL_PATH


# ==============================================================================
# 👁️ HAND TRACKER CLASS
# ==============================================================================

class HandTracker:
    """
    Orchestrates MediaPipe landmark inference, temporal smoothing,
    gesture extraction, and customizable holographic skeleton rendering.
    """

    SKELETON_STYLES = ["auto", "neon", "hologram", "fire", "matrix", "minimal", "rainbow", "stealth"]

    def __init__(self, max_hands=2, det_conf=0.35, track_conf=0.35,
                 smooth_min_cutoff=1.2, smooth_beta=0.4):
        self.use_tasks_api = not hasattr(mp, "solutions")
        if self.use_tasks_api:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            model_file = _ensure_task_model()
            base_options = python.BaseOptions(model_asset_path=model_file)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=max_hands,
                min_hand_detection_confidence=det_conf,
                min_tracking_confidence=track_conf,
            )
            self._detector = vision.HandLandmarker.create_from_options(options)
            self._hands = None
            self.mp_drawing = None
            self.mp_styles = None
        else:
            self._mp_hands = mp.solutions.hands
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_hands,
                min_detection_confidence=det_conf,
                min_tracking_confidence=track_conf,
            )
            self._detector = None
            self.mp_drawing = getattr(mp.solutions, "drawing_utils", None)
            self.mp_styles = getattr(mp.solutions, "drawing_styles", None)

        from smoothing import LandmarkSmoother
        self._smoothers = {
            "Left": LandmarkSmoother(21, min_cutoff=smooth_min_cutoff, beta=smooth_beta),
            "Right": LandmarkSmoother(21, min_cutoff=smooth_min_cutoff, beta=smooth_beta),
        }

        self.hands = {"Left": HandState("Left"), "Right": HandState("Right")}
        self.detected_list = []

    def close(self):
        """Releases underlying MediaPipe memory handles."""
        if self._hands is not None:
            self._hands.close()
        if self._detector is not None:
            self._detector.close()

    # --------------------------------------------------------------------------
    # 🔄 FRAME PROCESSING & SMOOTHING
    # --------------------------------------------------------------------------

    def process(self, frame_bgr, t=None):
        """
        Executes hand detection on a BGR video frame.
        Maps detected landmarks to screen pixels, smooths coordinates,
        and assigns Left/Right slots deterministically by screen X-position.
        """
        h, w = frame_bgr.shape[:2]

        for hs in self.hands.values():
            hs.present = False
        self.detected_list = []

        raw_hands = []

        if self.use_tasks_api:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._detector.detect(mp_image)

            if result.hand_landmarks:
                for idx, lm_set in enumerate(result.hand_landmarks):
                    pts = np.array([[p.x * w, p.y * h] for p in lm_set], dtype=np.float64)
                    raw_hands.append(pts)
        else:
            frame_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
            frame_rgb.flags.writeable = False
            result = self._hands.process(frame_rgb)

            if result.multi_hand_landmarks:
                for lm_set in result.multi_hand_landmarks:
                    pts = np.array([[p.x * w, p.y * h] for p in lm_set.landmark], dtype=np.float64)
                    raw_hands.append(pts)

        if len(raw_hands) == 1:
            pts = raw_hands[0]
            palm_x = pts[WRIST][0]
            slot = "Left" if palm_x < w / 2 else "Right"
            smoothed = self._smoothers[slot].smooth(pts, t=t)
            hs = self.hands[slot]
            hs.landmarks_px = smoothed
            hs.present = True
            self.detected_list.append(hs)
        elif len(raw_hands) >= 2:
            raw_hands.sort(key=lambda p: p[WRIST][0])
            for slot, pts in zip(["Left", "Right"], raw_hands[:2]):
                smoothed = self._smoothers[slot].smooth(pts, t=t)
                hs = self.hands[slot]
                hs.landmarks_px = smoothed
                hs.present = True
                self.detected_list.append(hs)

        return result

    # ==========================================================================
    # 🌟 ADVANCED HAND SKELETON & BORDER RENDERER
    # ==========================================================================

    def draw_landmarks(self, frame, style="neon", theme_color=(0, 230, 255), show_border=True):
        """
        Renders the active hand skeleton style and hand perimeter border.
        
        Args:
            frame: Live BGR camera frame to draw onto.
            style: One of ['neon', 'hologram', 'fire', 'matrix', 'minimal', 'rainbow', 'stealth'].
            theme_color: (B, G, R) color tuple matching active visual style.
            show_border: Boolean, whether to render glowing hand perimeter border.
        """
        style = str(style).lower()

        for hs in self.detected_list:
            if not hs.present or hs.landmarks_px is None:
                continue

            lm = hs.landmarks_px.astype(np.int32)

            # 1. Draw glowing hand boundary border / perimeter shield
            if show_border and style != "stealth":
                self._draw_hand_border(frame, lm, theme_color)

            # 2. Draw interactive pinch proximity halo
            self._draw_pinch_proximity(frame, lm, theme_color)

            # 3. Draw specific hand skeleton design
            if style == "hologram":
                self._draw_hologram_skeleton(frame, lm, theme_color)
            elif style == "fire":
                self._draw_fire_skeleton(frame, lm)
            elif style == "matrix":
                self._draw_matrix_skeleton(frame, lm)
            elif style == "minimal":
                self._draw_minimal_skeleton(frame, lm, theme_color)
            elif style == "rainbow":
                self._draw_rainbow_skeleton(frame, lm)
            elif style == "stealth":
                self._draw_stealth_skeleton(frame, lm, theme_color)
            else:  # Default to "neon"
                self._draw_neon_skeleton(frame, lm, theme_color)

    # --------------------------------------------------------------------------
    # 🛡️ HAND PERIMETER BORDER / SHIELD
    # --------------------------------------------------------------------------

    @staticmethod
    def _draw_hand_border(frame, lm, color):
        """
        Computes 2D convex hull of the hand landmarks and renders a glowing
        soft perimeter shield with corner accent ticks around the hand.
        """
        hull = cv2.convexHull(lm)
        if len(hull) < 3:
            return

        # Faint glowing outline (wide thickness)
        dim_color = (int(color[0] * 0.25), int(color[1] * 0.25), int(color[2] * 0.25))
        cv2.polylines(frame, [hull], isClosed=True, color=dim_color, thickness=4, lineType=cv2.LINE_AA)

        # Fine crisp boundary wire
        cv2.polylines(frame, [hull], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)

        # Corner accent markers on hull vertices
        for pt in hull[::2]:
            cv2.circle(frame, tuple(pt[0]), 2, (255, 255, 255), -1, cv2.LINE_AA)

    # --------------------------------------------------------------------------
    # 🤏 INTERACTIVE PINCH PROXIMITY RING
    # --------------------------------------------------------------------------

    @staticmethod
    def _draw_pinch_proximity(frame, lm, color):
        """
        Measures distance between thumb tip and index tip.
        As fingers approach each other, renders a bright proximity halo and bridge.
        """
        thumb_tip = lm[THUMB_TIP]
        index_tip = lm[INDEX_TIP]
        dist = np.linalg.norm(thumb_tip - index_tip)

        wrist = lm[WRIST]
        index_mcp = lm[INDEX_MCP]
        hand_scale = np.linalg.norm(index_mcp - wrist) + 1e-6
        ratio = dist / hand_scale

        # Proximity threshold: glow triggers when fingers are within 0.7 hand-scale
        if ratio < 0.7:
            intensity = max(0.0, min(1.0, 1.0 - (ratio / 0.7)))
            mid_pt = ((thumb_tip + index_tip) // 2).astype(np.int32)
            glow_radius = int(8 + 14 * intensity)
            glow_color = (
                min(255, int(color[0] + 120 * intensity)),
                min(255, int(color[1] + 120 * intensity)),
                min(255, int(color[2] + 120 * intensity)),
            )

            # Energy bridge connecting thumb and index
            cv2.line(frame, tuple(thumb_tip), tuple(index_tip), glow_color, 2 if ratio > 0.3 else 3, cv2.LINE_AA)

            # Pulsing proximity circle at center
            cv2.circle(frame, tuple(mid_pt), glow_radius, glow_color, 1, cv2.LINE_AA)
            if ratio < 0.35:
                # Solid lock dot when pinched
                cv2.circle(frame, tuple(mid_pt), 4, (255, 255, 255), -1, cv2.LINE_AA)

    # --------------------------------------------------------------------------
    # 1. CYBER NEON SKELETON
    # --------------------------------------------------------------------------

    @staticmethod
    def _draw_neon_skeleton(frame, lm, color):
        """Dual-layer blooming neon bones + emerald fingertip halos."""
        dim_color = (int(color[0] * 0.4), int(color[1] * 0.4), int(color[2] * 0.4))

        # 1. Wide outer glow bone lines
        all_conns = [c for chain in FINGER_CHAINS.values() for c in chain] + PALM_BASE_CHAIN
        for p1, p2 in all_conns:
            cv2.line(frame, tuple(lm[p1]), tuple(lm[p2]), dim_color, 5, cv2.LINE_AA)

        # 2. Bright core lines
        for p1, p2 in all_conns:
            cv2.line(frame, tuple(lm[p1]), tuple(lm[p2]), color, 2, cv2.LINE_AA)

        # 3. Joint nodes
        for i, pt in enumerate(lm):
            if i in (4, 8, 12, 16, 20):  # Fingertips
                cv2.circle(frame, tuple(pt), 7, (0, 255, 120), -1, cv2.LINE_AA)
                cv2.circle(frame, tuple(pt), 3, (255, 255, 255), -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, tuple(pt), 3, (255, 220, 0), -1, cv2.LINE_AA)

    # --------------------------------------------------------------------------
    # 2. HOLOGRAM HUD (JARVIS / IRON MAN)
    # --------------------------------------------------------------------------

    @staticmethod
    def _draw_hologram_skeleton(frame, lm, color):
        """Palm arc-reactor concentric telemetry reticle, knuckle gear rings, technical wireframe."""
        holo_color = (0, 210, 255)  # Holographic Amber-Cyan

        all_conns = [c for chain in FINGER_CHAINS.values() for c in chain] + PALM_BASE_CHAIN
        for p1, p2 in all_conns:
            cv2.line(frame, tuple(lm[p1]), tuple(lm[p2]), holo_color, 1, cv2.LINE_AA)

        # Palm Center Arc-Reactor Reticle
        palm_center = ((lm[WRIST] + lm[MIDDLE_MCP]) // 2).astype(np.int32)
        pc_tuple = tuple(palm_center)
        cv2.circle(frame, pc_tuple, 14, holo_color, 1, cv2.LINE_AA)
        cv2.circle(frame, pc_tuple, 7, holo_color, 1, cv2.LINE_AA)
        cv2.circle(frame, pc_tuple, 2, (255, 255, 255), -1, cv2.LINE_AA)

        # Reticle radial tick marks (+)
        cv2.line(frame, (palm_center[0] - 18, palm_center[1]), (palm_center[0] - 14, palm_center[1]), holo_color, 1)
        cv2.line(frame, (palm_center[0] + 14, palm_center[1]), (palm_center[0] + 18, palm_center[1]), holo_color, 1)
        cv2.line(frame, (palm_center[0], palm_center[1] - 18), (palm_center[0], palm_center[1] - 14), holo_color, 1)
        cv2.line(frame, (palm_center[0], palm_center[1] + 14), (palm_center[0], palm_center[1] + 18), holo_color, 1)

        # Knuckle telemetry rings
        for kn in (INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP):
            cv2.circle(frame, tuple(lm[kn]), 5, holo_color, 1, cv2.LINE_AA)

        # Fingertip targeting brackets
        for tip in (4, 8, 12, 16, 20):
            pt = tuple(lm[tip])
            cv2.circle(frame, pt, 6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(frame, pt, 2, holo_color, -1, cv2.LINE_AA)

    # --------------------------------------------------------------------------
    # 3. SOLAR FLAME SKELETON
    # --------------------------------------------------------------------------

    @staticmethod
    def _draw_fire_skeleton(frame, lm):
        """Thermal fire gradient: Crimson wrist -> Molten orange -> Yellow tips with glowing embers."""
        # Color stages along each finger branch (BGR)
        palette = [
            (20, 20, 230),   # Stage 0: Crimson Red
            (0, 100, 255),   # Stage 1: Molten Orange
            (0, 180, 255),   # Stage 2: Fire Amber
            (40, 240, 255),  # Stage 3: Bright Molten Yellow
        ]

        for chain in FINGER_CHAINS.values():
            for idx, (p1, p2) in enumerate(chain):
                c = palette[min(idx, len(palette) - 1)]
                cv2.line(frame, tuple(lm[p1]), tuple(lm[p2]), c, 3, cv2.LINE_AA)

        for p1, p2 in PALM_BASE_CHAIN:
            cv2.line(frame, tuple(lm[p1]), tuple(lm[p2]), (20, 60, 220), 2, cv2.LINE_AA)

        # Ember dots on joints
        for i, pt in enumerate(lm):
            if i in (4, 8, 12, 16, 20):
                cv2.circle(frame, tuple(pt), 7, (0, 200, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, tuple(pt), 3, (255, 255, 255), -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, tuple(pt), 3, (0, 100, 255), -1, cv2.LINE_AA)

    # --------------------------------------------------------------------------
    # 4. MATRIX STREAM SKELETON
    # --------------------------------------------------------------------------

    @staticmethod
    def _draw_matrix_skeleton(frame, lm):
        """Dashed terminal code green segments with digital square pixel nodes."""
        green_phosp = (0, 255, 80)
        green_dark = (0, 140, 40)

        all_conns = [c for chain in FINGER_CHAINS.values() for c in chain] + PALM_BASE_CHAIN
        for p1, p2 in all_conns:
            pt1 = lm[p1].astype(np.float32)
            pt2 = lm[p2].astype(np.float32)
            # Draw segmented dashed bone link
            mid1 = (pt1 * 0.7 + pt2 * 0.3).astype(np.int32)
            mid2 = (pt1 * 0.3 + pt2 * 0.7).astype(np.int32)
            cv2.line(frame, tuple(pt1.astype(np.int32)), tuple(mid1), green_phosp, 2, cv2.LINE_AA)
            cv2.line(frame, tuple(mid2), tuple(pt2.astype(np.int32)), green_phosp, 2, cv2.LINE_AA)

        # Square digital pixel joint nodes
        for i, pt in enumerate(lm):
            if i in (4, 8, 12, 16, 20):
                # Larger square at tips
                cv2.rectangle(frame, (pt[0] - 4, pt[1] - 4), (pt[0] + 4, pt[1] + 4), green_phosp, -1)
                cv2.rectangle(frame, (pt[0] - 2, pt[1] - 2), (pt[0] + 2, pt[1] + 2), (255, 255, 255), -1)
            else:
                cv2.rectangle(frame, (pt[0] - 2, pt[1] - 2), (pt[0] + 2, pt[1] + 2), green_dark, -1)

    # --------------------------------------------------------------------------
    # 5. MINIMAL TECH SKELETON
    # --------------------------------------------------------------------------

    @staticmethod
    def _draw_minimal_skeleton(frame, lm, color):
        """Ultra-clean thin silver wireframe with precision crosshair '+' nodes."""
        silver = (220, 220, 220)

        all_conns = [c for chain in FINGER_CHAINS.values() for c in chain] + PALM_BASE_CHAIN
        for p1, p2 in all_conns:
            cv2.line(frame, tuple(lm[p1]), tuple(lm[p2]), silver, 1, cv2.LINE_AA)

        # Crosshair '+' markers on joints
        for pt in lm:
            cv2.line(frame, (pt[0] - 3, pt[1]), (pt[0] + 3, pt[1]), silver, 1, cv2.LINE_AA)
            cv2.line(frame, (pt[0], pt[1] - 3), (pt[0], pt[1] + 3), silver, 1, cv2.LINE_AA)

        # Fine circle at fingertips
        for tip in (4, 8, 12, 16, 20):
            cv2.circle(frame, tuple(lm[tip]), 4, color, 1, cv2.LINE_AA)

    # --------------------------------------------------------------------------
    # 6. RAINBOW SPECTRUM SKELETON
    # --------------------------------------------------------------------------

    @staticmethod
    def _draw_rainbow_skeleton(frame, lm):
        """5-color distinct spectrum mapping per finger branch."""
        colors = {
            "thumb": (30, 40, 255),    # Red
            "index": (20, 160, 255),   # Orange
            "middle": (50, 240, 80),   # Green
            "ring": (255, 220, 30),    # Cyan
            "pinky": (240, 60, 200),   # Violet
        }

        for fname, chain in FINGER_CHAINS.items():
            c = colors[fname]
            for p1, p2 in chain:
                cv2.line(frame, tuple(lm[p1]), tuple(lm[p2]), c, 3, cv2.LINE_AA)

        for p1, p2 in PALM_BASE_CHAIN:
            cv2.line(frame, tuple(lm[p1]), tuple(lm[p2]), (240, 240, 240), 2, cv2.LINE_AA)

        for i, pt in enumerate(lm):
            cv2.circle(frame, tuple(pt), 4 if i in (4, 8, 12, 16, 20) else 2, (255, 255, 255), -1, cv2.LINE_AA)

    # --------------------------------------------------------------------------
    # 7. STEALTH WIREFRAME
    # --------------------------------------------------------------------------

    @staticmethod
    def _draw_stealth_skeleton(frame, lm, color):
        """Subtle translucent dotted wireframe for minimal visual obstruction."""
        faint_color = (int(color[0] * 0.4), int(color[1] * 0.4), int(color[2] * 0.4))
        all_conns = [c for chain in FINGER_CHAINS.values() for c in chain]
        for p1, p2 in all_conns:
            cv2.line(frame, tuple(lm[p1]), tuple(lm[p2]), faint_color, 1, cv2.LINE_AA)

        for tip in (4, 8, 12, 16, 20):
            cv2.circle(frame, tuple(lm[tip]), 3, faint_color, -1, cv2.LINE_AA)

    # ==========================================================================
    # 📐 SPATIAL AR FRAME GESTURE RECOGNITION
    # ==========================================================================

    def get_frame_gesture(self):
        """
        Extracts 4-point quadrilateral anchored deterministically between both hands:
          - Left Hand provides Top-Left and Bottom-Left corners.
          - Right Hand provides Top-Right and Bottom-Right corners.
        This spatial sorting guarantees zero corner-flipping or inverted rotations.
        """
        present = [h for h in self.detected_list if h.present and h.landmarks_px is not None]

        if len(present) >= 2:
            h_left = min(present[:2], key=lambda h: h.landmarks_px[WRIST][0])
            h_right = max(present[:2], key=lambda h: h.landmarks_px[WRIST][0])

            l_wrist = h_left.landmarks_px[WRIST]
            r_wrist = h_right.landmarks_px[WRIST]
            if np.linalg.norm(l_wrist - r_wrist) < 80:
                return None

            l_pts = [h_left.landmarks_px[INDEX_TIP], h_left.landmarks_px[THUMB_TIP]]
            r_pts = [h_right.landmarks_px[INDEX_TIP], h_right.landmarks_px[THUMB_TIP]]

            tl = min(l_pts, key=lambda p: p[1])
            bl = max(l_pts, key=lambda p: p[1])

            tr = min(r_pts, key=lambda p: p[1])
            br = max(r_pts, key=lambda p: p[1])

            return np.array([tl, tr, br, bl], dtype=np.float64)

        return None

    # ==========================================================================
    # 🤏 PINCH GESTURE RECOGNITION
    # ==========================================================================

    def get_pinch(self, hand_label=None, thresh_ratio=0.45):
        """
        Detects if thumb tip and index tip of any visible hand are touching.
        Uses hand-scale normalization (wrist to index MCP distance) so detection
        is invariant to how close or far the hand is from the camera.
        """
        candidates = [h for h in self.detected_list if h.present and h.landmarks_px is not None]
        if hand_label:
            candidates = [h for h in candidates if h.label == hand_label]

        for hs in candidates:
            lm = hs.landmarks_px
            thumb_tip, index_tip = lm[THUMB_TIP], lm[INDEX_TIP]
            wrist, index_mcp = lm[WRIST], lm[INDEX_MCP]
            hand_scale = np.linalg.norm(index_mcp - wrist) + 1e-6
            dist = np.linalg.norm(thumb_tip - index_tip)
            if (dist / hand_scale) < thresh_ratio:
                return True
        return False

    def get_swipe_hand_center(self, hand_label="Left"):
        """Returns palm center (middle finger MCP) pixel coordinate for swipe tracking."""
        hs = self.hands.get(hand_label)
        if not hs or not hs.present:
            return None
        return hs.landmarks_px[MIDDLE_MCP]