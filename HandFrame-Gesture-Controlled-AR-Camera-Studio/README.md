# 🖐️ HandFrame AI — Gesture-Controlled AR Camera Studio

An interactive, GPU-accelerated computer vision application that creates a **floating holographic AR frame pinned between your hands**, tracks your movements in 3D perspective, and transforms your webcam feed into **32 artistic AI styles** across **5 distinct AR frame designs** and **7 customizable hand skeleton styles** in real time!

Built with **Python**, **OpenCV (OpenCL GPU)**, and **Google MediaPipe Tasks**.

---

## 🎥 Live Demo Preview

<div align="center">
  <video src="live.mp4" width="100%" controls autoplay loop muted></video>
  <p>
    ▶️ <strong><a href="live.mp4">Watch live.mp4 Demonstration Video</a></strong> — Gesture-controlled 3D floating AR frame with real-time artistic AI styles & neon hand tracking.
  </p>
</div>

---

## ✨ Features

- 🦾 **NVIDIA GeForce RTX GPU Acceleration**: Hardware-accelerated image warping, matrix transforms, and filter pipelines via OpenCL 3.0 CUDA.
- 🖐️ **7 Holographic Hand Skeleton Styles & Perimeter Borders** (Cycle with **`H`**):
  - **Cyber Neon Glow**: Multi-layered blooming glow bones with glowing fingertip halos.
  - **Hologram HUD (Jarvis / Iron Man)**: Concentric palm arc-reactor telemetry reticle, knuckle gear rings, and holographic tech markings.
  - **Solar Flame**: Heat gradient from crimson wrist to fiery orange bones to bright yellow tips with glowing embers.
  - **Matrix Stream**: Dashed terminal code green segments with digital square pixel nodes.
  - **Minimal Tech**: Ultra-clean fine silver wireframe with precision crosshair `+` vertices.
  - **Rainbow Spectrum**: 5-color distinct spectrum mapping per finger branch.
  - **Stealth Wireframe**: Subtle translucent dotted wireframe for minimal visual obstruction.
  - **Glowing Hand Borders & Pinch Halos**: Soft perimeter convex hull shield around each hand and interactive proximity halos connecting thumb and index tips.
- 🪟 **Jitter-Free Spatial Floating Frame**: Deterministic left-to-right hand anchoring prevents corner flipping, jitter, and inverted rotation.
- 🖼️ **5 Unique AR Floating Frame Styles** (Cycle with **`F`**):
  - **Cyber HUD**: High-tech corner brackets `[ ]`, edge midpoint ticks, and tactical targeting crosshairs.
  - **Neon Glow**: Multi-layered blooming glow aura with corner glowing halo rings.
  - **Classic Ornate**: Double-inset border with beveled diagonal corner joints.
  - **Film Reel**: Vintage 35mm cinema filmstrip with perforated sprocket holes.
  - **Minimalist**: Clean modern segmented dashed boundary with fine crosshair vertices.
- 🎨 **32 Live Artistic Styles**: Every single style has its own dedicated, optimized computer vision algorithm:
  - **Cyber Glitch Hologram** (RGB chromatic aberration + VHS scanlines)
  - **Matrix Digital Rain** (Cascading green digital code stream)
  - **Solar Inferno / Lava** (Volcanic thermal radiation & fire embers)
  - **Sin City Noir** (Stark black & white with selective vibrant red isolation)
  - **Blueprint CAD** (Technical drafting cyan background with white architectural linework and grid)
  - **80s Synthwave Horizon** (Outrun sunset gradient with glowing perspective wireframe road)
  - **Manga Screentone** (Authentic screentone dot shading + dynamic speedlines)
  - **Night Vision Phosphor** (Military PVS-14 green phosphor glow with optical vignette and noise)
  - **Glacial Frost** (Sub-zero ice shimmer tint + diamond crystal edge highlights)
  - **Vintage 1920s Silent Film** (Sepia silver-gelatin + vignette + vertical projector scratches)
  - **Tron Wireframe** (Pitch black void with luminous electric cyan/orange edge traces)
  - **Predator Thermal HUD** (Alien multi-spectrum rainbow thermal vision + triangular lock reticle)
  - **Anime**, **Van Gogh**, **Pop Art**, **Pixel Art**, **Golden Hour Flare**, and 15 more!
- ⚡ **Ultra-Fast 30–60 FPS Performance**: Optimized local ROI bounding-box rendering delivering a 35x speed boost with zero lag.
- 📱 **Multi-Camera & Phone Support**:
  - Automatically selects 1080p Full HD camera when available.
  - Supports phone webcams via **Iriun / DroidCam USB** or **Wi-Fi IP Webcam**.
  - Toggle cameras on the fly with the **`C`** key!
- 📸 **High-Resolution Snapshot Saving with Shutter Flash**: Press **`S`** to instantly save full-res photos to `saved_scans/` with a visual shutter flash.

---

## 🚀 Quick Start (1-Click Launchers)

Double-click any launcher in the folder:

| Launcher | Description |
|---|---|
| **[`run.bat`](run.bat)** | **Interactive Selector** (choose between PC webcam, USB phone, or Wi-Fi phone). |
| **[`run_pc.bat`](run_pc.bat)** | **Direct PC Webcam** (Index 0). |
| **[`run_phone_usb.bat`](run_phone_usb.bat)** | **Direct Phone USB** (Index 1 - Iriun / DroidCam / Camo). |
| **[`run_phone_wifi.bat`](run_phone_wifi.bat)** | **Direct Phone Wi-Fi** (prompts for `http://IP:8080/video`). |

### Running from Terminal / PowerShell:
```powershell
# Default launch (auto-selects 1080p camera)
python app.py

# Force specific camera index
python app.py --camera 1

# Stream from wireless phone IP Webcam
python app.py --camera "http://192.168.1.15:8080/video"
```

---

## 🎮 Controls & Gestures

| Action | Control | What it does |
|---|---|---|
| 🤲 **Raise Both Hands** | Hand Gesture | Anchors the floating AR panel between your fingertips. |
| 🤏 **Quick Pinch** | Thumb + Index | Instantly cycles to the next artistic filter. |
| 🤏 **Hold Pinch (0.6s)** | Hold Pinch | Submits high-res capture to cloud AI (fal.ai FLUX.2). |
| **`F`** | Keyboard | **Cycle Frame Style** (`auto`, `cyber`, `neon`, `classic`, `film`, `minimal`). |
| **`H`** | Keyboard | **Cycle Hand Skeleton Style** (`auto`, `neon`, `hologram`, `fire`, `matrix`, `minimal`, `rainbow`, `stealth`). |
| **`[`** / **`]`** | Keyboard | Cycle backward / forward through all 32 styles. |
| **`S`** | Keyboard | **Save Snapshot** photo to `saved_scans/` with visual shutter flash. |
| **`C`** | Keyboard | **Switch Camera** (toggle between Camera 0 and Camera 1). |
| **`R`** | Keyboard | Reset floating frame to live camera. |
| **`Q`** / **`ESC`** | Keyboard | Quit application cleanly. |

---

## 🖐️ Hand Skeleton Style Gallery

| Skeleton Style | Visual Aesthetic | Features |
|---|---|---|
| **Neon Glow** | Cyberpunk Glow | Dual-layer blooming glow lines, glowing joint nodes, and emerald fingertip halos. |
| **Hologram HUD** | Iron Man / Jarvis | Concentric palm arc-reactor telemetry reticle with tick marks, knuckle gear rings. |
| **Solar Flame** | Thermal Ember | Multi-stage heat gradient from Crimson wrist to Molten orange to Yellow tips. |
| **Matrix Stream** | Digital Code | Dashed terminal phosphor green bone segments with digital square pixel nodes. |
| **Minimal Tech** | Precision Wireframe | Ultra-clean thin silver wireframe with precision crosshair `+` vertices. |
| **Rainbow Wave** | Spectrum Chromatic | Distinct color per finger branch (Red, Orange, Green, Cyan, Violet). |
| **Stealth** | Ghost Wireframe | Subtle translucent dotted wireframe for minimal visual obstruction. |

---

## 🖼️ AR Frame Style Gallery

| Frame Style | Aesthetic | Features |
|---|---|---|
| **Cyber HUD** | High-Tech Sci-Fi | L-shaped corner brackets `[ ]`, edge midpoint tick marks, and crosshairs. |
| **Neon Glow** | Cyberpunk Electric | 3-layer optical bloom with outer aura, bright core line, and corner halo rings. |
| **Classic Ornate** | Fine Art Gallery | Double-inset border with beveled diagonal corner connectors and corner accents. |
| **Film Reel** | Vintage Cinema | 35mm filmstrip border with perforated rectangular sprocket cutouts. |
| **Minimalist** | Modern Sleek | Segmented dashed boundaries with fine crosshair vertex marks. |

---

## 🎨 Complete Style Library (32 Distinct Styles)

1. **Cyber Glitch Hologram**: RGB split chromatic aberration, digital glitch artifacts, and horizontal VHS scanlines.
2. **Matrix Digital Rain**: Inverted high-contrast green terminal code stream cascading down the screen.
3. **Solar Inferno / Lava**: Fiery volcanic thermal radiation with intense molten orange/red embers.
4. **Ultraviolet X-Ray**: Electric bio-luminescent negative glow with cyan/violet contours.
5. **Terminator Cyber HUD**: Red tactical cyborg night-vision scanner with target crosshair reticles.
6. **Thermal / Heatmap**: Multi-color infrared heatmap with digital telemetry grid overlay.
7. **Anime**: Crisp cel-shading with bilateral smoothing, multi-level posterization, and bold dark ink contours.
8. **Classic Oil Painting**: Rich chiaroscuro Rembrandt brushwork with impasto texture and warm vignette.
9. **Pop Art**: Bold Warhol-style color blocking, posterized palette, and high-contrast pop outlines.
10. **Psychedelic Swirl**: High-saturation kaleidoscopic rainbow swirl noise with hypersaturated subject colors.
11. **Watercolor Sketch**: Soft pastel color wash bleeding, textured paper background, and loose ink lines.
12. **Cyberpunk Neon**: Rain-slicked dark cityscape background with neon magenta/cyan duotone lighting and reflection bloom.
13. **Van Gogh**: Swirling directional brushstroke simulation with Starry Night cobalt blues and yellow highlights.
14. **Graffiti**: Urban spray-paint stencil mural with aerosol splatter texture, bold dripping outlines, and vivid saturation.
15. **Pencil Sketch**: Detailed graphite pencil sketch with fine crosshatching and realistic paper texture.
16. **Pixel Art**: True 16-bit retro arcade video game character sprites with quantized 16-color palette.
17. **Comic Book**: Black ink outlines with Ben-Day color halftone dot shading and punchy graphic tones.
18. **Low Poly**: Faceted geometric polygon block shading with crystalline cell quantization.
19. **Stained Glass**: Leaded black came outlines with jewel-toned cathedral translucent glass panels.
20. **Charcoal**: Expressive smudged dramatic shadows with high-contrast chiaroscuro carbon grain.
21. **Vaporwave**: 80s synthwave pastel pink and teal duotone gradient with CRT scanlines and soft dream bloom.
22. **Studio Ghibli Sky**: Dreamy painterly anime palette with warm golden-hour tones, soft sky contrast, and gentle lighting.
23. **Sin City Noir**: Stark black-and-white graphic novel noir with selective vibrant red isolation.
24. **Blueprint CAD**: Technical architectural cyan blueprint background with white linework and millimeter grid.
25. **80s Synthwave Horizon**: Outrun sunset gradient horizon with neon sun and glowing perspective wireframe grid.
26. **Manga Screentone**: Japanese manga ink with authentic screentone dot patterns in midtones and speedlines.
27. **Night Vision Phosphor**: Military NVG PVS-14 green phosphor monochrome with optical vignette falloff and sensor grain.
28. **Glacial Frost**: Sub-zero ice shimmer tint with diamond crystal edge reflections and icy bloom.
29. **Vintage 1920s Silent Film**: Antique sepia silver-gelatin tones with vignette darkening and vertical scratch flicker.
30. **Tron Wireframe**: Pitch black void with luminous electric cyan and orange wireframe edge tracing.
31. **Golden Hour Flare**: Warm cinematic sunlight wash with soft skin diffusion and golden bloom.
32. **Predator Thermal HUD**: Alien multi-spectrum rainbow thermal vision with triangular tactical lock reticle.

---

## 📱 How to Use Your Phone as a 1080p Webcam

### Method 1: Wireless Wi-Fi (No PC driver needed)
1. Install **IP Webcam** (Android) or **IP Camera Lite** (iOS).
2. Connect phone to the same Wi-Fi as your PC.
3. Tap **Start Server** in the app.
4. Run `run_phone_wifi.bat` and paste the URL shown on your phone (e.g. `http://192.168.1.15:8080/video`).

### Method 2: USB Cable via Iriun / DroidCam (Zero lag)
1. Install **Iriun Webcam** or **DroidCam** on phone & PC.
2. Connect via USB cable.
3. Run `run_phone_usb.bat` (or launch `run.bat` and press `2`).

---

## 📂 Project Structure

```
HandFrame-AI-Master/
├── app.py                           # Main application loop, camera manager & input router
├── hand_tracker.py                  # 7 hand skeleton styles, borders & gesture extraction
├── perspective.py                   # 35x accelerated ROI perspective warper & 5 AR frame designs
├── inference.py                     # All 32 offline fast filter algorithms & cloud AI engine
├── styles.py                        # 32 style presets, frame types & dynamic theme colors
├── smoothing.py                     # OneEuroFilter temporal landmark smoothing
├── ui_overlay.py                    # Glassmorphism HUD, theme accents & GPU telemetry
├── haarcascade_frontalface_default.xml # Local OpenCV face detector
├── hand_landmarker.task             # Google MediaPipe hand tracking weights
├── requirements.txt                 # Python dependencies
├── run.bat                          # Universal launcher with camera selector
├── run_pc.bat                       # Direct PC webcam launcher
├── run_phone_usb.bat                # Direct Phone USB launcher
├── run_phone_wifi.bat               # Direct Phone Wi-Fi stream launcher
├── saved_scans/                     # Folder where your saved snapshots go
└── README.md                        # Documentation & user guide
```

---

## ⚙️ Technical Specifications

- **Target Resolution**: 1920×1080 Full HD
- **Hardware Decoder**: High-Bitrate MJPG (`cv2.CAP_PROP_FOURCC = 'MJPG'`)
- **Acceleration**: NVIDIA GeForce RTX via OpenCL 3.0 CUDA
- **Hand Detector**: MediaPipe Tasks HandLandmarker Float16
- **Tracking Latency**: ~30ms total frame time (~30–60 FPS)
- **License**: MIT
