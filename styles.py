"""
styles.py
----------
Defines the artistic style presets and visual metadata for HandFrame AI.
Each style includes:
  - 'name': Human-readable display title shown in the HUD.
  - 'prompt': Detailed generative prompt sent to fal.ai cloud FLUX.2 inference.
  - 'frame_type': Preferred floating AR frame border design:
      * 'cyber': High-tech HUD with corner bracket notches and crosshairs.
      * 'neon': Multi-layered blooming glow aura.
      * 'classic': Double-inset beveled ornate border.
      * 'film': Perforated vintage filmstrip border.
      * 'minimal': Clean dotted/dashed sleek modern border.
  - 'theme_color': Primary BGR tuple used for dynamic glowing frame borders,
                   skeleton highlights, and HUD accent lighting.
"""

# ==============================================================================
# 🎨 32 DISTINCT STYLE PRESETS & METADATA
# ==============================================================================

STYLES = [
    # --------------------------------------------------------------------------
    # 1. Cyber Glitch Hologram
    # --------------------------------------------------------------------------
    {
        "name": "Cyber Glitch Hologram",
        "prompt": "futuristic cyan and magenta chromatic aberration holographic portrait, vhs scanlines, digital cyber artifacting",
        "frame_type": "cyber",
        "theme_color": (255, 230, 0),  # Electric Cyan
    },
    # --------------------------------------------------------------------------
    # 2. Matrix Digital Rain
    # --------------------------------------------------------------------------
    {
        "name": "Matrix Digital Rain",
        "prompt": "matrix terminal digital green code stream cascading over a portrait, cybernetic green phosphor glow, hacker aesthetic",
        "frame_type": "cyber",
        "theme_color": (0, 255, 100),  # Terminal Phosphor Green
    },
    # --------------------------------------------------------------------------
    # 3. Solar Inferno / Lava
    # --------------------------------------------------------------------------
    {
        "name": "Solar Inferno / Lava",
        "prompt": "molten lava inferno burning fire portrait, intense radiant orange and crimson heat glow, volcanic embers",
        "frame_type": "neon",
        "theme_color": (0, 120, 255),  # Molten Flame Orange
    },
    # --------------------------------------------------------------------------
    # 4. Ultraviolet X-Ray
    # --------------------------------------------------------------------------
    {
        "name": "Ultraviolet X-Ray",
        "prompt": "glowing neon ultraviolet x-ray electric negative portrait, bright cyan and violet bio-luminescent contours",
        "frame_type": "neon",
        "theme_color": (255, 180, 50),  # Bioluminescent Cyan-Blue
    },
    # --------------------------------------------------------------------------
    # 5. Terminator Cyber HUD
    # --------------------------------------------------------------------------
    {
        "name": "Terminator Cyber HUD",
        "prompt": "terminator cyborg red tactical night-vision scanner portrait, futuristic targeting crosshairs and telemetry grid",
        "frame_type": "cyber",
        "theme_color": (30, 30, 255),  # Crimson Red Alert
    },
    # --------------------------------------------------------------------------
    # 6. Thermal / Heatmap
    # --------------------------------------------------------------------------
    {
        "name": "Thermal / Heatmap",
        "prompt": "thermal infrared heatmap portrait, glowing purple and blue grid overlay, futuristic scan effect, high contrast",
        "frame_type": "cyber",
        "theme_color": (255, 60, 200),  # Infrared Magenta
    },
    # --------------------------------------------------------------------------
    # 7. Anime
    # --------------------------------------------------------------------------
    {
        "name": "Anime",
        "prompt": "clean anime illustration portrait, crisp cel shading, studio ghibli inspired lighting, vibrant colors",
        "frame_type": "minimal",
        "theme_color": (255, 160, 240),  # Blossom Pink
    },
    # --------------------------------------------------------------------------
    # 8. Classic Oil Painting
    # --------------------------------------------------------------------------
    {
        "name": "Classic Oil Painting",
        "prompt": "renaissance oil painting portrait, rembrandt chiaroscuro lighting, rich dark background, museum quality brushwork",
        "frame_type": "classic",
        "theme_color": (60, 160, 220),  # Antique Gold / Amber
    },
    # --------------------------------------------------------------------------
    # 9. Pop Art
    # --------------------------------------------------------------------------
    {
        "name": "Pop Art",
        "prompt": "vibrant pop art portrait, bold geometric shapes, andy warhol inspired color blocking, halftone texture",
        "frame_type": "minimal",
        "theme_color": (0, 240, 255),  # Bright Canary Yellow
    },
    # --------------------------------------------------------------------------
    # 10. Psychedelic Swirl
    # --------------------------------------------------------------------------
    {
        "name": "Psychedelic Swirl",
        "prompt": "psychedelic rainbow swirl portrait, kaleidoscopic color noise background, trippy poster art, high saturation",
        "frame_type": "neon",
        "theme_color": (255, 50, 180),  # Trippy Electric Purple
    },
    # --------------------------------------------------------------------------
    # 11. Watercolor Sketch
    # --------------------------------------------------------------------------
    {
        "name": "Watercolor Sketch",
        "prompt": "delicate watercolor portrait sketch, soft pastel washes, loose ink linework, paper texture background",
        "frame_type": "classic",
        "theme_color": (200, 200, 120),  # Soft Aqua Teal
    },
    # --------------------------------------------------------------------------
    # 12. Cyberpunk Neon
    # --------------------------------------------------------------------------
    {
        "name": "Cyberpunk Neon",
        "prompt": "cyberpunk neon city portrait, glowing magenta and cyan signage reflections, rain-soaked street background",
        "frame_type": "cyber",
        "theme_color": (255, 40, 220),  # Cyber Neon Magenta
    },
    # --------------------------------------------------------------------------
    # 13. Van Gogh
    # --------------------------------------------------------------------------
    {
        "name": "Van Gogh",
        "prompt": "post-impressionist portrait in the style of van gogh, swirling thick brushstrokes, vivid starry night palette",
        "frame_type": "classic",
        "theme_color": (30, 210, 255),  # Starry Night Yellow-Gold
    },
    # --------------------------------------------------------------------------
    # 14. Graffiti
    # --------------------------------------------------------------------------
    {
        "name": "Graffiti",
        "prompt": "urban graffiti mural style portrait, spray paint texture, bold outlines, street art color palette",
        "frame_type": "neon",
        "theme_color": (40, 240, 180),  # Street Spray Lime
    },
    # --------------------------------------------------------------------------
    # 15. Pencil Sketch
    # --------------------------------------------------------------------------
    {
        "name": "Pencil Sketch",
        "prompt": "graphite pencil sketch portrait, fine crosshatching, realistic shading, white paper background",
        "frame_type": "classic",
        "theme_color": (180, 180, 180),  # Graphite Silver
    },
    # --------------------------------------------------------------------------
    # 16. Pixel Art
    # --------------------------------------------------------------------------
    {
        "name": "Pixel Art",
        "prompt": "16-bit pixel art portrait, limited color palette, retro video game character style",
        "frame_type": "minimal",
        "theme_color": (100, 255, 100),  # Retro Arcade Green
    },
    # --------------------------------------------------------------------------
    # 17. Comic Book
    # --------------------------------------------------------------------------
    {
        "name": "Comic Book",
        "prompt": "comic book ink portrait, bold black outlines, ben-day dot shading, dramatic action-panel lighting",
        "frame_type": "minimal",
        "theme_color": (40, 80, 240),  # Graphic Scarlet Red
    },
    # --------------------------------------------------------------------------
    # 18. Low Poly
    # --------------------------------------------------------------------------
    {
        "name": "Low Poly",
        "prompt": "low poly 3D portrait, faceted geometric shading, flat color gradients, modern minimalist art",
        "frame_type": "cyber",
        "theme_color": (240, 180, 60),  # Geometric Cyan
    },
    # --------------------------------------------------------------------------
    # 19. Stained Glass
    # --------------------------------------------------------------------------
    {
        "name": "Stained Glass",
        "prompt": "stained glass window portrait, leaded black outlines, jewel-toned translucent panels, cathedral lighting",
        "frame_type": "classic",
        "theme_color": (220, 140, 40),  # Cathedral Cobalt Azure
    },
    # --------------------------------------------------------------------------
    # 20. Charcoal
    # --------------------------------------------------------------------------
    {
        "name": "Charcoal",
        "prompt": "expressive charcoal drawing portrait, smudged dramatic shadows, textured paper, high contrast monochrome",
        "frame_type": "classic",
        "theme_color": (140, 140, 140),  # Carbon Grey
    },
    # --------------------------------------------------------------------------
    # 21. Vaporwave
    # --------------------------------------------------------------------------
    {
        "name": "Vaporwave",
        "prompt": "vaporwave aesthetic portrait, pastel pink and teal gradient, retro 80s grid background, glitch accents",
        "frame_type": "neon",
        "theme_color": (240, 100, 250),  # Synth Pastel Pink
    },
    # --------------------------------------------------------------------------
    # 22. Studio Ghibli Sky
    # --------------------------------------------------------------------------
    {
        "name": "Studio Ghibli Sky",
        "prompt": "soft hand-painted anime portrait, dreamy cloud sky background, warm golden hour lighting, gentle color palette",
        "frame_type": "classic",
        "theme_color": (240, 210, 140),  # Ghibli Sky Cerulean
    },
    # --------------------------------------------------------------------------
    # 23. Sin City Noir (NEW)
    # --------------------------------------------------------------------------
    {
        "name": "Sin City Noir",
        "prompt": "frank miller sin city graphic novel portrait, high contrast black and white noir with selective vibrant scarlet red",
        "frame_type": "minimal",
        "theme_color": (40, 40, 230),  # Blood Red
    },
    # --------------------------------------------------------------------------
    # 24. Blueprint CAD (NEW)
    # --------------------------------------------------------------------------
    {
        "name": "Blueprint CAD",
        "prompt": "technical architectural cyan blueprint drawing, white grid lines, technical drafting measurements, engineering schematic",
        "frame_type": "cyber",
        "theme_color": (240, 160, 40),  # Blueprint Drafting Blue
    },
    # --------------------------------------------------------------------------
    # 25. 80s Synthwave Horizon (NEW)
    # --------------------------------------------------------------------------
    {
        "name": "80s Synthwave Horizon",
        "prompt": "outrun synthwave retro 80s portrait, neon sun horizon, purple and orange sunset gradient, wireframe road",
        "frame_type": "neon",
        "theme_color": (80, 60, 255),  # Sunset Radiant Orange
    },
    # --------------------------------------------------------------------------
    # 26. Manga Screentone (NEW)
    # --------------------------------------------------------------------------
    {
        "name": "Manga Screentone",
        "prompt": "japanese manga ink portrait, authentic screentone dot patterns, dramatic action speed lines, stark monochrome",
        "frame_type": "minimal",
        "theme_color": (220, 220, 220),  # Manga Silver Tone
    },
    # --------------------------------------------------------------------------
    # 27. Night Vision Phosphor (NEW)
    # --------------------------------------------------------------------------
    {
        "name": "Night Vision Phosphor",
        "prompt": "military night vision goggle portrait, intense monochrome green phosphor glow, heavy sensor noise, optical vignetting",
        "frame_type": "cyber",
        "theme_color": (50, 255, 80),  # PVS-14 Night Vision Green
    },
    # --------------------------------------------------------------------------
    # 28. Glacial Frost (NEW)
    # --------------------------------------------------------------------------
    {
        "name": "Glacial Frost",
        "prompt": "crystalline ice frost portrait, sub-zero frozen edge shimmer, glittering diamond frost crystals, pale glacial cyan glow",
        "frame_type": "neon",
        "theme_color": (255, 240, 160),  # Glacial Ice Cyan
    },
    # --------------------------------------------------------------------------
    # 29. Vintage 1920s Silent Film (NEW)
    # --------------------------------------------------------------------------
    {
        "name": "Vintage 1920s Silent Film",
        "prompt": "1920s early cinema silent film portrait, warm sepia silver gelatin tones, heavy film grain, vignette, vertical projector scratches",
        "frame_type": "film",
        "theme_color": (120, 170, 210),  # Antique Sepia Amber
    },
    # --------------------------------------------------------------------------
    # 30. Tron Wireframe (NEW)
    # --------------------------------------------------------------------------
    {
        "name": "Tron Wireframe",
        "prompt": "tron legacy digital grid portrait, pitch black void with luminous electric cyan and orange wireframe edge glow",
        "frame_type": "cyber",
        "theme_color": (255, 200, 0),  # Tron Cyan Circuit
    },
    # --------------------------------------------------------------------------
    # 31. Golden Hour Flare (NEW)
    # --------------------------------------------------------------------------
    {
        "name": "Golden Hour Flare",
        "prompt": "cinematic golden hour sunlight portrait, warm amber backlight flare, soft glowing skin diffusion, dreamy sunset warmth",
        "frame_type": "classic",
        "theme_color": (60, 200, 255),  # Golden Sun Flare
    },
    # --------------------------------------------------------------------------
    # 32. Predator Thermal HUD (NEW)
    # --------------------------------------------------------------------------
    {
        "name": "Predator Thermal HUD",
        "prompt": "predator alien thermal vision scanner portrait, rainbow infrared spectrum heat signature, triangular targeting reticle",
        "frame_type": "cyber",
        "theme_color": (0, 80, 255),  # Tactical Lock Crimson/Orange
    },
]


# ==============================================================================
# 🛠️ HELPER FUNCTIONS
# ==============================================================================

def get_style(index: int) -> dict:
    """
    Safely retrieves a style preset by index with wrap-around modulo.
    Ensures index cycling is always within bounds [0, style_count() - 1].
    """
    return STYLES[index % len(STYLES)]


def style_count() -> int:
    """Returns total number of available style presets."""
    return len(STYLES)
