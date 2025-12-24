"""
Upscale Module

Provides the SeedVR2 upscaling tab with Image and Video upscaling functionality.
"""

import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import gradio as gr
import httpx
import imageio
from PIL import Image

if TYPE_CHECKING:
    from modules import SharedServices

logger = logging.getLogger(__name__)

# Module metadata
TAB_ID = "upscale"
TAB_LABEL = "🔍 Upscale"
TAB_ORDER = 1

# Session temp directory for results (auto-cleaned on exit)
# Using a persistent TemporaryDirectory so files have nice names for Gradio download
_results_temp_dir = tempfile.TemporaryDirectory(prefix="upscale_results_")

# SeedVR2 Upscaler models (auto-download on demand by node)
SEEDVR2_DIT_MODELS = [
    "seedvr2_ema_3b-Q4_K_M.gguf",
    "seedvr2_ema_3b-Q8_0.gguf",
    "seedvr2_ema_7b-Q4_K_M.gguf",
    "seedvr2_ema_7b_sharp-Q4_K_M.gguf",
    "seedvr2_ema_3b_fp16.safetensors",
    "seedvr2_ema_7b_fp16.safetensors",
    "seedvr2_ema_7b_sharp_fp16.safetensors",
]
DEFAULT_SEEDVR2_DIT = "seedvr2_ema_3b-Q4_K_M.gguf"

# Built-in defaults (used if no user preset exists)
UPSCALE_BUILTIN_DEFAULTS = {
    "Image Default": {
        "dit_model": DEFAULT_SEEDVR2_DIT,
        "blocks_to_swap": 32,
        "attention_mode": "flash_attn_2",
        "batch_size": 1,
        "uniform_batch": False,
        "color_correction": "lab",
        "temporal_overlap": 0,
        "input_noise": 0.0,
        "latent_noise": 0.0,
        "encode_tiled": True,
        "encode_tile_size": 1024,
        "encode_tile_overlap": 128,
        "decode_tiled": True,
        "decode_tile_size": 1024,
        "decode_tile_overlap": 128,
        # Video export (not used for image, but included for consistency)
        "video_format": "H.264 (MP4)",
        "video_crf": 19,
        "video_pix_fmt": "yuv420p",
        "prores_profile": "hq",
        "save_png_sequence": False,
        "save_to_comfyui": True,
        # Resolution
        "image_resolution": 3072,
        "image_max_resolution": 4096,
        "video_resolution": 1080,
    },
    "Video Default": {
        "dit_model": "seedvr2_ema_3b_fp16.safetensors",
        "blocks_to_swap": 32,
        "attention_mode": "flash_attn_2",
        "batch_size": 33,
        "uniform_batch": True,
        "color_correction": "lab",
        "temporal_overlap": 3,
        "input_noise": 0.0,
        "latent_noise": 0.0,
        "encode_tiled": True,
        "encode_tile_size": 1024,
        "encode_tile_overlap": 128,
        "decode_tiled": True,
        "decode_tile_size": 768,
        "decode_tile_overlap": 128,
        # Video export defaults
        "video_format": "H.264 (MP4)",
        "video_crf": 19,
        "video_pix_fmt": "yuv420p",
        "prores_profile": "hq",
        "save_png_sequence": False,
        "save_to_comfyui": True,
        # Resolution
        "image_resolution": 3072,
        "image_max_resolution": 4096,
        "video_resolution": 1080,
    },
}

# Setting keys for preset serialization
UPSCALE_SETTING_KEYS = [
    "dit_model", "blocks_to_swap", "attention_mode", "batch_size", "uniform_batch",
    "color_correction", "temporal_overlap", "input_noise", "latent_noise",
    "encode_tiled", "encode_tile_size", "encode_tile_overlap",
    "decode_tiled", "decode_tile_size", "decode_tile_overlap",
    # Video export settings
    "video_format", "video_crf", "video_pix_fmt", "prores_profile", "save_png_sequence",
    "save_to_comfyui",
    # Resolution settings
    "image_resolution", "image_max_resolution", "video_resolution",
]


def new_random_seed_32bit():
    """Generate a new random seed (32-bit max for SeedVR2)."""
    return random.randint(0, 4294967295)


def get_seedvr2_max_blocks(dit_model: str) -> int:
    """Get max block swap value based on model size (3B=32, 7B=36)."""
    return 32 if "3b" in dit_model.lower() else 36


def extract_meaningful_filename(filepath: str) -> str:
    """Extract a meaningful filename, filtering out temp file patterns."""
    if not filepath:
        return "image"
    
    stem = Path(filepath).stem
    
    # Detect Gradio/system temp file patterns (tmp*, random hex strings, etc.)
    is_temp = (
        stem.lower().startswith('tmp') or
        stem.lower().startswith('temp') or
        (len(stem) < 12 and not any(c.isalpha() for c in stem[:3]))
    )
    
    if is_temp:
        return "image"
    
    # Truncate if too long
    if len(stem) > 50:
        stem = stem[:50]
    
    return stem


def save_upscale_to_outputs(image_path: str, original_path: str, resolution: int, 
                            outputs_dir: Path, subfolder: str = "upscaled") -> str:
    """Save upscaled image preserving original name with upscale details."""
    timestamp = datetime.now().strftime("%H%M%S")
    
    # Extract meaningful filename, filtering out temp patterns
    original_name = extract_meaningful_filename(original_path)
    
    target_dir = outputs_dir / subfolder
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Format: originalname_4Kup_HHMMSS.png
    res_label = f"{resolution // 1000}K" if resolution >= 1000 else f"{resolution}p"
    filename = f"{original_name}_{res_label}up_{timestamp}.png"
    output_path = target_dir / filename
    shutil.copy2(image_path, output_path)
    logger.info(f"Saved upscale to: {output_path}")
    return str(output_path)


async def download_image_from_url(url: str) -> str:
    """Download image from ComfyUI URL to a local temp file."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        suffix = Path(url).suffix or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(response.content)
            return f.name


def copy_to_temp_with_name(image_path: str, original_path: str, resolution: int) -> str:
    """Copy image to session temp dir with a meaningful name for Gradio download."""
    timestamp = datetime.now().strftime("%H%M%S")
    original_name = extract_meaningful_filename(original_path)
    res_label = f"{resolution // 1000}K" if resolution >= 1000 else f"{resolution}p"
    filename = f"{original_name}_{res_label}up_{timestamp}.png"
    temp_path = Path(_results_temp_dir.name) / filename
    shutil.copy2(image_path, temp_path)
    return str(temp_path)


async def upscale_image(
    services: "SharedServices",
    input_image,
    seed: int,
    randomize_seed: bool,
    resolution: int,
    max_resolution: int,
    dit_model: str,
    blocks_to_swap: int,
    attention_mode: str,
    # VAE settings
    encode_tiled: bool,
    encode_tile_size: int,
    encode_tile_overlap: int,
    decode_tiled: bool,
    decode_tile_size: int,
    decode_tile_overlap: int,
    # Upscaler settings
    batch_size: int,
    uniform_batch_size: bool,
    color_correction: str,
    temporal_overlap: int,
    input_noise_scale: float,
    latent_noise_scale: float,
    autosave: bool,
) -> tuple:
    """Upscale an image using SeedVR2. Returns (slider_tuple, status, seed, upscaled_path, original_path, resolution)."""
    outputs_dir = services.get_outputs_dir()
    
    try:
        if input_image is None:
            return None, "❌ Please upload an image to upscale", seed, None, None, None
        
        # SeedVR2 uses 32-bit seed max (4294967295)
        actual_seed = new_random_seed_32bit() if randomize_seed else min(int(seed), 4294967295)
        
        workflow_path = services.workflows_dir / "SeedVR2_4K_image_upscale.json"
        if not workflow_path.exists():
            return None, "❌ Upscale workflow not found", seed, None, None, None
        
        logger.info(f"Upscaling image with SeedVR2: {dit_model}, res={resolution}, max={max_resolution}")
        
        params = {
            "image": input_image,
            "seed": actual_seed,
            "resolution": int(resolution),
            "max_resolution": int(max_resolution),
            "dit_model": dit_model,
            "blocks_to_swap": int(blocks_to_swap),
            "attention_mode": attention_mode,
            # VAE settings
            "encode_tiled": encode_tiled,
            "encode_tile_size": int(encode_tile_size),
            "encode_tile_overlap": int(encode_tile_overlap),
            "decode_tiled": decode_tiled,
            "decode_tile_size": int(decode_tile_size),
            "decode_tile_overlap": int(decode_tile_overlap),
            # Upscaler settings
            "batch_size": int(batch_size),
            "uniform_batch_size": uniform_batch_size,
            "color_correction": color_correction,
            "temporal_overlap": int(temporal_overlap),
            "input_noise_scale": float(input_noise_scale),
            "latent_noise_scale": float(latent_noise_scale),
        }
        
        result = await services.kit.execute(str(workflow_path), params)
        
        if result.status == "error":
            return None, f"❌ Upscale failed: {result.msg}", actual_seed, None, None, None
        
        if not result.images:
            return None, "❌ No images generated", actual_seed, None, None, None
        
        image_path = result.images[0]
        if image_path.startswith("http"):
            image_path = await download_image_from_url(image_path)
        
        # Copy to temp with meaningful name so Gradio download button works nicely
        image_path = copy_to_temp_with_name(image_path, input_image, resolution)
        
        # Autosave
        if autosave:
            save_upscale_to_outputs(image_path, input_image, resolution, outputs_dir)
            status = f"✓ {result.duration:.1f}s | Saved" if result.duration else "✓ Saved"
        else:
            status = f"✓ {result.duration:.1f}s" if result.duration else "✓ Done"
        
        # Return tuple for ImageSlider (original, upscaled) + upscaled path for save button
        return (input_image, image_path), status, actual_seed, image_path, input_image, resolution
        
    except Exception as e:
        logger.error(f"Upscale error: {e}", exc_info=True)
        if "connect" in str(e).lower():
            return None, "❌ Cannot connect to ComfyUI", seed, None, None, None
        return None, f"❌ {str(e)}", seed, None, None, None


async def upscale_video(
    services: "SharedServices",
    input_video,
    seed: int,
    randomize_seed: bool,
    resolution: int,
    # Export settings
    video_format: str,
    video_crf: int,
    video_pix_fmt: str,
    prores_profile: str,
    save_png_sequence: bool,
    save_to_comfyui: bool,
    filename_prefix: str,
    # Model settings
    dit_model: str,
    blocks_to_swap: int,
    attention_mode: str,
    # VAE settings
    encode_tiled: bool,
    encode_tile_size: int,
    encode_tile_overlap: int,
    decode_tiled: bool,
    decode_tile_size: int,
    decode_tile_overlap: int,
    # Upscaler settings
    batch_size: int,
    uniform_batch_size: bool,
    color_correction: str,
    temporal_overlap: int,
    input_noise_scale: float,
    latent_noise_scale: float,
) -> tuple:
    """Upscale a video using SeedVR2 with VHS export. Returns (video_path, status, seed, output_path)."""
    outputs_dir = services.get_outputs_dir()
    
    try:
        if input_video is None:
            return None, "❌ Please upload a video to upscale", seed, None
        
        # SeedVR2 uses 32-bit seed max (4294967295)
        actual_seed = new_random_seed_32bit() if randomize_seed else min(int(seed), 4294967295)
        
        workflow_path = services.workflows_dir / "SeedVR2_HD_video_upscale.json"
        if not workflow_path.exists():
            return None, "❌ Video upscale workflow not found", seed, None
        
        # Map UI format choice to VHS format string and file extension
        format_map = {
            "H.264 (MP4)": ("video/h264-mp4", ".mp4"),
            "H.265 (MP4)": ("video/h265-mp4", ".mp4"),
            "ProRes (MOV)": ("video/ProRes", ".mov"),
        }
        vhs_format, file_ext = format_map.get(video_format, ("video/h264-mp4", ".mp4"))
        
        # Extract meaningful name from input video
        input_video_name = extract_meaningful_filename(input_video)
        if input_video_name == "image":
            input_video_name = "video"  # Better default for videos
        
        # Optional tag prefix from user
        tag = filename_prefix.strip() if filename_prefix else ""
        
        # Build output filename: [tag_]inputname_resolution_timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if tag:
            output_basename = f"{tag}_{input_video_name}_{resolution}p_{timestamp}"
        else:
            output_basename = f"{input_video_name}_{resolution}p_{timestamp}"
        
        # For ComfyUI/VHS, use a temp prefix (we'll copy to our folder after)
        comfyui_prefix = f"seedvr2_temp_{timestamp}"
        png_prefix = f"{comfyui_prefix}_png/{comfyui_prefix}"
        
        logger.info(f"Upscaling video with SeedVR2: {dit_model}, res={resolution}, format={vhs_format}, attn={attention_mode}")
        
        params = {
            "video": input_video,
            "seed": actual_seed,
            "resolution": int(resolution),
            "dit_model": dit_model,
            "blocks_to_swap": int(blocks_to_swap),
            "attention_mode": attention_mode,
            # VAE settings
            "encode_tiled": encode_tiled,
            "encode_tile_size": int(encode_tile_size),
            "encode_tile_overlap": int(encode_tile_overlap),
            "decode_tiled": decode_tiled,
            "decode_tile_size": int(decode_tile_size),
            "decode_tile_overlap": int(decode_tile_overlap),
            # Upscaler settings
            "batch_size": int(batch_size),
            "uniform_batch_size": uniform_batch_size,
            "color_correction": color_correction,
            "temporal_overlap": int(temporal_overlap),
            "input_noise_scale": float(input_noise_scale),
            "latent_noise_scale": float(latent_noise_scale),
            # Export settings - VHS saves to ComfyUI output folder with temp prefix
            "filename_prefix": comfyui_prefix,
            "video_format": vhs_format,
            "video_crf": int(video_crf),
            "video_pix_fmt": video_pix_fmt,
            "prores_profile": prores_profile,
            # Redundancy - also save to ComfyUI output folder
            "save_video_to_comfyui": save_to_comfyui,
            # PNG sequence settings - save to ComfyUI first, we'll copy after
            "save_png_sequence": save_png_sequence,
            "png_filename_prefix": png_prefix,
        }
        
        result = await services.kit.execute(str(workflow_path), params)
        
        if result.status == "error":
            return None, f"❌ Video upscale failed: {result.msg}", actual_seed, None
        
        if not result.videos:
            return None, "❌ No video generated", actual_seed, None
        
        video_url = result.videos[0]
        
        # Save video to our outputs folder with proper naming
        output_filename = f"{output_basename}{file_ext}"
        output_dir = outputs_dir / "upscaled"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename
        
        if video_url.startswith("http"):
            async with httpx.AsyncClient(timeout=300) as client:
                response = await client.get(video_url)
                response.raise_for_status()
                with open(output_path, "wb") as f:
                    f.write(response.content)
        else:
            # Local path - copy to our outputs
            shutil.copy2(video_url, output_path)
        
        logger.info(f"Saved upscaled video to: {output_path}")
        
        # For Gradio display, copy to temp file to prevent Gradio's MP4 conversion
        temp_display_path = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext).name
        shutil.copy2(output_path, temp_display_path)
        
        # Build status
        time_str = f"{result.duration:.1f}s" if result.duration else ""
        format_str = video_format.split(" ")[0]  # "H.264" from "H.264 (MP4)"
        
        status_parts = [f"✓ {format_str}"]
        if time_str:
            status_parts.append(time_str)
        if save_png_sequence:
            status_parts.append("+ PNG seq")
        
        status = " | ".join(status_parts)
        status += f"\n📁 {output_path}"
        
        # Return temp path for Gradio display, actual output path for state
        return temp_display_path, status, actual_seed, str(output_path)
        
    except Exception as e:
        logger.error(f"Video upscale error: {e}", exc_info=True)
        if "connect" in str(e).lower():
            return None, "❌ Cannot connect to ComfyUI", seed, None
        return None, f"❌ {str(e)}", seed, None


def get_upscale_preset(name: str, settings_manager) -> dict:
    """Get preset by name - checks user presets first, then built-in defaults."""
    user_presets = settings_manager.get("upscale_presets", {})
    if name in user_presets:
        return user_presets[name]
    return UPSCALE_BUILTIN_DEFAULTS.get(name, UPSCALE_BUILTIN_DEFAULTS["Image Default"])


def _build_preset_choices(user_presets: list) -> list:
    """Build preset dropdown choices, filtering duplicates from built-in list."""
    builtin_presets = [p for p in UPSCALE_BUILTIN_DEFAULTS.keys() if p not in user_presets]
    if user_presets and builtin_presets:
        return user_presets + ["─────────"] + builtin_presets
    return user_presets + builtin_presets


def save_upscale_preset(name: str, preset: dict, settings_manager) -> tuple[str, list]:
    """Save an upscale preset. Returns (status_message, updated_choices)."""
    if not name or not name.strip():
        return "❌ Enter a preset name", []
    name = name.strip()
    
    # Load existing settings, update, save
    settings = settings_manager.load()
    if "upscale_presets" not in settings:
        settings["upscale_presets"] = {}
    settings["upscale_presets"][name] = preset
    settings_manager.save(settings)
    
    # Return updated choices (filter duplicates from built-in)
    user_presets = list(settings["upscale_presets"].keys())
    return f"✓ Saved '{name}'", _build_preset_choices(user_presets)


def delete_upscale_preset(name: str, settings_manager) -> tuple[str, list, str]:
    """Delete a user preset. Returns (status_message, updated_choices, new_selection)."""
    # Check if it's a separator
    if name == "─────────":
        return f"❌ Cannot delete separator", [], name
    
    settings = settings_manager.load()
    user_presets = settings.get("upscale_presets", {})
    
    # Check if it's a user preset (can be deleted) or only exists as built-in (cannot delete)
    if name not in user_presets:
        if name in UPSCALE_BUILTIN_DEFAULTS:
            return f"❌ Cannot delete built-in preset '{name}'", [], name
        return f"❌ Preset '{name}' not found", [], name
    
    del user_presets[name]
    settings["upscale_presets"] = user_presets
    settings_manager.save(settings)
    
    # Update dropdown choices (filter duplicates from built-in)
    remaining = list(user_presets.keys())
    return f"✓ Deleted '{name}'", _build_preset_choices(remaining), "Image Default"


def open_folder(folder_path: Path):
    """Cross-platform folder opener."""
    folder_path.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        os.startfile(folder_path)
    elif sys.platform == "darwin":
        subprocess.run(["open", str(folder_path)])
    else:
        subprocess.run(["xdg-open", str(folder_path)])


# =============================================================================
# Media Analysis
# =============================================================================

# Color schemes for analysis panels (dark = left/saturated, light = right/faded)
ANALYSIS_COLOR_SCHEMES = {
    "purple": {
        "dark": "#bbc1f2",
        "light": "rgba(220, 222, 250, 0.6)",
        "divider": "#c0c6e8",
        "text": "#362e54",
    },
    "blue": {
        "dark": "#6586c7",
        "light": "rgba(230, 245, 248, 0.6)",
        "divider": "#6586c7",
        "text": "#12316e",
    },
    "coral": {
        "dark": "#f8d0d4",
        "light": "rgba(252, 235, 237, 0.6)",
        "divider": "#f0c0c6",
        "text": "#8b3a4a",
    },
    "teal": {
        "dark": "#b2dfdb",
        "light": "rgba(220, 242, 240, 0.6)",
        "divider": "#a8d8d2",
        "text": "#00695c",
    },
}


def _stat_box(label: str, value: str, text_color: str) -> str:
    """Generate a single stat box HTML (transparent bg, inherits parent gradient)."""
    return f'''<div style="flex: 1; min-width: 80px; padding: 6px 8px;">
<div style="font-size: 0.7em; color: #292626; margin-bottom: 1px;">{label}</div>
<div style="font-size: 0.95em; font-weight: 600; color: {text_color};">{value}</div>
</div>'''


def _stat_pair(label1: str, val1: str, label2: str, val2: str, scheme: dict, reverse: bool = False) -> str:
    """Generate a pair of stat boxes with gradient. reverse=True for light→dark."""
    if reverse:
        bg_start, bg_end = scheme["light"], scheme["dark"]
    else:
        bg_start, bg_end = scheme["dark"], scheme["light"]
    
    box1 = _stat_box(label1, val1, scheme["text"])
    box2 = _stat_box(label2, val2, scheme["text"])
    sep = f'<div style="width: 1px; background: {scheme["divider"]}; margin: 4px 0;"></div>'
    
    return f'''<div style="flex: 1; min-width: 160px; display: flex; gap: 0; background: linear-gradient(90deg, {bg_start} 0%, {bg_end} 100%); border-radius: 4px; overflow: hidden;">
{box1}{sep}{box2}
</div>'''


def analyze_media(file_path: str, is_video: bool = False, color_scheme: str = "purple") -> gr.update:
    """Unified media analysis. Returns single gr.HTML update with paired stat groups."""
    if not file_path:
        return gr.update(value="", visible=False)

    scheme = ANALYSIS_COLOR_SCHEMES.get(color_scheme, ANALYSIS_COLOR_SCHEMES["purple"])

    try:
        resolved_path = str(Path(file_path).resolve())

        # File size
        file_size = "N/A"
        if os.path.exists(resolved_path):
            size_bytes = os.path.getsize(resolved_path)
            if size_bytes < 1024**2:
                file_size = f"{size_bytes/1024:.1f} KB"
            elif size_bytes < 1024**3:
                file_size = f"{size_bytes/1024**2:.1f} MB"
            else:
                file_size = f"{size_bytes/1024**3:.2f} GB"

        if is_video:
            reader = imageio.get_reader(resolved_path)
            meta = reader.get_meta_data()
            duration, fps = meta.get("duration", 0), meta.get("fps", 30)
            size = meta.get("size", (0, 0))
            width, height = (int(size[0]), int(size[1])) if isinstance(size, tuple) else (0, 0)
            nframes = meta.get("nframes")
            frames = int(nframes) if nframes and nframes != float("inf") else int(duration * fps) if duration and fps else 0
            reader.close()
            # Same gradient direction: dark→light | dark→light
            pair1 = _stat_pair("RESOLUTION", f"{width}×{height}", "FRAMES", str(frames), scheme)
            pair2 = _stat_pair("DURATION", f"{duration:.2f}s @ {fps:.1f} FPS", "FILE SIZE", file_size, scheme)
        else:
            img = Image.open(resolved_path)
            width, height = img.size
            mp = (width * height) / 1_000_000
            # Same gradient direction: dark→light | dark→light
            pair1 = _stat_pair("RESOLUTION", f"{width}×{height}", "MEGAPIXELS", f"{mp:.2f} MP", scheme)
            pair2 = _stat_pair("FORMAT", img.format or "Unknown", "FILE SIZE", file_size, scheme)

        html = f'''<div style="display: flex; flex-wrap: wrap; gap: 4px; margin: -8px 0; font-family: 'Segoe UI', sans-serif;">{pair1}{pair2}</div>'''
        return gr.update(value=html, visible=True)

    except Exception as e:
        error_html = f'<div style="padding: 8px; background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 6px; color: #721c24; font-size: 0.9em;">❌ Error: {str(e)}</div>'
        return gr.update(value=error_html, visible=True)


def create_tab(services: "SharedServices") -> gr.TabItem:
    """
    Create the Upscale tab with Image and Video sub-tabs.
    
    Args:
        services: SharedServices instance with all dependencies
        
    Returns:
        gr.TabItem containing the complete Upscale interface
    """
    outputs_dir = services.get_outputs_dir()
    comfyui_output_dir = services.app_dir / "comfyui" / "output"
    
    # Load existing user presets for dropdown (user presets override built-in defaults)
    user_presets = list(services.settings.get("upscale_presets", {}).keys())
    preset_choices = _build_preset_choices(user_presets)
    
    # Load saved "Image Default" preset for initial UI values (or fall back to built-in)
    initial_preset = get_upscale_preset("Image Default", services.settings)
    
    def apply_upscale_preset(preset: dict):
        """Convert preset dict to tuple of values for UI components."""
        max_blocks = get_seedvr2_max_blocks(preset.get("dit_model", DEFAULT_SEEDVR2_DIT))
        video_format = preset.get("video_format", "H.264 (MP4)")
        is_prores = "ProRes" in video_format
        return (
            preset.get("dit_model", DEFAULT_SEEDVR2_DIT),
            gr.update(value=preset.get("blocks_to_swap", 36), maximum=max_blocks),
            preset.get("attention_mode", "flash_attn_2"),
            preset.get("batch_size", 1),
            preset.get("uniform_batch", False),
            preset.get("color_correction", "lab"),
            preset.get("temporal_overlap", 0),
            preset.get("input_noise", 0.0),
            preset.get("latent_noise", 0.0),
            preset.get("encode_tiled", True),
            preset.get("encode_tile_size", 1024),
            preset.get("encode_tile_overlap", 128),
            preset.get("decode_tiled", True),
            preset.get("decode_tile_size", 1024),
            preset.get("decode_tile_overlap", 128),
            # Video export settings
            video_format,
            gr.update(value=preset.get("video_crf", 19), visible=not is_prores),
            gr.update(value=preset.get("video_pix_fmt", "yuv420p"), visible=not is_prores),
            gr.update(value=preset.get("prores_profile", "hq"), visible=is_prores),
            preset.get("save_png_sequence", False),
            preset.get("save_to_comfyui", True),
            # Resolution settings
            preset.get("image_resolution", 3072),
            preset.get("image_max_resolution", 4096),
            preset.get("video_resolution", 1080),
        )
    
    with gr.TabItem(TAB_LABEL, id=TAB_ID) as tab:
        with gr.Row():
            with gr.Column(scale=1):
                # Image/Video input tabs
                with gr.Tabs() as upscale_input_tabs:
                    with gr.TabItem("🖼️ Image", id="upscale_image_tab"):
                        upscale_input_image = gr.Image(label="Input Image", type="filepath", elem_classes="image-window")
                        input_image_analysis = gr.HTML(visible=False, elem_classes="analysis-panel")
                        with gr.Row():
                            upscale_resolution = gr.Slider(
                                label="Resolution",
                                value=initial_preset.get("image_resolution", 3072),
                                minimum=1024,
                                maximum=4096,
                                step=8,
                                info="Target short-side resolution"
                            )
                            upscale_max_resolution = gr.Slider(
                                label="Max Resolution",
                                value=initial_preset.get("image_max_resolution", 4096),
                                minimum=1024,
                                maximum=7680,
                                step=8,
                                info="Maximum long-side resolution"
                            )
                        upscale_btn = gr.Button("🔍 Upscale Image", variant="primary", size="sm")
                    
                    with gr.TabItem("🎬 Video", id="upscale_video_tab"):
                        upscale_input_video = gr.Video(label="Input Video", elem_classes="video-window")
                        input_video_analysis = gr.HTML(visible=False, elem_classes="analysis-panel")
                        with gr.Row():
                            upscale_video_resolution = gr.Slider(
                                label="Resolution",
                                value=initial_preset.get("video_resolution", 1080),
                                minimum=640,
                                maximum=2160,
                                step=2,
                                info="Target short-side resolution",
                                scale=3
                            )
                            upscale_video_res_720_btn = gr.Button("720", size="sm", scale=0, min_width=50)
                            upscale_video_res_1080_btn = gr.Button("1080", size="sm", scale=0, min_width=50)
                        
                        # Video Export Settings
                        initial_video_format = initial_preset.get("video_format", "H.264 (MP4)")
                        initial_is_prores = "ProRes" in initial_video_format
                        with gr.Accordion("📹 Export Settings", open=False):
                            upscale_video_format = gr.Dropdown(
                                label="Format",
                                choices=["H.264 (MP4)", "H.265 (MP4)", "ProRes (MOV)"],
                                value=initial_video_format,
                                info="Output video format"
                            )
                            # H.264/H.265 options
                            upscale_video_crf = gr.Slider(
                                label="Quality (CRF)",
                                value=initial_preset.get("video_crf", 19),
                                minimum=0,
                                maximum=51,
                                step=1,
                                info="Lower = better quality, larger file. 19 is visually lossless",
                                visible=not initial_is_prores
                            )
                            upscale_video_pix_fmt = gr.Dropdown(
                                label="Pixel Format",
                                choices=["yuv420p", "yuv420p10le"],
                                value=initial_preset.get("video_pix_fmt", "yuv420p"),
                                info="10-bit (10le) for higher quality, 8-bit for compatibility",
                                visible=not initial_is_prores
                            )
                            # ProRes options
                            upscale_prores_profile = gr.Dropdown(
                                label="ProRes Profile",
                                choices=["lt", "standard", "hq", "4444", "4444xq"],
                                value=initial_preset.get("prores_profile", "hq"),
                                info="HQ for most uses, 4444/4444XQ for maximum quality",
                                visible=initial_is_prores
                            )
                            # Redundancy options - save to ComfyUI output folder
                            gr.Markdown("**Redundancy Options** *(saves to ComfyUI output)*")
                            upscale_save_png_sequence = gr.Checkbox(
                                label="Also save PNG sequence (16-bit lossless)",
                                value=initial_preset.get("save_png_sequence", False),
                                info="Failsafe for long videos - saves frames as individual PNGs"
                            )
                            upscale_save_to_comfyui = gr.Checkbox(
                                label="Also save video to ComfyUI output folder",
                                value=initial_preset.get("save_to_comfyui", True),
                                info="Backup copy saved alongside PNG sequence if enabled"
                            )
                            open_comfyui_output_btn = gr.Button("📂 Open ComfyUI Output Folder", size="sm")
                            
                            # Optional filename tag
                            upscale_video_filename = gr.Textbox(
                                label="Filename Tag (optional)",
                                value="",
                                placeholder="e.g. test1, final",
                                info="Optional prefix tag added to output filename"
                            )
                        
                        upscale_video_btn = gr.Button("🎬 Upscale Video", variant="primary", size="lg")
                
                with gr.Accordion("🔧 SeedVR2 Settings", open=True):
                    initial_dit_model = initial_preset.get("dit_model", DEFAULT_SEEDVR2_DIT)
                    initial_max_blocks = get_seedvr2_max_blocks(initial_dit_model)
                    upscale_dit_model = gr.Dropdown(
                        label="DIT Model",
                        choices=SEEDVR2_DIT_MODELS,
                        value=initial_dit_model,
                        info="Models auto-download on first use"
                    )
                    with gr.Row():
                        upscale_blocks_to_swap = gr.Slider(
                            label="Block Swap",
                            value=initial_preset.get("blocks_to_swap", initial_max_blocks),
                            minimum=0,
                            maximum=initial_max_blocks,
                            step=1,
                            info="Higher = less VRAM, slower"
                        )
                        upscale_attention_mode = gr.Dropdown(
                            label="Attention",
                            choices=["sdpa", "flash_attn_2", "sageattn_2"],
                            value=initial_preset.get("attention_mode", "sdpa"),
                            info="flash_attn_2/sageattn_2 faster if available"
                        )
                
                with gr.Accordion("🎛️ Advanced Settings", open=False):
                    with gr.Row():
                        upscale_batch_size = gr.Slider(
                            label="Batch Size",
                            value=initial_preset.get("batch_size", 1),
                            minimum=1,
                            maximum=64,
                            step=1,
                            info="Frames per batch (video: ~33)"
                        )
                        upscale_uniform_batch = gr.Checkbox(
                            label="Uniform Batch",
                            value=initial_preset.get("uniform_batch", False),
                            info="Equal batch sizes"
                        )
                    with gr.Row():
                        upscale_color_correction = gr.Dropdown(
                            label="Color Correction",
                            choices=["none", "lab", "wavelet", "adain"],
                            value=initial_preset.get("color_correction", "lab"),
                            info="Color matching method"
                        )
                        upscale_temporal_overlap = gr.Slider(
                            label="Temporal Overlap",
                            value=initial_preset.get("temporal_overlap", 0),
                            minimum=0,
                            maximum=16,
                            step=1,
                            info="Frame overlap (video: ~3)"
                        )
                    with gr.Row():
                        upscale_input_noise = gr.Slider(
                            label="Input Noise",
                            value=initial_preset.get("input_noise", 0.0),
                            minimum=0.0,
                            maximum=0.2,
                            step=0.001,
                            info="Low levels (<0.1) can add detail"
                        )
                        upscale_latent_noise = gr.Slider(
                            label="Latent Noise",
                            value=initial_preset.get("latent_noise", 0.0),
                            minimum=0.0,
                            maximum=1.0,
                            step=0.001,
                            info="Not recommended for most use"
                        )
                
                with gr.Accordion("🎛️ VAE Tiling", open=False):
                    with gr.Row():
                        upscale_encode_tiled = gr.Checkbox(label="Encode Tiled", value=initial_preset.get("encode_tiled", True))
                        upscale_decode_tiled = gr.Checkbox(label="Decode Tiled", value=initial_preset.get("decode_tiled", True))
                    with gr.Row():
                        upscale_encode_tile_size = gr.Slider(
                            label="Encode Tile Size",
                            value=initial_preset.get("encode_tile_size", 1024),
                            minimum=256,
                            maximum=2048,
                            step=64
                        )
                        upscale_encode_tile_overlap = gr.Slider(
                            label="Encode Overlap",
                            value=initial_preset.get("encode_tile_overlap", 128),
                            minimum=0,
                            maximum=512,
                            step=16
                        )
                    with gr.Row():
                        upscale_decode_tile_size = gr.Slider(
                            label="Decode Tile Size",
                            value=initial_preset.get("decode_tile_size", 1024),
                            minimum=256,
                            maximum=2048,
                            step=64
                        )
                        upscale_decode_tile_overlap = gr.Slider(
                            label="Decode Overlap",
                            value=initial_preset.get("decode_tile_overlap", 128),
                            minimum=0,
                            maximum=512,
                            step=16
                        )
                
                with gr.Accordion("💾 Presets", open=False):
                    # Track which input tab is active for "Save as Default"
                    upscale_active_tab = gr.State(value="Image")
                    
                    upscale_save_default_btn = gr.Button("⭐ Save current settings as default (loads on app start)", size="sm")
                    gr.Markdown("---")
                    with gr.Row():
                        upscale_preset_dropdown = gr.Dropdown(
                            label="Load Preset",
                            choices=preset_choices,
                            value="Image Default",
                            scale=2
                        )
                        upscale_load_preset_btn = gr.Button("📂 Load", size="sm", scale=1)
                        upscale_delete_preset_btn = gr.Button("🗑️", size="sm", scale=0, min_width=40)
                    with gr.Row():
                        upscale_preset_name = gr.Textbox(
                            label="Preset Name",
                            placeholder="my_preset",
                            scale=2
                        )
                        upscale_save_preset_btn = gr.Button("💾 Save", size="sm", scale=1)
                    upscale_preset_status = gr.Textbox(label="", interactive=False, show_label=False)
                
                with gr.Row():
                    upscale_seed = gr.Number(label="Seed", value=new_random_seed_32bit(), minimum=0, maximum=4294967295, step=1)
                    upscale_randomize_seed = gr.Checkbox(label="🎲 Randomize", value=True)
            
            with gr.Column(scale=1):
                # Output tabs matching input
                with gr.Tabs() as upscale_output_tabs:
                    with gr.TabItem("🖼️ Image Result", id="upscale_image_result"):
                        upscale_slider = gr.ImageSlider(
                            label="Before / After",
                            type="filepath",
                            elem_classes="image-window",
                            show_download_button=False
                        )
                        output_image_analysis = gr.HTML(visible=False, elem_classes="analysis-panel")
                        with gr.Row():                          
                            upscale_save_btn = gr.Button("💾 Save", size="sm", variant="primary")
                        upscale_autosave = gr.Checkbox(label="Auto-save", value=False)
                    
                    with gr.TabItem("🎬 Video Result", id="upscale_video_result"):
                        upscale_output_video = gr.Video(label="Upscaled Video", show_download_button=False, elem_classes="video-window")
                        output_video_analysis = gr.HTML(visible=False, elem_classes="analysis-panel")
                        gr.Markdown(
                            "*All upscaled videos are automatically saved to the output folder. "
                            "Note: Gradio converts H.265/ProRes to MP4 for browser preview — "
                            "the saved file retains full quality.*",
                            elem_classes=["video-note"]
                        )

                upscale_status = gr.Textbox(label="Status", interactive=False, show_label=False, lines=2)
                upscale_open_folder_btn = gr.Button("📂 Open Output Folder", size="sm")

                # System monitor
                with gr.Row():
                    with gr.Column(scale=1, min_width=200):
                        gpu_monitor = gr.Textbox(
                            value="Loading...",
                            lines=4.5,
                            container=False,
                            interactive=False,
                            show_label=False,
                            elem_classes="monitor-box gpu-monitor"
                        )
                    with gr.Column(scale=1, min_width=200):
                        cpu_monitor = gr.Textbox(
                            value="Loading...",
                            lines=4,
                            container=False,
                            interactive=False,
                            show_label=False,
                            elem_classes="monitor-box cpu-monitor"
                        )

                # Hidden state for upscaled paths and original info (for save naming)
                upscale_result_path = gr.State(value=None)
                upscale_original_path = gr.State(value=None)
                upscale_result_resolution = gr.State(value=None)
                upscale_video_result_path = gr.State(value=None)
                
                with gr.Accordion("ℹ️ Upscale Help", open=False):
                    gr.Markdown("""
**Running Out of VRAM (OOM errors)?**
1. Reduce **Resolution** to 2048 or lower
2. Increase **Block Swap** to maximum (32 for 3B, 36 for 7B)
3. Reduce **VAE Tile Size** to 512 or 256
4. Use a **3B GGUF** model instead of 7B

**Performance Tips**
- **Block Swap**: Lower values = faster but uses more VRAM
- **Batch Size**: Higher = faster video upscaling (if VRAM allows)
- Defaults are tuned for lower-end hardware

**Presets**: Save your settings with **💾 Presets** — use **⭐ Save as default** to auto-load on startup
""")
                    gr.Button("🎬 SeedVR2 Tutorial Video", size="sm", link="https://www.youtube.com/watch?v=MBtWYXq_r60")
        
        # ===== EVENT HANDLERS =====
        
        # Helper to get current color scheme from settings
        def get_analysis_color():
            return services.settings.get("analysis_color_scheme", "purple")
        
        # Input analysis - triggered when image/video is uploaded
        upscale_input_image.change(
            fn=lambda p: analyze_media(p, is_video=False, color_scheme=get_analysis_color()),
            inputs=[upscale_input_image],
            outputs=[input_image_analysis]
        )
        upscale_input_video.change(
            fn=lambda p: analyze_media(p, is_video=True, color_scheme=get_analysis_color()),
            inputs=[upscale_input_video],
            outputs=[input_video_analysis]
        )
        
        # All settings components for preset system
        upscale_all_settings = [
            upscale_dit_model,
            upscale_blocks_to_swap,
            upscale_attention_mode,
            upscale_batch_size,
            upscale_uniform_batch,
            upscale_color_correction,
            upscale_temporal_overlap,
            upscale_input_noise,
            upscale_latent_noise,
            upscale_encode_tiled,
            upscale_encode_tile_size,
            upscale_encode_tile_overlap,
            upscale_decode_tiled,
            upscale_decode_tile_size,
            upscale_decode_tile_overlap,
            # Video export settings
            upscale_video_format,
            upscale_video_crf,
            upscale_video_pix_fmt,
            upscale_prores_profile,
            upscale_save_png_sequence,
            upscale_save_to_comfyui,
            # Resolution settings
            upscale_resolution,
            upscale_max_resolution,
            upscale_video_resolution,
        ]
        
        # Update block swap slider max based on DIT model selection
        def update_block_swap_max(dit_model):
            max_blocks = get_seedvr2_max_blocks(dit_model)
            return gr.update(maximum=max_blocks, value=max_blocks)
        
        upscale_dit_model.change(
            fn=update_block_swap_max,
            inputs=[upscale_dit_model],
            outputs=[upscale_blocks_to_swap]
        )
        
        # Tab switching loads from preset system and tracks active tab
        def on_upscale_tab_select(evt: gr.SelectData):
            """Switch presets based on which tab is selected."""
            if evt.value == "🖼️ Image":
                preset = get_upscale_preset("Image Default", services.settings)
                active_tab = "Image"
            elif evt.value == "🎬 Video":
                preset = get_upscale_preset("Video Default", services.settings)
                active_tab = "Video"
            else:
                return (gr.update(),) * len(upscale_all_settings) + (gr.update(),)
            
            return apply_upscale_preset(preset) + (active_tab,)
        
        upscale_input_tabs.select(
            fn=on_upscale_tab_select,
            outputs=upscale_all_settings + [upscale_active_tab]
        )
        
        def save_as_default(active_tab, *values):
            """Save current settings as the default for the active tab (Image/Video)."""
            preset_name = f"{active_tab} Default"
            
            # Build preset dict
            preset = dict(zip(UPSCALE_SETTING_KEYS, values))
            
            # Save preset
            status, choices = save_upscale_preset(preset_name, preset, services.settings)
            return status, gr.update(choices=choices, value=preset_name)
        
        upscale_save_default_btn.click(
            fn=save_as_default,
            inputs=[upscale_active_tab] + upscale_all_settings,
            outputs=[upscale_preset_status, upscale_preset_dropdown]
        )
        
        def save_preset_handler(name, *values):
            """Save current upscale settings as a preset."""
            preset = dict(zip(UPSCALE_SETTING_KEYS, values))
            status, choices = save_upscale_preset(name, preset, services.settings)
            if choices:
                return status, gr.update(choices=choices, value=name)
            return status, gr.update()
        
        def load_preset_handler(name):
            """Load a preset's settings via button click."""
            if name == "─────────":
                return ("",) + (gr.update(),) * len(upscale_all_settings)
            
            preset = get_upscale_preset(name, services.settings)
            return (f"✓ Loaded '{name}'",) + apply_upscale_preset(preset)
        
        upscale_save_preset_btn.click(
            fn=save_preset_handler,
            inputs=[upscale_preset_name] + upscale_all_settings,
            outputs=[upscale_preset_status, upscale_preset_dropdown]
        )
        
        upscale_load_preset_btn.click(
            fn=load_preset_handler,
            inputs=[upscale_preset_dropdown],
            outputs=[upscale_preset_status] + upscale_all_settings
        )
        
        def delete_preset_handler(name):
            """Delete a user preset (cannot delete built-in defaults)."""
            status, choices, new_selection = delete_upscale_preset(name, services.settings)
            if choices:
                return status, gr.update(choices=choices, value=new_selection)
            return status, gr.update()
        
        upscale_delete_preset_btn.click(
            fn=delete_preset_handler,
            inputs=[upscale_preset_dropdown],
            outputs=[upscale_preset_status, upscale_preset_dropdown]
        )
        
        # Shared upscale inputs (SeedVR2 settings) - for image upscale
        upscale_common_inputs = [
            upscale_dit_model,
            upscale_blocks_to_swap,
            upscale_attention_mode,
            # VAE settings
            upscale_encode_tiled,
            upscale_encode_tile_size,
            upscale_encode_tile_overlap,
            upscale_decode_tiled,
            upscale_decode_tile_size,
            upscale_decode_tile_overlap,
            # Upscaler settings
            upscale_batch_size,
            upscale_uniform_batch,
            upscale_color_correction,
            upscale_temporal_overlap,
            upscale_input_noise,
            upscale_latent_noise,
            upscale_autosave,
        ]
        
        # Video upscale inputs - includes attention_mode, no autosave (always saves)
        upscale_video_common_inputs = [
            upscale_dit_model,
            upscale_blocks_to_swap,
            upscale_attention_mode,
            # VAE settings
            upscale_encode_tiled,
            upscale_encode_tile_size,
            upscale_encode_tile_overlap,
            upscale_decode_tiled,
            upscale_decode_tile_size,
            upscale_decode_tile_overlap,
            # Upscaler settings
            upscale_batch_size,
            upscale_uniform_batch,
            upscale_color_correction,
            upscale_temporal_overlap,
            upscale_input_noise,
            upscale_latent_noise,
        ]
        
        # Hidden state to pass status from main handler to finalize (avoids spinner on status)
        upscale_pending_status = gr.State(value="")
        upscale_video_pending_status = gr.State(value="")
        
        # Image Upscale - initial handler sets status message and switches tab immediately
        def upscale_image_start():
            return "⏳ Upscaling image... (check terminal for details)", gr.Tabs(selected="upscale_image_result")
        
        # Image Upscale - main function returns primary outputs (status goes to State, not Textbox)
        async def upscale_image_main(
            input_image, seed, randomize_seed, resolution, max_resolution,
            dit_model, blocks_to_swap, attention_mode,
            encode_tiled, encode_tile_size, encode_tile_overlap,
            decode_tiled, decode_tile_size, decode_tile_overlap,
            batch_size, uniform_batch, color_correction, temporal_overlap,
            input_noise, latent_noise, autosave
        ):
            result = await upscale_image(
                services, input_image, seed, randomize_seed, resolution, max_resolution,
                dit_model, blocks_to_swap, attention_mode,
                encode_tiled, encode_tile_size, encode_tile_overlap,
                decode_tiled, decode_tile_size, decode_tile_overlap,
                batch_size, uniform_batch, color_correction, temporal_overlap,
                input_noise, latent_noise, autosave
            )
            # result = (slider_tuple, status, seed, upscaled_path, original_path, resolution)
            # Return: slider, seed, status->State, hidden states
            return result[0], result[2], result[1], result[3], result[4], result[5]
        
        # Secondary handler for status, analysis (no progress spinner on these)
        def upscale_image_finalize(pending_status, upscaled_path):
            analysis = analyze_media(upscaled_path, is_video=False, color_scheme=get_analysis_color())
            return pending_status, analysis
        
        upscale_btn.click(
            fn=upscale_image_start,
            outputs=[upscale_status, upscale_output_tabs]
        ).then(
            fn=upscale_image_main,
            inputs=[
                upscale_input_image,
                upscale_seed,
                upscale_randomize_seed,
                upscale_resolution,
                upscale_max_resolution,
            ] + upscale_common_inputs,
            outputs=[upscale_slider, upscale_seed, upscale_pending_status, upscale_result_path, upscale_original_path, upscale_result_resolution]
        ).then(
            fn=upscale_image_finalize,
            inputs=[upscale_pending_status, upscale_result_path],
            outputs=[upscale_status, output_image_analysis]
        )
        
        # Video export inputs (before common inputs)
        upscale_video_export_inputs = [
            upscale_video_format,
            upscale_video_crf,
            upscale_video_pix_fmt,
            upscale_prores_profile,
            upscale_save_png_sequence,
            upscale_save_to_comfyui,
            upscale_video_filename,
        ]
        
        # Video Upscale - initial handler sets status message and switches tab immediately
        def upscale_video_start():
            return "⏳ Upscaling video... (check terminal for details)", gr.Tabs(selected="upscale_video_result")
        
        # Video Upscale - main function returns primary outputs (status goes to State, not Textbox)
        async def upscale_video_main(
            input_video, seed, randomize_seed, resolution,
            video_format, video_crf, video_pix_fmt, prores_profile,
            save_png_sequence, save_to_comfyui, filename_prefix,
            dit_model, blocks_to_swap, attention_mode,
            encode_tiled, encode_tile_size, encode_tile_overlap,
            decode_tiled, decode_tile_size, decode_tile_overlap,
            batch_size, uniform_batch, color_correction, temporal_overlap,
            input_noise, latent_noise
        ):
            result = await upscale_video(
                services, input_video, seed, randomize_seed, resolution,
                video_format, video_crf, video_pix_fmt, prores_profile,
                save_png_sequence, save_to_comfyui, filename_prefix,
                dit_model, blocks_to_swap, attention_mode,
                encode_tiled, encode_tile_size, encode_tile_overlap,
                decode_tiled, decode_tile_size, decode_tile_overlap,
                batch_size, uniform_batch, color_correction, temporal_overlap,
                input_noise, latent_noise
            )
            # result = (video_path, status, seed, output_path)
            # Return: video, seed, status->State, hidden state
            return result[0], result[2], result[1], result[3]
        
        # Secondary handler for status, analysis (no progress spinner on these)
        def upscale_video_finalize(pending_status, output_path):
            analysis = analyze_media(output_path, is_video=True, color_scheme=get_analysis_color())
            return pending_status, analysis
        
        upscale_video_btn.click(
            fn=upscale_video_start,
            outputs=[upscale_status, upscale_output_tabs]
        ).then(
            fn=upscale_video_main,
            inputs=[
                upscale_input_video,
                upscale_seed,
                upscale_randomize_seed,
                upscale_video_resolution,
            ] + upscale_video_export_inputs + upscale_video_common_inputs,
            outputs=[upscale_output_video, upscale_seed, upscale_video_pending_status, upscale_video_result_path]
        ).then(
            fn=upscale_video_finalize,
            inputs=[upscale_video_pending_status, upscale_video_result_path],
            outputs=[upscale_status, output_video_analysis]
        )
        
        # Video format change handler - show/hide format-specific options
        def on_video_format_change(format_choice):
            is_prores = "ProRes" in format_choice
            is_h265 = "H.265" in format_choice
            # CRF default: 19 for H.264, 22 for H.265
            crf_default = 22 if is_h265 else 19
            return (
                gr.update(visible=not is_prores),  # CRF slider
                gr.update(visible=not is_prores),  # Pixel format
                gr.update(visible=is_prores),      # ProRes profile
                gr.update(value=crf_default) if not is_prores else gr.update(),  # Update CRF default
            )
        
        upscale_video_format.change(
            fn=on_video_format_change,
            inputs=[upscale_video_format],
            outputs=[upscale_video_crf, upscale_video_pix_fmt, upscale_prores_profile, upscale_video_crf]
        )
        
        # Video resolution quick buttons
        upscale_video_res_720_btn.click(fn=lambda: 720, outputs=[upscale_video_resolution])
        upscale_video_res_1080_btn.click(fn=lambda: 1080, outputs=[upscale_video_resolution])
        
        # Save upscaled image
        def save_upscaled_image(image_path, original_path, resolution):
            if not image_path:
                return "❌ No image to save"
            saved_path = save_upscale_to_outputs(image_path, original_path, resolution or 4096, outputs_dir)
            return f"✓ Saved: {Path(saved_path).name}"
        
        upscale_save_btn.click(
            fn=save_upscaled_image,
            inputs=[upscale_result_path, upscale_original_path, upscale_result_resolution],
            outputs=[upscale_status]
        )
        
        # Open folder helpers
        def open_outputs_folder():
            open_folder(outputs_dir / "upscaled")
        
        def open_comfyui_folder():
            open_folder(comfyui_output_dir)
        
        upscale_open_folder_btn.click(fn=open_outputs_folder, outputs=[])
        open_comfyui_output_btn.click(fn=open_comfyui_folder, outputs=[])
        
        # Register as an image receiver for inter-module transfers
        services.inter_module.image_transfer.register_receiver(
            tab_id=TAB_ID,
            label=TAB_LABEL,
            input_component=upscale_input_image,
            status_component=upscale_status
        )
        
        # Fallback: check for pending images when tab is selected
        tab.select(
            fn=services.inter_module.image_transfer.create_tab_select_handler(TAB_ID),
            outputs=[upscale_input_image, upscale_status]
        )
        
        # System Monitor
        def update_monitor():
            if services.system_monitor:
                gpu_info, cpu_info = services.system_monitor.get_system_info()
                return gpu_info, cpu_info
            return "N/A", "N/A"
        
        monitor_timer = gr.Timer(2, active=True)
        monitor_timer.tick(fn=update_monitor, outputs=[gpu_monitor, cpu_monitor])
    
    return tab
