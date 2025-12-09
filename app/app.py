"""
Z-Image Turbo - Comfy-Gradio App

Fast, high-quality image generation using the Z-Image 6B turbo model.
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

import gradio as gr
import httpx
import webbrowser
from comfykit import ComfyKit
from huggingface_hub import hf_hub_download

# Custom modules
from modules.prompt_assistant import PromptAssistant
from modules.system_monitor import SystemMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Paths
APP_DIR = Path(__file__).parent
UI_SETTINGS_FILE = APP_DIR / "ui_settings.json"


def load_ui_settings() -> dict:
    """Load UI settings from JSON file."""
    import json
    if UI_SETTINGS_FILE.exists():
        try:
            with open(UI_SETTINGS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load UI settings: {e}")
    return {}


def save_ui_settings(settings: dict) -> None:
    """Save UI settings to JSON file."""
    import json
    try:
        with open(UI_SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save UI settings: {e}")
OUTPUTS_DIR = APP_DIR / "outputs" / "z-image-fusion"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
MODULES_DIR = APP_DIR / "modules"
CAMERA_PROMPTS_DIR = APP_DIR / "CameraPromptsGenerator"

# Initialize Prompt Assistant
prompt_assistant = PromptAssistant(
    settings_file=str(MODULES_DIR / "llm_settings.json"),
    ckpt_dir=str(MODULES_DIR / "llm_ckpts")
)

# Model folders (directly in ComfyUI)
MODELS_DIR = APP_DIR / "comfyui" / "models"
DIFFUSION_DIR = MODELS_DIR / "diffusion_models"
TEXT_ENCODERS_DIR = MODELS_DIR / "text_encoders"
VAE_DIR = MODELS_DIR / "vae"
LORAS_DIR = MODELS_DIR / "loras"

# Initialize ComfyKit client
try:
    kit = ComfyKit()
    logger.info(f"ComfyKit initialized: {kit.comfyui_url}")
except Exception as e:
    logger.error(f"Failed to initialize ComfyKit: {e}")
    kit = None

# Fallback sampler/scheduler lists (used if ComfyUI not available)
FALLBACK_SAMPLERS = ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde", "res_multistep"]
FALLBACK_SCHEDULERS = ["simple", "normal", "karras", "exponential", "sgm_uniform"]

# Preferred defaults (will be selected if available)
PREFERRED_SAMPLER = "euler"
PREFERRED_SCHEDULER = "simple"


def fetch_comfyui_options() -> dict:
    """Fetch available samplers and schedulers from ComfyUI's object_info API."""
    result = {
        "samplers": FALLBACK_SAMPLERS.copy(),
        "schedulers": FALLBACK_SCHEDULERS.copy()
    }
    
    if kit is None:
        return result
    
    try:
        with httpx.Client(timeout=5) as client:
            response = client.get(f"{kit.comfyui_url}/object_info/KSampler")
            if response.status_code == 200:
                data = response.json()
                ksampler_info = data.get("KSampler", {}).get("input", {}).get("required", {})
                
                # Extract sampler names
                sampler_info = ksampler_info.get("sampler_name", [])
                if sampler_info and isinstance(sampler_info[0], list):
                    result["samplers"] = sampler_info[0]
                
                # Extract scheduler names
                scheduler_info = ksampler_info.get("scheduler", [])
                if scheduler_info and isinstance(scheduler_info[0], list):
                    result["schedulers"] = scheduler_info[0]
                    
                logger.info(f"Loaded {len(result['samplers'])} samplers, {len(result['schedulers'])} schedulers from ComfyUI")
    except Exception as e:
        logger.warning(f"Could not fetch ComfyUI options, using fallbacks: {e}")
    
    return result

# Default models
DEFAULT_DIFFUSION = "z_image_turbo_bf16.safetensors"
DEFAULT_DIFFUSION_GGUF = "z_image_turbo-Q4_K_M.gguf"
DEFAULT_TEXT_ENCODER = "qwen_3_4b.safetensors"
DEFAULT_TEXT_ENCODER_GGUF = "Qwen3-4B-Q6_K.gguf"
DEFAULT_VAE = "ae.safetensors"

# File extensions by mode
STANDARD_EXTENSIONS = (".safetensors", ".ckpt", ".pt")
GGUF_EXTENSIONS = (".gguf",)

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


def scan_models(folder: Path, extensions: tuple = (".safetensors", ".ckpt", ".pt", ".gguf"), name_filter: str = None) -> list:
    """Scan folder recursively for model files, returning relative paths.
    
    Args:
        folder: Directory to scan
        extensions: File extensions to include
        name_filter: Optional case-insensitive substring to filter by (e.g. "z-image", "qwen")
    """
    if not folder.exists():
        return []
    models = []
    for ext in extensions:
        for f in folder.rglob(f"*{ext}"):
            rel_path = str(f.relative_to(folder))
            # Apply name filter if specified
            if name_filter is None or name_filter.lower() in rel_path.lower():
                models.append(rel_path)
    return sorted(models)


# Name filters for Z-Image compatible models
ZIMAGE_FILTERS = {
    "diffusion": "z-image",  # z-image, z_image variants
    "text_encoder": "qwen",  # Qwen3 text encoder
    "vae": "ae",             # ae.safetensors
}


def get_models_by_mode(folder: Path, is_gguf: bool, default_standard: str, default_gguf: str, name_filter: str = None) -> list:
    """Get models filtered by mode (standard vs GGUF) and optional name filter."""
    extensions = GGUF_EXTENSIONS if is_gguf else STANDARD_EXTENSIONS
    default = default_gguf if is_gguf else default_standard
    models = scan_models(folder, extensions, name_filter)
    return models or [default]


def get_model_choices():
    """Get available models from local folders (all extensions)."""
    return {
        "diffusion": scan_models(DIFFUSION_DIR) or [DEFAULT_DIFFUSION],
        "text_encoder": scan_models(TEXT_ENCODERS_DIR) or [DEFAULT_TEXT_ENCODER],
        "vae": scan_models(VAE_DIR, (".safetensors",)) or [DEFAULT_VAE],
        "lora": scan_models(LORAS_DIR, (".safetensors",)) or [],
    }


def get_default_model(choices: list, preferred: str) -> str:
    """Get default model, preferring the specified one if available."""
    if preferred in choices:
        return preferred
    return choices[0] if choices else None


def check_models_installed() -> dict:
    """Check which model sets are installed and return status info."""
    # Check for Z-Image compatible models
    standard_diffusion = scan_models(DIFFUSION_DIR, STANDARD_EXTENSIONS, ZIMAGE_FILTERS["diffusion"])
    standard_te = scan_models(TEXT_ENCODERS_DIR, STANDARD_EXTENSIONS, ZIMAGE_FILTERS["text_encoder"])
    gguf_diffusion = scan_models(DIFFUSION_DIR, GGUF_EXTENSIONS, ZIMAGE_FILTERS["diffusion"])
    gguf_te = scan_models(TEXT_ENCODERS_DIR, GGUF_EXTENSIONS, ZIMAGE_FILTERS["text_encoder"])
    vae = scan_models(VAE_DIR, (".safetensors",), ZIMAGE_FILTERS["vae"])
    
    has_standard = bool(standard_diffusion and standard_te and vae)
    has_gguf = bool(gguf_diffusion and gguf_te and vae)
    
    return {
        "has_standard": has_standard,
        "has_gguf": has_gguf,
        "has_any": has_standard or has_gguf,
        "recommended_mode": has_gguf and not has_standard,  # True = GGUF, False = Standard
    }


def validate_selected_models(unet_name: str, clip_name: str, vae_name: str) -> tuple[bool, str]:
    """Validate that selected model files actually exist. Returns (valid, error_message)."""
    unet_path = DIFFUSION_DIR / unet_name
    clip_path = TEXT_ENCODERS_DIR / clip_name
    vae_path = VAE_DIR / vae_name
    
    missing = []
    if not unet_path.exists():
        missing.append(f"Diffusion: {unet_name}")
    if not clip_path.exists():
        missing.append(f"Text Encoder: {clip_name}")
    if not vae_path.exists():
        missing.append(f"VAE: {vae_name}")
    
    if missing:
        return False, "Missing models:\n• " + "\n• ".join(missing) + "\n\nDownload models in the 🔧 Models section."
    return True, ""


async def download_image_from_url(url: str) -> str:
    """Download image from ComfyUI URL to a local temp file."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        suffix = Path(url).suffix or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(response.content)
            return f.name


def save_image_to_outputs(image_path: str, prompt: str, subfolder: str = None) -> str:
    """Save image to outputs folder with prompt and short timestamp."""
    timestamp = datetime.now().strftime("%H%M%S")
    safe_prompt = "".join(c if c.isalnum() or c in " -_" else "" for c in prompt[:30]).strip()
    safe_prompt = safe_prompt.replace(" ", "_") if safe_prompt else "image"
    
    # Use subfolder if specified
    target_dir = OUTPUTS_DIR / subfolder if subfolder else OUTPUTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Format: prompt_HHMMSS.png (timestamp at end, less obtrusive)
    filename = f"{safe_prompt}_{timestamp}.png"
    output_path = target_dir / filename
    shutil.copy2(image_path, output_path)
    logger.info(f"Saved to: {output_path}")
    return str(output_path)


def extract_meaningful_filename(filepath: str) -> str:
    """Extract a meaningful filename, filtering out temp file patterns."""
    if not filepath:
        return "image"
    
    stem = Path(filepath).stem
    
    # Detect Gradio/system temp file patterns (tmp*, random hex strings, etc.)
    # These typically start with 'tmp' or are short random strings
    is_temp = (
        stem.lower().startswith('tmp') or
        stem.lower().startswith('temp') or
        (len(stem) < 12 and not any(c.isalpha() for c in stem[:3]))  # Short random strings
    )
    
    if is_temp:
        return "image"
    
    # Truncate if too long
    if len(stem) > 50:
        stem = stem[:50]
    
    return stem


def save_upscale_to_outputs(image_path: str, original_path: str, resolution: int, subfolder: str = "upscaled") -> str:
    """Save upscaled image preserving original name with upscale details."""
    timestamp = datetime.now().strftime("%H%M%S")
    
    # Extract meaningful filename, filtering out temp patterns
    original_name = extract_meaningful_filename(original_path)
    
    target_dir = OUTPUTS_DIR / subfolder
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Format: originalname_4Kup_HHMMSS.png
    res_label = f"{resolution // 1000}K" if resolution >= 1000 else f"{resolution}p"
    filename = f"{original_name}_{res_label}up_{timestamp}.png"
    output_path = target_dir / filename
    shutil.copy2(image_path, output_path)
    logger.info(f"Saved upscale to: {output_path}")
    return str(output_path)


def new_random_seed():
    """Generate a new random seed."""
    return random.randint(0, 999999999999)


def new_random_seed_32bit():
    """Generate a new random seed (32-bit max for SeedVR2)."""
    return random.randint(0, 4294967295)


def get_workflow_file(gen_type: str, use_lora: bool, use_gguf: bool) -> str:
    """Determine which workflow file to use based on settings."""
    parts = ["z_image"]
    if use_gguf:
        parts.append("gguf")
    parts.append(gen_type)
    if use_lora:
        parts.append("lora")
    return "_".join(parts) + ".json"


async def generate_image(
    prompt: str,
    gen_type: str,
    use_gguf: bool,
    # t2i params
    width: int, height: int,
    # i2i params
    input_image, megapixels: float, denoise: float,
    # common params
    steps: int, seed: int, randomize_seed: bool,
    cfg: float, shift: float,
    sampler_name: str, scheduler: str,
    # model params
    unet_name: str, clip_name: str, vae_name: str,
    # lora params
    use_lora: bool, lora_name: str, lora_strength: float,
    # output params
    autosave: bool
):
    """Generate an image using the selected workflow. Yields (image, status, seed) tuples."""
    # Handle seed early so we can yield it immediately
    actual_seed = new_random_seed() if randomize_seed else int(seed)
    
    # Validate models exist before attempting generation
    models_valid, error_msg = validate_selected_models(unet_name, clip_name, vae_name)
    if not models_valid:
        yield None, f"❌ {error_msg}", actual_seed
        return
    
    # Validate i2i has input image
    if gen_type == "i2i" and input_image is None:
        yield None, "❌ Please upload an input image for Image→Image", actual_seed
        return
    
    # Select workflow
    workflow_file = get_workflow_file(gen_type, use_lora, use_gguf)
    workflow_path = APP_DIR / "workflows" / workflow_file
    
    if not workflow_path.exists():
        yield None, f"❌ Workflow not found: {workflow_file}", actual_seed
        return
    
    # Allow empty prompt - model will generate without guidance
    prompt_text = prompt.strip() if prompt else ""
    
    logger.info(f"Using workflow: {workflow_file}")
    logger.info(f"Generating: '{prompt_text[:50]}{'...' if len(prompt_text) > 50 else ''}', seed={actual_seed}")
    
    # Yield initial state - shows "Generating..." status, locks in seed, clears spinner on seed/status
    yield None, "⏳ Generating...", actual_seed
    
    try:
        # Build params dict
        params = {
            "prompt": prompt_text,
            "steps": int(steps),
            "seed": actual_seed,
            "cfg": cfg,
            "shift": shift,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "unet_name": unet_name,
            "clip_name": clip_name,
            "vae_name": vae_name,
        }
        
        # Add type-specific params
        if gen_type == "t2i":
            params["width"] = int(width)
            params["height"] = int(height)
        else:  # i2i
            params["image"] = input_image
            params["megapixels"] = megapixels
            params["denoise"] = denoise
        
        # Add lora params if enabled
        if use_lora and lora_name:
            params["lora_name"] = lora_name
            params["lora_strength"] = lora_strength
        
        # Execute workflow
        result = await kit.execute(str(workflow_path), params)
        
        if result.status == "error":
            yield None, f"❌ Generation failed: {result.msg}", actual_seed
            return
        
        if not result.images:
            yield None, "❌ No images generated", actual_seed
            return
        
        # Get image
        image_path = result.images[0]
        if image_path.startswith("http"):
            image_path = await download_image_from_url(image_path)
        
        # Autosave
        if autosave:
            save_image_to_outputs(image_path, prompt_text or "image")
            status = f"✓ {result.duration:.1f}s | Saved" if result.duration else "✓ Saved"
        else:
            status = f"✓ {result.duration:.1f}s" if result.duration else "✓ Done"
        
        yield image_path, status, actual_seed
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        if "connect" in str(e).lower():
            yield None, "❌ Cannot connect to ComfyUI", actual_seed
        else:
            yield None, f"❌ {str(e)}", actual_seed


def get_seedvr2_max_blocks(dit_model: str) -> int:
    """Get max block swap value based on model size (3B=32, 7B=36)."""
    return 32 if "3b" in dit_model.lower() else 36


async def upscale_image(
    input_image,
    seed: int,
    randomize_seed: bool,
    resolution: int,
    max_resolution: int,
    dit_model: str,
    blocks_to_swap: int,
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
    try:
        if input_image is None:
            return None, "❌ Please upload an image to upscale", seed, None, None, None
        
        # SeedVR2 uses 32-bit seed max (4294967295)
        actual_seed = new_random_seed_32bit() if randomize_seed else min(int(seed), 4294967295)
        
        workflow_path = APP_DIR / "workflows" / "SeedVR2_4K_image_upscale.json"
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
        
        result = await kit.execute(str(workflow_path), params)
        
        if result.status == "error":
            return None, f"❌ Upscale failed: {result.msg}", actual_seed, None, None, None
        
        if not result.images:
            return None, "❌ No images generated", actual_seed, None, None, None
        
        image_path = result.images[0]
        if image_path.startswith("http"):
            image_path = await download_image_from_url(image_path)
        
        # Autosave
        if autosave:
            save_upscale_to_outputs(image_path, input_image, resolution)
            status = f"✓ {result.duration:.1f}s | Saved" if result.duration else "✓ Saved"
        else:
            status = f"✓ {result.duration:.1f}s" if result.duration else "✓ Done"
        
        # Return tuple for ImageSlider (original, upscaled) + upscaled path for save button
        # Also return input_image path and resolution for manual save
        return (input_image, image_path), status, actual_seed, image_path, input_image, resolution
        
    except Exception as e:
        logger.error(f"Upscale error: {e}", exc_info=True)
        if "connect" in str(e).lower():
            return None, "❌ Cannot connect to ComfyUI", seed, None, None, None
        return None, f"❌ {str(e)}", seed, None, None, None


async def upscale_video(
    input_video,
    seed: int,
    randomize_seed: bool,
    resolution: int,
    dit_model: str,
    blocks_to_swap: int,
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
    """Upscale a video using SeedVR2. Returns (video_path, status, seed)."""
    try:
        if input_video is None:
            return None, "❌ Please upload a video to upscale", seed
        
        # SeedVR2 uses 32-bit seed max (4294967295)
        actual_seed = new_random_seed_32bit() if randomize_seed else min(int(seed), 4294967295)
        
        workflow_path = APP_DIR / "workflows" / "SeedVR2_HD_video_upscale.json"
        if not workflow_path.exists():
            return None, "❌ Video upscale workflow not found", seed
        
        logger.info(f"Upscaling video with SeedVR2: {dit_model}, res={resolution}")
        
        params = {
            "video": input_video,
            "seed": actual_seed,
            "resolution": int(resolution),
            "dit_model": dit_model,
            "blocks_to_swap": int(blocks_to_swap),
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
        
        result = await kit.execute(str(workflow_path), params)
        
        if result.status == "error":
            return None, f"❌ Video upscale failed: {result.msg}", actual_seed
        
        if not result.videos:
            return None, "❌ No video generated", actual_seed
        
        video_path = result.videos[0]
        if video_path.startswith("http"):
            # Download video from ComfyUI
            async with httpx.AsyncClient() as client:
                response = await client.get(video_path)
                response.raise_for_status()
                suffix = Path(video_path).suffix or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                    f.write(response.content)
                    video_path = f.name
        
        # Autosave
        if autosave:
            save_video_to_outputs(video_path, input_video, resolution)
            status = f"✓ {result.duration:.1f}s | Saved" if result.duration else "✓ Saved"
        else:
            status = f"✓ {result.duration:.1f}s" if result.duration else "✓ Done"
        
        return video_path, status, actual_seed
        
    except Exception as e:
        logger.error(f"Video upscale error: {e}", exc_info=True)
        if "connect" in str(e).lower():
            return None, "❌ Cannot connect to ComfyUI", seed
        return None, f"❌ {str(e)}", seed


def save_video_to_outputs(video_path: str, original_path: str = None, resolution: int = None) -> str:
    """Save upscaled video preserving original name with upscale details."""
    timestamp = datetime.now().strftime("%H%M%S")
    target_dir = OUTPUTS_DIR / "upscaled"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract meaningful filename, filtering out temp patterns
    original_name = extract_meaningful_filename(original_path)
    if original_name == "image":
        original_name = "video"  # Better default for videos
    
    suffix = Path(video_path).suffix or ".mp4"
    res_label = f"{resolution}p" if resolution else "HDup"
    filename = f"{original_name}_{res_label}_{timestamp}{suffix}"
    output_path = target_dir / filename
    shutil.copy2(video_path, output_path)
    logger.info(f"Saved video to: {output_path}")
    return str(output_path)


async def unload_models() -> str:
    """Unload all models from ComfyUI to free VRAM."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{kit.comfyui_url}/free",
                json={"unload_models": True, "free_memory": True}
            )
            if response.status_code == 200:
                return "✓ ComfyUI models unloaded, VRAM freed"
            return f"❌ Failed: {response.status_code}"
    except Exception as e:
        return f"❌ Error: {e}"


# Model definitions for huggingface_hub downloads
MODEL_DOWNLOADS = {
    "diffusion_bf16": {
        "repo_id": "Comfy-Org/z_image_turbo",
        "filename": "split_files/diffusion_models/z_image_turbo_bf16.safetensors",
        "local_name": "z_image_turbo_bf16.safetensors",
        "folder": DIFFUSION_DIR,
        "label": "Diffusion (bf16)",
    },
    "te_bf16": {
        "repo_id": "Comfy-Org/z_image_turbo",
        "filename": "split_files/text_encoders/qwen_3_4b.safetensors",
        "local_name": "qwen_3_4b.safetensors",
        "folder": TEXT_ENCODERS_DIR,
        "label": "Text Encoder (bf16)",
    },
    "vae": {
        "repo_id": "Comfy-Org/z_image_turbo",
        "filename": "split_files/vae/ae.safetensors",
        "local_name": "ae.safetensors",
        "folder": VAE_DIR,
        "label": "VAE",
    },
    "diffusion_gguf": {
        "repo_id": "gguf-org/z-image-gguf",
        "filename": "z-image-turbo-q4_k_m.gguf",
        "local_name": "z-image-turbo-q4_k_m.gguf",
        "folder": DIFFUSION_DIR,
        "label": "Diffusion (Q4 GGUF)",
    },
    "te_gguf": {
        "repo_id": "Qwen/Qwen3-4B-GGUF",
        "filename": "Qwen3-4B-Q4_K_M.gguf",
        "local_name": "Qwen3-4B-Q4_K_M.gguf",
        "folder": TEXT_ENCODERS_DIR,
        "label": "Qwen3 TE (Q4 GGUF)",
    },
}


def download_model(model_key: str, progress=gr.Progress()):
    """Download a model from HuggingFace Hub to the appropriate folder."""
    if model_key not in MODEL_DOWNLOADS:
        return f"❌ Unknown model: {model_key}"
    
    info = MODEL_DOWNLOADS[model_key]
    repo_id = info["repo_id"]
    filename = info["filename"]
    local_name = info["local_name"]
    folder = info["folder"]
    label = info["label"]
    dest_path = folder / local_name
    
    # Check if already exists
    if dest_path.exists():
        return f"✓ {label} already installed"
    
    # Ensure folder exists
    folder.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"Downloading {label}: {repo_id}/{filename}")
        progress(0, desc=f"Downloading {label}...")
        
        # Download using huggingface_hub (handles caching, resume, xet acceleration)
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=folder,
            local_dir_use_symlinks=False,
        )
        
        # If HF downloaded to a subfolder structure, move to expected location
        downloaded_path = Path(downloaded_path)
        if downloaded_path != dest_path and downloaded_path.exists():
            shutil.move(str(downloaded_path), str(dest_path))
            # Clean up any empty parent dirs created by HF
            for parent in downloaded_path.parents:
                if parent != folder and parent.is_dir() and not any(parent.iterdir()):
                    parent.rmdir()
        
        logger.info(f"Downloaded: {dest_path}")
        progress(1, desc="Done")
        return f"✓ {label} downloaded successfully"
        
    except Exception as e:
        logger.error(f"Download failed for {label}: {e}")
        return f"❌ {label} download failed: {e}"


def download_all_standard(progress=gr.Progress()):
    """Download all standard (bf16) models."""
    results = []
    keys = ["diffusion_bf16", "te_bf16", "vae"]
    for i, key in enumerate(keys):
        progress((i / len(keys)), desc=f"Downloading {MODEL_DOWNLOADS[key]['label']}...")
        result = download_model(key, progress)
        results.append(result)
    progress(1, desc="Done")
    return "\n".join(results)


def download_all_gguf(progress=gr.Progress()):
    """Download all GGUF (Q4) models + VAE."""
    results = []
    keys = ["diffusion_gguf", "te_gguf", "vae"]
    for i, key in enumerate(keys):
        progress((i / len(keys)), desc=f"Downloading {MODEL_DOWNLOADS[key]['label']}...")
        result = download_model(key, progress)
        results.append(result)
    progress(1, desc="Done")
    return "\n".join(results)


def get_saved_model_defaults() -> dict | None:
    """Load saved model defaults from ui_settings.json if they exist."""
    settings = load_ui_settings()
    return settings.get("model_defaults")


def create_interface() -> gr.Blocks:
    """Create the Gradio interface."""
    models = get_model_choices()
    comfyui_options = fetch_comfyui_options()
    samplers = comfyui_options["samplers"]
    schedulers = comfyui_options["schedulers"]
    
    # Check installed models to determine startup state
    model_status = check_models_installed()
    show_setup_banner = not model_status["has_any"]
    
    # Load saved model defaults, or fall back to auto-detection
    saved_defaults = get_saved_model_defaults()
    if saved_defaults:
        # Use saved preferences
        default_gguf_mode = saved_defaults.get("use_gguf", model_status["recommended_mode"])
        default_diffusion = saved_defaults.get("diffusion")
        default_te = saved_defaults.get("text_encoder")
        default_vae = saved_defaults.get("vae")
    else:
        # Auto-detect based on installed models
        default_gguf_mode = model_status["recommended_mode"]
        default_diffusion = None
        default_te = None
        default_vae = None
    
    # CSS fix for textarea scrollbars not appearing on initial content
    css = """
    textarea {
        overflow-y: auto !important;
        resize: vertical;
    }
    .setup-banner {
        background: linear-gradient(90deg, #ff6b35, #f7931e);
        color: white;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 12px;
        font-weight: 500;
    }

    /* Enhanced Monitor Textboxes */
    .monitor-box {
        min-width: 0 !important;
    }
    .monitor-box textarea {
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace !important;
        font-size: 0.85em !important;
        line-height: 1.6 !important;
        padding: 12px !important;
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%) !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06) !important;
        resize: none !important;
        font-weight: 500 !important;
        overflow-y: hidden !important;  /* Override global textarea scroll */
    }
    .gpu-monitor textarea {
        border-left: 3px solid #667eea !important;
        background: linear-gradient(135deg, #667eea08 0%, #ffffff 100%) !important;
    }
    .cpu-monitor textarea {
        border-left: 3px solid #f5576c !important;
        background: linear-gradient(135deg, #f5576c08 0%, #ffffff 100%) !important;
    }
    .monitor-box textarea:focus {
        outline: none !important;
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .monitor-box textarea {
            background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%) !important;
            border-color: #4a5568 !important;
            color: #e2e8f0 !important;
        }
        .gpu-monitor textarea {
            background: linear-gradient(135deg, #667eea15 0%, #2d3748 100%) !important;
        }
        .cpu-monitor textarea {
            background: linear-gradient(135deg, #f5576c15 0%, #2d3748 100%) !important;
        }
    }    
    """
    
    with gr.Blocks(title="Z-Image Turbo", css=css) as interface:
        
        # Main tabs for different tools
        with gr.Tabs() as main_tabs:
            # ===== Z-IMAGE GENERATION TAB =====
            with gr.TabItem("⚡ Z-Image Turbo", id="tab_generate"):
                # Setup banner - shown when no models installed
                setup_banner = gr.Markdown(
                    "⚠️ **Setup Required** — Download models in the **🔧 Models** section below to get started!",
                    visible=show_setup_banner,
                    elem_classes=["setup-banner"]
                )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe your image...",
                            lines=3
                        )
                        
                        # Generation type tabs
                        with gr.Tabs():
                            with gr.TabItem("Text → Image"):
                                generate_t2i_btn = gr.Button("⚡ Generate", variant="primary", size="lg")
                                with gr.Row():
                                    width = gr.Slider(label="Width", value=1024, minimum=512, maximum=2048, step=64)
                                    height = gr.Slider(label="Height", value=1024, minimum=512, maximum=2048, step=64)
                            
                            with gr.TabItem("Image → Image"):
                                input_image = gr.Image(label="Input Image", type="filepath", height=360)
                                with gr.Row():
                                    generate_i2i_btn = gr.Button("⚡ Generate", variant="primary", size="lg", scale=2)
                                    i2i_describe_btn = gr.Button("🖼️ Describe", variant="secondary", size="lg", scale=1)
                                with gr.Row():                                    
                                    i2i_assist_status = gr.Textbox(
                                        value="💡 Tip: Use Describe to generate a prompt describing your image. A low denoise img2img pass can greatly enhance existing images. Add a character LoRA for powerful transformations!",
                                        lines=2.5,
                                        interactive=False,
                                        show_label=False
                                    )
                                with gr.Row():
                                    megapixels = gr.Slider(label="Megapixels", info="Scales against input image to maintain aspect ratio", value=2, minimum=0.5, maximum=3.0, step=0.1)
                                    denoise = gr.Slider(label="Denoise", value=0.67, minimum=0.0, maximum=1.0, step=0.01)
                            
                            with gr.TabItem("🤖 Prompt Assistant"):
                                # Mode selection
                                assist_mode = gr.Radio(
                                    choices=["Enhance Text", "Describe Image"],
                                    value="Enhance Text",
                                    label="Mode",
                                    interactive=True
                                )
                                # Image input for describe mode (hidden by default)
                                assist_image = gr.Image(
                                    label="Image to Describe",
                                    type="filepath",
                                    height=200,
                                    visible=False
                                )
                                with gr.Row():
                                    enhance_btn = gr.Button("✨ Enhance", variant="primary", size="lg", scale=2)
                                    describe_btn = gr.Button("🖼️ Describe", variant="secondary", size="lg", scale=2, visible=False)
                                    clear_prompt_btn = gr.Button("🗑️ Clear", size="lg", scale=1)
                                assist_status = gr.Textbox(label="Status", value="Ready", interactive=False, show_label=False)

                        
                        # Generation settings - compact group
                        
                        with gr.Accordion("Settings"):
                            with gr.Group():
                                with gr.Row():
                                    steps = gr.Slider(label="Steps", value=9, minimum=1, maximum=50, step=1)
                                    cfg = gr.Slider(label="CFG", value=1.0, minimum=1.0, maximum=10.0, step=0.5)
                                    shift = gr.Slider(label="Shift", value=3.0, minimum=1.0, maximum=10.0, step=0.5)
                            with gr.Group():
                                with gr.Row():
                                    sampler_name = gr.Dropdown(label="Sampler", choices=samplers, value=PREFERRED_SAMPLER if PREFERRED_SAMPLER in samplers else samplers[0], scale=2)
                                    scheduler = gr.Dropdown(label="Scheduler", choices=schedulers, value=PREFERRED_SCHEDULER if PREFERRED_SCHEDULER in schedulers else schedulers[0], scale=2)
                            with gr.Row():                                   
                                seed = gr.Number(label="Seed", value=new_random_seed(), minimum=0, step=1, scale=3)
                                randomize_seed = gr.Checkbox(label="🎲", value=True, scale=1)
                        
                        # LoRA settings
                        with gr.Accordion("🎨 LoRA", open=False):
                            with gr.Row():
                                use_lora = gr.Checkbox(label="Enable", value=False, scale=0, min_width=80)
                                lora_name = gr.Dropdown(
                                    label="Model",
                                    show_label=False,
                                    container=False,
                                    choices=models["lora"],
                                    value=None,
                                    interactive=True,
                                    scale=3,
                                    allow_custom_value=True
                                )
                                refresh_loras_btn = gr.Button("🔄", size="sm", scale=0, min_width=40)
                            with gr.Row():                              
                                lora_strength = gr.Slider(label="Strength", value=1.0, minimum=0.0, maximum=2.0, step=0.05, scale=2)
                            with gr.Row():
                                open_loras_btn = gr.Button("📂 Open LoRAs Folder", size="sm")
                            gr.Markdown("*[🔗 Browse CivitAI LoRAs](https://civitai.com/models) — Filter by 'Z-Image' (top-right)*")
                        
                        # Model selection - auto-open if setup needed
                        with gr.Accordion("🔧 Models", open=show_setup_banner):
                            with gr.Row():
                                use_gguf = gr.Radio(choices=[("Standard", False), ("GGUF", True)], value=default_gguf_mode, label="Mode", scale=1)
                                show_all_models = gr.Checkbox(label="Show all", value=False, scale=1, min_width=80, info="Show non-Z-Image models")
                            
                            # Compute initial values - use saved defaults if available, else auto-detect
                            initial_diffusion = get_models_by_mode(DIFFUSION_DIR, default_gguf_mode, DEFAULT_DIFFUSION, DEFAULT_DIFFUSION_GGUF, ZIMAGE_FILTERS["diffusion"])
                            initial_diffusion_value = default_diffusion if default_diffusion and default_diffusion in initial_diffusion else get_default_model(initial_diffusion, DEFAULT_DIFFUSION_GGUF if default_gguf_mode else DEFAULT_DIFFUSION)
                            
                            initial_te = get_models_by_mode(TEXT_ENCODERS_DIR, default_gguf_mode, DEFAULT_TEXT_ENCODER, DEFAULT_TEXT_ENCODER_GGUF, ZIMAGE_FILTERS["text_encoder"])
                            initial_te_value = default_te if default_te and default_te in initial_te else get_default_model(initial_te, DEFAULT_TEXT_ENCODER_GGUF if default_gguf_mode else DEFAULT_TEXT_ENCODER)
                            
                            initial_vae = scan_models(VAE_DIR, (".safetensors",), ZIMAGE_FILTERS["vae"]) or [DEFAULT_VAE]
                            initial_vae_value = default_vae if default_vae and default_vae in initial_vae else get_default_model(initial_vae, DEFAULT_VAE)
                            
                            # Check if initial selections exist on disk
                            def get_status_icon(folder: Path, filename: str) -> str:
                                return "✓" if (folder / filename).exists() else "❌"
                            
                            initial_diff_status = get_status_icon(DIFFUSION_DIR, initial_diffusion_value) if initial_diffusion_value else "❌"
                            initial_te_status = get_status_icon(TEXT_ENCODERS_DIR, initial_te_value) if initial_te_value else "❌"
                            initial_vae_status = get_status_icon(VAE_DIR, initial_vae_value) if initial_vae_value else "❌"
                            
                            with gr.Row():
                                unet_name = gr.Dropdown(label="Diffusion Model", choices=initial_diffusion, value=initial_diffusion_value, scale=3)
                                unet_status = gr.Textbox(value=initial_diff_status, show_label=False, container=False, scale=0, min_width=30, interactive=False)
                            with gr.Row():
                                clip_name = gr.Dropdown(label="Text Encoder", choices=initial_te, value=initial_te_value, scale=3)
                                clip_status = gr.Textbox(value=initial_te_status, show_label=False, container=False, scale=0, min_width=30, interactive=False)
                            with gr.Row():
                                vae_name = gr.Dropdown(label="VAE", choices=initial_vae, value=initial_vae_value, scale=3)
                                vae_status = gr.Textbox(value=initial_vae_status, show_label=False, container=False, scale=0, min_width=30, interactive=False)
                            
                            with gr.Row():
                                save_model_defaults_btn = gr.Button("⭐ Save as default", size="sm")
                                model_defaults_status = gr.Textbox(value="", show_label=False, container=False, interactive=False, scale=2)
                            
                            # Model Management - auto-open if setup needed
                            with gr.Accordion("📦 Model Management", open=show_setup_banner):
                                with gr.Row():
                                    open_diffusion_btn = gr.Button("📂 diffusion_models", size="sm")
                                    open_te_btn = gr.Button("📂 text_encoders", size="sm")
                                    open_vae_btn = gr.Button("📂 vae", size="sm")
 
                                download_status = gr.Textbox(label="Status", interactive=False, show_label=False, lines=2) 
                                gr.Markdown("**=> Download Starter Models** *(check terminal for progress)*")
                                gr.Markdown("---")
                                gr.Markdown("*Standard (bf16) complete set — Full precision, ~20GB total*")
                                with gr.Row():
                                    dl_all_standard_btn = gr.Button("⬇️ Download All (bf16)", variant="primary", size="sm")
                                gr.Markdown("*Or download individually*")                                     
                                with gr.Row():
                                    dl_diffusion_bf16_btn = gr.Button("⬇️ Diffusion", size="sm")
                                    dl_te_bf16_btn = gr.Button("⬇️ Text Encoder", size="sm")
                                    dl_vae_btn = gr.Button("⬇️ VAE", size="sm")
                                gr.Markdown("---")

                                gr.Markdown("*GGUF (Q4) complete set — Low VRAM, ~7GB total*")
                                with gr.Row():
                                    dl_all_gguf_btn = gr.Button("⬇️ Download All (GGUF)", variant="primary", size="sm")
                                gr.Markdown("*Or download individually*")                         
                                with gr.Row():
                                    dl_diffusion_gguf_btn = gr.Button("⬇️ Diffusion Q4", size="sm")
                                    dl_te_gguf_btn = gr.Button("⬇️ Qwen3 TE Q4", size="sm")
                                    dl_vae_gguf_btn = gr.Button("⬇️ VAE", size="sm")
                                gr.Markdown("---")

                                gr.Markdown("*Advanced — Browse repos for other quants*")
                                with gr.Row():
                                    gr.Button("🔗 Z-Image GGUF Repo", size="sm", link="https://huggingface.co/gguf-org/z-image-gguf/tree/main")
                                    gr.Button("🔗 Qwen3 GGUF Repo", size="sm", link="https://huggingface.co/Qwen/Qwen3-4B-GGUF/tree/main")
                        
                    with gr.Column(scale=1):
                        output_image = gr.Image(label="Generated Image", type="filepath", interactive=False, height=512)
                        with gr.Row():
                            save_btn = gr.Button("💾 Save", size="sm")
                            send_to_upscale_btn = gr.Button("🔍 Send to Upscale", size="sm")
                        autosave = gr.Checkbox(label="Auto-save", value=False)
                        gen_status = gr.Textbox(label="Status", interactive=False, show_label=False)
                        open_folder_btn = gr.Button("📂 Open Output Folder", size="sm")
                       
                        with gr.Row():
                            with gr.Column(scale=1, min_width=200):                            
                                gpu_monitor = gr.Textbox(
                                    lines=4.5,
                                    container=False,
                                    interactive=False,
                                    show_label=False,
                                    elem_classes="monitor-box gpu-monitor"
                                )
                            with gr.Column(scale=1, min_width=200):
                                cpu_monitor = gr.Textbox(
                                    lines=4,
                                    container=False,
                                    interactive=False,
                                    show_label=False,
                                    elem_classes="monitor-box cpu-monitor"
                                )                            
                        unload_btn = gr.Button("🧹 Unload ComfyUI Models", size="sm")



                        with gr.Accordion("📸 Camera Prompts", open=False):
                            gr.Markdown("Visual guide to camera angles, shots & composition. *Opens in browser.*")
                            open_camera_prompts_btn = gr.Button("🔗 Open Reference Tool", size="sm")
                        
                        with gr.Accordion("ℹ️ Getting Started", open=False):
                            gr.Markdown("""
**First Time Setup**
1. Download models in **🔧 Models** section (left panel)
2. Choose **GGUF** for lower VRAM (8GB) or **Standard** for full precision (16GB+)
3. Click the download button — check Pinokio's `->_ Terminal` button (top bar) for progress

**Already have ComfyUI via Pinokio?**  
Your models & LoRAs are automatically shared — no re-download needed!

**🤖 Prompt Assistant**
- **Enhance Text**: Expands simple prompts into detailed descriptions
- **Describe Image**: Generates prompts from uploaded images (uses VL model)
- Defaults work great, but you can change LLMs in the **⚙️ LLM Settings** tab
- After changing models there, click **Save All Settings** to apply

**Tips**
- Default settings are tuned for the Z-Image Turbo model
- Use 🧹 **Unload ComfyUI Models** to keep Z-Image-Fusion active while freeing resources for other activities.
- Check the GPU/CPU monitor to track resource usage
""")
                            # open_docs_btn = gr.Button("📖 Open Documentation", size="sm")
            
            # ===== UPSCALE TAB =====
            with gr.TabItem("🔍 Upscale", id="tab_upscale"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Image/Video input tabs
                        with gr.Tabs() as upscale_input_tabs:
                            with gr.TabItem("🖼️ Image", id="upscale_image_tab"):
                                upscale_input_image = gr.Image(label="Input Image", type="filepath", height=300)
                                with gr.Row():
                                    upscale_resolution = gr.Slider(
                                        label="Resolution",
                                        value=3072,
                                        minimum=1024,
                                        maximum=8192,
                                        step=256,
                                        info="Target resolution"
                                    )
                                    upscale_max_resolution = gr.Slider(
                                        label="Max Resolution",
                                        value=4096,
                                        minimum=1024,
                                        maximum=8192,
                                        step=256,
                                        info="Maximum output resolution"
                                    )
                                upscale_btn = gr.Button("🔍 Upscale Image", variant="primary", size="lg")
                            
                            with gr.TabItem("🎬 Video", id="upscale_video_tab"):
                                upscale_input_video = gr.Video(label="Input Video", height=300)
                                upscale_video_resolution = gr.Slider(
                                    label="Resolution",
                                    value=1080,
                                    minimum=480,
                                    maximum=2160,
                                    step=120,
                                    info="Target resolution (height)"
                                )
                                upscale_video_btn = gr.Button("🎬 Upscale Video", variant="primary", size="lg")
                        
                        with gr.Accordion("🔧 SeedVR2 Settings", open=True):
                            upscale_dit_model = gr.Dropdown(
                                label="DIT Model",
                                choices=SEEDVR2_DIT_MODELS,
                                value=DEFAULT_SEEDVR2_DIT,
                                info="Models auto-download on first use"
                            )
                            upscale_blocks_to_swap = gr.Slider(
                                label="Block Swap",
                                value=36,
                                minimum=0,
                                maximum=36,
                                step=1,
                                info="Higher = less VRAM, slower. Lower = faster, more VRAM"
                            )
                        
                        with gr.Accordion("🎛️ Advanced Settings", open=False):
                            with gr.Row():
                                upscale_batch_size = gr.Slider(
                                    label="Batch Size",
                                    value=1,
                                    minimum=1,
                                    maximum=64,
                                    step=1,
                                    info="Frames per batch (video: ~33)"
                                )
                                upscale_uniform_batch = gr.Checkbox(
                                    label="Uniform Batch",
                                    value=False,
                                    info="Equal batch sizes"
                                )
                            with gr.Row():
                                upscale_color_correction = gr.Dropdown(
                                    label="Color Correction",
                                    choices=["none", "lab", "wavelet", "adain"],
                                    value="lab",
                                    info="Color matching method"
                                )
                                upscale_temporal_overlap = gr.Slider(
                                    label="Temporal Overlap",
                                    value=0,
                                    minimum=0,
                                    maximum=16,
                                    step=1,
                                    info="Frame overlap (video: ~3)"
                                )
                            with gr.Row():
                                upscale_input_noise = gr.Slider(
                                    label="Input Noise",
                                    value=0.0,
                                    minimum=0.0,
                                    maximum=0.2,
                                    step=0.001,
                                    info="Low levels (<0.1) can add detail"
                                )
                                upscale_latent_noise = gr.Slider(
                                    label="Latent Noise",
                                    value=0.0,
                                    minimum=0.0,
                                    maximum=1.0,
                                    step=0.001,
                                    info="Not recommended for most use"
                                )
                        
                        with gr.Accordion("🎛️ VAE Tiling", open=False):
                            with gr.Row():
                                upscale_encode_tiled = gr.Checkbox(label="Encode Tiled", value=True)
                                upscale_decode_tiled = gr.Checkbox(label="Decode Tiled", value=True)
                            with gr.Row():
                                upscale_encode_tile_size = gr.Slider(
                                    label="Encode Tile Size",
                                    value=1024,
                                    minimum=256,
                                    maximum=2048,
                                    step=64
                                )
                                upscale_encode_tile_overlap = gr.Slider(
                                    label="Encode Overlap",
                                    value=128,
                                    minimum=0,
                                    maximum=512,
                                    step=16
                                )
                            with gr.Row():
                                upscale_decode_tile_size = gr.Slider(
                                    label="Decode Tile Size",
                                    value=1024,
                                    minimum=256,
                                    maximum=2048,
                                    step=64
                                )
                                upscale_decode_tile_overlap = gr.Slider(
                                    label="Decode Overlap",
                                    value=128,
                                    minimum=0,
                                    maximum=512,
                                    step=16
                                )
                        
                        with gr.Accordion("💾 Presets", open=False):
                            # Load existing user presets + built-in defaults for dropdown
                            ui_settings = load_ui_settings()
                            user_presets = list(ui_settings.get("upscale_presets", {}).keys())
                            builtin_presets = ["Image Default", "Video Default"]
                            preset_choices = user_presets + (["─────────"] if user_presets else []) + builtin_presets
                            
                            # Track which input tab is active for "Save as Default"
                            upscale_active_tab = gr.State(value="Image")
                            
                            upscale_save_default_btn = gr.Button("⭐ Save current settings as default", size="sm")
                            gr.Markdown("---")
                            with gr.Row():
                                upscale_preset_dropdown = gr.Dropdown(
                                    label="Load Preset",
                                    choices=preset_choices,
                                    value="Image Default",
                                    scale=2
                                )
                                upscale_load_preset_btn = gr.Button("📂 Load", size="sm", scale=1)
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
                                    show_download_button=True
                                )
                            with gr.TabItem("🎬 Video Result", id="upscale_video_result"):
                                upscale_output_video = gr.Video(label="Upscaled Video")

                        with gr.Row():                          
                            upscale_save_btn = gr.Button("💾 Save", size="sm")
                            upscale_open_folder_btn = gr.Button("📂 Open Folder", size="sm")
                     
                        upscale_autosave = gr.Checkbox(label="Auto-save", value=False)
                        upscale_status = gr.Textbox(label="Status", interactive=False, show_label=False)

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
            
            # ===== LLM SETTINGS TAB =====
            with gr.TabItem("⚙️ LLM Settings"):
                prompt_assistant.render_settings_ui()
            
        
        # ===== EVENT HANDLERS =====
        
        # Prompt Assistant mode toggle
        def toggle_assist_mode(mode):
            is_describe = mode == "Describe Image"
            return (
                gr.update(visible=is_describe),  # assist_image
                gr.update(visible=not is_describe),  # enhance_btn
                gr.update(visible=is_describe),  # describe_btn
            )
        
        assist_mode.change(
            fn=toggle_assist_mode,
            inputs=[assist_mode],
            outputs=[assist_image, enhance_btn, describe_btn]
        )
        
        # Prompt Assistant buttons
        enhance_btn.click(
            fn=prompt_assistant.enhance_prompt,
            inputs=[prompt],
            outputs=[prompt, assist_status]
        )
        
        describe_btn.click(
            fn=prompt_assistant.describe_image,
            inputs=[assist_image, prompt],
            outputs=[prompt, assist_status]
        )
        
        # I2I tab describe button - uses the input_image directly
        i2i_describe_btn.click(
            fn=prompt_assistant.describe_image,
            inputs=[input_image, prompt],
            outputs=[prompt, i2i_assist_status]
        )
        
        clear_prompt_btn.click(
            fn=lambda: "",
            outputs=[prompt]
        )
        
        # Status indicator helpers
        def check_model_status(folder: Path, filename: str) -> str:
            """Return ✓ if file exists, ❌ if not."""
            if not filename:
                return "❌"
            return "✓" if (folder / filename).exists() else "❌"
        
        def update_unet_status(unet):
            return check_model_status(DIFFUSION_DIR, unet)
        
        def update_clip_status(clip):
            return check_model_status(TEXT_ENCODERS_DIR, clip)
        
        def update_vae_status(vae):
            return check_model_status(VAE_DIR, vae)
        
        # Update status when dropdown selection changes
        unet_name.change(fn=update_unet_status, inputs=[unet_name], outputs=[unet_status])
        clip_name.change(fn=update_clip_status, inputs=[clip_name], outputs=[clip_status])
        vae_name.change(fn=update_vae_status, inputs=[vae_name], outputs=[vae_status])
        
        # Update model dropdowns based on GGUF mode and show_all filter
        def update_model_dropdowns(is_gguf, show_all):
            """Refresh model dropdowns with current filter settings, including status indicators."""
            diff_filter = None if show_all else ZIMAGE_FILTERS["diffusion"]
            te_filter = None if show_all else ZIMAGE_FILTERS["text_encoder"]
            vae_filter = None if show_all else ZIMAGE_FILTERS["vae"]
            
            diffusion_models = get_models_by_mode(DIFFUSION_DIR, is_gguf, DEFAULT_DIFFUSION, DEFAULT_DIFFUSION_GGUF, diff_filter)
            clip_models = get_models_by_mode(TEXT_ENCODERS_DIR, is_gguf, DEFAULT_TEXT_ENCODER, DEFAULT_TEXT_ENCODER_GGUF, te_filter)
            vae_models = scan_models(VAE_DIR, (".safetensors",), vae_filter) or [DEFAULT_VAE]
            
            default_diffusion = DEFAULT_DIFFUSION_GGUF if is_gguf else DEFAULT_DIFFUSION
            default_clip = DEFAULT_TEXT_ENCODER_GGUF if is_gguf else DEFAULT_TEXT_ENCODER
            
            # Get the values that will be selected
            diff_value = get_default_model(diffusion_models, default_diffusion)
            clip_value = get_default_model(clip_models, default_clip)
            vae_value = get_default_model(vae_models, DEFAULT_VAE)
            
            return (
                gr.update(choices=diffusion_models, value=diff_value),
                gr.update(choices=clip_models, value=clip_value),
                gr.update(choices=vae_models, value=vae_value),
                check_model_status(DIFFUSION_DIR, diff_value),
                check_model_status(TEXT_ENCODERS_DIR, clip_value),
                check_model_status(VAE_DIR, vae_value),
            )
        
        model_dropdown_outputs = [unet_name, clip_name, vae_name, unet_status, clip_status, vae_status]
        
        use_gguf.change(
            fn=update_model_dropdowns,
            inputs=[use_gguf, show_all_models],
            outputs=model_dropdown_outputs
        )
        
        show_all_models.change(
            fn=update_model_dropdowns,
            inputs=[use_gguf, show_all_models],
            outputs=model_dropdown_outputs
        )
        
        # Save model defaults button
        def save_model_defaults(is_gguf, diffusion, text_encoder, vae):
            """Save current model selection as defaults for next startup."""
            settings = load_ui_settings()
            settings["model_defaults"] = {
                "use_gguf": is_gguf,
                "diffusion": diffusion,
                "text_encoder": text_encoder,
                "vae": vae,
            }
            save_ui_settings(settings)
            return "✓ Saved as default"
        
        save_model_defaults_btn.click(
            fn=save_model_defaults,
            inputs=[use_gguf, unet_name, clip_name, vae_name],
            outputs=[model_defaults_status]
        )
        
        # Refresh LoRA list
        def refresh_loras():
            loras = scan_models(LORAS_DIR, (".safetensors",))
            return gr.update(choices=loras, value=loras[0] if loras else None)
        
        refresh_loras_btn.click(
            fn=refresh_loras,
            outputs=[lora_name]
        )
        
        # Update block swap slider max based on DIT model selection
        def update_block_swap_max(dit_model):
            max_blocks = get_seedvr2_max_blocks(dit_model)
            return gr.update(maximum=max_blocks, value=max_blocks)
        
        upscale_dit_model.change(
            fn=update_block_swap_max,
            inputs=[upscale_dit_model],
            outputs=[upscale_blocks_to_swap]
        )
        
        # Unified upscale preset system
        upscale_all_settings = [
            upscale_dit_model,
            upscale_blocks_to_swap,
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
        ]
        upscale_setting_keys = [
            "dit_model", "blocks_to_swap", "batch_size", "uniform_batch",
            "color_correction", "temporal_overlap", "input_noise", "latent_noise",
            "encode_tiled", "encode_tile_size", "encode_tile_overlap",
            "decode_tiled", "decode_tile_size", "decode_tile_overlap",
        ]
        
        # Built-in defaults (used if no user preset exists)
        UPSCALE_BUILTIN_DEFAULTS = {
            "Image Default": {
                "dit_model": DEFAULT_SEEDVR2_DIT,
                "blocks_to_swap": 36,
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
            },
            "Video Default": {
                "dit_model": "seedvr2_ema_3b_fp16.safetensors",
                "blocks_to_swap": 32,
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
            },
        }
        
        def get_upscale_preset(name: str) -> dict:
            """Get preset by name - checks user presets first, then built-in defaults."""
            settings = load_ui_settings()
            user_presets = settings.get("upscale_presets", {})
            if name in user_presets:
                return user_presets[name]
            return UPSCALE_BUILTIN_DEFAULTS.get(name, UPSCALE_BUILTIN_DEFAULTS["Image Default"])
        
        def apply_upscale_preset(preset: dict):
            """Convert preset dict to tuple of values for UI components."""
            max_blocks = get_seedvr2_max_blocks(preset.get("dit_model", DEFAULT_SEEDVR2_DIT))
            return (
                preset.get("dit_model", DEFAULT_SEEDVR2_DIT),
                gr.update(value=preset.get("blocks_to_swap", 36), maximum=max_blocks),
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
            )
        
        # Tab switching loads from preset system and tracks active tab
        def on_upscale_tab_select(evt: gr.SelectData):
            """Switch presets based on which tab is selected."""
            if evt.value == "🖼️ Image":
                preset = get_upscale_preset("Image Default")
                active_tab = "Image"
            elif evt.value == "🎬 Video":
                preset = get_upscale_preset("Video Default")
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
            preset = dict(zip(upscale_setting_keys, values))
            
            # Load existing settings, update, save
            settings = load_ui_settings()
            if "upscale_presets" not in settings:
                settings["upscale_presets"] = {}
            settings["upscale_presets"][preset_name] = preset
            save_ui_settings(settings)
            
            # Update dropdown choices
            user_presets = list(settings["upscale_presets"].keys())
            choices = user_presets + ["─────────"] + list(UPSCALE_BUILTIN_DEFAULTS.keys())
            return f"✓ Saved as '{preset_name}'", gr.update(choices=choices, value=preset_name)
        
        upscale_save_default_btn.click(
            fn=save_as_default,
            inputs=[upscale_active_tab] + upscale_all_settings,
            outputs=[upscale_preset_status, upscale_preset_dropdown]
        )
        
        def save_upscale_preset(name, *values):
            """Save current upscale settings as a preset."""
            if not name or not name.strip():
                return "❌ Enter a preset name", gr.update()
            name = name.strip()
            
            # Build preset dict
            preset = dict(zip(upscale_setting_keys, values))
            
            # Load existing settings, update, save
            settings = load_ui_settings()
            if "upscale_presets" not in settings:
                settings["upscale_presets"] = {}
            settings["upscale_presets"][name] = preset
            save_ui_settings(settings)
            
            # Update dropdown choices - user presets + built-in defaults
            user_presets = list(settings["upscale_presets"].keys())
            choices = user_presets + ["─────────"] + list(UPSCALE_BUILTIN_DEFAULTS.keys())
            return f"✓ Saved '{name}'", gr.update(choices=choices, value=name)
        
        def load_upscale_preset_btn(name):
            """Load a preset's settings via button click."""
            if name == "─────────":
                return ("",) + (gr.update(),) * len(upscale_all_settings)
            
            preset = get_upscale_preset(name)
            return (f"✓ Loaded '{name}'",) + apply_upscale_preset(preset)
        
        upscale_save_preset_btn.click(
            fn=save_upscale_preset,
            inputs=[upscale_preset_name] + upscale_all_settings,
            outputs=[upscale_preset_status, upscale_preset_dropdown]
        )
        
        upscale_load_preset_btn.click(
            fn=load_upscale_preset_btn,
            inputs=[upscale_preset_dropdown],
            outputs=[upscale_preset_status] + upscale_all_settings
        )
        
        # Shared inputs for both generate buttons
        common_inputs = [
            use_gguf,
            steps, seed, randomize_seed, cfg, shift,
            sampler_name, scheduler,
            unet_name, clip_name, vae_name,
            use_lora, lora_name, lora_strength,
            autosave
        ]
        
        # Wrapper functions for async generate (async generators)
        async def generate_t2i(p, w, h, gguf, *args):
            async for result in generate_image(p, "t2i", gguf, w, h, None, 2.0, 0.67, *args):
                yield result
        
        async def generate_i2i(p, img, mp, dn, gguf, *args):
            async for result in generate_image(p, "i2i", gguf, 1024, 1024, img, mp, dn, *args):
                yield result
        
        # T2I generate
        generate_t2i_btn.click(
            fn=generate_t2i,
            inputs=[prompt, width, height] + common_inputs,
            outputs=[output_image, gen_status, seed]
        )
        
        # I2I generate
        generate_i2i_btn.click(
            fn=generate_i2i,
            inputs=[prompt, input_image, megapixels, denoise] + common_inputs,
            outputs=[output_image, gen_status, seed]
        )
        
        # Shared upscale inputs (SeedVR2 settings)
        upscale_common_inputs = [
            upscale_dit_model,
            upscale_blocks_to_swap,
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
        
        # Image Upscale - wrapper to also switch output tab
        async def upscale_image_and_switch(*args):
            result = await upscale_image(*args)
            # Return result + tab switch to image result
            return result + (gr.Tabs(selected="upscale_image_result"),)
        
        upscale_btn.click(
            fn=upscale_image_and_switch,
            inputs=[
                upscale_input_image,
                upscale_seed,
                upscale_randomize_seed,
                upscale_resolution,
                upscale_max_resolution,
            ] + upscale_common_inputs,
            outputs=[upscale_slider, upscale_status, upscale_seed, upscale_result_path, upscale_original_path, upscale_result_resolution, upscale_output_tabs]
        )
        
        # Video Upscale - wrapper to also switch output tab
        async def upscale_video_and_switch(*args):
            result = await upscale_video(*args)
            # Return result + tab switch to video result
            return result + (gr.Tabs(selected="upscale_video_result"),)
        
        upscale_video_btn.click(
            fn=upscale_video_and_switch,
            inputs=[
                upscale_input_video,
                upscale_seed,
                upscale_randomize_seed,
                upscale_video_resolution,
            ] + upscale_common_inputs,
            outputs=[upscale_output_video, upscale_status, upscale_seed, upscale_output_tabs]
        )
        
        unload_btn.click(
            fn=unload_models,
            outputs=[gen_status]
        )
        
        # Save generated image
        def save_current_image(image_path, prompt_text):
            if not image_path:
                return "❌ No image to save"
            saved_path = save_image_to_outputs(image_path, prompt_text or "image")
            return f"✓ Saved: {Path(saved_path).name}"
        
        save_btn.click(
            fn=save_current_image,
            inputs=[output_image, prompt],
            outputs=[gen_status]
        )
        
        # Save upscaled image
        def save_upscaled_image(image_path, original_path, resolution):
            if not image_path:
                return "❌ No image to save"
            saved_path = save_upscale_to_outputs(image_path, original_path, resolution or 4096)
            return f"✓ Saved: {Path(saved_path).name}"
        
        upscale_save_btn.click(
            fn=save_upscaled_image,
            inputs=[upscale_result_path, upscale_original_path, upscale_result_resolution],
            outputs=[upscale_status]
        )
        
        # Send to Upscale - copy image and switch tab
        send_to_upscale_btn.click(
            fn=lambda img: (img, gr.Tabs(selected="tab_upscale")),
            inputs=[output_image],
            outputs=[upscale_input_image, main_tabs]
        )
        
        # Open folder helpers
        def open_folder(folder_path: Path):
            """Cross-platform folder opener."""
            folder_path.mkdir(parents=True, exist_ok=True)
            if sys.platform == "win32":
                os.startfile(folder_path)
            elif sys.platform == "darwin":
                subprocess.run(["open", str(folder_path)])
            else:
                subprocess.run(["xdg-open", str(folder_path)])
        
        def open_outputs_folder():
            open_folder(OUTPUTS_DIR)
        
        def open_upscaled_folder():
            open_folder(OUTPUTS_DIR / "upscaled")
        
        open_folder_btn.click(fn=open_outputs_folder)
        upscale_open_folder_btn.click(fn=open_upscaled_folder)
        
        # Model folder buttons
        open_loras_btn.click(fn=lambda: open_folder(LORAS_DIR))
        open_diffusion_btn.click(fn=lambda: open_folder(DIFFUSION_DIR))
        open_te_btn.click(fn=lambda: open_folder(TEXT_ENCODERS_DIR))
        open_vae_btn.click(fn=lambda: open_folder(VAE_DIR))
        
        # Open documentation file
        def open_documentation():
            doc_path = APP_DIR / "DOCUMENTATION.md"
            if doc_path.exists():
                if sys.platform == "win32":
                    os.startfile(doc_path)
                elif sys.platform == "darwin":
                    subprocess.run(["open", str(doc_path)])
                else:
                    subprocess.run(["xdg-open", str(doc_path)])
        
        # open_docs_btn.click(fn=open_documentation)

        # System Monitor 
        def update_monitor():
            gpu_info, cpu_info = SystemMonitor.get_system_info()
            # Return same info for both video and image tabs
            return gpu_info, cpu_info
            
        monitor_timer = gr.Timer(2, active=True)
        monitor_timer.tick(fn=update_monitor, outputs=[gpu_monitor, cpu_monitor]) 
        
        # Model download buttons - refresh dropdowns and hide banner after download
        def download_and_refresh(is_gguf, show_all, model_key, progress=gr.Progress()):
            """Download a single model and refresh dropdowns."""
            status = download_model(model_key, progress)
            dropdowns = update_model_dropdowns(is_gguf, show_all)
            # Check if we should hide the setup banner
            model_status = check_models_installed()
            banner_visible = gr.update(visible=not model_status["has_any"])
            return (status,) + dropdowns + (banner_visible,)
        
        def download_all_and_refresh(is_gguf_mode, show_all, progress=gr.Progress()):
            """Download all models for a mode, switch to that mode, and refresh dropdowns."""
            if is_gguf_mode:
                status = download_all_gguf(progress)
            else:
                status = download_all_standard(progress)
            # Update dropdowns for the mode we just downloaded
            dropdowns = update_model_dropdowns(is_gguf_mode, show_all)
            # Check if we should hide the setup banner
            model_status = check_models_installed()
            banner_visible = gr.update(visible=not model_status["has_any"])
            # Also switch the radio to match the downloaded mode
            mode_switch = gr.update(value=is_gguf_mode)
            return (status,) + dropdowns + (banner_visible, mode_switch)
        
        # download_outputs includes dropdowns + status indicators + banner
        download_outputs = [download_status, unet_name, clip_name, vae_name, unet_status, clip_status, vae_status, setup_banner]
        download_inputs = [use_gguf, show_all_models]
        
        # Individual download buttons (don't change mode)
        dl_diffusion_bf16_btn.click(
            fn=download_and_refresh,
            inputs=download_inputs + [gr.State("diffusion_bf16")],
            outputs=download_outputs
        )
        dl_te_bf16_btn.click(
            fn=download_and_refresh,
            inputs=download_inputs + [gr.State("te_bf16")],
            outputs=download_outputs
        )
        dl_vae_btn.click(
            fn=download_and_refresh,
            inputs=download_inputs + [gr.State("vae")],
            outputs=download_outputs
        )
        dl_diffusion_gguf_btn.click(
            fn=download_and_refresh,
            inputs=download_inputs + [gr.State("diffusion_gguf")],
            outputs=download_outputs
        )
        dl_te_gguf_btn.click(
            fn=download_and_refresh,
            inputs=download_inputs + [gr.State("te_gguf")],
            outputs=download_outputs
        )
        dl_vae_gguf_btn.click(
            fn=download_and_refresh,
            inputs=download_inputs + [gr.State("vae")],
            outputs=download_outputs
        )
        
        # Download all buttons - also switch mode to match
        download_all_outputs = download_outputs + [use_gguf]
        dl_all_standard_btn.click(
            fn=lambda show_all, progress=gr.Progress(): download_all_and_refresh(False, show_all, progress),
            inputs=[show_all_models], outputs=download_all_outputs
        )
        dl_all_gguf_btn.click(
            fn=lambda show_all, progress=gr.Progress(): download_all_and_refresh(True, show_all, progress),
            inputs=[show_all_models], outputs=download_all_outputs
        )
        
        # Camera prompts - open in browser
        def open_camera_prompts():
            camera_html = CAMERA_PROMPTS_DIR / "index.html"
            if camera_html.exists():
                webbrowser.open(camera_html.as_uri())
                return "✓ Opened in browser"
            return "❌ Camera prompts not found"
        
        open_camera_prompts_btn.click(fn=open_camera_prompts)
    
    return interface


def main():
    """Main entry point."""
    if kit is None:
        print("\n❌ Failed to initialize ComfyKit")
        print("   Make sure ComfyUI is running at http://127.0.0.1:8188\n")
        return
    
    print("\n" + "="*50)
    print("⚡ Z-Image Turbo")
    print("="*50)
    print(f"ComfyUI: {kit.comfyui_url}")
    print(f"Models:  {MODELS_DIR}")
    print(f"Outputs: {OUTPUTS_DIR}")
    print("="*50 + "\n")
    
    interface = create_interface()
    interface.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
