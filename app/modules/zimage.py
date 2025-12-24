"""
Z-Image Generation Module

Provides the Z-Image Turbo generation tab with Text→Image, Image→Image,
and Prompt Assistant functionality.
"""

import logging
import os
import random
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import gradio as gr
import httpx
from huggingface_hub import hf_hub_download

if TYPE_CHECKING:
    from modules import SharedServices

logger = logging.getLogger(__name__)

# Module metadata
TAB_ID = "zimage"
TAB_LABEL = "⚡ Z-Image"
TAB_ORDER = 0

# Default models
DEFAULT_DIFFUSION = "z_image_turbo_bf16.safetensors"
DEFAULT_DIFFUSION_GGUF = "z_image_turbo-Q4_K_M.gguf"
DEFAULT_TEXT_ENCODER = "qwen_3_4b.safetensors"
DEFAULT_TEXT_ENCODER_GGUF = "Qwen3-4B-Q4_K_M.gguf"
DEFAULT_VAE = "ae.safetensors"

# File extensions by mode
STANDARD_EXTENSIONS = (".safetensors", ".ckpt", ".pt")
GGUF_EXTENSIONS = (".gguf",)

# Dummy lora filename (used when lora disabled - strength 0 bypasses it)
DUMMY_LORA = "none.safetensors"

# Name filters for Z-Image compatible models
ZIMAGE_FILTERS = {
    "diffusion": "z-image",  # z-image, z_image variants
    "text_encoder": "qwen",  # Qwen3 text encoder
    "vae": "ae",             # ae.safetensors
}

# Fallback sampler/scheduler lists (used if ComfyUI not available)
FALLBACK_SAMPLERS = ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde", "res_multistep"]
FALLBACK_SCHEDULERS = ["simple", "normal", "karras", "exponential", "sgm_uniform"]

# Preferred defaults (will be selected if available)
PREFERRED_SAMPLER = "euler"
PREFERRED_SCHEDULER = "simple"

# Generation cancellation flag (set by stop button, checked by batch loop)
_cancel_generation = False

# Session temp directory for results (auto-cleaned on exit)
# Using a persistent TemporaryDirectory so files have nice names for Gradio download
_results_temp_dir = tempfile.TemporaryDirectory(prefix="zimage_results_")

# Resolution presets by base size - format: "WxH ( AR )"
RES_CHOICES = {
    "1024": [
        "1024x1024 ( 1:1 )",
        "1152x896 ( 9:7 )",
        "896x1152 ( 7:9 )",
        "1152x864 ( 4:3 )",
        "864x1152 ( 3:4 )",
        "1248x832 ( 3:2 )",
        "832x1248 ( 2:3 )",
        "1280x720 ( 16:9 )",
        "720x1280 ( 9:16 )",
        "1344x576 ( 21:9 )",
        "576x1344 ( 9:21 )",
    ],
    "1280": [
        "1280x1280 ( 1:1 )",
        "1440x1120 ( 9:7 )",
        "1120x1440 ( 7:9 )",
        "1472x1104 ( 4:3 )",
        "1104x1472 ( 3:4 )",
        "1536x1024 ( 3:2 )",
        "1024x1536 ( 2:3 )",
        "1536x864 ( 16:9 )",
        "864x1536 ( 9:16 )",
        "1680x720 ( 21:9 )",
        "720x1680 ( 9:21 )",
    ],
    "1536": [
        "1536x1536 ( 1:1 )",
        "1728x1344 ( 9:7 )",
        "1344x1728 ( 7:9 )",
        "1728x1296 ( 4:3 )",
        "1296x1728 ( 3:4 )",
        "1872x1248 ( 3:2 )",
        "1248x1872 ( 2:3 )",
        "2048x1152 ( 16:9 )",
        "1152x2048 ( 9:16 )",
        "2016x864 ( 21:9 )",
        "864x2016 ( 9:21 )",
    ],
}

def parse_resolution(res_string: str) -> tuple[int, int]:
    """Parse resolution string like '1024x1024 ( 1:1 )' into (width, height)."""
    # Extract WxH part before the parenthesis
    dims = res_string.split("(")[0].strip()
    w, h = dims.split("x")
    return int(w), int(h)

def get_resolution_dropdown_choices(base: str) -> list[tuple[str, str]]:
    """Get formatted dropdown choices with landscape/portrait grouping."""
    choices = RES_CHOICES.get(base, RES_CHOICES["1024"])
    # First item is always square
    result = [choices[0]]  # Square
    # Add landscape options (indices 1,3,5,7,9 - odd width > height)
    result.append("── Landscape ──")
    for i in [1, 3, 5, 7, 9]:
        result.append(choices[i])
    # Add portrait options (indices 2,4,6,8,10 - odd height > width)
    result.append("── Portrait ──")
    for i in [2, 4, 6, 8, 10]:
        result.append(choices[i])
    return result


# Model definitions for huggingface_hub downloads
MODEL_DOWNLOADS = {
    "diffusion_bf16": {
        "repo_id": "Comfy-Org/z_image_turbo",
        "filename": "split_files/diffusion_models/z_image_turbo_bf16.safetensors",
        "local_name": "z_image_turbo_bf16.safetensors",
        "folder_key": "diffusion",
        "label": "Diffusion (bf16)",
    },
    "te_bf16": {
        "repo_id": "Comfy-Org/z_image_turbo",
        "filename": "split_files/text_encoders/qwen_3_4b.safetensors",
        "local_name": "qwen_3_4b.safetensors",
        "folder_key": "text_encoder",
        "label": "Text Encoder (bf16)",
    },
    "vae": {
        "repo_id": "Comfy-Org/z_image_turbo",
        "filename": "split_files/vae/ae.safetensors",
        "local_name": "ae.safetensors",
        "folder_key": "vae",
        "label": "VAE",
    },
    "diffusion_gguf": {
        "repo_id": "gguf-org/z-image-gguf",
        "filename": "z-image-turbo-q4_k_m.gguf",
        "local_name": "z-image-turbo-q4_k_m.gguf",
        "folder_key": "diffusion",
        "label": "Diffusion (Q4 GGUF)",
    },
    "te_gguf": {
        "repo_id": "Qwen/Qwen3-4B-GGUF",
        "filename": "Qwen3-4B-Q4_K_M.gguf",
        "local_name": "Qwen3-4B-Q4_K_M.gguf",
        "folder_key": "text_encoder",
        "label": "Qwen3 TE (Q4 GGUF)",
    },
}


def scan_models(folder: Path, extensions: tuple = (".safetensors", ".ckpt", ".pt", ".gguf"), name_filter: str = None) -> list:
    """Scan folder recursively for model files, returning relative paths.
    
    Args:
        folder: Directory to scan
        extensions: File extensions to include
        name_filter: Optional case-insensitive substring to filter by (e.g. "z-image", "qwen")
    
    Returns:
        List of relative paths to model files that exist on disk
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


def get_models_by_mode(folder: Path, is_gguf: bool, default_standard: str, default_gguf: str, name_filter: str = None) -> list:
    """Get models filtered by mode (standard vs GGUF) and optional name filter."""
    extensions = GGUF_EXTENSIONS if is_gguf else STANDARD_EXTENSIONS
    default = default_gguf if is_gguf else default_standard
    models = scan_models(folder, extensions, name_filter)
    return models or [default]


def get_default_model(choices: list, preferred: str) -> str:
    """Get default model, preferring the specified one if available."""
    if preferred in choices:
        return preferred
    return choices[0] if choices else None


def new_random_seed():
    """Generate a new random seed."""
    return random.randint(0, 999999999999)


def get_workflow_file(gen_type: str, use_gguf: bool) -> str:
    """Determine which workflow file to use based on settings.
    
    Always uses lora workflows - loras are disabled by setting strength to 0.
    
    Args:
        gen_type: "t2i" or "i2i"
        use_gguf: Whether to use GGUF workflow variant
        
    Returns:
        Workflow filename matching pattern z_image_{gguf_}?{gen_type}_lora.json
    """
    parts = ["z_image"]
    if use_gguf:
        parts.append("gguf")
    parts.append(gen_type)
    parts.append("lora")  # Always use lora workflow
    return "_".join(parts) + ".json"


def extract_png_metadata(image_path: str) -> dict:
    """Extract ComfyUI metadata from PNG text chunks.
    
    Returns dict with 'prompt', 'workflow', and parsed generation params if available.
    """
    from PIL import Image
    import json
    
    result = {
        "prompt_text": "",
        "params": {},
        "raw_prompt": None,
        "raw_workflow": None,
        "error": None
    }
    
    try:
        with Image.open(image_path) as img:
            # PNG text chunks are in img.info
            if not hasattr(img, 'info') or not img.info:
                result["error"] = "No metadata found in image"
                return result
            
            # Get raw metadata
            raw_prompt = img.info.get("prompt")
            raw_workflow = img.info.get("workflow")
            
            if raw_prompt:
                result["raw_prompt"] = raw_prompt
                try:
                    prompt_data = json.loads(raw_prompt)
                    # ComfyUI prompt is a dict of node_id -> node_data
                    # Look for text encoder / CLIP nodes that contain the prompt
                    for node_id, node_data in prompt_data.items():
                        if isinstance(node_data, dict):
                            inputs = node_data.get("inputs", {})
                            class_type = node_data.get("class_type", "")
                            
                            # Extract prompt text from various node types
                            if "text" in inputs and isinstance(inputs["text"], str):
                                if len(inputs["text"]) > len(result["prompt_text"]):
                                    result["prompt_text"] = inputs["text"]
                            
                            # Extract generation params from sampler nodes
                            if "KSampler" in class_type or "sampler" in class_type.lower():
                                if "seed" in inputs:
                                    result["params"]["seed"] = inputs["seed"]
                                if "steps" in inputs:
                                    result["params"]["steps"] = inputs["steps"]
                                if "cfg" in inputs:
                                    result["params"]["cfg"] = inputs["cfg"]
                                if "sampler_name" in inputs:
                                    result["params"]["sampler"] = inputs["sampler_name"]
                                if "scheduler" in inputs:
                                    result["params"]["scheduler"] = inputs["scheduler"]
                                if "shift" in inputs:
                                    result["params"]["shift"] = inputs["shift"]
                                if "denoise" in inputs:
                                    result["params"]["denoise"] = inputs["denoise"]
                            
                            # Extract dimensions from empty latent or image nodes
                            if "width" in inputs and "height" in inputs:
                                result["params"]["width"] = inputs["width"]
                                result["params"]["height"] = inputs["height"]
                            
                            # Extract model names
                            if "unet_name" in inputs:
                                result["params"]["diffusion"] = inputs["unet_name"]
                            if "clip_name" in inputs:
                                result["params"]["text_encoder"] = inputs["clip_name"]
                            
                            # Extract LoRA info from loader nodes
                            if "LoraLoader" in class_type or "lora" in class_type.lower():
                                lora_name = inputs.get("lora_name")
                                strength = inputs.get("strength_model", inputs.get("strength", 1.0))
                                if lora_name and lora_name != "none.safetensors":
                                    if "loras" not in result["params"]:
                                        result["params"]["loras"] = []
                                    result["params"]["loras"].append({
                                        "name": lora_name,
                                        "strength": strength
                                    })
                                
                except json.JSONDecodeError:
                    result["error"] = "Could not parse prompt metadata"
            
            if raw_workflow:
                result["raw_workflow"] = raw_workflow
            
            if not raw_prompt and not raw_workflow:
                result["error"] = "No ComfyUI metadata found"
                
    except Exception as e:
        result["error"] = f"Error reading image: {str(e)}"
    
    return result


def format_metadata_display(metadata: dict) -> str:
    """Format extracted metadata for display."""
    lines = []
    
    if metadata.get("error"):
        return f"⚠️ {metadata['error']}"
    
    if metadata.get("prompt_text"):
        lines.append(f"📝 Prompt:\n{metadata['prompt_text']}\n")
    
    params = metadata.get("params", {})
    if params:
        lines.append("⚙️ Settings:")
        if "seed" in params:
            lines.append(f"  Seed: {params['seed']}")
        if "steps" in params:
            lines.append(f"  Steps: {params['steps']}")
        if "cfg" in params:
            lines.append(f"  CFG: {params['cfg']}")
        if "sampler" in params:
            lines.append(f"  Sampler: {params['sampler']}")
        if "scheduler" in params:
            lines.append(f"  Scheduler: {params['scheduler']}")
        if "shift" in params:
            lines.append(f"  Shift: {params['shift']}")
        if "width" in params and "height" in params:
            lines.append(f"  Size: {params['width']}x{params['height']}")
        if "denoise" in params:
            lines.append(f"  Denoise: {params['denoise']}")
        if "diffusion" in params:
            lines.append(f"  Model: {params['diffusion']}")
        if "loras" in params and params["loras"]:
            lines.append("\n🎨 LoRAs:")
            for lora in params["loras"]:
                lines.append(f"  {lora['name']} (strength: {lora['strength']})")
    
    if not lines:
        return "No generation parameters found in metadata"
    
    return "\n".join(lines)


def save_image_to_outputs(image_path: str, prompt: str, outputs_dir: Path, subfolder: str = None) -> str:
    """Save image to outputs folder with prompt and short timestamp.
    
    Args:
        image_path: Path to the source image
        prompt: Prompt text used for filename
        outputs_dir: Base outputs directory from SharedServices
        subfolder: Optional subfolder within outputs_dir
        
    Returns:
        Path to the saved image file
    """
    timestamp = datetime.now().strftime("%H%M%S")
    safe_prompt = "".join(c if c.isalnum() or c in " -_" else "" for c in prompt[:30]).strip()
    safe_prompt = safe_prompt.replace(" ", "_") if safe_prompt else "image"
    
    # Use subfolder if specified
    target_dir = outputs_dir / subfolder if subfolder else outputs_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Format: prompt_HHMMSS.png (timestamp at end, less obtrusive)
    filename = f"{safe_prompt}_{timestamp}.png"
    output_path = target_dir / filename
    shutil.copy2(image_path, output_path)
    logger.info(f"Saved to: {output_path}")
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


def copy_to_temp_with_name(image_path: str, prompt: str, seed: int) -> str:
    """Copy image to session temp dir with a meaningful name for Gradio download."""
    timestamp = datetime.now().strftime("%H%M%S")
    safe_prompt = "".join(c if c.isalnum() or c in " -_" else "" for c in prompt[:30]).strip()
    safe_prompt = safe_prompt.replace(" ", "_") if safe_prompt else "image"
    filename = f"{safe_prompt}_{seed}_{timestamp}.png"
    temp_path = Path(_results_temp_dir.name) / filename
    shutil.copy2(image_path, temp_path)
    return str(temp_path)


def check_models_installed(diffusion_dir: Path, text_encoders_dir: Path, vae_dir: Path) -> dict:
    """Check which model sets are installed and return status info."""
    # Check for Z-Image compatible models
    standard_diffusion = scan_models(diffusion_dir, STANDARD_EXTENSIONS, ZIMAGE_FILTERS["diffusion"])
    standard_te = scan_models(text_encoders_dir, STANDARD_EXTENSIONS, ZIMAGE_FILTERS["text_encoder"])
    gguf_diffusion = scan_models(diffusion_dir, GGUF_EXTENSIONS, ZIMAGE_FILTERS["diffusion"])
    gguf_te = scan_models(text_encoders_dir, GGUF_EXTENSIONS, ZIMAGE_FILTERS["text_encoder"])
    vae = scan_models(vae_dir, (".safetensors",), ZIMAGE_FILTERS["vae"])
    
    has_standard = bool(standard_diffusion and standard_te and vae)
    has_gguf = bool(gguf_diffusion and gguf_te and vae)
    
    return {
        "has_standard": has_standard,
        "has_gguf": has_gguf,
        "has_any": has_standard or has_gguf,
        "recommended_mode": has_gguf and not has_standard,  # True = GGUF, False = Standard
    }


def validate_selected_models(unet_name: str, clip_name: str, vae_name: str, 
                             diffusion_dir: Path, text_encoders_dir: Path, vae_dir: Path) -> tuple[bool, str]:
    """Validate that selected model files actually exist. Returns (valid, error_message)."""
    unet_path = diffusion_dir / unet_name
    clip_path = text_encoders_dir / clip_name
    vae_path = vae_dir / vae_name
    
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


def fetch_comfyui_options(kit) -> dict:
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


def ensure_dummy_lora(loras_dir: Path):
    """Create a minimal dummy lora file for disabled slots."""
    dummy_path = loras_dir / DUMMY_LORA
    if dummy_path.exists():
        return
    
    try:
        loras_dir.mkdir(parents=True, exist_ok=True)
        import torch
        from safetensors.torch import save_file
        save_file({"__placeholder__": torch.zeros(1)}, str(dummy_path))
        logger.info(f"Created dummy lora: {dummy_path}")
    except Exception as e:
        logger.warning(f"Could not create dummy lora: {e}")


def download_model(model_key: str, models_dir: Path, progress=gr.Progress()):
    """Download a model from HuggingFace Hub to the appropriate folder."""
    if model_key not in MODEL_DOWNLOADS:
        return f"❌ Unknown model: {model_key}"
    
    info = MODEL_DOWNLOADS[model_key]
    repo_id = info["repo_id"]
    filename = info["filename"]
    local_name = info["local_name"]
    folder_key = info["folder_key"]
    label = info["label"]
    
    # Determine folder based on key
    folder_map = {
        "diffusion": models_dir / "diffusion_models",
        "text_encoder": models_dir / "text_encoders",
        "vae": models_dir / "vae",
    }
    folder = folder_map.get(folder_key, models_dir)
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


def download_all_standard(models_dir: Path, progress=gr.Progress()):
    """Download all standard (bf16) models."""
    results = []
    keys = ["diffusion_bf16", "te_bf16", "vae"]
    for i, key in enumerate(keys):
        progress((i / len(keys)), desc=f"Downloading {MODEL_DOWNLOADS[key]['label']}...")
        result = download_model(key, models_dir, progress)
        results.append(result)
    progress(1, desc="Done")
    return "\n".join(results)


def download_all_gguf(models_dir: Path, progress=gr.Progress()):
    """Download all GGUF (Q4) models + VAE."""
    results = []
    keys = ["diffusion_gguf", "te_gguf", "vae"]
    for i, key in enumerate(keys):
        progress((i / len(keys)), desc=f"Downloading {MODEL_DOWNLOADS[key]['label']}...")
        result = download_model(key, models_dir, progress)
        results.append(result)
    progress(1, desc="Done")
    return "\n".join(results)


async def generate_image(
    services: "SharedServices",
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
    # lora params (6 slots with enable flags)
    lora1_enabled: bool, lora1_name: str, lora1_strength: float,
    lora2_enabled: bool, lora2_name: str, lora2_strength: float,
    lora3_enabled: bool, lora3_name: str, lora3_strength: float,
    lora4_enabled: bool, lora4_name: str, lora4_strength: float,
    lora5_enabled: bool, lora5_name: str, lora5_strength: float,
    lora6_enabled: bool, lora6_name: str, lora6_strength: float,
    # output params
    autosave: bool,
    # batch params
    batch_count: int = 1,
    # seed variance params
    sv_enabled: bool = False,
    sv_noise_insert: str = "noise on beginning steps",
    sv_randomize_percent: float = 50.0,
    sv_strength: float = 20.0,
    sv_steps_switchover_percent: float = 20.0,
    sv_seed: int = 0,
    sv_mask_starts_at: str = "beginning",
    sv_mask_percent: float = 0.0
):
    """Generate images using the selected workflow. Yields (gallery_images, status, seed) tuples."""
    # Get paths from services
    diffusion_dir = services.models_dir / "diffusion_models"
    text_encoders_dir = services.models_dir / "text_encoders"
    vae_dir = services.models_dir / "vae"
    outputs_dir = services.get_outputs_dir()
    
    # Handle seed early so we can yield it immediately
    base_seed = new_random_seed() if randomize_seed else int(seed)
    batch_count = max(1, min(int(batch_count), 100))  # Clamp to 1-100
    
    # Validate models exist before attempting generation
    models_valid, error_msg = validate_selected_models(
        unet_name, clip_name, vae_name,
        diffusion_dir, text_encoders_dir, vae_dir
    )
    if not models_valid:
        yield [], f"❌ {error_msg}", base_seed
        return
    
    # Validate i2i has input image
    if gen_type == "i2i" and input_image is None:
        yield [], "❌ Please upload an input image for Image→Image", base_seed
        return
    
    # Select workflow
    workflow_file = get_workflow_file(gen_type, use_gguf)
    workflow_path = services.workflows_dir / workflow_file
    
    if not workflow_path.exists():
        yield [], f"❌ Workflow not found: {workflow_file}", base_seed
        return
    
    # Allow empty prompt - model will generate without guidance
    prompt_text = prompt.strip() if prompt else ""
    
    logger.info(f"Using workflow: {workflow_file}")
    logger.info(f"Batch generation: {batch_count} images starting at seed={base_seed}")
    
    # Yield initial state
    status_prefix = f"[1/{batch_count}] " if batch_count > 1 else ""
    yield [], f"⏳ {status_prefix}Generating...", base_seed
    
    generated_images = []
    total_duration = 0.0
    
    try:
        global _cancel_generation
        _cancel_generation = False  # Reset at start of generation
        
        for i in range(batch_count):
            # Check for cancellation
            if _cancel_generation:
                _cancel_generation = False  # Reset for next run
                yield generated_images, "⏹️ Generation cancelled", base_seed
                return
            
            current_seed = base_seed + i
            
            # Update status for batch progress
            if batch_count > 1:
                yield generated_images, f"⏳ [{i+1}/{batch_count}] Generating (seed: {current_seed})...", base_seed
            
            # Build params dict
            params = {
                "prompt": prompt_text,
                "steps": int(steps),
                "seed": int(current_seed),
                "cfg": cfg,
                "shift": shift,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "unet_name": unet_name,
                "clip_name": clip_name,
                "vae_name": vae_name,
            }
            
            # Debug: log generation params
            logger.info(f"Generation params: seed={params['seed']}, steps={params['steps']}, cfg={params['cfg']}, shift={params['shift']}, sampler={params['sampler_name']}, scheduler={params['scheduler']}")
            
            # Add type-specific params
            if gen_type == "t2i":
                params["width"] = int(width)
                params["height"] = int(height)
            else:  # i2i
                params["image"] = input_image
                params["megapixels"] = megapixels
                params["denoise"] = denoise
            
            # Add lora params (6 slots - use dummy lora with strength 0 when disabled)
            params["lora1_name"] = lora1_name if (lora1_enabled and lora1_name) else DUMMY_LORA
            params["lora1_strength"] = lora1_strength if (lora1_enabled and lora1_name) else 0
            params["lora2_name"] = lora2_name if (lora2_enabled and lora2_name) else DUMMY_LORA
            params["lora2_strength"] = lora2_strength if (lora2_enabled and lora2_name) else 0
            params["lora3_name"] = lora3_name if (lora3_enabled and lora3_name) else DUMMY_LORA
            params["lora3_strength"] = lora3_strength if (lora3_enabled and lora3_name) else 0
            params["lora4_name"] = lora4_name if (lora4_enabled and lora4_name) else DUMMY_LORA
            params["lora4_strength"] = lora4_strength if (lora4_enabled and lora4_name) else 0
            params["lora5_name"] = lora5_name if (lora5_enabled and lora5_name) else DUMMY_LORA
            params["lora5_strength"] = lora5_strength if (lora5_enabled and lora5_name) else 0
            params["lora6_name"] = lora6_name if (lora6_enabled and lora6_name) else DUMMY_LORA
            params["lora6_strength"] = lora6_strength if (lora6_enabled and lora6_name) else 0
            
            # Debug: log lora params (first 3 for brevity)
            logger.info(f"LoRA params: lora1={params['lora1_name']} ({params['lora1_strength']}), lora2={params['lora2_name']} ({params['lora2_strength']}), lora3={params['lora3_name']} ({params['lora3_strength']})")
            
            # Add seed variance params
            # When disabled, pass "disabled" to make the node a passthrough
            params["sv_noise_insert"] = sv_noise_insert if sv_enabled else "disabled"
            params["sv_randomize_percent"] = sv_randomize_percent
            params["sv_strength"] = sv_strength
            params["sv_steps_switchover_percent"] = sv_steps_switchover_percent
            # Use main seed if sv_seed is 0, otherwise use the specified variance seed
            params["sv_seed"] = current_seed if sv_seed == 0 else int(sv_seed)
            params["sv_mask_starts_at"] = sv_mask_starts_at
            params["sv_mask_percent"] = sv_mask_percent
            
            # Execute workflow using services.kit
            result = await services.kit.execute(str(workflow_path), params)
            
            if result.status == "error":
                if batch_count == 1:
                    yield [], f"❌ Generation failed: {result.msg}", base_seed
                    return
                else:
                    # Continue batch on error, just log it
                    logger.warning(f"Batch item {i+1} failed: {result.msg}")
                    continue
            
            if not result.images:
                if batch_count == 1:
                    yield [], "❌ No images generated", base_seed
                    return
                else:
                    continue
            
            # Get image
            image_path = result.images[0]
            if image_path.startswith("http"):
                image_path = await download_image_from_url(image_path)
            
            # Copy to temp with meaningful name so Gradio download button works nicely
            image_path = copy_to_temp_with_name(image_path, prompt_text or "image", current_seed)
            
            # Track duration
            if result.duration:
                total_duration += result.duration
            
            # Autosave each image as it completes
            if autosave:
                save_image_to_outputs(image_path, prompt_text or "image", outputs_dir)
            
            # Add to gallery with caption showing seed
            generated_images.append((image_path, f"seed: {current_seed}"))
            
            # Yield progress update
            if batch_count > 1:
                yield generated_images, f"✓ [{i+1}/{batch_count}] Complete", base_seed
        
        # Final status
        if not generated_images:
            yield [], "❌ No images generated", base_seed
            return
        
        # Build final status message
        count_str = f"{len(generated_images)} images" if len(generated_images) > 1 else ""
        time_str = f"{total_duration:.1f}s" if total_duration else ""
        save_str = "Saved" if autosave else ""
        
        status_parts = [p for p in [count_str, time_str, save_str] if p]
        status = "✓ " + " | ".join(status_parts) if status_parts else "✓ Done"
        
        yield generated_images, status, base_seed
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        if "connect" in str(e).lower():
            yield generated_images or [], "❌ Cannot connect to ComfyUI", base_seed
        else:
            yield generated_images or [], f"❌ {str(e)}", base_seed


def create_tab(services: "SharedServices") -> gr.TabItem:
    """
    Create the Z-Image generation tab with all UI components and event handlers.
    
    Args:
        services: SharedServices instance with all dependencies
        
    Returns:
        gr.TabItem containing the complete Z-Image generation interface
    """
    # Get model directories
    diffusion_dir = services.models_dir / "diffusion_models"
    text_encoders_dir = services.models_dir / "text_encoders"
    vae_dir = services.models_dir / "vae"
    loras_dir = services.models_dir / "loras"
    
    # Ensure dummy lora exists
    ensure_dummy_lora(loras_dir)
    
    # Get model choices
    def get_model_choices():
        """Get available models from local folders (all extensions)."""
        return {
            "diffusion": scan_models(diffusion_dir) or [DEFAULT_DIFFUSION],
            "text_encoder": scan_models(text_encoders_dir) or [DEFAULT_TEXT_ENCODER],
            "vae": scan_models(vae_dir, (".safetensors",)) or [DEFAULT_VAE],
            "lora": scan_models(loras_dir, (".safetensors",)) or [],
        }
    
    models = get_model_choices()
    comfyui_options = fetch_comfyui_options(services.kit)
    samplers = comfyui_options["samplers"]
    schedulers = comfyui_options["schedulers"]
    
    # Check installed models to determine startup state
    model_status = check_models_installed(diffusion_dir, text_encoders_dir, vae_dir)
    show_setup_banner = not model_status["has_any"]
    
    # Load saved model defaults, or fall back to auto-detection
    saved_defaults = services.settings.get("model_defaults")
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
    
    with gr.TabItem(TAB_LABEL, id=TAB_ID) as tab:
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
                        with gr.Row():
                            generate_t2i_btn = gr.Button("⚡ Generate", variant="primary", size="sm", scale=3)
                            enhance_btn = gr.Button("✨ Enhance", variant="huggingface", size="sm", scale=1)
                        with gr.Group():                        
                            with gr.Row():
                                width = gr.Slider(label="Width", value=1024, minimum=512, maximum=2048, step=32)
                                height = gr.Slider(label="Height", value=1024, minimum=512, maximum=2048, step=32)
                                  
                            with gr.Row():
                                res_base = gr.Radio(
                                    choices=["1024", "1280", "1536"],
                                    value="1024",
                                    label="Resolution",
                                    show_label=False,
                                    scale=1,
                                    min_width=180,
                                    elem_classes=["res-radio-compact"]
                                )
                                res_preset = gr.Dropdown(
                                    choices=get_resolution_dropdown_choices("1024"),
                                    value="1024x1024 ( 1:1 )",
                                    label="Aspect Ratio",
                                    show_label=False,
                                    scale=1,
                                    interactive=True
                                )
                    
                    with gr.TabItem("Image → Image"):
                        input_image = gr.Image(label="Input Image", type="filepath", height=360)
                        with gr.Row():
                            generate_i2i_btn = gr.Button("⚡ Generate", variant="primary", size="sm", scale=3)
                            i2i_describe_btn = gr.Button("🖼️ Describe", variant="huggingface", size="sm", scale=1)
                        with gr.Row():                                    
                            i2i_assist_status = gr.Textbox(
                                value="💡 Use Describe to generate a prompt describing your image. A low denoise img2img pass can greatly enhance existing images. Add a character LoRA for powerful transformations!",
                                lines=2.5,
                                interactive=False,
                                show_label=False
                            )
                        with gr.Row():
                            megapixels = gr.Slider(label="Megapixels", info="Scales against input image to maintain aspect ratio", value=1.5, minimum=0.5, maximum=3.0, step=0.1)
                            denoise = gr.Slider(label="Denoise", value=0.67, minimum=0.0, maximum=1.0, step=0.01)
                    


                
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
                        seed = gr.Number(label="Seed", value=new_random_seed(), minimum=0, step=1, scale=1)
                        randomize_seed = gr.Checkbox(label="🎲", value=True, scale=0, min_width=60)
                        batch_count = gr.Slider(label="Batch", value=1, minimum=1, maximum=100, step=1, scale=2, info="Images to generate")
                
                # Seed Variance - adds noise to text embeddings for more variation
                with gr.Accordion("🎲 Seed Variance", open=False):
                    gr.Markdown("*Add controlled noise to text embeddings for more variation across seeds*")
                    with gr.Row():
                        sv_enabled = gr.Checkbox(label="Enable", value=False, scale=0, min_width=80)
                        sv_noise_insert = gr.Dropdown(
                            label="Noise Insert",
                            choices=["noise on beginning steps", "noise on ending steps", "noise on all steps"],
                            value="noise on beginning steps",
                            scale=2,
                            info="Which steps use noisy embeddings"
                        )
                    with gr.Row():
                        sv_randomize_percent = gr.Slider(
                            label="Randomize %",
                            value=50.0, minimum=0.0, maximum=100.0, step=1,
                            info="Percentage of embedding values to add noise to"
                        )
                        sv_strength = gr.Slider(
                            label="Strength",
                            value=20.0, minimum=0.0, maximum=100.0, step=0.5,
                            info="Scale of the random noise"
                        )
                    with gr.Row():
                        sv_steps_switchover_percent = gr.Slider(
                            label="Steps Switchover %",
                            value=20.0, minimum=0.0, maximum=100.0, step=1,
                            info="When to switch between noisy and original embeddings"
                        )
                        sv_seed = gr.Number(
                            label="Variance Seed",
                            value=0, minimum=0, step=1,
                            info="Seed for noise generation (0 = use main seed)"
                        )
                    with gr.Row():
                        sv_mask_starts_at = gr.Dropdown(
                            label="Mask Starts At",
                            choices=["beginning", "end"],
                            value="beginning",
                            info="Which part of prompt to protect from noise"
                        )
                        sv_mask_percent = gr.Slider(
                            label="Mask %",
                            value=0.0, minimum=0.0, maximum=100.0, step=1,
                            info="Percentage of prompt protected from noise"
                        )
                
                # LoRA settings (6 slots with progressive reveal via lora_ui module)
                from modules.lora_ui import create_lora_ui, setup_lora_handlers, get_lora_inputs
                lora_components = create_lora_ui(loras_dir, accordion_open=False, initial_visible=1)
                
                # Model selection - auto-open if setup needed
                with gr.Accordion("🔧 Models", open=show_setup_banner):
                    with gr.Row():
                        use_gguf = gr.Radio(choices=[("Standard", False), ("GGUF", True)], value=default_gguf_mode, label="Mode", scale=1)
                        show_all_models = gr.Checkbox(label="Show all", value=False, scale=1, min_width=80, info="Show non-Z-Image models")
                    
                    # Compute initial values - use saved defaults if available, else auto-detect
                    initial_diffusion = get_models_by_mode(diffusion_dir, default_gguf_mode, DEFAULT_DIFFUSION, DEFAULT_DIFFUSION_GGUF, ZIMAGE_FILTERS["diffusion"])
                    initial_diffusion_value = default_diffusion if default_diffusion and default_diffusion in initial_diffusion else get_default_model(initial_diffusion, DEFAULT_DIFFUSION_GGUF if default_gguf_mode else DEFAULT_DIFFUSION)
                    
                    initial_te = get_models_by_mode(text_encoders_dir, default_gguf_mode, DEFAULT_TEXT_ENCODER, DEFAULT_TEXT_ENCODER_GGUF, ZIMAGE_FILTERS["text_encoder"])
                    initial_te_value = default_te if default_te and default_te in initial_te else get_default_model(initial_te, DEFAULT_TEXT_ENCODER_GGUF if default_gguf_mode else DEFAULT_TEXT_ENCODER)
                    
                    initial_vae = scan_models(vae_dir, (".safetensors",), ZIMAGE_FILTERS["vae"]) or [DEFAULT_VAE]
                    initial_vae_value = default_vae if default_vae and default_vae in initial_vae else get_default_model(initial_vae, DEFAULT_VAE)
                    
                    # Check if initial selections exist on disk
                    def get_status_icon(folder: Path, filename: str) -> str:
                        return "✓" if (folder / filename).exists() else "❌"
                    
                    initial_diff_status = get_status_icon(diffusion_dir, initial_diffusion_value) if initial_diffusion_value else "❌"
                    initial_te_status = get_status_icon(text_encoders_dir, initial_te_value) if initial_te_value else "❌"
                    initial_vae_status = get_status_icon(vae_dir, initial_vae_value) if initial_vae_value else "❌"
                    
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
                            dl_diffusion_bf16_btn = gr.Button("⬇️ Diffusion (bf16)", size="sm")
                            dl_te_bf16_btn = gr.Button("⬇️ Text Encoder (bf16)", size="sm")
                            dl_vae_btn = gr.Button("⬇️ VAE", size="sm")
                        gr.Markdown("---")
                        gr.Markdown("*GGUF (Q4) complete set — Quantized, ~8GB total, lower VRAM*")
                        with gr.Row():
                            dl_all_gguf_btn = gr.Button("⬇️ Download All (GGUF)", variant="primary", size="sm")
                        gr.Markdown("*Or download individually*")
                        with gr.Row():
                            dl_diffusion_gguf_btn = gr.Button("⬇️ Diffusion (GGUF)", size="sm")
                            dl_te_gguf_btn = gr.Button("⬇️ Text Encoder (GGUF)", size="sm")
                            dl_vae_gguf_btn = gr.Button("⬇️ VAE", size="sm")

                        gr.Markdown("---")

                        gr.Markdown("*Advanced — Browse repos for other quants*")
                        with gr.Row():
                            gr.Button("🔗 Z-Image GGUF Repo", size="sm", link="https://huggingface.co/gguf-org/z-image-gguf/tree/main")
                            gr.Button("🔗 Qwen3 GGUF Repo", size="sm", link="https://huggingface.co/Qwen/Qwen3-4B-GGUF/tree/main")
                                    
            # Right column - output
            with gr.Column(scale=1):
                output_gallery = gr.Gallery(
                    label="Generated Images",
                    columns=4,
                    rows=2,
                    height=400,
                    object_fit="contain",
                    show_download_button=True,
                    show_share_button=False,
                    preview=True,
                    elem_id="output-gallery"
                )
                
                # Hidden state for selected gallery image
                selected_gallery_image = gr.State(value=None)
                
                with gr.Row():
                    save_btn = gr.Button("💾 Save", size="sm", variant="primary")
                    send_to_upscale_btn = gr.Button("🔍 Send to Upscale", size="sm", variant="huggingface")
                    open_folder_btn = gr.Button("📂 Open Folder", size="sm")
                
                with gr.Row():
                    autosave = gr.Checkbox(label="Auto-save", value=False, elem_classes="checkbox-compact")
                    gen_status = gr.Textbox(label="Status", interactive=False, show_label=False, lines=2)
                with gr.Row():
                    stop_btn = gr.Button("⏹️ Stop Generation", size="sm", variant="stop")
                    unload_btn = gr.Button("🗑️ Unload Comfyui Models", size="sm")
                
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
                
                # Image metadata reader
                with gr.Accordion("🔍 Read Image Metadata", open=False):
                    gr.Markdown("*Drop a ComfyUI-generated image to extract prompt & settings*")
                    meta_image = gr.Image(label="Drop image here", type="filepath", height=250)
                    meta_output = gr.Textbox(label="Metadata", lines=10, interactive=False, placeholder="Note that comfyui images not generated in z-image-fusion may not have compatible metadata, ie from workflows with parameters set in custom nodes etc.  SeedVR2 upscaled images don't contain generation metadata.", show_copy_button=True)
                    with gr.Row():
                        meta_to_prompt_btn = gr.Button("📋 Copy Prompt", size="sm", variant="huggingface")
                        meta_to_settings_btn = gr.Button("⚙️ Apply Settings", size="sm", variant="primary")
                
                # Camera prompts helper
                with gr.Accordion("📷 Camera Prompts", open=False):
                    gr.Markdown("*Visual reference for camera angles, shots, and compositions*")
                    open_camera_prompts_btn = gr.Button("🔗 Open Camera Prompts Generator", size="sm")
                
                # Getting Started guide
                with gr.Accordion("ℹ️ Getting Started", open=False):
                    gr.Markdown("""
**First Time Setup**
1. Download models in **🔧 Models** section (left panel)
2. Choose **GGUF** for lower VRAM (8GB) or **Standard** for full precision (16GB+)
3. Click the download button — check Pinokio's `->_ Terminal` button (top bar) for progress

**Already have ComfyUI via Pinokio?**  
Your models & LoRAs are automatically shared — no re-download needed!

**✨ Prompt Enhance**
- Click **Enhance** next to Generate to expand simple prompts into detailed descriptions
- Use **Describe** in Image→Image to generate prompts from uploaded images
- Defaults work great, but you can change LLMs in the **⚙️ LLM Settings** tab

**🎲 Seed Variance**
Distilled "turbo" models can produce similar images across different seeds, especially with detailed prompts. Seed Variance fixes this by adding controlled noise to text embeddings, giving you more diverse outputs.
- **When to use**: Enable when your batch generations look too similar
- **Start with**: Strength 15-30, Randomize 50%, Switchover 20%
- **Key insight**: Detailed prompts = more variation (more values to randomize)
- **Advanced**: Use masking to protect important parts of your prompt from noise

**Tips**
- Default settings are tuned for the Z-Image Turbo model
- Use 🧹 **Unload ComfyUI Models** to keep Z-Image-Fusion active while freeing resources for other activities.
- Check the GPU/CPU monitor to track resource usage
""")
        
        # ===== EVENT HANDLERS =====
        _setup_event_handlers(
            services=services,
            # UI components
            prompt=prompt,
            setup_banner=setup_banner,
            # T2I components
            generate_t2i_btn=generate_t2i_btn,
            width=width,
            height=height,
            res_base=res_base,
            res_preset=res_preset,
            # I2I components
            generate_i2i_btn=generate_i2i_btn,
            input_image=input_image,
            i2i_describe_btn=i2i_describe_btn,
            i2i_assist_status=i2i_assist_status,
            megapixels=megapixels,
            denoise=denoise,
            # Prompt enhance
            enhance_btn=enhance_btn,
            # Settings
            steps=steps,
            cfg=cfg,
            shift=shift,
            sampler_name=sampler_name,
            scheduler=scheduler,
            seed=seed,
            randomize_seed=randomize_seed,
            batch_count=batch_count,
            # Seed variance
            sv_enabled=sv_enabled,
            sv_noise_insert=sv_noise_insert,
            sv_randomize_percent=sv_randomize_percent,
            sv_strength=sv_strength,
            sv_steps_switchover_percent=sv_steps_switchover_percent,
            sv_seed=sv_seed,
            sv_mask_starts_at=sv_mask_starts_at,
            sv_mask_percent=sv_mask_percent,
            # LoRA (using lora_ui module)
            lora_components=lora_components,
            # Models
            use_gguf=use_gguf,
            show_all_models=show_all_models,
            unet_name=unet_name,
            unet_status=unet_status,
            clip_name=clip_name,
            clip_status=clip_status,
            vae_name=vae_name,
            vae_status=vae_status,
            save_model_defaults_btn=save_model_defaults_btn,
            model_defaults_status=model_defaults_status,
            # Model management
            open_diffusion_btn=open_diffusion_btn,
            open_te_btn=open_te_btn,
            open_vae_btn=open_vae_btn,
            download_status=download_status,
            dl_all_standard_btn=dl_all_standard_btn,
            dl_diffusion_bf16_btn=dl_diffusion_bf16_btn,
            dl_te_bf16_btn=dl_te_bf16_btn,
            dl_vae_btn=dl_vae_btn,
            dl_all_gguf_btn=dl_all_gguf_btn,
            dl_diffusion_gguf_btn=dl_diffusion_gguf_btn,
            dl_te_gguf_btn=dl_te_gguf_btn,
            dl_vae_gguf_btn=dl_vae_gguf_btn,
            # Output
            output_gallery=output_gallery,
            selected_gallery_image=selected_gallery_image,
            save_btn=save_btn,
            send_to_upscale_btn=send_to_upscale_btn,
            open_folder_btn=open_folder_btn,
            autosave=autosave,
            stop_btn=stop_btn,
            unload_btn=unload_btn,
            gen_status=gen_status,
            gpu_monitor=gpu_monitor,
            cpu_monitor=cpu_monitor,
            # Metadata reader
            meta_image=meta_image,
            meta_output=meta_output,
            meta_to_prompt_btn=meta_to_prompt_btn,
            meta_to_settings_btn=meta_to_settings_btn,
            open_camera_prompts_btn=open_camera_prompts_btn,
            # Directories
            diffusion_dir=diffusion_dir,
            text_encoders_dir=text_encoders_dir,
            vae_dir=vae_dir,
            loras_dir=loras_dir,
            # Valid options for metadata apply
            samplers=samplers,
            schedulers=schedulers,
        )
    
    return tab


def _setup_event_handlers(
    services: "SharedServices",
    # All UI components passed as kwargs
    **components
):
    """Set up all event handlers for the Z-Image tab."""
    import subprocess
    import sys
    import webbrowser
    
    # Extract components
    prompt = components["prompt"]
    setup_banner = components["setup_banner"]
    generate_t2i_btn = components["generate_t2i_btn"]
    width = components["width"]
    height = components["height"]
    res_base = components["res_base"]
    res_preset = components["res_preset"]
    generate_i2i_btn = components["generate_i2i_btn"]
    input_image = components["input_image"]
    i2i_describe_btn = components["i2i_describe_btn"]
    i2i_assist_status = components["i2i_assist_status"]
    megapixels = components["megapixels"]
    denoise = components["denoise"]
    enhance_btn = components["enhance_btn"]
    steps = components["steps"]
    cfg = components["cfg"]
    shift = components["shift"]
    sampler_name = components["sampler_name"]
    scheduler = components["scheduler"]
    seed = components["seed"]
    randomize_seed = components["randomize_seed"]
    batch_count = components["batch_count"]
    sv_enabled = components["sv_enabled"]
    sv_noise_insert = components["sv_noise_insert"]
    sv_randomize_percent = components["sv_randomize_percent"]
    sv_strength = components["sv_strength"]
    sv_steps_switchover_percent = components["sv_steps_switchover_percent"]
    sv_seed = components["sv_seed"]
    sv_mask_starts_at = components["sv_mask_starts_at"]
    sv_mask_percent = components["sv_mask_percent"]
    # LoRA components from lora_ui module
    lora_components = components["lora_components"]
    use_gguf = components["use_gguf"]
    show_all_models = components["show_all_models"]
    unet_name = components["unet_name"]
    unet_status = components["unet_status"]
    clip_name = components["clip_name"]
    clip_status = components["clip_status"]
    vae_name = components["vae_name"]
    vae_status = components["vae_status"]
    save_model_defaults_btn = components["save_model_defaults_btn"]
    model_defaults_status = components["model_defaults_status"]
    open_diffusion_btn = components["open_diffusion_btn"]
    open_te_btn = components["open_te_btn"]
    open_vae_btn = components["open_vae_btn"]
    download_status = components["download_status"]
    dl_all_standard_btn = components["dl_all_standard_btn"]
    dl_diffusion_bf16_btn = components["dl_diffusion_bf16_btn"]
    dl_te_bf16_btn = components["dl_te_bf16_btn"]
    dl_vae_btn = components["dl_vae_btn"]
    dl_all_gguf_btn = components["dl_all_gguf_btn"]
    dl_diffusion_gguf_btn = components["dl_diffusion_gguf_btn"]
    dl_te_gguf_btn = components["dl_te_gguf_btn"]
    dl_vae_gguf_btn = components["dl_vae_gguf_btn"]
    output_gallery = components["output_gallery"]
    selected_gallery_image = components["selected_gallery_image"]
    save_btn = components["save_btn"]
    send_to_upscale_btn = components["send_to_upscale_btn"]
    open_folder_btn = components["open_folder_btn"]
    autosave = components["autosave"]
    stop_btn = components["stop_btn"]
    unload_btn = components["unload_btn"]
    gen_status = components["gen_status"]
    gpu_monitor = components["gpu_monitor"]
    cpu_monitor = components["cpu_monitor"]
    meta_image = components["meta_image"]
    meta_output = components["meta_output"]
    meta_to_prompt_btn = components["meta_to_prompt_btn"]
    meta_to_settings_btn = components["meta_to_settings_btn"]
    open_camera_prompts_btn = components["open_camera_prompts_btn"]
    diffusion_dir = components["diffusion_dir"]
    text_encoders_dir = components["text_encoders_dir"]
    vae_dir = components["vae_dir"]
    loras_dir = components["loras_dir"]
    samplers = components["samplers"]
    schedulers = components["schedulers"]
    
    # Prompt Enhance button - outputs to gen_status
    if services.prompt_assistant:
        enhance_btn.click(
            fn=services.prompt_assistant.enhance_prompt,
            inputs=[prompt],
            outputs=[prompt, gen_status]
        )
        
        # I2I tab describe button - clears prompt first, then describes image
        i2i_describe_btn.click(
            fn=lambda: ("", "👁️ Preparing..."),
            outputs=[prompt, i2i_assist_status]
        ).then(
            fn=services.prompt_assistant.describe_image,
            inputs=[input_image, prompt],
            outputs=[prompt, i2i_assist_status]
        )
    
    # Resolution preset handlers
    def on_res_base_change(base):
        """Update dropdown choices when resolution base changes, keeping same AR if possible."""
        new_choices = get_resolution_dropdown_choices(base)
        # Default to square for the new base
        default_value = RES_CHOICES[base][0]
        return gr.update(choices=new_choices, value=default_value), *parse_resolution(default_value)
    
    def on_res_preset_change(preset):
        """Update width/height sliders when preset is selected."""
        # Skip divider labels
        if preset.startswith("──"):
            return gr.update(), gr.update()
        w, h = parse_resolution(preset)
        return w, h
    
    res_base.change(
        fn=on_res_base_change,
        inputs=[res_base],
        outputs=[res_preset, width, height]
    )
    
    res_preset.change(
        fn=on_res_preset_change,
        inputs=[res_preset],
        outputs=[width, height]
    )
    
    # Status indicator helpers
    def check_model_status(folder: Path, filename: str) -> str:
        """Return ✓ if file exists, ❌ if not."""
        if not filename:
            return "❌"
        return "✓" if (folder / filename).exists() else "❌"
    
    def update_unet_status(unet):
        return check_model_status(diffusion_dir, unet)
    
    def update_clip_status(clip):
        return check_model_status(text_encoders_dir, clip)
    
    def update_vae_status(vae):
        return check_model_status(vae_dir, vae)
    
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
        
        diffusion_models = get_models_by_mode(diffusion_dir, is_gguf, DEFAULT_DIFFUSION, DEFAULT_DIFFUSION_GGUF, diff_filter)
        clip_models = get_models_by_mode(text_encoders_dir, is_gguf, DEFAULT_TEXT_ENCODER, DEFAULT_TEXT_ENCODER_GGUF, te_filter)
        vae_models = scan_models(vae_dir, (".safetensors",), vae_filter) or [DEFAULT_VAE]
        
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
            check_model_status(diffusion_dir, diff_value),
            check_model_status(text_encoders_dir, clip_value),
            check_model_status(vae_dir, vae_value),
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
        settings = services.settings.load()
        settings["model_defaults"] = {
            "use_gguf": is_gguf,
            "diffusion": diffusion,
            "text_encoder": text_encoder,
            "vae": vae,
        }
        services.settings.save(settings)
        return "✓ Saved as default"
    
    save_model_defaults_btn.click(
        fn=save_model_defaults,
        inputs=[use_gguf, unet_name, clip_name, vae_name],
        outputs=[model_defaults_status]
    )
    
    # Set up LoRA handlers using lora_ui module
    from modules.lora_ui import setup_lora_handlers, get_lora_inputs
    setup_lora_handlers(lora_components, loras_dir)
    lora_inputs = get_lora_inputs(lora_components)
    
    # Metadata reader handlers
    # Store extracted metadata for use by buttons
    extracted_metadata = gr.State(value={})
    
    def on_meta_image_change(image_path):
        """Extract and display metadata when image is uploaded."""
        if not image_path:
            return "", {}
        metadata = extract_png_metadata(image_path)
        display = format_metadata_display(metadata)
        return display, metadata
    
    meta_image.change(
        fn=on_meta_image_change,
        inputs=[meta_image],
        outputs=[meta_output, extracted_metadata]
    )
    
    def copy_prompt_from_metadata(metadata):
        """Copy extracted prompt to the main prompt field."""
        if metadata and metadata.get("prompt_text"):
            return metadata["prompt_text"]
        return gr.update()
    
    meta_to_prompt_btn.click(
        fn=copy_prompt_from_metadata,
        inputs=[extracted_metadata],
        outputs=[prompt]
    )
    
    # Get current valid options for validation
    available_loras = scan_models(loras_dir, (".safetensors",))
    
    def apply_settings_from_metadata(metadata):
        """Apply extracted settings to the UI controls including prompt and LoRAs.
        
        Only applies values that are valid for the current setup (e.g., installed LoRAs,
        available samplers/schedulers).
        """
        # 28 outputs: prompt, seed, randomize_seed, steps, cfg, shift, sampler, scheduler, width, height,
        #             lora1-6 (enabled, name, strength) = 18 lora outputs
        no_update = [gr.update()] * 28
        
        if not metadata:
            return no_update
        
        params = metadata.get("params", {})
        prompt_text = metadata.get("prompt_text", "")
        loras_from_meta = params.get("loras", [])
        
        # Build LoRA updates (6 slots) - only apply if LoRA exists locally
        lora_updates = []
        for i in range(6):
            if i < len(loras_from_meta):
                lora = loras_from_meta[i]
                lora_name = lora["name"]
                # Check if this LoRA exists in our collection
                if lora_name in available_loras:
                    lora_updates.extend([
                        gr.update(value=True),  # enabled
                        gr.update(value=lora_name),  # name
                        gr.update(value=lora["strength"]),  # strength
                    ])
                else:
                    # LoRA not found - skip it (don't enable, don't change name)
                    logger.warning(f"LoRA not found locally, skipping: {lora_name}")
                    lora_updates.extend([
                        gr.update(value=False),  # disabled
                        gr.update(),  # keep current name
                        gr.update(),  # keep current strength
                    ])
            else:
                # Clear unused slots
                lora_updates.extend([
                    gr.update(value=False),  # disabled
                    gr.update(),  # keep name
                    gr.update(),  # keep strength
                ])
        
        # Only apply sampler/scheduler if they're in our available lists
        sampler_update = gr.update()
        if "sampler" in params and params["sampler"] in samplers:
            sampler_update = gr.update(value=params["sampler"])
        elif "sampler" in params:
            logger.warning(f"Sampler not available, skipping: {params['sampler']}")
        
        scheduler_update = gr.update()
        if "scheduler" in params and params["scheduler"] in schedulers:
            scheduler_update = gr.update(value=params["scheduler"])
        elif "scheduler" in params:
            logger.warning(f"Scheduler not available, skipping: {params['scheduler']}")
        
        # Helper to check if a value is a usable number (not a node reference like ['67', 0])
        def is_valid_number(val):
            return isinstance(val, (int, float)) and not isinstance(val, bool)
        
        def is_valid_int(val):
            return isinstance(val, int) and not isinstance(val, bool)
        
        return (
            gr.update(value=prompt_text) if prompt_text else gr.update(),
            gr.update(value=int(params["seed"])) if "seed" in params and is_valid_number(params["seed"]) else gr.update(),
            gr.update(value=False) if "seed" in params and is_valid_number(params["seed"]) else gr.update(),  # Only uncheck if we have a valid seed
            gr.update(value=int(params["steps"])) if "steps" in params and is_valid_int(params["steps"]) else gr.update(),
            gr.update(value=params["cfg"]) if "cfg" in params and is_valid_number(params["cfg"]) else gr.update(),
            gr.update(value=params["shift"]) if "shift" in params and is_valid_number(params["shift"]) else gr.update(),
            sampler_update,
            scheduler_update,
            gr.update(value=int(params["width"])) if "width" in params and is_valid_int(params["width"]) else gr.update(),
            gr.update(value=int(params["height"])) if "height" in params and is_valid_int(params["height"]) else gr.update(),
            *lora_updates
        )
    
    # Build lora output components list for metadata apply
    lora_output_components = []
    for slot in lora_components.slots:
        lora_output_components.extend([slot.enabled, slot.name, slot.strength])
    
    meta_to_settings_btn.click(
        fn=apply_settings_from_metadata,
        inputs=[extracted_metadata],
        outputs=[
            prompt, seed, randomize_seed, steps, cfg, shift, sampler_name, scheduler, width, height,
            *lora_output_components
        ]
    )
    
    # Shared inputs for both generate buttons
    common_inputs = [
        use_gguf,
        steps, seed, randomize_seed, cfg, shift,
        sampler_name, scheduler,
        unet_name, clip_name, vae_name,
        # 6 lora slots (enabled, name, strength) via lora_inputs
        *lora_inputs,
        autosave,
        batch_count,
        # Seed variance params
        sv_enabled, sv_noise_insert, sv_randomize_percent, sv_strength,
        sv_steps_switchover_percent, sv_seed, sv_mask_starts_at, sv_mask_percent
    ]
    
    # Wrapper functions for async generate (async generators)
    # Returns (gallery, status, seed, selected_image) - selected_image is always None to clear stale selection
    async def generate_t2i(p, w, h, gguf, *args):
        async for gallery, status, seed_val in generate_image(services, p, "t2i", gguf, w, h, None, 2.0, 0.67, *args):
            yield gallery, status, seed_val, None  # Clear selected_gallery_image on each yield
    
    async def generate_i2i(p, img, mp, dn, gguf, *args):
        async for gallery, status, seed_val in generate_image(services, p, "i2i", gguf, 1024, 1024, img, mp, dn, *args):
            yield gallery, status, seed_val, None  # Clear selected_gallery_image on each yield
    
    # T2I generate - clears selected_gallery_image to prevent stale selection bug
    generate_t2i_btn.click(
        fn=generate_t2i,
        inputs=[prompt, width, height] + common_inputs,
        outputs=[output_gallery, gen_status, seed, selected_gallery_image]
    )
    
    # I2I generate - clears selected_gallery_image to prevent stale selection bug
    generate_i2i_btn.click(
        fn=generate_i2i,
        inputs=[prompt, input_image, megapixels, denoise] + common_inputs,
        outputs=[output_gallery, gen_status, seed, selected_gallery_image]
    )
    
    # Unload models
    async def unload_models() -> str:
        """Unload all models from ComfyUI to free VRAM."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{services.kit.comfyui_url}/free",
                    json={"unload_models": True, "free_memory": True}
                )
                if response.status_code == 200:
                    return "✓ ComfyUI models unloaded, VRAM freed"
                return f"❌ Failed: {response.status_code}"
        except Exception as e:
            return f"❌ Error: {e}"
    
    unload_btn.click(
        fn=unload_models,
        outputs=[gen_status]
    )
    
    # Stop generation (interrupt ComfyUI and cancel batch loop)
    async def stop_generation() -> str:
        """Interrupt current ComfyUI generation and cancel batch loop."""
        global _cancel_generation
        _cancel_generation = True  # Signal batch loop to stop
        try:
            async with httpx.AsyncClient() as client:
                # Interrupt current generation
                response = await client.post(f"{services.kit.comfyui_url}/interrupt")
                if response.status_code == 200:
                    return "⏹️ Stopping generation..."
                return f"❌ Failed: {response.status_code}"
        except Exception as e:
            return f"❌ Error: {e}"
    
    stop_btn.click(
        fn=stop_generation,
        outputs=[gen_status]
    )
    
    # Save selected image from gallery (or first if none selected)
    def save_selected_image(selected_img, gallery_data, prompt_text):
        image_to_save = None
        
        # Prefer explicitly selected image
        if selected_img:
            image_to_save = selected_img
        # Fall back to first gallery image
        elif gallery_data:
            item = gallery_data[0]
            image_to_save = item[0] if isinstance(item, (list, tuple)) else item
        
        if not image_to_save:
            return "❌ No image to save"
        
        outputs_dir = services.get_outputs_dir()
        saved_path = save_image_to_outputs(image_to_save, prompt_text or "image", outputs_dir)
        return f"✓ Saved: {Path(saved_path).name}"
    
    save_btn.click(
        fn=save_selected_image,
        inputs=[selected_gallery_image, output_gallery, prompt],
        outputs=[gen_status]
    )
    
    # Track gallery selection for "Send to Upscale"
    def on_gallery_select(evt: gr.SelectData, gallery_data):
        """Store the selected image path when user clicks on gallery item."""
        if gallery_data and evt.index < len(gallery_data):
            item = gallery_data[evt.index]
            image_path = item[0] if isinstance(item, (list, tuple)) else item
            return image_path
        return None
    
    output_gallery.select(
        fn=on_gallery_select,
        inputs=[output_gallery],
        outputs=[selected_gallery_image]
    )
    
    # Register components for post-load cross-module wiring of "Send to X" buttons
    # The actual click handlers are wired up in app.py after all modules are loaded
    services.inter_module.register_component("zimage_send_to_upscale_btn", send_to_upscale_btn)
    services.inter_module.register_component("zimage_selected_gallery_image", selected_gallery_image)
    services.inter_module.register_component("zimage_output_gallery", output_gallery)
    services.inter_module.register_component("zimage_gen_status", gen_status)
    
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
        open_folder(services.get_outputs_dir())
    
    open_folder_btn.click(fn=open_outputs_folder)
    
    # Model folder buttons
    open_diffusion_btn.click(fn=lambda: open_folder(diffusion_dir))
    open_te_btn.click(fn=lambda: open_folder(text_encoders_dir))
    open_vae_btn.click(fn=lambda: open_folder(vae_dir))
    
    # System Monitor 
    def update_monitor():
        if services.system_monitor:
            gpu_info, cpu_info = services.system_monitor.get_system_info()
            return gpu_info, cpu_info
        return "N/A", "N/A"
        
    monitor_timer = gr.Timer(2, active=True)
    monitor_timer.tick(fn=update_monitor, outputs=[gpu_monitor, cpu_monitor])
    
    # Model download buttons - refresh dropdowns and hide banner after download
    def download_and_refresh(is_gguf, show_all, model_key, progress=gr.Progress()):
        """Download a single model and refresh dropdowns."""
        status = download_model(model_key, services.models_dir, progress)
        dropdowns = update_model_dropdowns(is_gguf, show_all)
        # Check if we should hide the setup banner
        model_status = check_models_installed(diffusion_dir, text_encoders_dir, vae_dir)
        banner_visible = gr.update(visible=not model_status["has_any"])
        return (status,) + dropdowns + (banner_visible,)
    
    def download_all_and_refresh(is_gguf_mode, show_all, progress=gr.Progress()):
        """Download all models for a mode, switch to that mode, and refresh dropdowns."""
        if is_gguf_mode:
            status = download_all_gguf(services.models_dir, progress)
        else:
            status = download_all_standard(services.models_dir, progress)
        # Update dropdowns for the mode we just downloaded
        dropdowns = update_model_dropdowns(is_gguf_mode, show_all)
        # Check if we should hide the setup banner
        model_status = check_models_installed(diffusion_dir, text_encoders_dir, vae_dir)
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
        camera_html = services.app_dir / "CameraPromptsGenerator" / "index.html"
        if camera_html.exists():
            webbrowser.open(camera_html.as_uri())
            return "✓ Opened in browser"
        return "❌ Camera prompts not found"
    
    open_camera_prompts_btn.click(fn=open_camera_prompts)
