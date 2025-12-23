"""
Experimental Module

Provides a sandbox environment for testing new ComfyUI workflows before
production integration. First workflow: z_image_upscaleAny.json using
the FlowMatchEulerDiscreteScheduler custom node.

This module manages its own custom node dependencies, avoiding bloat
in the main install/update scripts.
"""

import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import gradio as gr
import httpx

if TYPE_CHECKING:
    from modules import SharedServices

logger = logging.getLogger(__name__)

# Module metadata
TAB_ID = "experimental"
TAB_LABEL = "🧪 Experimental"
TAB_ORDER = 2

# Custom node configuration
CUSTOM_NODE_NAME = "ComfyUI-EulerDiscreteScheduler"
CUSTOM_NODE_REPO = "https://github.com/erosDiffusion/ComfyUI-EulerDiscreteScheduler.git"

# Status message constants
STATUS_UPSCALING = "⏳ Upscaling..."
STATUS_SUCCESS_PREFIX = "✓"
STATUS_ERROR_PREFIX = "❌"

# Default samplers (fallback if ComfyUI not available)
DEFAULT_SAMPLERS = ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_sde"]

# Batch processing cancellation flag
_cancel_batch = False

# Session temp directory for batch results (auto-cleaned on exit)
# Using a persistent TemporaryDirectory so files have nice names for Gradio download
_batch_temp_dir = tempfile.TemporaryDirectory(prefix="experimental_batch_")

# Model defaults - Standard
DEFAULT_DIFFUSION = "z_image_turbo_bf16.safetensors"
DEFAULT_CLIP = "qwen_3_4b.safetensors"
DEFAULT_VAE = "ae.safetensors"

# Model defaults - GGUF
DEFAULT_DIFFUSION_GGUF = "z-image-turbo-q4_k_m.gguf"
DEFAULT_CLIP_GGUF = "Qwen3-4B-Q4_K_M.gguf"

# File extensions by mode
STANDARD_EXTENSIONS = (".safetensors", ".ckpt", ".pt")
GGUF_EXTENSIONS = (".gguf",)
MODEL_EXTENSIONS = (".safetensors", ".ckpt", ".pt", ".gguf")

# Supported image extensions for batch processing
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")

# Name filters for Z-Image compatible models
ZIMAGE_FILTERS = {
    "diffusion": "z",
    "text_encoder": "qwen",
    "vae": "ae",
}


def scan_models(folder: Path, extensions: tuple = MODEL_EXTENSIONS, name_filter: str = None) -> list:
    """Scan folder recursively for model files, returning relative paths."""
    if not folder.exists():
        return []
    models = []
    for ext in extensions:
        for f in folder.rglob(f"*{ext}"):
            rel_path = str(f.relative_to(folder))
            if name_filter is None or name_filter.lower() in rel_path.lower():
                models.append(rel_path)
    return sorted(models)


def get_default_model(choices: list, preferred: str) -> str:
    """Get default model, preferring the specified one if available."""
    if preferred in choices:
        return preferred
    return choices[0] if choices else preferred


def get_models_by_mode(folder: Path, is_gguf: bool, default_standard: str, default_gguf: str, name_filter: str = None) -> list:
    """Get models filtered by mode (standard vs GGUF) and optional name filter."""
    extensions = GGUF_EXTENSIONS if is_gguf else STANDARD_EXTENSIONS
    default = default_gguf if is_gguf else default_standard
    models = scan_models(folder, extensions, name_filter)
    return models or [default]


def get_upscale_workflow(use_gguf: bool) -> str:
    """Get the appropriate upscale workflow based on GGUF mode."""
    return "z_image_upscaleAny_gguf.json" if use_gguf else "z_image_upscaleAny.json"


def format_status_success(duration: float, saved: bool = False) -> str:
    """Format status message for successful upscale operation."""
    if saved:
        return f"{STATUS_SUCCESS_PREFIX} {duration:.1f}s | Saved"
    return f"{STATUS_SUCCESS_PREFIX} {duration:.1f}s"


def format_status_error(error_message: str) -> str:
    """Format status message for failed upscale operation."""
    return f"{STATUS_ERROR_PREFIX} {error_message}"


def new_random_seed():
    """Generate a new random seed."""
    return random.randint(0, 999999999999)


def get_images_from_folder(folder_path: str) -> List[str]:
    """Get list of image files from a folder path."""
    if not folder_path or not folder_path.strip():
        return []
    path = Path(folder_path.strip())
    if not path.exists() or not path.is_dir():
        return []
    images = set()  # Use set to avoid duplicates (Windows glob is case-insensitive)
    for ext in IMAGE_EXTENSIONS:
        images.update(str(f) for f in path.glob(f"*{ext}"))
        images.update(str(f) for f in path.glob(f"*{ext.upper()}"))
    return sorted(images)


def get_batch_images(batch_files: Optional[List], folder_path: str) -> List[str]:
    """Combine images from file upload and folder path."""
    images = []
    # From file upload
    if batch_files:
        for f in batch_files:
            if hasattr(f, 'name'):
                images.append(f.name)
            elif isinstance(f, str):
                images.append(f)
    # From folder path
    images.extend(get_images_from_folder(folder_path))
    return images


def check_custom_node_installed(custom_nodes_dir: Path, node_name: str) -> bool:
    """Check if a custom node directory exists."""
    node_path = custom_nodes_dir / node_name
    return node_path.exists() and node_path.is_dir()


def install_custom_node(custom_nodes_dir: Path, repo_url: str, node_name: str) -> tuple[bool, str]:
    """Clone a custom node repository."""
    node_path = custom_nodes_dir / node_name
    if node_path.exists():
        return False, f"Custom node '{node_name}' already exists"
    try:
        custom_nodes_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cloning {repo_url} to {node_path}")
        result = subprocess.run(
            ["git", "clone", repo_url, str(node_path)],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip() or "Unknown error"
            return False, f"Failed to clone: {error_msg}"
        return True, f"✓ Installed '{node_name}'. Please restart ComfyUI to load the node."
    except subprocess.TimeoutExpired:
        return False, "Installation timed out. Check your network connection."
    except FileNotFoundError:
        return False, "Git is not installed or not in PATH"
    except Exception as e:
        return False, f"Installation failed: {str(e)}"


def update_custom_node(custom_nodes_dir: Path, node_name: str) -> tuple[bool, str]:
    """Git pull an existing custom node."""
    node_path = custom_nodes_dir / node_name
    if not node_path.exists():
        return False, f"Custom node '{node_name}' is not installed"
    try:
        result = subprocess.run(
            ["git", "pull"], cwd=str(node_path),
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            return False, f"Failed to update: {result.stderr.strip() or 'Unknown error'}"
        if "Already up to date" in result.stdout:
            return True, f"✓ '{node_name}' is already up to date"
        return True, f"✓ Updated '{node_name}'. Please restart ComfyUI to apply changes."
    except subprocess.TimeoutExpired:
        return False, "Update timed out."
    except Exception as e:
        return False, f"Update failed: {str(e)}"


async def download_image_from_url(url: str) -> str:
    """Download image from ComfyUI URL to a local temp file."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        suffix = Path(url).suffix or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(response.content)
            return f.name


def copy_to_temp_with_name(image_path: str, original_path: str) -> str:
    """Copy image to session temp dir with a meaningful name for Gradio download."""
    timestamp = datetime.now().strftime("%H%M%S")
    if original_path:
        original_stem = Path(original_path).stem[:30]
        safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in original_stem)
    else:
        safe_stem = "image"
    filename = f"{safe_stem}_enhanced_{timestamp}.png"
    temp_path = Path(_batch_temp_dir.name) / filename
    shutil.copy2(image_path, temp_path)
    return str(temp_path)


def save_experimental_output(image_path: str, original_path: str, outputs_dir: Path) -> str:
    """Save enhanced image to outputs/experimental folder."""
    timestamp = datetime.now().strftime("%H%M%S")
    if original_path:
        original_stem = Path(original_path).stem[:30]
        safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in original_stem)
    else:
        safe_stem = "image"
    target_dir = outputs_dir / "experimental"
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{safe_stem}_enhanced_{timestamp}.png"
    output_path = target_dir / filename
    shutil.copy2(image_path, output_path)
    logger.info(f"Saved experimental output to: {output_path}")
    return str(output_path)


def open_folder(folder_path: Path):
    """Cross-platform folder opener."""
    folder_path.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        os.startfile(folder_path)
    elif sys.platform == "darwin":
        subprocess.run(["open", str(folder_path)])
    else:
        subprocess.run(["xdg-open", str(folder_path)])


async def experimental_upscale_single(
    services: "SharedServices",
    input_image: str,
    prompt: str,
    seed: int,
    megapixels: float,
    scale_by: float,
    steps: int,
    start_at_step: int,
    end_at_step: int,
    shift: float,
    cfg: float,
    sampler_name: str,
    use_gguf: bool,
    unet_name: str,
    clip_name: str,
    vae_name: str,
    base_shift: float,
    max_shift: float,
    use_karras_sigmas: str,
    stochastic_sampling: str,
    autosave: bool,
    lora_params: dict,
) -> tuple:
    """
    Execute single image upscale. Returns (result_path, status, duration).
    """
    from modules.lora_ui import DUMMY_LORA
    outputs_dir = services.get_outputs_dir()
    start_time = time.time()
    
    workflow_file = get_upscale_workflow(use_gguf)
    workflow_path = services.workflows_dir / workflow_file
    if not workflow_path.exists():
        return None, f"Workflow not found: {workflow_file}", 0
    
    params = {
        "image": input_image,
        "prompt": prompt.strip() if prompt else "",
        "seed": int(seed),
        "cfg": float(cfg),
        "scale_by": float(scale_by),
        "megapixels": float(megapixels),
        "steps": int(steps),
        "start_at_step": int(start_at_step),
        "end_at_step": int(end_at_step),
        "shift": float(shift),
        "sampler_name": sampler_name,
        "unet_name": unet_name,
        "clip_name": clip_name,
        "vae_name": vae_name,
        "base_shift": float(base_shift),
        "max_shift": float(max_shift),
        "use_karras_sigmas": use_karras_sigmas,
        "stochastic_sampling": stochastic_sampling,
    }
    params.update(lora_params)
    
    try:
        result = await services.kit.execute(str(workflow_path), params)
        if result.status == "error":
            return None, f"Failed: {result.msg}", 0
        if not result.images:
            return None, "No images generated", 0
        
        image_path = result.images[0]
        if image_path.startswith("http"):
            image_path = await download_image_from_url(image_path)
        
        # Copy to temp with meaningful name so Gradio download button works nicely
        image_path = copy_to_temp_with_name(image_path, input_image)
        
        duration = time.time() - start_time
        
        if autosave:
            save_experimental_output(image_path, input_image, outputs_dir)
        
        return image_path, "success", duration
    except Exception as e:
        return None, str(e), 0


async def experimental_upscale(
    services: "SharedServices",
    input_image: str,
    prompt: str,
    seed: int,
    randomize_seed: bool,
    megapixels: float,
    scale_by: float,
    steps: int,
    start_at_step: int,
    end_at_step: int,
    shift: float,
    cfg: float,
    sampler_name: str,
    use_gguf: bool,
    unet_name: str,
    clip_name: str,
    vae_name: str,
    base_shift: float,
    max_shift: float,
    use_karras_sigmas: str,
    stochastic_sampling: str,
    autosave: bool,
    lora1_enabled: bool = False,
    lora1_name: str = None,
    lora1_strength: float = 1.0,
    lora2_enabled: bool = False,
    lora2_name: str = None,
    lora2_strength: float = 1.0,
    lora3_enabled: bool = False,
    lora3_name: str = None,
    lora3_strength: float = 1.0,
    lora4_enabled: bool = False,
    lora4_name: str = None,
    lora4_strength: float = 1.0,
    lora5_enabled: bool = False,
    lora5_name: str = None,
    lora5_strength: float = 1.0,
    lora6_enabled: bool = False,
    lora6_name: str = None,
    lora6_strength: float = 1.0,
):
    """Execute single image upscale workflow. Yields (slider_tuple, status, seed, result_path)."""
    from modules.lora_ui import get_lora_params
    
    actual_seed = new_random_seed() if randomize_seed else int(seed)
    
    if input_image is None:
        yield None, format_status_error("Please upload an image"), actual_seed, None
        return
    
    yield None, STATUS_UPSCALING, actual_seed, None
    
    lora_params = get_lora_params(
        lora1_enabled, lora1_name, lora1_strength,
        lora2_enabled, lora2_name, lora2_strength,
        lora3_enabled, lora3_name, lora3_strength,
        lora4_enabled, lora4_name, lora4_strength,
        lora5_enabled, lora5_name, lora5_strength,
        lora6_enabled, lora6_name, lora6_strength,
    )
    
    result_path, status_msg, duration = await experimental_upscale_single(
        services, input_image, prompt, actual_seed, megapixels, scale_by,
        steps, start_at_step, end_at_step, shift, cfg, sampler_name,
        use_gguf, unet_name, clip_name, vae_name,
        base_shift, max_shift, use_karras_sigmas, stochastic_sampling,
        autosave, lora_params
    )
    
    if result_path is None:
        yield None, format_status_error(status_msg), actual_seed, None
    else:
        status = format_status_success(duration, saved=autosave)
        yield (input_image, result_path), status, actual_seed, result_path


async def experimental_upscale_batch(
    services: "SharedServices",
    batch_files: Optional[List],
    folder_path: str,
    prompt: str,
    seed: int,
    randomize_seed: bool,
    megapixels: float,
    scale_by: float,
    steps: int,
    start_at_step: int,
    end_at_step: int,
    shift: float,
    cfg: float,
    sampler_name: str,
    use_gguf: bool,
    unet_name: str,
    clip_name: str,
    vae_name: str,
    base_shift: float,
    max_shift: float,
    use_karras_sigmas: str,
    stochastic_sampling: str,
    autosave: bool,
    lora1_enabled: bool = False,
    lora1_name: str = None,
    lora1_strength: float = 1.0,
    lora2_enabled: bool = False,
    lora2_name: str = None,
    lora2_strength: float = 1.0,
    lora3_enabled: bool = False,
    lora3_name: str = None,
    lora3_strength: float = 1.0,
    lora4_enabled: bool = False,
    lora4_name: str = None,
    lora4_strength: float = 1.0,
    lora5_enabled: bool = False,
    lora5_name: str = None,
    lora5_strength: float = 1.0,
    lora6_enabled: bool = False,
    lora6_name: str = None,
    lora6_strength: float = 1.0,
):
    """Execute batch upscale workflow. Yields (gallery_images, status, seed)."""
    global _cancel_batch
    from modules.lora_ui import get_lora_params
    
    images = get_batch_images(batch_files, folder_path)
    if not images:
        yield [], format_status_error("No images found. Upload files or enter a folder path."), seed
        return
    
    base_seed = new_random_seed() if randomize_seed else int(seed)
    _cancel_batch = False
    
    lora_params = get_lora_params(
        lora1_enabled, lora1_name, lora1_strength,
        lora2_enabled, lora2_name, lora2_strength,
        lora3_enabled, lora3_name, lora3_strength,
        lora4_enabled, lora4_name, lora4_strength,
        lora5_enabled, lora5_name, lora5_strength,
        lora6_enabled, lora6_name, lora6_strength,
    )
    
    results = []
    total = len(images)
    total_duration = 0.0
    
    for i, img_path in enumerate(images):
        if _cancel_batch:
            _cancel_batch = False
            yield results, f"⏹️ Cancelled after {i}/{total} images", base_seed
            return
        
        current_seed = base_seed + i
        yield results, f"⏳ [{i+1}/{total}] Processing {Path(img_path).name}...", base_seed
        
        result_path, status_msg, duration = await experimental_upscale_single(
            services, img_path, prompt, current_seed, megapixels, scale_by,
            steps, start_at_step, end_at_step, shift, cfg, sampler_name,
            use_gguf, unet_name, clip_name, vae_name,
            base_shift, max_shift, use_karras_sigmas, stochastic_sampling,
            autosave, lora_params
        )
        
        if result_path:
            results.append(result_path)
            total_duration += duration
        else:
            logger.warning(f"Batch item {i+1} failed: {status_msg}")
    
    avg_time = total_duration / len(results) if results else 0
    status = f"✓ {len(results)}/{total} images | {total_duration:.1f}s total ({avg_time:.1f}s avg)"
    if autosave:
        status += " | Saved"
    yield results, status, base_seed


def create_tab(services: "SharedServices") -> gr.TabItem:
    """Create the Experimental tab with sub-tabs for different workflows."""
    from modules.lora_ui import create_lora_ui, setup_lora_handlers, get_lora_inputs
    
    custom_nodes_dir = services.app_dir / "comfyui" / "custom_nodes"
    outputs_dir = services.get_outputs_dir()
    experimental_dir = outputs_dir / "experimental"
    
    # Model directories
    diffusion_dir = services.models_dir / "diffusion_models"
    text_encoders_dir = services.models_dir / "text_encoders"
    vae_dir = services.models_dir / "vae"
    loras_dir = services.models_dir / "loras"
    
    # Determine default mode
    has_standard = bool(scan_models(diffusion_dir, STANDARD_EXTENSIONS, ZIMAGE_FILTERS["diffusion"]))
    has_gguf = bool(scan_models(diffusion_dir, GGUF_EXTENSIONS, ZIMAGE_FILTERS["diffusion"]))
    default_gguf_mode = has_gguf and not has_standard
    
    # Initial model lists
    diffusion_models = get_models_by_mode(diffusion_dir, default_gguf_mode, DEFAULT_DIFFUSION, DEFAULT_DIFFUSION_GGUF, ZIMAGE_FILTERS["diffusion"])
    clip_models = get_models_by_mode(text_encoders_dir, default_gguf_mode, DEFAULT_CLIP, DEFAULT_CLIP_GGUF, ZIMAGE_FILTERS["text_encoder"])
    vae_models = scan_models(vae_dir, STANDARD_EXTENSIONS, ZIMAGE_FILTERS["vae"]) or [DEFAULT_VAE]
    
    is_installed = check_custom_node_installed(custom_nodes_dir, CUSTOM_NODE_NAME)
    
    # Fetch samplers
    samplers = DEFAULT_SAMPLERS.copy()
    try:
        with httpx.Client(timeout=5) as client:
            response = client.get(f"{services.kit.comfyui_url}/object_info/KSamplerSelect")
            if response.status_code == 200:
                data = response.json()
                info = data.get("KSamplerSelect", {}).get("input", {}).get("required", {}).get("sampler_name", [])
                if info and isinstance(info[0], list):
                    samplers = info[0]
    except Exception:
        pass
    
    with gr.TabItem(TAB_LABEL, id=TAB_ID) as tab:
        gr.Markdown("## 🧪 Experimental Workflows")
        
        with gr.Tabs():
            with gr.TabItem("🔍 UpscaleAny", id="upscale_any"):
                with gr.Row():
                    # ===== LEFT COLUMN =====
                    with gr.Column(scale=1):
                        # Input tabs: Single / Batch
                        with gr.Tabs() as input_tabs:
                            with gr.TabItem("📷 Single", id="single_input"):
                                input_image = gr.Image(
                                    label="Input Image",
                                    type="filepath",
                                    elem_classes="image-window"
                                )
                                # Single enhance button
                                with gr.Row():
                                    single_enhance_btn = gr.Button("🔍 Enhance", variant="primary", size="sm", scale=3)
                                    single_stop_btn = gr.Button("⏹️ Stop", size="sm", variant="stop", scale=1)
                            
                            with gr.TabItem("📁 Batch", id="batch_input"):
                                batch_files = gr.File(
                                    label="Upload Images",
                                    file_count="multiple",
                                    file_types=["image"],
                                    type="filepath"
                                )
                                batch_folder = gr.Textbox(
                                    label="Or Enter Folder Path",
                                    placeholder="C:\\path\\to\\images or /path/to/images",
                                    info="Process all images in a folder"
                                )
                                # Batch enhance button
                                with gr.Row():
                                    batch_enhance_btn = gr.Button("🔍 Enhance Batch", variant="primary", size="sm", scale=3)
                                    batch_stop_btn = gr.Button("⏹️ Stop", size="sm", variant="stop", scale=1)

                        # Prompt
                        prompt = gr.Textbox(
                            label="Prompt (Optional)",
                            placeholder="Leave empty, or guide the enhancement...",
                            lines=2
                        )
                        
                        # Main parameters
                        with gr.Row():
                            megapixels = gr.Slider(label="Megapixels", value=1.0, minimum=0.5, maximum=2.0, step=0.1)
                            scale_by = gr.Slider(label="Scale Factor", value=1.5, minimum=1.1, maximum=2.0, step=0.1)
                        
                        # Seed controls
                        with gr.Row():
                            seed = gr.Number(label="Seed", value=new_random_seed(), minimum=0, step=1, scale=2)
                            randomize_seed = gr.Checkbox(label="🎲 Random", value=True, scale=0, min_width=100)
                        
                        # LoRA section
                        lora_components = create_lora_ui(loras_dir, accordion_open=False)
                        
                        # Advanced Settings
                        with gr.Accordion("⚙️ Advanced Settings", open=False):
                            with gr.Group():
                                with gr.Row():
                                    steps = gr.Slider(
                                        label="Steps", value=10, minimum=5, maximum=20, step=1,
                                        info="Total diffusion steps for the sigma schedule"
                                    )
                                    cfg = gr.Slider(
                                        label="CFG", value=1.0, minimum=1.0, maximum=5.0, step=0.1,
                                        info="Classifier-free guidance scale"
                                    )
                            with gr.Group():
                                with gr.Row():
                                    start_at_step = gr.Slider(
                                        label="Start Step", value=5, minimum=0, maximum=20, step=1,
                                        info="Starting step index (0 = beginning)"
                                    )
                                    end_at_step = gr.Slider(
                                        label="End Step", value=10, minimum=0, maximum=20, step=1,
                                        info="Ending step index (set higher than steps to use all)"
                                    )
                            with gr.Group():
                                with gr.Row():
                                    shift = gr.Slider(
                                        label="Shift", value=3.0, minimum=1.0, maximum=10.0, step=0.5,
                                        info="Global timestep shift. Z-Image-Turbo optimal: 3.0"
                                    )
                                    sampler_name = gr.Dropdown(label="Sampler", choices=samplers, value="dpmpp_sde" if "dpmpp_sde" in samplers else samplers[0])
                            gr.Markdown("##### Scheduler Fine-Tuning")
                            with gr.Group():
                                with gr.Row():
                                    base_shift = gr.Slider(
                                        label="Base Shift", value=0.5, minimum=0.0, maximum=2.0, step=0.01,
                                        info="Stabilizes generation. Higher = more consistent outputs"
                                    )
                                    max_shift = gr.Slider(
                                        label="Max Shift", value=1.15, minimum=0.5, maximum=3.0, step=0.01,
                                        info="Max variation. Higher = more stylized results"
                                    )
                            with gr.Group():
                                with gr.Row():
                                    use_karras_sigmas = gr.Dropdown(
                                        label="Karras Sigmas", choices=["disable", "enable"], value="disable",
                                        info="Uses Karras noise schedule for smoother results. (doesn't seem viable here)"
                                    )
                                    stochastic_sampling = gr.Dropdown(
                                        label="Stochastic Sampling", choices=["disable", "enable"], value="disable",
                                        info="Adds randomness for more varied outputs"
                                    )

                        # Model Selection
                        with gr.Accordion("Model Selection", open=True):
                            with gr.Row():
                                use_gguf = gr.Radio(choices=[("Standard", False), ("GGUF", True)], value=default_gguf_mode, label="Mode", info="GGUF uses less VRAM")
                            with gr.Group():
                                default_diff = DEFAULT_DIFFUSION_GGUF if default_gguf_mode else DEFAULT_DIFFUSION
                                default_te = DEFAULT_CLIP_GGUF if default_gguf_mode else DEFAULT_CLIP
                                unet_name = gr.Dropdown(label="Diffusion Model", choices=diffusion_models, value=get_default_model(diffusion_models, default_diff))
                                clip_name = gr.Dropdown(label="Text Encoder", choices=clip_models, value=get_default_model(clip_models, default_te))
                                vae_name = gr.Dropdown(label="VAE", choices=vae_models, value=get_default_model(vae_models, DEFAULT_VAE))
                        
                        # Custom node status
                        with gr.Accordion("📦 Custom Node Status", open=True):
                            node_status = gr.Textbox(
                                label="ComfyUI-EulerDiscreteScheduler",
                                value="✓ Installed" if is_installed else "⚠️ Not installed - Required for workflow",
                                interactive=False
                            )
                            with gr.Row():
                                install_btn = gr.Button("📥 Install", visible=not is_installed, variant="primary")
                                update_btn = gr.Button("🔄 Update", visible=is_installed)
                    
                    # ===== RIGHT COLUMN =====
                    with gr.Column(scale=1):
                        # Output tabs: Single / Batch (each with its own save/send buttons)
                        with gr.Tabs() as output_tabs:
                            with gr.TabItem("📷 Single Result", id="single_output"):
                                output_slider = gr.ImageSlider(label="Before / After", type="filepath", elem_classes="image-window", show_download_button=True)
                                with gr.Row():
                                    single_save_btn = gr.Button("💾 Save", size="sm", variant="primary")
                                    single_send_btn = gr.Button("🔍 Send to SeedVR2", size="sm", variant="huggingface")
                            with gr.TabItem("📁 Batch Results", id="batch_output"):
                                output_gallery = gr.Gallery(label="Results", columns=4, rows=2, height=400, object_fit="contain", preview=True, elem_id="output-gallery", show_download_button=True)
                                with gr.Row():
                                    batch_save_btn = gr.Button("💾 Save Selected", size="sm", variant="primary")
                                    batch_save_all_btn = gr.Button("💾 Save All", size="sm", variant="secondary")
                                    batch_send_btn = gr.Button("🔍 Send to SeedVR2", size="sm", variant="huggingface")
                        
                        # Shared output controls
                        with gr.Row():
                            autosave = gr.Checkbox(label="Auto-save", container=False, value=False)
                            open_folder_btn = gr.Button("📂 Open Folder", size="sm")                            
                        
                        status = gr.Textbox(label="Status", interactive=False, show_label=False, lines=1)
                        
                        # System monitor
                        with gr.Row():
                            with gr.Column(scale=1, min_width=200):
                                gpu_monitor = gr.Textbox(value="Loading...", lines=4.5, container=False, interactive=False, show_label=False, elem_classes="monitor-box gpu-monitor")
                            with gr.Column(scale=1, min_width=200):
                                cpu_monitor = gr.Textbox(value="Loading...", lines=4, container=False, interactive=False, show_label=False, elem_classes="monitor-box cpu-monitor")
                        
                        # Hidden states (separate for single vs batch to avoid cross-contamination)
                        single_result_state = gr.State(value=None)
                        single_original_state = gr.State(value=None)
                        selected_gallery_image = gr.State(value=None)

        # ===== EVENT HANDLERS =====
        setup_lora_handlers(lora_components, loras_dir)
        lora_inputs = get_lora_inputs(lora_components)
        
        # Install/Update handlers
        def on_install():
            success, msg = install_custom_node(custom_nodes_dir, CUSTOM_NODE_REPO, CUSTOM_NODE_NAME)
            if success:
                return msg, gr.update(visible=False), gr.update(visible=True)
            return msg, gr.update(), gr.update()
        
        install_btn.click(fn=on_install, outputs=[node_status, install_btn, update_btn])
        update_btn.click(fn=lambda: update_custom_node(custom_nodes_dir, CUSTOM_NODE_NAME)[1], outputs=[node_status])
        
        # Step range update
        def update_step_ranges(steps_val):
            return gr.update(maximum=steps_val), gr.update(maximum=steps_val, value=min(steps_val, 10))
        steps.change(fn=update_step_ranges, inputs=[steps], outputs=[start_at_step, end_at_step])
        
        # Model dropdown update
        def update_model_dropdowns(is_gguf):
            new_diff = get_models_by_mode(diffusion_dir, is_gguf, DEFAULT_DIFFUSION, DEFAULT_DIFFUSION_GGUF, ZIMAGE_FILTERS["diffusion"])
            new_clip = get_models_by_mode(text_encoders_dir, is_gguf, DEFAULT_CLIP, DEFAULT_CLIP_GGUF, ZIMAGE_FILTERS["text_encoder"])
            d_diff = DEFAULT_DIFFUSION_GGUF if is_gguf else DEFAULT_DIFFUSION
            d_clip = DEFAULT_CLIP_GGUF if is_gguf else DEFAULT_CLIP
            return gr.update(choices=new_diff, value=get_default_model(new_diff, d_diff)), gr.update(choices=new_clip, value=get_default_model(new_clip, d_clip))
        use_gguf.change(fn=update_model_dropdowns, inputs=[use_gguf], outputs=[unet_name, clip_name])
        
        # Stop buttons
        async def stop_generation():
            global _cancel_batch
            _cancel_batch = True
            try:
                async with httpx.AsyncClient() as client:
                    await client.post(f"{services.kit.comfyui_url}/interrupt")
                return "⏹️ Stopping..."
            except Exception as e:
                return f"⏹️ Stop requested ({e})"
        single_stop_btn.click(fn=stop_generation, outputs=[status])
        batch_stop_btn.click(fn=stop_generation, outputs=[status])

        # Single image upscale handler
        async def run_single_upscale(img, prompt_text, seed_val, randomize, mp, scale, steps_val, start_step, end_step, shift_val, cfg_val, sampler, is_gguf, unet, clip, vae, base_shift_val, max_shift_val, karras, stochastic, auto, *lora_args):
            async for result in experimental_upscale(
                services, img, prompt_text, seed_val, randomize, mp, scale,
                steps_val, start_step, end_step, shift_val, cfg_val, sampler,
                is_gguf, unet, clip, vae, base_shift_val, max_shift_val, karras, stochastic, auto,
                *lora_args
            ):
                slider_tuple, status_msg, actual_seed, res_path = result
                yield slider_tuple, status_msg, actual_seed, res_path, img
        
        # Batch upscale handler
        async def run_batch_upscale(files, folder, prompt_text, seed_val, randomize, mp, scale, steps_val, start_step, end_step, shift_val, cfg_val, sampler, is_gguf, unet, clip, vae, base_shift_val, max_shift_val, karras, stochastic, auto, *lora_args):
            async for result in experimental_upscale_batch(
                services, files, folder, prompt_text, seed_val, randomize, mp, scale,
                steps_val, start_step, end_step, shift_val, cfg_val, sampler,
                is_gguf, unet, clip, vae, base_shift_val, max_shift_val, karras, stochastic, auto,
                *lora_args
            ):
                gallery, status_msg, actual_seed = result
                yield gallery, status_msg, actual_seed

        # Common inputs for both buttons (excluding image sources)
        common_inputs = [
            prompt, seed, randomize_seed, megapixels, scale_by,
            steps, start_at_step, end_at_step, shift, cfg, sampler_name,
            use_gguf, unet_name, clip_name, vae_name,
            base_shift, max_shift, use_karras_sigmas, stochastic_sampling, autosave,
        ] + lora_inputs

        # Wire single enhance button
        single_enhance_btn.click(
            fn=run_single_upscale,
            inputs=[input_image] + common_inputs,
            outputs=[output_slider, status, seed, single_result_state, single_original_state]
        )
        
        # Wire batch enhance button (no longer touches single result state)
        batch_enhance_btn.click(
            fn=run_batch_upscale,
            inputs=[batch_files, batch_folder] + common_inputs,
            outputs=[output_gallery, status, seed]
        )
        
        # Gallery selection handler
        def on_gallery_select(evt: gr.SelectData, gallery_data):
            if gallery_data and evt.index < len(gallery_data):
                item = gallery_data[evt.index]
                # Gallery items can be tuples (path, caption) or just paths
                if isinstance(item, tuple):
                    return item[0]  # Return just the path
                return item
            return None
        output_gallery.select(fn=on_gallery_select, inputs=[output_gallery], outputs=[selected_gallery_image])
        
        # Single save button - only saves from single result state
        def on_save_single(res_path, orig_path):
            if res_path is None:
                return "❌ No image to save"
            try:
                current_outputs_dir = services.get_outputs_dir()
                saved = save_experimental_output(res_path, orig_path, current_outputs_dir)
                return f"✓ Saved: {Path(saved).name}"
            except Exception as e:
                return f"❌ Save failed: {e}"
        single_save_btn.click(fn=on_save_single, inputs=[single_result_state, single_original_state], outputs=[status])
        
        # Batch save selected button - saves currently selected gallery image
        def on_save_batch_selected(selected_img, gallery_data):
            # Use selected image, or fall back to first gallery image
            image_to_save = selected_img
            if not image_to_save and gallery_data:
                item = gallery_data[0]
                image_to_save = item[0] if isinstance(item, (list, tuple)) else item
            
            if image_to_save is None:
                return "❌ No image selected"
            try:
                current_outputs_dir = services.get_outputs_dir()
                saved = save_experimental_output(image_to_save, image_to_save, current_outputs_dir)
                return f"✓ Saved: {Path(saved).name}"
            except Exception as e:
                return f"❌ Save failed: {e}"
        batch_save_btn.click(fn=on_save_batch_selected, inputs=[selected_gallery_image, output_gallery], outputs=[status])
        
        # Batch save all button - saves all images in gallery
        def on_save_batch_all(gallery_data):
            if not gallery_data:
                return "❌ No images to save"
            try:
                current_outputs_dir = services.get_outputs_dir()
                saved_count = 0
                for item in gallery_data:
                    img_path = item[0] if isinstance(item, (list, tuple)) else item
                    save_experimental_output(img_path, img_path, current_outputs_dir)
                    saved_count += 1
                return f"✓ Saved {saved_count} images"
            except Exception as e:
                return f"❌ Save failed: {e}"
        batch_save_all_btn.click(fn=on_save_batch_all, inputs=[output_gallery], outputs=[status])
        
        # Open folder - get path dynamically to respect settings changes
        def on_open_folder():
            current_outputs_dir = services.get_outputs_dir()
            open_folder(current_outputs_dir / "experimental")
        open_folder_btn.click(fn=on_open_folder)
        
        # Register components for inter-module transfer (separate buttons for single vs batch)
        services.inter_module.register_component("experimental_single_send_btn", single_send_btn)
        services.inter_module.register_component("experimental_batch_send_btn", batch_send_btn)
        services.inter_module.register_component("experimental_selected_image", selected_gallery_image)
        services.inter_module.register_component("experimental_single_result", single_result_state)
        services.inter_module.register_component("experimental_status", status)
        
        # System Monitor
        def update_monitor():
            if services.system_monitor:
                return services.system_monitor.get_system_info()
            return "N/A", "N/A"
        monitor_timer = gr.Timer(2, active=True)
        monitor_timer.tick(fn=update_monitor, outputs=[gpu_monitor, cpu_monitor])
    
    return tab
