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
from comfykit import ComfyKit

# Prompt Assistant module
from modules.prompt_assistant import PromptAssistant

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Paths
APP_DIR = Path(__file__).parent
OUTPUTS_DIR = APP_DIR / "outputs" / "z-image-fusion"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
MODULES_DIR = APP_DIR / "modules"

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
DEFAULT_SEEDVR2_DIT = "seedvr2_ema_7b_sharp-Q4_K_M.gguf"


def scan_models(folder: Path, extensions: tuple = (".safetensors", ".ckpt", ".pt", ".gguf")) -> list:
    """Scan folder recursively for model files, returning relative paths."""
    if not folder.exists():
        return []
    models = []
    for ext in extensions:
        for f in folder.rglob(f"*{ext}"):
            rel_path = f.relative_to(folder)
            models.append(str(rel_path))
    return sorted(models)


def get_models_by_mode(folder: Path, is_gguf: bool, default_standard: str, default_gguf: str) -> list:
    """Get models filtered by mode (standard vs GGUF)."""
    extensions = GGUF_EXTENSIONS if is_gguf else STANDARD_EXTENSIONS
    default = default_gguf if is_gguf else default_standard
    models = scan_models(folder, extensions)
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
    """Save image to outputs folder with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = "".join(c if c.isalnum() or c in " -_" else "" for c in prompt[:30]).strip()
    safe_prompt = safe_prompt.replace(" ", "_") if safe_prompt else "image"
    
    # Use subfolder if specified
    target_dir = OUTPUTS_DIR / subfolder if subfolder else OUTPUTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{timestamp}_{safe_prompt}.png"
    output_path = target_dir / filename
    shutil.copy2(image_path, output_path)
    logger.info(f"Saved to: {output_path}")
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
    autosave: bool,
) -> tuple:
    """Upscale an image using SeedVR2. Returns (slider_tuple, status, seed, upscaled_path)."""
    try:
        if input_image is None:
            return None, "❌ Please upload an image to upscale", seed, None
        
        # SeedVR2 uses 32-bit seed max (4294967295)
        actual_seed = new_random_seed_32bit() if randomize_seed else min(int(seed), 4294967295)
        
        workflow_path = APP_DIR / "workflows" / "SeedVR2_4K_image_upscale.json"
        if not workflow_path.exists():
            return None, "❌ Upscale workflow not found", seed, None
        
        logger.info(f"Upscaling with SeedVR2: {dit_model}, res={resolution}, max={max_resolution}")
        
        params = {
            "image": input_image,
            "seed": actual_seed,
            "resolution": int(resolution),
            "max_resolution": int(max_resolution),
            "dit_model": dit_model,
            "blocks_to_swap": int(blocks_to_swap),
        }
        
        result = await kit.execute(str(workflow_path), params)
        
        if result.status == "error":
            return None, f"❌ Upscale failed: {result.msg}", actual_seed, None
        
        if not result.images:
            return None, "❌ No images generated", actual_seed, None
        
        image_path = result.images[0]
        if image_path.startswith("http"):
            image_path = await download_image_from_url(image_path)
        
        # Autosave
        if autosave:
            save_image_to_outputs(image_path, "upscale", subfolder="upscaled")
            status = f"✓ {result.duration:.1f}s | Saved" if result.duration else "✓ Saved"
        else:
            status = f"✓ {result.duration:.1f}s" if result.duration else "✓ Done"
        
        # Return tuple for ImageSlider (original, upscaled) + upscaled path for save button
        return (input_image, image_path), status, actual_seed, image_path
        
    except Exception as e:
        logger.error(f"Upscale error: {e}", exc_info=True)
        if "connect" in str(e).lower():
            return None, "❌ Cannot connect to ComfyUI", seed, None
        return None, f"❌ {str(e)}", seed, None


async def unload_models() -> str:
    """Unload all models from ComfyUI to free VRAM."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{kit.comfyui_url}/free",
                json={"unload_models": True, "free_memory": True}
            )
            if response.status_code == 200:
                return "✓ Models unloaded, VRAM freed"
            return f"❌ Failed: {response.status_code}"
    except Exception as e:
        return f"❌ Error: {e}"


def create_interface() -> gr.Blocks:
    """Create the Gradio interface."""
    models = get_model_choices()
    comfyui_options = fetch_comfyui_options()
    samplers = comfyui_options["samplers"]
    schedulers = comfyui_options["schedulers"]
    
    # CSS fix for textarea scrollbars not appearing on initial content
    css = """
    textarea {
        overflow-y: auto !important;
        resize: vertical;
    }
    """
    
    with gr.Blocks(title="Z-Image Turbo", css=css) as interface:
        
        # Main tabs for different tools
        with gr.Tabs() as main_tabs:
            # ===== Z-IMAGE GENERATION TAB =====
            with gr.TabItem("⚡ Z-Image Turbo", id="tab_generate"):
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
                                generate_i2i_btn = gr.Button("⚡ Generate", variant="primary", size="lg")                                
                                with gr.Row():
                                    megapixels = gr.Slider(label="Megapixels", info="Scales against input image to maintain aspect ratio", value=2.0, minimum=0.5, maximum=4.0, step=0.1)
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
                                    value=models["lora"][0] if models["lora"] else None,
                                    interactive=True,
                                    scale=3
                                )
                                refresh_loras_btn = gr.Button("🔄", size="sm", scale=0, min_width=40)
                            with gr.Row():                              
                                lora_strength = gr.Slider(label="Strength", value=1.0, minimum=0.0, maximum=2.0, step=0.05, scale=2)
                        
                        # Model selection
                        with gr.Accordion("🔧 Models", open=False):
                            with gr.Row():
                                use_gguf = gr.Radio(choices=[("Standard", False), ("GGUF", True)], value=False, label="Mode", scale=0)
                                unet_name = gr.Dropdown(label="Diffusion", choices=get_models_by_mode(DIFFUSION_DIR, False, DEFAULT_DIFFUSION, DEFAULT_DIFFUSION_GGUF), value=DEFAULT_DIFFUSION, scale=2)
                            with gr.Row():
                                clip_name = gr.Dropdown(label="Text Encoder", choices=get_models_by_mode(TEXT_ENCODERS_DIR, False, DEFAULT_TEXT_ENCODER, DEFAULT_TEXT_ENCODER_GGUF), value=DEFAULT_TEXT_ENCODER)
                                vae_name = gr.Dropdown(label="VAE", choices=models["vae"], value=get_default_model(models["vae"], DEFAULT_VAE))
                        
                    with gr.Column(scale=1):
                        output_image = gr.Image(label="Generated Image", type="filepath", interactive=False, height=512)
                        gen_status = gr.Textbox(label="Status", interactive=False, show_label=False)
                        autosave = gr.Checkbox(label="Auto-save", value=False)                        
                        with gr.Row():
                            save_btn = gr.Button("💾 Save", size="sm")
                            open_folder_btn = gr.Button("📂 Open Folder", size="sm")
                            send_to_upscale_btn = gr.Button("🔍 Send to Upscale", size="sm")
                            unload_btn = gr.Button("🧹 Unload Models", size="sm")
            
            # ===== UPSCALE TAB =====
            with gr.TabItem("🔍 Upscale", id="tab_upscale"):
                with gr.Row():
                    with gr.Column(scale=1):
                        upscale_input_image = gr.Image(label="Input Image", type="filepath", height=360)
                        
                        with gr.Row():
                            upscale_resolution = gr.Slider(
                                label="Resolution",
                                value=4096,
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
                                info="3B: 0-32, 7B: 0-36 (higher = less VRAM)"
                            )
                        
                        with gr.Row():
                            upscale_seed = gr.Number(label="Seed", value=new_random_seed_32bit(), minimum=0, maximum=4294967295, step=1)
                            upscale_randomize_seed = gr.Checkbox(label="🎲 Randomize", value=True)
                        
                        upscale_btn = gr.Button("🔍 Upscale", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        upscale_slider = gr.ImageSlider(
                            label="Before / After",
                            type="filepath",
                            show_download_button=True
                        )
                        upscale_status = gr.Textbox(label="Status", interactive=False)
                        upscale_autosave = gr.Checkbox(label="Auto-save", value=False)
                        with gr.Row():
                            upscale_save_btn = gr.Button("💾 Save", size="sm")
                            upscale_open_folder_btn = gr.Button("📂 Open Folder", size="sm")
                        # Hidden state for upscaled image path
                        upscale_result_path = gr.State(value=None)
            
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
        
        clear_prompt_btn.click(
            fn=lambda: "",
            outputs=[prompt]
        )
        
        # Toggle GGUF mode - update model dropdowns
        def toggle_gguf_mode(is_gguf):
            diffusion_models = get_models_by_mode(DIFFUSION_DIR, is_gguf, DEFAULT_DIFFUSION, DEFAULT_DIFFUSION_GGUF)
            clip_models = get_models_by_mode(TEXT_ENCODERS_DIR, is_gguf, DEFAULT_TEXT_ENCODER, DEFAULT_TEXT_ENCODER_GGUF)
            default_diffusion = DEFAULT_DIFFUSION_GGUF if is_gguf else DEFAULT_DIFFUSION
            default_clip = DEFAULT_TEXT_ENCODER_GGUF if is_gguf else DEFAULT_TEXT_ENCODER
            return (
                gr.update(choices=diffusion_models, value=get_default_model(diffusion_models, default_diffusion)),
                gr.update(choices=clip_models, value=get_default_model(clip_models, default_clip)),
            )
        
        use_gguf.change(
            fn=toggle_gguf_mode,
            inputs=[use_gguf],
            outputs=[unet_name, clip_name]
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
        
        # Upscale
        upscale_btn.click(
            fn=upscale_image,
            inputs=[
                upscale_input_image,
                upscale_seed,
                upscale_randomize_seed,
                upscale_resolution,
                upscale_max_resolution,
                upscale_dit_model,
                upscale_blocks_to_swap,
                upscale_autosave,
            ],
            outputs=[upscale_slider, upscale_status, upscale_seed, upscale_result_path]
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
        def save_upscaled_image(image_path):
            if not image_path:
                return "❌ No image to save"
            saved_path = save_image_to_outputs(image_path, "upscale", subfolder="upscaled")
            return f"✓ Saved: {Path(saved_path).name}"
        
        upscale_save_btn.click(
            fn=save_upscaled_image,
            inputs=[upscale_result_path],
            outputs=[upscale_status]
        )
        
        # Send to Upscale - copy image and switch tab
        send_to_upscale_btn.click(
            fn=lambda img: (img, gr.Tabs(selected="tab_upscale")),
            inputs=[output_image],
            outputs=[upscale_input_image, main_tabs]
        )
        
        # Open folder helpers
        def open_outputs_folder():
            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            if sys.platform == "win32":
                os.startfile(OUTPUTS_DIR)
            elif sys.platform == "darwin":
                subprocess.run(["open", str(OUTPUTS_DIR)])
            else:
                subprocess.run(["xdg-open", str(OUTPUTS_DIR)])
        
        def open_upscaled_folder():
            upscaled_dir = OUTPUTS_DIR / "upscaled"
            upscaled_dir.mkdir(parents=True, exist_ok=True)
            if sys.platform == "win32":
                os.startfile(upscaled_dir)
            elif sys.platform == "darwin":
                subprocess.run(["open", str(upscaled_dir)])
            else:
                subprocess.run(["xdg-open", str(upscaled_dir)])
        
        open_folder_btn.click(fn=open_outputs_folder)
        upscale_open_folder_btn.click(fn=open_upscaled_folder)
    
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
