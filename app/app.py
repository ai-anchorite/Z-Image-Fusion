"""
Z-Image Turbo - Comfy-Gradio App

Fast, high-quality image generation using the Z-Image 6B turbo model.

This is the main orchestrator that:
- Initializes shared services (ComfyKit, paths, settings)
- Discovers and loads feature modules
- Creates the Gradio interface with module tabs
- Launches the server
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
from comfykit import ComfyKit

# Import shared services infrastructure
from modules import SharedServices, SettingsManager, discover_modules
from modules.prompt_assistant import PromptAssistant
from modules.system_monitor import SystemMonitor
from modules.output_gallery import create_output_gallery

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

# =============================================================================
# Path Configuration
# =============================================================================

APP_DIR = Path(__file__).parent
MODULES_DIR = APP_DIR / "modules"
UI_SETTINGS_FILE = APP_DIR / "ui_settings.json"
MODELS_DIR = APP_DIR / "comfyui" / "models"
WORKFLOWS_DIR = APP_DIR / "workflows"
DEFAULT_OUTPUTS_DIR = APP_DIR / "outputs" / "z-image-fusion"

# Gradio temp directory (uses GRADIO_TEMP_DIR env var if set, else system temp)
GRADIO_TEMP_DIR = Path(os.environ.get("GRADIO_TEMP_DIR", tempfile.gettempdir()))


# =============================================================================
# Initialization Helpers
# =============================================================================

def init_comfykit() -> ComfyKit | None:
    """Initialize ComfyKit client, returning None on failure."""
    try:
        kit = ComfyKit()
        logger.info(f"ComfyKit initialized: {kit.comfyui_url}")
        return kit
    except Exception as e:
        logger.error(f"Failed to initialize ComfyKit: {e}")
        return None


def init_settings() -> SettingsManager:
    """Initialize the settings manager."""
    return SettingsManager(UI_SETTINGS_FILE)


def init_prompt_assistant() -> PromptAssistant:
    """Initialize the Prompt Assistant."""
    return PromptAssistant(
        settings_file=str(MODULES_DIR / "llm_settings.json"),
        ckpt_dir=str(MODULES_DIR / "llm_ckpts")
    )


def get_outputs_dir(settings: SettingsManager) -> Path:
    """Get the outputs directory from settings, or use default."""
    custom_path = settings.get("outputs_dir")
    if custom_path:
        path = Path(custom_path)
        if path.is_absolute():
            path.mkdir(parents=True, exist_ok=True)
            return path
    # Default path
    DEFAULT_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_OUTPUTS_DIR


def clear_temp_folder() -> tuple[bool, str]:
    """Clear the Gradio temp folder. Returns (success, message)."""
    try:
        temp_path = GRADIO_TEMP_DIR.resolve()
        logger.info(f"Clearing temp folder: {temp_path}")
        
        if temp_path.exists():
            file_count = sum(1 for _ in temp_path.rglob("*") if _.is_file())
            shutil.rmtree(temp_path)
            temp_path.mkdir(parents=True, exist_ok=True)
            return True, f"✓ Cleared {file_count} files from {temp_path}"
        return True, f"✓ Temp folder empty ({temp_path})"
    except Exception as e:
        logger.warning(f"Failed to clear temp folder: {e}")
        return False, f"❌ Failed to clear temp: {e}"


def maybe_clear_temp_on_start(settings: SettingsManager) -> None:
    """Clear temp folder on startup if enabled in settings."""
    if settings.get("clear_temp_on_start", False):
        success, msg = clear_temp_folder()
        if success:
            logger.info(f"Startup: {msg}")


# =============================================================================
# Theme Configuration
# =============================================================================

# Built-in Gradio themes
BUILTIN_THEMES = {
    "Default": gr.themes.Default(),
    "Soft": gr.themes.Soft(),
    "Monochrome": gr.themes.Monochrome(),
    "Glass": gr.themes.Glass(),
    "Base": gr.themes.Base(),
    "Ocean": gr.themes.Ocean(),
    "Origin": gr.themes.Origin(),
    "Citrus": gr.themes.Citrus(),
}

# Community themes from Hugging Face Spaces
COMMUNITY_THEMES = {
    "Miku": "NoCrypt/miku",
    "Interstellar": "Nymbo/Interstellar",
    "xkcd": "gstaff/xkcd",
}


def get_theme(settings: SettingsManager) -> Any:
    """Get the Gradio theme based on settings."""
    theme_name = settings.get("ui_theme", "Default")
    
    # Check built-in themes first
    if theme_name in BUILTIN_THEMES:
        logger.info(f"Using built-in theme: {theme_name}")
        return BUILTIN_THEMES[theme_name]
    
    # Check community themes
    if theme_name in COMMUNITY_THEMES:
        theme_id = COMMUNITY_THEMES[theme_name]
        logger.info(f"Using community theme: {theme_name} ({theme_id})")
        return theme_id
    
    # Fallback to default
    logger.warning(f"Unknown theme '{theme_name}', using Default")
    return gr.themes.Default()


# =============================================================================
# Interface Creation
# =============================================================================

def create_interface(services: SharedServices, theme: Any = None) -> gr.Blocks:
    """
    Create the Gradio interface by discovering and loading modules.
    
    Args:
        services: SharedServices instance with all dependencies
        theme: Gradio theme to apply (built-in theme object or HF theme string)
        
    Returns:
        gr.Blocks interface ready to launch
    """
    # CSS for the interface
    css = """
    
    .video-window {
        min-height: 300px !important;
        height: auto !important;
    }

    .video-window video, .image-window img {
        max-height: 60vh !important;
        object-fit: contain;
        width: 100%;
    }
    .video-window .source-selection,
    .image-window .source-selection {
        display: none !important;
    }    

    textarea {
        overflow-y: auto !important;
        resize: vertical;
    }
    /* Hide upload overlay on output gallery while keeping preview functional */
    #output-gallery > button[aria-label*="upload"],
    #output-gallery > button[aria-dropeffect="copy"] {
        display: none !important;
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
        overflow-y: hidden !important;
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

    /* Analysis panels - tight vertical spacing */
    .analysis-panel {
        margin-top: -9px !important;
        margin-bottom: -9px !important;
    }

    /* Resolution preset radio - inline compact style */
    .res-radio-compact {
        min-width: 0 !important;
    }
    .res-radio-compact .wrap {
        gap: 0 !important;
    }
    .res-radio-compact .wrap > label {
        border: none !important;
        background: transparent !important;
        padding: 6px 12px !important;
        margin: 0 !important;
        border-radius: 0 !important;
    }
    .res-radio-compact .wrap > label:first-child {
        border-radius: 8px 0 0 8px !important;
    }
    .res-radio-compact .wrap > label:last-child {
        border-radius: 0 8px 8px 0 !important;
    }

    /* Compact inline checkbox */
    .checkbox-compact {
        min-width: fit-content !important;
        flex-grow: 0 !important;
    }
    .checkbox-compact > label {
        padding: 0 !important;
        gap: 6px !important;
    }

    /* Fix fullscreen/lightbox image toolbar being covered by scrollbar */
    .icon-button-wrapper {
        right: 28px !important;
    }

    /* Shared output gallery - compact when collapsed */
    #output-gallery-shared {
        margin-top: 8px;
    }
    #output-gallery-shared .thumbnail-item {
        border-radius: 6px;
        transition: transform 0.15s ease;
    }
    #output-gallery-shared .thumbnail-item:hover {
        transform: scale(1.02);
    }
    """
    
    with gr.Blocks(title="Z-Image Turbo", css=css, theme=theme) as interface:
        # Discover and load modules
        modules = discover_modules(MODULES_DIR)
        
        if not modules:
            gr.Markdown("## ❌ No modules found\n\nPlease check the `app/modules/` directory.")
            return interface
        
        # Create tabs from discovered modules
        with gr.Tabs() as main_tabs:
            # Register main_tabs for inter-module tab switching
            services.inter_module.main_tabs = main_tabs
            services.inter_module.register_component("main_tabs", main_tabs)
            
            loaded_tabs = []
            
            for module_info in modules:
                try:
                    tab = module_info.create_tab(services)
                    loaded_tabs.append((module_info, tab))
                    logger.info(f"Loaded module tab: {module_info.tab_label}")
                except Exception as e:
                    logger.error(f"Failed to create tab for module {module_info.name}: {e}")
                    # Create an error placeholder tab
                    with gr.TabItem(f"❌ {module_info.name}", id=f"error_{module_info.tab_id}"):
                        gr.Markdown(f"## Module Load Error\n\n`{module_info.name}` failed to load:\n\n```\n{e}\n```")
        
        # Log summary
        logger.info(f"Loaded {len(loaded_tabs)} module tabs")
        
        # Post-load cross-module wiring for image transfer buttons
        _wire_image_transfers(services)
        
        # Shared output gallery at the bottom (outside tabs, always visible on generation tabs)
        output_gallery_components = create_output_gallery(services)
        logger.info("Created shared output gallery")
        
        # Wire up gallery refresh after save operations from tabs
        _wire_gallery_refresh(services, output_gallery_components)
        
        # Hide gallery on settings tabs (llm_settings, app_settings)
        settings_tabs = {"llm_settings", "app_settings"}
        gallery_accordion = output_gallery_components["accordion"]
        
        def on_tab_change(evt: gr.SelectData):
            """Hide gallery accordion on settings tabs."""
            selected_tab = evt.value if hasattr(evt, 'value') else None
            # evt.value contains the tab label, but we need tab_id
            # Check if any settings tab label is selected
            hide_gallery = any(s in str(selected_tab).lower() for s in ["settings", "llm"])
            return gr.update(visible=not hide_gallery)
        
        main_tabs.select(
            fn=on_tab_change,
            outputs=[gallery_accordion]
        )
    
    return interface


def _wire_image_transfers(services: SharedServices):
    """
    Wire up all "Send to X" buttons after all modules are loaded.
    
    This uses the ImageTransfer system to connect sender modules to receiver modules.
    Each receiver module registers itself, and sender modules register their buttons.
    """
    image_transfer = services.inter_module.image_transfer
    
    # Wire Z-Image "Send to Upscale" button
    send_btn = services.inter_module.get_component("zimage_send_to_upscale_btn")
    selected_img = services.inter_module.get_component("zimage_selected_gallery_image")
    gallery = services.inter_module.get_component("zimage_output_gallery")
    zimage_status = services.inter_module.get_component("zimage_gen_status")
    
    if all([send_btn, selected_img, gallery, zimage_status]):
        success = image_transfer.wire_send_button(
            button=send_btn,
            target_tab_id="upscale",
            source_selected=selected_img,
            source_gallery=gallery,
            source_status=zimage_status
        )
        if success:
            logger.info("Wired Z-Image -> Upscale image transfer")
        else:
            logger.warning("Failed to wire Z-Image -> Upscale (receiver not ready)")
    else:
        logger.warning("Could not wire Z-Image -> Upscale - some components not registered")
    
    # Wire Experimental "Send to SeedVR2" buttons (separate for single vs batch)
    exp_single_send_btn = services.inter_module.get_component("experimental_single_send_btn")
    exp_batch_send_btn = services.inter_module.get_component("experimental_batch_send_btn")
    exp_selected = services.inter_module.get_component("experimental_selected_image")
    exp_single_result = services.inter_module.get_component("experimental_single_result")
    exp_status = services.inter_module.get_component("experimental_status")
    
    receiver = image_transfer.get_receiver("upscale")
    if receiver:
        # Single result send handler - only uses single_result_state
        def exp_single_send_handler(result_path):
            import gradio as gr
            if not result_path:
                return "❌ No image to send", gr.update(), gr.update(), gr.update()
            
            receiver = image_transfer.get_receiver("upscale")
            if receiver:
                image_transfer.set_pending("upscale", result_path)
                return (
                    f"✓ Sent to {receiver.label}",
                    result_path,
                    f"✓ Received image",
                    gr.Tabs(selected="upscale")
                )
            return "❌ Upscale tab not available", gr.update(), gr.update(), gr.update()
        
        # Batch result send handler - only uses selected_gallery_image
        def exp_batch_send_handler(selected_img):
            import gradio as gr
            if not selected_img:
                return "❌ No image selected", gr.update(), gr.update(), gr.update()
            
            receiver = image_transfer.get_receiver("upscale")
            if receiver:
                image_transfer.set_pending("upscale", selected_img)
                return (
                    f"✓ Sent to {receiver.label}",
                    selected_img,
                    f"✓ Received image",
                    gr.Tabs(selected="upscale")
                )
            return "❌ Upscale tab not available", gr.update(), gr.update(), gr.update()
        
        # Build outputs list
        outputs = [exp_status, receiver.input_component]
        if receiver.status_component:
            outputs.append(receiver.status_component)
        else:
            import gradio as gr
            outputs.append(gr.State())
        outputs.append(services.inter_module.main_tabs)
        
        # Wire single send button
        if exp_single_send_btn and exp_single_result:
            exp_single_send_btn.click(
                fn=exp_single_send_handler,
                inputs=[exp_single_result],
                outputs=outputs
            )
            logger.info("Wired Experimental Single -> Upscale image transfer")
        
        # Wire batch send button
        if exp_batch_send_btn and exp_selected:
            exp_batch_send_btn.click(
                fn=exp_batch_send_handler,
                inputs=[exp_selected],
                outputs=outputs
            )
            logger.info("Wired Experimental Batch -> Upscale image transfer")
    else:
        logger.warning("Failed to wire Experimental -> Upscale (receiver not ready)")


def _wire_gallery_refresh(services: SharedServices, gallery_components: dict):
    """
    Wire up the shared output gallery send buttons to target tabs.
    """
    import gradio as gr
    
    selected_info = gallery_components["selected_info"]
    get_selected_path = gallery_components["get_selected_path"]
    gallery_status = gallery_components["status"]
    send_to_upscale_btn = gallery_components["send_to_upscale_btn"]
    send_to_experimental_btn = gallery_components["send_to_experimental_btn"]
    
    image_transfer = services.inter_module.image_transfer
    main_tabs = services.inter_module.main_tabs
    
    # Wire "Send to Upscale" button
    upscale_receiver = image_transfer.get_receiver("upscale")
    if upscale_receiver and main_tabs:
        def send_to_upscale(info_str):
            img_path = get_selected_path(info_str)
            if not img_path:
                return "❌ No image selected", gr.update(), gr.update(), gr.update()
            return (
                f"✓ Sent to Upscale",
                img_path,
                "✓ Received image",
                gr.Tabs(selected="upscale")
            )
        
        outputs = [gallery_status, upscale_receiver.input_component]
        if upscale_receiver.status_component:
            outputs.append(upscale_receiver.status_component)
        else:
            outputs.append(gr.State())
        outputs.append(main_tabs)
        
        send_to_upscale_btn.click(
            fn=send_to_upscale,
            inputs=[selected_info],
            outputs=outputs
        )
        logger.info("Wired Output Gallery -> Upscale")
    
    # Wire "Send to Experimental" button
    exp_receiver = image_transfer.get_receiver("experimental")
    if exp_receiver and main_tabs:
        def send_to_experimental(info_str):
            img_path = get_selected_path(info_str)
            if not img_path:
                return "❌ No image selected", gr.update(), gr.update(), gr.update()
            return (
                f"✓ Sent to Experimental",
                img_path,
                "✓ Received image",
                gr.Tabs(selected="experimental")
            )
        
        outputs = [gallery_status, exp_receiver.input_component]
        if exp_receiver.status_component:
            outputs.append(exp_receiver.status_component)
        else:
            outputs.append(gr.State())
        outputs.append(main_tabs)
        
        send_to_experimental_btn.click(
            fn=send_to_experimental,
            inputs=[selected_info],
            outputs=outputs
        )
        logger.info("Wired Output Gallery -> Experimental")
    
    logger.info("Output gallery ready with auto-refresh")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point - initialize services and launch the app."""
    # Initialize ComfyKit
    kit = init_comfykit()
    if kit is None:
        print("\n❌ Failed to initialize ComfyKit")
        print("   Make sure ComfyUI is running at http://127.0.0.1:8188\n")
        return
    
    # Initialize settings
    settings = init_settings()
    
    # Clear temp on startup if enabled
    maybe_clear_temp_on_start(settings)
    
    # Initialize prompt assistant
    prompt_assistant = init_prompt_assistant()
    
    # Get outputs directory
    outputs_dir = get_outputs_dir(settings)
    
    # Create SharedServices container
    services = SharedServices(
        kit=kit,
        app_dir=APP_DIR,
        models_dir=MODELS_DIR,
        outputs_dir=outputs_dir,
        workflows_dir=WORKFLOWS_DIR,
        settings=settings,
        prompt_assistant=prompt_assistant,
        system_monitor=SystemMonitor,
    )
    
    # Get theme from settings
    theme = get_theme(settings)
    theme_name = settings.get("ui_theme", "Default")
    
    # Print startup banner
    print("\n" + "="*50)
    print("⚡ Z-Image Turbo")
    print("="*50)
    print(f"ComfyUI: {kit.comfyui_url}")
    print(f"Models:  {MODELS_DIR}")
    print(f"Outputs: {outputs_dir}")
    print(f"Theme:   {theme_name}")
    print("="*50 + "\n")
    
    # Create and launch interface
    interface = create_interface(services, theme=theme)
    interface.launch(server_name="127.0.0.1", share=False)


if __name__ == "__main__":
    main()
