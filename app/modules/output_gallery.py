"""
Shared Output Gallery Component

A collapsible gallery at the bottom of the UI that displays all saved outputs.
Supports drag-and-drop to upscale tabs and image deletion.
"""

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import gradio as gr

if TYPE_CHECKING:
    from modules import SharedServices

logger = logging.getLogger(__name__)

# Track last scan results for change detection
_last_image_count = 0
_last_newest_mtime = 0.0


def scan_output_images(outputs_dir: Path, max_images: int = 100) -> list[str]:
    """
    Scan outputs directory recursively for images, sorted by modification time (newest first).
    
    Args:
        outputs_dir: Base outputs directory
        max_images: Maximum number of images to return
        
    Returns:
        List of image paths, newest first
    """
    if not outputs_dir.exists():
        return []
    
    images = []
    extensions = {".png", ".jpg", ".jpeg", ".webp"}
    
    for f in outputs_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in extensions:
            images.append(f)
    
    # Sort by modification time, newest first
    images.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    # Return paths as strings, limited to max_images
    return [str(p) for p in images[:max_images]]


def create_output_gallery(services: "SharedServices") -> dict:
    """
    Create the shared output gallery accordion.
    
    Args:
        services: SharedServices instance
        
    Returns:
        Dict with gallery component and related UI elements
    """
    outputs_dir = services.get_outputs_dir()
    initial_images = scan_output_images(outputs_dir)
    
    # Get configurable gallery height from settings (default 400)
    gallery_height = services.settings.get("output_gallery_height", 600)
    
    # Initialize change detection state
    global _last_image_count, _last_newest_mtime
    _last_image_count = len(initial_images)
    _last_newest_mtime = 0.0
    if initial_images:
        try:
            _last_newest_mtime = Path(initial_images[0]).stat().st_mtime
        except:
            pass
    
    with gr.Accordion("📁 Output Gallery", open=False) as accordion:
        gallery = gr.Gallery(
            value=initial_images,
            label="Saved Outputs",
            columns=6,
            rows=2,
            height=gallery_height,
            object_fit="cover",
            show_download_button=True,
            show_share_button=False,
            preview=True,
            allow_preview=True,
            elem_id="output-gallery-shared"
        )
        
        with gr.Row():
            send_to_upscale_btn = gr.Button("🔍 Send to Upscale", size="sm", variant="huggingface", scale=0)
            send_to_experimental_btn = gr.Button("🧪 Send to Experimental", size="sm", variant="huggingface", scale=0)
            delete_btn = gr.Button("🗑️ Delete", size="sm", variant="stop", scale=0)
            refresh_btn = gr.Button("🔄 Refresh", size="sm", scale=0)
            open_folder_btn = gr.Button("📂 Open Folder", size="sm", scale=0)
            # Shows currently selected filename
            selected_display = gr.Textbox(
                value="",
                placeholder="No selection",
                show_label=False,
                container=False,
                interactive=False,
                scale=1
            )
            # Shows image count and status messages
            gallery_status = gr.Textbox(
                value=f"{len(initial_images)} images" if initial_images else "No images",
                show_label=False,
                container=False,
                interactive=False,
                scale=1
            )
        
        # Hidden state for selection tracking (stores "index|path" for internal use)
        selected_info = gr.Textbox(
            value="",
            visible=False
        )
    
    # Track original paths by index (Gradio gallery may use temp copies)
    current_image_paths = gr.State(value=initial_images)
    
    # Event handlers
    def refresh_gallery():
        """Refresh gallery with current outputs."""
        current_outputs_dir = services.get_outputs_dir()
        images = scan_output_images(current_outputs_dir)
        count_msg = f"{len(images)} images" if images else "No images"
        return images, count_msg, "", "", images  # Clear selection
    
    def on_gallery_select(evt: gr.SelectData, image_paths):
        """Track selected image via info textbox."""
        if evt.value and 'image' in evt.value and 'path' in evt.value['image']:
            # Get the index and use our tracked original path
            idx = evt.index
            if image_paths and idx < len(image_paths):
                original_path = image_paths[idx]
                filename = Path(original_path).name
                # Return display name and internal tracking string
                return filename, f"{idx}|{original_path}"
        return "", ""
    
    def delete_selected(selected_info_str, image_paths):
        """Delete the selected image file from outputs directory."""
        if not selected_info_str or "|" not in selected_info_str:
            return gr.update(), "❌ No image selected", gr.update(), "", gr.update()
        
        # Parse the selection info
        try:
            idx_str, image_path = selected_info_str.split("|", 1)
            selected_idx = int(idx_str)
        except (ValueError, IndexError):
            return gr.update(), "❌ Invalid selection", gr.update(), "", gr.update()
        
        if not image_path:
            return gr.update(), "❌ No image path", gr.update(), "", gr.update()
        
        # Safety check: only delete files in outputs directory
        current_outputs_dir = services.get_outputs_dir()
        path = Path(image_path)
        
        try:
            path.relative_to(current_outputs_dir)
        except ValueError:
            logger.warning(f"Blocked delete attempt outside outputs dir: {path}")
            return gr.update(), "❌ Can only delete files in outputs folder", gr.update(), "", gr.update()
        
        try:
            if path.exists():
                filename = path.name
                path.unlink()
                logger.info(f"Deleted: {path}")
                
                # Refresh gallery after deletion
                images = scan_output_images(current_outputs_dir)
                count_msg = f"✓ Deleted {filename} · {len(images)} images"
                
                # Auto-select next image
                new_display = ""
                new_info = ""
                if images:
                    next_idx = min(selected_idx, len(images) - 1)
                    next_path = images[next_idx]
                    new_display = Path(next_path).name
                    new_info = f"{next_idx}|{next_path}"
                
                return images, count_msg, new_display, new_info, images
            else:
                return gr.update(), "❌ File not found", "", "", gr.update()
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return gr.update(), f"❌ Delete failed: {e}", "", "", gr.update()
    
    def get_selected_path(selected_info_str):
        """Extract path from selection info for send buttons."""
        if selected_info_str and "|" in selected_info_str:
            try:
                _, path = selected_info_str.split("|", 1)
                return path
            except:
                pass
        return None
    
    def open_outputs_folder():
        """Open the outputs folder in file explorer."""
        import subprocess
        import sys
        
        folder = services.get_outputs_dir()
        folder.mkdir(parents=True, exist_ok=True)
        
        if sys.platform == "win32":
            os.startfile(folder)
        elif sys.platform == "darwin":
            subprocess.run(["open", str(folder)])
        else:
            subprocess.run(["xdg-open", str(folder)])
    
    # Wire up events
    refresh_btn.click(
        fn=refresh_gallery,
        outputs=[gallery, gallery_status, selected_display, selected_info, current_image_paths]
    )
    
    gallery.select(
        fn=on_gallery_select,
        inputs=[current_image_paths],
        outputs=[selected_display, selected_info]
    )
    
    delete_btn.click(
        fn=delete_selected,
        inputs=[selected_info, current_image_paths],
        outputs=[gallery, gallery_status, selected_display, selected_info, current_image_paths]
    )
    
    open_folder_btn.click(fn=open_outputs_folder)
    
    # Auto-refresh timer - checks for new images every 5 seconds
    def auto_refresh_check(current_paths):
        """Check if new images were added and refresh if so."""
        global _last_image_count, _last_newest_mtime
        
        current_outputs_dir = services.get_outputs_dir()
        images = scan_output_images(current_outputs_dir)
        
        # Check if anything changed
        current_count = len(images)
        current_newest_mtime = 0.0
        if images:
            try:
                current_newest_mtime = Path(images[0]).stat().st_mtime
            except:
                pass
        
        # Only update if count changed or newest file is different
        if current_count != _last_image_count or current_newest_mtime != _last_newest_mtime:
            _last_image_count = current_count
            _last_newest_mtime = current_newest_mtime
            count_msg = f"{current_count} images" if images else "No images"
            return images, count_msg, images
        
        # No change - return gr.update() to avoid unnecessary UI updates
        return gr.update(), gr.update(), gr.update()
    
    auto_refresh_timer = gr.Timer(5, active=True)
    auto_refresh_timer.tick(
        fn=auto_refresh_check,
        inputs=[current_image_paths],
        outputs=[gallery, gallery_status, current_image_paths]
    )
    
    # Register components for inter-module access
    services.inter_module.register_component("output_gallery", gallery)
    services.inter_module.register_component("output_gallery_status", gallery_status)
    services.inter_module.register_component("output_gallery_selected_info", selected_info)
    services.inter_module.register_component("output_gallery_get_selected_path", get_selected_path)
    services.inter_module.register_component("output_gallery_send_upscale_btn", send_to_upscale_btn)
    services.inter_module.register_component("output_gallery_send_experimental_btn", send_to_experimental_btn)
    
    return {
        "accordion": accordion,
        "gallery": gallery,
        "refresh_btn": refresh_btn,
        "delete_btn": delete_btn,
        "send_to_upscale_btn": send_to_upscale_btn,
        "send_to_experimental_btn": send_to_experimental_btn,
        "status": gallery_status,
        "selected_info": selected_info,
        "get_selected_path": get_selected_path,
    }
