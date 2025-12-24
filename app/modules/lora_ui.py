"""
LoRA UI Components

Reusable Gradio components for LoRA selection across modules.
Provides a consistent UI pattern for 6-slot LoRA selection with
enable checkboxes, dropdowns, strength sliders, and progressive reveal.
"""

import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import gradio as gr

if TYPE_CHECKING:
    from modules import SharedServices

logger = logging.getLogger(__name__)

# Dummy lora filename (used when lora disabled - strength 0 bypasses it)
DUMMY_LORA = "none.safetensors"

# Maximum number of LoRA slots
MAX_LORA_SLOTS = 6


def scan_loras(loras_dir: Path) -> list:
    """Scan loras directory for available LoRA files."""
    if not loras_dir.exists():
        return []
    loras = []
    for f in loras_dir.rglob("*.safetensors"):
        rel_path = str(f.relative_to(loras_dir))
        if rel_path != DUMMY_LORA:  # Exclude dummy
            loras.append(rel_path)
    return sorted(loras)


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


def open_folder(folder_path: Path):
    """Cross-platform folder opener."""
    folder_path.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        os.startfile(folder_path)
    elif sys.platform == "darwin":
        subprocess.run(["open", str(folder_path)])
    else:
        subprocess.run(["xdg-open", str(folder_path)])


@dataclass
class LoraSlot:
    """Container for a single LoRA slot's UI components."""
    row: gr.Row  # Container row for visibility control
    enabled: gr.Checkbox
    name: gr.Dropdown
    strength: gr.Slider


@dataclass
class LoraComponents:
    """Container for LoRA UI components returned by create_lora_ui."""
    # All 6 LoRA slots
    slots: list  # List of LoraSlot objects
    # Buttons
    add_btn: gr.Button
    refresh_btn: gr.Button
    open_folder_btn: gr.Button
    # State tracking visible count
    visible_count: gr.State


def create_lora_ui(loras_dir: Path, accordion_open: bool = False, initial_visible: int = 1) -> LoraComponents:
    """
    Create the LoRA accordion UI with 6 slots and progressive reveal.
    
    Args:
        loras_dir: Path to the loras directory
        accordion_open: Whether the accordion should be open by default
        initial_visible: Number of LoRA slots visible initially (1-6)
        
    Returns:
        LoraComponents dataclass with all UI components
    """
    # Ensure dummy lora exists
    ensure_dummy_lora(loras_dir)
    
    # Scan available loras
    loras = scan_loras(loras_dir)
    
    # Clamp initial visible
    initial_visible = max(1, min(initial_visible, MAX_LORA_SLOTS))
    
    slots = []
    
    with gr.Accordion("🎨 LoRA", open=accordion_open):
        # Default strengths per slot - decreasing for stacking
        default_strengths = [1.0, 0.5, 0.5, 0.25, 0.25, 0.25]
        
        # Create 6 LoRA slots
        for i in range(1, MAX_LORA_SLOTS + 1):
            # First slot always visible, others based on initial_visible
            is_visible = i <= initial_visible
            default_strength = default_strengths[i - 1]
            
            with gr.Row(visible=is_visible) as row:
                enabled = gr.Checkbox(label="", value=False, scale=0, min_width=30)
                name = gr.Dropdown(
                    label=f"LoRA {i}",
                    choices=loras,
                    value=None,
                    interactive=True,
                    scale=3,
                    allow_custom_value=True
                )
                strength = gr.Slider(
                    label="Strength",
                    value=default_strength,
                    minimum=-2.0,
                    maximum=2.0,
                    step=0.05,
                    scale=1
                )
            
            slots.append(LoraSlot(row=row, enabled=enabled, name=name, strength=strength))
        
        # Add LoRA button - hidden when all 6 are visible
        add_btn = gr.Button(
            "➕ Add LoRA", 
            size="sm", 
            variant="secondary",
            visible=(initial_visible < MAX_LORA_SLOTS)
        )
        
        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh", size="sm", scale=0)
            open_folder_btn = gr.Button("📂 Open LoRAs Folder", size="sm", scale=1)
        
        gr.Markdown("*⭐ Tip: Distilled models don't stack LoRAs well. Try lowering strength when using multiple.*")
        gr.Markdown("*[🔗 CivitAI LoRAs](https://civitai.com/models) (filter 'Z-Image') · [🔗 Character LoRAs](https://huggingface.co/spaces/malcolmrey/browser) (filter 'ZImage')*")
        
        # State to track how many slots are visible
        visible_count = gr.State(value=initial_visible)
    
    return LoraComponents(
        slots=slots,
        add_btn=add_btn,
        refresh_btn=refresh_btn,
        open_folder_btn=open_folder_btn,
        visible_count=visible_count,
    )


def setup_lora_handlers(lora_components: LoraComponents, loras_dir: Path):
    """
    Wire up event handlers for LoRA UI components.
    
    Args:
        lora_components: LoraComponents from create_lora_ui
        loras_dir: Path to the loras directory
    """
    slots = lora_components.slots
    
    # Refresh button - updates all 6 dropdowns
    def refresh_loras():
        loras = scan_loras(loras_dir)
        return tuple(gr.update(choices=loras) for _ in range(MAX_LORA_SLOTS))
    
    lora_components.refresh_btn.click(
        fn=refresh_loras,
        outputs=[slot.name for slot in slots]
    )
    
    # Open folder button
    lora_components.open_folder_btn.click(
        fn=lambda: open_folder(loras_dir),
        outputs=[]
    )
    
    # Add LoRA button - reveals next hidden slot
    def add_lora_slot(current_count):
        new_count = min(current_count + 1, MAX_LORA_SLOTS)
        
        # Build visibility updates for all 6 rows
        row_updates = []
        for i in range(1, MAX_LORA_SLOTS + 1):
            row_updates.append(gr.update(visible=(i <= new_count)))
        
        # Hide add button when all slots visible
        add_btn_visible = new_count < MAX_LORA_SLOTS
        
        return (new_count, gr.update(visible=add_btn_visible), *row_updates)
    
    lora_components.add_btn.click(
        fn=add_lora_slot,
        inputs=[lora_components.visible_count],
        outputs=[
            lora_components.visible_count,
            lora_components.add_btn,
            *[slot.row for slot in slots]
        ]
    )


def get_lora_params(
    lora1_enabled: bool, lora1_name: str, lora1_strength: float,
    lora2_enabled: bool, lora2_name: str, lora2_strength: float,
    lora3_enabled: bool, lora3_name: str, lora3_strength: float,
    lora4_enabled: bool = False, lora4_name: str = None, lora4_strength: float = 1.0,
    lora5_enabled: bool = False, lora5_name: str = None, lora5_strength: float = 1.0,
    lora6_enabled: bool = False, lora6_name: str = None, lora6_strength: float = 1.0,
) -> dict:
    """
    Build LoRA params dict for workflow execution.
    
    Returns dict with lora1_name, lora1_strength, lora2_name, etc. for all 6 slots.
    Uses DUMMY_LORA with strength 0 for disabled slots.
    """
    def get_slot_params(enabled, name, strength):
        if enabled and name:
            return name, strength
        return DUMMY_LORA, 0
    
    l1_name, l1_str = get_slot_params(lora1_enabled, lora1_name, lora1_strength)
    l2_name, l2_str = get_slot_params(lora2_enabled, lora2_name, lora2_strength)
    l3_name, l3_str = get_slot_params(lora3_enabled, lora3_name, lora3_strength)
    l4_name, l4_str = get_slot_params(lora4_enabled, lora4_name, lora4_strength)
    l5_name, l5_str = get_slot_params(lora5_enabled, lora5_name, lora5_strength)
    l6_name, l6_str = get_slot_params(lora6_enabled, lora6_name, lora6_strength)
    
    return {
        "lora1_name": l1_name, "lora1_strength": l1_str,
        "lora2_name": l2_name, "lora2_strength": l2_str,
        "lora3_name": l3_name, "lora3_strength": l3_str,
        "lora4_name": l4_name, "lora4_strength": l4_str,
        "lora5_name": l5_name, "lora5_strength": l5_str,
        "lora6_name": l6_name, "lora6_strength": l6_str,
    }


def get_lora_inputs(lora_components: LoraComponents) -> list:
    """
    Get list of LoRA input components for use in gr.Button.click() inputs.
    
    Returns list in order: [enabled1, name1, strength1, enabled2, name2, strength2, ...]
    for all 6 slots (18 components total).
    """
    inputs = []
    for slot in lora_components.slots:
        inputs.extend([slot.enabled, slot.name, slot.strength])
    return inputs
