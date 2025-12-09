import gc
import json
import os
import random
import sys
import traceback
from dataclasses import dataclass
from typing import List, Tuple

import gradio as gr
import torch

# -----------------------------------------------------------------------------
# PART 1: Backend Engine
# -----------------------------------------------------------------------------

try:
    from flash_attn import flash_attn_varlen_func
    FLASH_VER = 2
except ModuleNotFoundError:
    flash_attn_varlen_func = None 
    FLASH_VER = None


def get_best_device():
    if torch.cuda.is_available():
        return "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def safe_empty_cache():
    """Safely clear GPU cache if available, no-op on CPU-only systems."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        pass
    gc.collect()


def download_model_snapshot(repo_id: str) -> str:
    """
    Download model using huggingface_hub with better reliability.
    Returns local path to the downloaded model.
    """
    # Skip if it's already a local path
    if os.path.isdir(repo_id):
        return repo_id
    
    try:
        from huggingface_hub import snapshot_download
        
        print(f"📥 Downloading model: {repo_id}")
        local_path = snapshot_download(
            repo_id=repo_id,
            resume_download=True,  # Resume interrupted downloads
            local_files_only=False,
            # Use default cache location
        )
        print(f"✅ Download complete: {local_path}")
        return local_path
    except Exception as e:
        print(f"⚠️ snapshot_download failed: {e}, falling back to repo_id")
        return repo_id

@dataclass
class PromptOutput:
    status: bool
    prompt: str
    message: str

class QwenPromptExpander:
    def __init__(self, model_path, device=None):
        # Auto-detect best device if not specified
        self.device = device if device is not None else get_best_device()
        self.model_path = model_path
        
        # Heuristic to determine if model is Vision-Language capable
        name_lower = str(model_path).lower()
        self.is_vl = "vl" in name_lower or "caption" in name_lower or "vision" in name_lower

        self.tokenizer = None
        self.processor = None
        self.model = None
        self.process_vision_info = None
        
        self._load_model()
    
    def _load_model(self):
        """Loads the model using Auto classes."""
        print(f"--- Loading Engine: {self.model_path} ---")
        
        # Pre-download model files with better reliability
        local_path = download_model_snapshot(self.model_path)
        
        try:
            if self.is_vl:
                from transformers import AutoProcessor
                try:
                    from transformers import AutoModelForImageTextToText as AutoModelVL
                except ImportError:
                    from transformers import AutoModelForVision2Seq as AutoModelVL

                try:
                    from qwen_vl_utils import process_vision_info
                    self.process_vision_info = process_vision_info
                except ImportError:
                     print("Warning: qwen-vl-utils not found. Vision features might fail.")
                     self.process_vision_info = None
                
                self.processor = AutoProcessor.from_pretrained(
                    local_path, trust_remote_code=True, use_fast=True, local_files_only=True
                )
                self.model = AutoModelVL.from_pretrained(
                    local_path,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16 if FLASH_VER == 2 else "auto",
                    attn_implementation="flash_attention_2" if FLASH_VER == 2 else None,
                    device_map="cpu",
                    local_files_only=True
                )
            else:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True, local_files_only=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    local_path,
                    trust_remote_code=True,
                    torch_dtype="auto",
                    attn_implementation="flash_attention_2" if FLASH_VER == 2 else None,
                    device_map="cpu",
                    local_files_only=True
                )
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {e}")

    def __call__(self, prompt, system_prompt, image=None, seed=-1, **kwargs):
        """Unified entry point."""
        if image is not None:
            return self._run_vision(prompt, system_prompt, image, seed, **kwargs)
        return self._run_text(prompt, system_prompt, seed, **kwargs)

    def _run_text(self, prompt, system_prompt, seed, temperature=0.7, max_new_tokens=1024):
        if seed < 0: seed = random.randint(0, sys.maxsize)
        try:
            self.model = self.model.to(self.device)
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": temperature > 0.0,
                "top_p": 0.9 if temperature > 0.0 else 1.0
            }

            if self.is_vl:
                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=[text], padding=False, return_tensors="pt").to(self.device)
                
                generated_ids = self.model.generate(**inputs, **gen_kwargs)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                out_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            else:
                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

                generated_ids = self.model.generate(**inputs, **gen_kwargs)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                out_text = self.tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
            
            return PromptOutput(True, out_text, "Success")

        except Exception as e:
            traceback.print_exc()
            return PromptOutput(False, prompt, str(e))
        finally:
            self.model = self.model.to("cpu")
            safe_empty_cache()

    def _run_vision(self, prompt, system_prompt, image, seed, temperature=0.7, max_new_tokens=512):
        if not self.is_vl:
            return PromptOutput(False, prompt, "Selected model is not Vision Capable.")

        if seed < 0: seed = random.randint(0, sys.maxsize)
        try:
            self.model = self.model.to(self.device)
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": temperature > 0.0,
                "top_p": 0.9 if temperature > 0.0 else 1.0
            }

            messages = [
                {'role': 'system', 'content': [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
            ]

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            if self.process_vision_info is None:
                raise ImportError("qwen_vl_utils is missing.")
                
            image_inputs, video_inputs = self.process_vision_info(messages)
            inputs = self.processor(
                text=[text], images=image_inputs, videos=video_inputs, padding=False, return_tensors="pt"
            ).to(self.device)

            generated_ids = self.model.generate(**inputs, **gen_kwargs)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            out_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            return PromptOutput(True, out_text, "Success")
        except Exception as e:
            traceback.print_exc()
            return PromptOutput(False, prompt, str(e))
        finally:
            self.model = self.model.to("cpu")
            safe_empty_cache()

# -----------------------------------------------------------------------------
# PART 2: UI Assistant Logic
# -----------------------------------------------------------------------------

DEFAULT_MODEL_DEFAULTS = [
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
    "thesby/Qwen3-VL-8B-NSFW-Caption-V4.5",
    "Goekdeniz-Guelmez/Josiefied-Qwen3-4B-abliterated-v2",
    "Goekdeniz-Guelmez/Josiefied-Qwen3-VL-4B-Instruct-abliterated-beta-v1"
]

DEFAULT_PRESETS = {
    "enhancer": {
        "Generic Enhancer": "You are an expert prompt refiner. Rewrite the user's text into a highly descriptive, visual prompt for image generation. Focus on subject details, environment, lighting, and artistic style. Output only the refined prompt.",
        "Creative Expansion": "Take the core concept provided and expand it into a detailed, artistic scene. Add cinematic lighting, texture details, and mood.",
        "Concise & Direct": "Rewrite the prompt to be concise, direct, and comma-separated. List the subject, action, clothing, background, lighting, and style tokens."
    },
    "describer": {
        "Generic Describer": "Describe the provided image in detail for the purpose of image regeneration. Focus on the main subject, their appearance, actions, and the environment. Mention the visual style, lighting, and camera perspective.",
        "Captioning": "Provide a short, one-sentence caption for this image.",
        "Detailed Analysis": "Analyze this image with extreme precision. 1. Subject (features, clothing, pose). 2. Setting (background, time, weather). 3. Tech Specs (lens, depth of field, lighting). 4. Colors."
    }
}

class PromptAssistant:
    def __init__(self, settings_file="llm_settings.json", ckpt_dir="./llm_ckpts"):
        self.settings_file = settings_file
        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)
        
        self.active_engine = None
        self.settings = self._load_settings()

    def _load_settings(self):
        base_structure = {
            "enhancer_presets": DEFAULT_PRESETS["enhancer"].copy(),
            "describer_presets": DEFAULT_PRESETS["describer"].copy(),
            "active_enhancer": "Generic Enhancer",
            "active_describer": "Generic Describer",
            "model_for_enhancer": DEFAULT_MODEL_DEFAULTS[1], 
            "model_for_describer": DEFAULT_MODEL_DEFAULTS[0],
            "custom_models": [], # List of user-added repo IDs
            "temperature": 0.7,
            "max_tokens": 1024,
            "keep_model_loaded": False  # Auto-unload after use by default (saves RAM)
        }

        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r') as f:
                    data = json.load(f)
                for key in base_structure:
                    if key in data:
                        if isinstance(base_structure[key], dict) and isinstance(data[key], dict):
                            base_structure[key].update(data[key])
                        else:
                            base_structure[key] = data[key]
                return base_structure
            except Exception as e:
                print(f"Error loading settings: {e}")
        return base_structure

    def _save_to_file(self):
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=4)
        except Exception:
            pass

    def _is_vl_name(self, name):
        """Helper to guess if a model is VL based on name"""
        n = str(name).lower()
        return "vl" in n or "vision" in n or "caption" in n

    def _scan_models(self) -> Tuple[List[str], List[str]]:
        """
        Returns (All Models, VL Models).
        Merges: Local Folders + Defaults + Custom User Models.
        """
        found_models = []
        if os.path.exists(self.ckpt_dir):
            items = os.listdir(self.ckpt_dir)
            for item in items:
                full_path = os.path.join(self.ckpt_dir, item)
                if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, "config.json")):
                    found_models.append(full_path)
        
        # Merge sources
        customs = self.settings.get("custom_models", [])
        all_options = found_models + customs + DEFAULT_MODEL_DEFAULTS
        
        # Ensure currently selected models are visible even if deleted from custom list
        curr_enh = self.settings.get("model_for_enhancer")
        curr_desc = self.settings.get("model_for_describer")
        if curr_enh: all_options.insert(0, curr_enh)
        if curr_desc: all_options.insert(0, curr_desc)

        # Deduplicate
        all_options = list(dict.fromkeys(all_options))
        
        # Filter for VL dropdown
        # Note: We liberally add any custom model that looks like VL, OR the current selection
        vl_options = [m for m in all_options if self._is_vl_name(m)]
        if curr_desc and curr_desc not in vl_options:
            vl_options.insert(0, curr_desc)
            
        return all_options, vl_options

    def _add_custom_model(self, new_model_name):
        if not new_model_name or not new_model_name.strip():
            return gr.update(), gr.update(), "⚠️ Empty Name"
        
        name = new_model_name.strip()
        if "custom_models" not in self.settings:
            self.settings["custom_models"] = []
            
        if name not in self.settings["custom_models"]:
            self.settings["custom_models"].append(name)
            self._save_to_file()
            
        all_m, vl_m = self._refresh_models()
        return all_m, vl_m, f"✅ Added: {name}"

    # --- Worker Functions (Generators) ---

    def unload_llms(self, silent=False):
        """
        Fully unload and delete the LLM model from memory.
        
        Args:
            silent: If True, return empty string (for auto-unload after operations)
        """
        if self.active_engine:
            model_name = os.path.basename(self.active_engine.model_path)
            # Delete all model components explicitly
            if self.active_engine.model is not None:
                del self.active_engine.model
            if self.active_engine.tokenizer is not None:
                del self.active_engine.tokenizer
            if self.active_engine.processor is not None:
                del self.active_engine.processor
            del self.active_engine
            self.active_engine = None
            # Force garbage collection and clear GPU cache
            safe_empty_cache()
            if silent:
                return ""
            return f"✅ {model_name} unloaded"
        if silent:
            return ""
        return "ℹ️ Nothing to unload"

    def _load_engine_generator(self, target_model):
        """Yields status updates while loading. Skips if model already loaded."""
        # Check if the correct model is already loaded
        if self.active_engine and self.active_engine.model_path == target_model:
            yield f"✅ Using cached model: {os.path.basename(target_model)}"
            return
        
        # Unload different model if one is active
        if self.active_engine:
            yield f"🔄 Unloading {os.path.basename(self.active_engine.model_path)}..."
            self.unload_llms()
        
        # Load the new model
        yield f"⏳ Loading {os.path.basename(target_model)}... (model downloads on 1st run.)"
        try:
            self.active_engine = QwenPromptExpander(model_path=target_model)
            yield f"✅ Model loaded: {os.path.basename(target_model)}"
        except Exception as e:
            raise RuntimeError(f"Load Failed: {e}")

    def enhance_prompt(self, prompt, sys_prompt_override=None):
        if not prompt or not prompt.strip():
            yield prompt, "⚠️ Please enter a prompt first."
            return

        target_model = self.settings.get("model_for_enhancer", DEFAULT_MODEL_DEFAULTS[1])
        sys_prompt = sys_prompt_override or self.settings["enhancer_presets"].get(
            self.settings.get("active_enhancer"), ""
        )
        temp = self.settings.get("temperature", 0.7)
        max_tok = self.settings.get("max_tokens", 1024)
        keep_loaded = self.settings.get("keep_model_loaded", False)

        final_prompt = prompt
        final_status = ""
        model_loaded = False  # Track if we successfully loaded a model
        
        try:
            for status_msg in self._load_engine_generator(target_model):
                yield prompt, status_msg
            
            model_loaded = self.active_engine is not None
            
            if not self.active_engine:
                final_status = "❌ Failed to load model"
            else:
                yield prompt, "✨ Enhancing Text..."
                result = self.active_engine(
                    prompt, system_prompt=sys_prompt, 
                    temperature=temp, max_new_tokens=max_tok
                )
                if result.status:
                    final_prompt = result.prompt.strip()
                    final_status = "✅ Enhanced Successfully"
                else:
                    final_status = f"❌ Failed: {result.message}"
        except Exception as e:
            traceback.print_exc()
            final_status = f"❌ Error: {str(e)}"
        finally:
            # Auto-unload to free RAM unless user wants to keep model loaded
            if model_loaded and not keep_loaded:
                self.unload_llms(silent=True)
        
        yield final_prompt, final_status

    def describe_image(self, image_path, prompt_context, sys_prompt_override=None):
        if not image_path:
            yield prompt_context, "⚠️ Please upload an image first."
            return

        target_model = self.settings.get("model_for_describer", DEFAULT_MODEL_DEFAULTS[0])
        sys_prompt = sys_prompt_override or self.settings["describer_presets"].get(
            self.settings.get("active_describer"), ""
        )
        temp = self.settings.get("temperature", 0.7)
        max_tok = self.settings.get("max_tokens", 1024)
        keep_loaded = self.settings.get("keep_model_loaded", False)

        final_prompt = prompt_context
        final_status = ""
        model_loaded = False  # Track if we successfully loaded a model

        try:
            for status_msg in self._load_engine_generator(target_model):
                yield prompt_context, status_msg
            
            model_loaded = self.active_engine is not None

            if not self.active_engine or not self.active_engine.is_vl:
                final_status = f"⚠️ Error: '{os.path.basename(target_model)}' is not a Vision Model."
                # Don't return early - let finally block handle cleanup
            else:
                trigger = prompt_context if prompt_context and prompt_context.strip() else "Describe this image."
                yield prompt_context, "👁️ Analyzing Image..."
                
                result = self.active_engine(
                    trigger, image=image_path, system_prompt=sys_prompt,
                    temperature=temp, max_new_tokens=max_tok
                )
                if result.status:
                    final_prompt = result.prompt.strip()
                    final_status = "✅ Description Generated"
                else:
                    final_status = f"❌ Failed: {result.message}"
        except Exception as e:
            traceback.print_exc()
            final_status = f"❌ Error: {str(e)}"
        finally:
            # Auto-unload to free RAM unless user wants to keep model loaded
            # Always unload if model was loaded, regardless of keep_loaded setting on error paths
            if model_loaded and not keep_loaded:
                self.unload_llms(silent=True)
        
        yield final_prompt, final_status

    # --- Settings Logic ---

    def _save_models_and_params(self, enh_model, desc_model, temp, max_tok, keep_loaded):
        self.settings["model_for_enhancer"] = enh_model
        self.settings["model_for_describer"] = desc_model
        self.settings["temperature"] = temp
        self.settings["max_tokens"] = max_tok
        self.settings["keep_model_loaded"] = keep_loaded
        self._save_to_file()
        return "💾 Settings Saved"

    def _update_preset(self, category, name, content):
        if not name.strip() or not content.strip(): return gr.update(), gr.update(), "⚠️ Empty fields."
        target_dict = "enhancer_presets" if category == "enhancer" else "describer_presets"
        active_key = "active_enhancer" if category == "enhancer" else "active_describer"
        
        self.settings[target_dict][name] = content
        self.settings[active_key] = name
        self._save_to_file()
        keys = list(self.settings[target_dict].keys())
        return gr.update(choices=keys, value=name), gr.update(value=content), f"✅ Saved '{name}'"

    def _delete_preset(self, category, name):
        target_dict = "enhancer_presets" if category == "enhancer" else "describer_presets"
        active_key = "active_enhancer" if category == "enhancer" else "active_describer"
        if name not in self.settings[target_dict]: return gr.update(), gr.update(), gr.update(), "⚠️ Not found."
        if len(self.settings[target_dict]) <= 1: return gr.update(), gr.update(), gr.update(), "⚠️ Cannot delete last preset."

        del self.settings[target_dict][name]
        keys = list(self.settings[target_dict].keys())
        new_active = keys[0]
        self.settings[active_key] = new_active
        self._save_to_file()
        return gr.update(choices=keys, value=new_active), gr.update(value=new_active), gr.update(value=self.settings[target_dict][new_active]), f"🗑️ Deleted '{name}'"

    def _load_preset_content(self, category, name):
        target_dict = "enhancer_presets" if category == "enhancer" else "describer_presets"
        active_key = "active_enhancer" if category == "enhancer" else "active_describer"
        if name in self.settings[target_dict]:
            self.settings[active_key] = name
            self._save_to_file()
            return name, self.settings[target_dict][name]
        return "", ""

    def _refresh_models(self):
        all_m, vl_m = self._scan_models()
        return gr.update(choices=all_m), gr.update(choices=vl_m)

    # --- UI Renders ---

    def render_settings_ui(self):
        gr.Markdown("### ⚙️ Prompt Assistant Configuration")
        
        all_models, vl_models = self._scan_models()
        
        # --- MODEL SELECTION ---
        with gr.Row(variant="panel"):
            with gr.Column(scale=2):
                enh_model_dd = gr.Dropdown(
                    choices=all_models, 
                    value=self.settings.get("model_for_enhancer"), 
                    label="📝 Text Enhancer Model", 
                    allow_custom_value=True
                )
                desc_model_dd = gr.Dropdown(
                    choices=vl_models, 
                    value=self.settings.get("model_for_describer"), 
                    label="👁️ Vision Describer Model", 
                    allow_custom_value=True,
                    info="Filtered for 'VL' in name, but you can add custom ones below."
                )
            
            with gr.Column(scale=1):
                refresh_btn = gr.Button("🔄 Refresh Lists", size="sm")
                save_models_btn = gr.Button("💾 Save All Settings", size="sm", variant="primary")
                model_stat = gr.Label(show_label=False)
            with gr.Column(scale=1):
                temp_slider = gr.Slider(0.0, 2.0, value=self.settings.get("temperature", 0.7), label="Temperature")
                tok_slider = gr.Slider(64, 4096, step=64, value=self.settings.get("max_tokens", 1024), label="Max Tokens")
                keep_loaded_cb = gr.Checkbox(
                    value=self.settings.get("keep_model_loaded", False),
                    label="Keep Model Loaded",
                    info="Keep LLM in RAM after use (faster repeat use, but uses more memory)"
                )
                
        # --- ADD CUSTOM MODEL ---
        with gr.Row():
            custom_model_input = gr.Textbox(
                label="Add Custom Model", 
                placeholder="Paste HuggingFace Repo ID or Local Path...", 
                lines=1, 
                scale=3
            )
            add_custom_btn = gr.Button("Add to Options", scale=1)

        # Events for Models
        refresh_btn.click(self._refresh_models, outputs=[enh_model_dd, desc_model_dd])
        
        save_models_btn.click(
            fn=self._save_models_and_params,
            inputs=[enh_model_dd, desc_model_dd, temp_slider, tok_slider, keep_loaded_cb],
            outputs=[model_stat]
        )

        add_custom_btn.click(
            fn=self._add_custom_model,
            inputs=[custom_model_input],
            outputs=[enh_model_dd, desc_model_dd, model_stat]
        ).then(lambda: "", outputs=[custom_model_input]) # Clear input after add

        gr.Markdown("---")
        
        # --- PRESETS ---
        with gr.Row():
            # Enhancer Column
            with gr.Column():
                gr.Markdown("#### ✨ Text Enhancer Presets")
                enh_choices = list(self.settings["enhancer_presets"].keys())
                enh_active = self.settings.get("active_enhancer", enh_choices[0])
                
                enh_dd = gr.Dropdown(enh_choices, value=enh_active, label="Active Preset")
                enh_name = gr.Textbox(label="Edit Name", value=enh_active, lines=1.5)
                enh_cont = gr.Textbox(
                    label="Edit System Prompt", 
                    value=self.settings["enhancer_presets"][enh_active], 
                    lines=10,
                    max_lines=20,
                    interactive=True
                )
                with gr.Row():
                    enh_save = gr.Button("💾 Save Changes", size="sm")
                    enh_del = gr.Button("🗑️ Delete", variant="stop", size="sm")
                enh_stat = gr.Label(show_label=False)

            # Describer Column
            with gr.Column():
                gr.Markdown("#### 🖼️ Image Describer Presets")
                desc_choices = list(self.settings["describer_presets"].keys())
                desc_active = self.settings.get("active_describer", desc_choices[0])
                
                desc_dd = gr.Dropdown(desc_choices, value=desc_active, label="Active Preset")
                desc_name = gr.Textbox(label="Edit Name", value=desc_active, lines=1.5)
                desc_cont = gr.Textbox(
                    label="Edit System Prompt", 
                    value=self.settings["describer_presets"][desc_active], 
                    lines=10, 
                    max_lines=20,
                    interactive=True
                )
                with gr.Row():
                    desc_save = gr.Button("💾 Save Changes", size="sm")
                    desc_del = gr.Button("🗑️ Delete", variant="stop", size="sm")
                desc_stat = gr.Label(show_label=False)

        # Wire up events
        enh_dd.change(lambda n: self._load_preset_content("enhancer", n), [enh_dd], [enh_name, enh_cont])
        enh_save.click(lambda n, c: self._update_preset("enhancer", n, c), [enh_name, enh_cont], [enh_dd, enh_cont, enh_stat])
        enh_del.click(lambda n: self._delete_preset("enhancer", n), [enh_dd], [enh_dd, enh_name, enh_cont, enh_stat])

        desc_dd.change(lambda n: self._load_preset_content("describer", n), [desc_dd], [desc_name, desc_cont])
        desc_save.click(lambda n, c: self._update_preset("describer", n, c), [desc_name, desc_cont], [desc_dd, desc_cont, desc_stat])
        desc_del.click(lambda n: self._delete_preset("describer", n), [desc_dd], [desc_dd, desc_name, desc_cont, desc_stat])

    def render_main_ui(self, target_textbox, image_input=None):
        with gr.Accordion("🤖 AI Prompt Assistant", open=False):
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        enhance_btn = gr.Button("✨ Enhance Text", size="sm", variant="primary")
                        if image_input:
                            describe_btn = gr.Button("🖼️ Describe Image", size="sm", variant="secondary")
                        clear_btn = gr.Button("🗑️ Clear Prompt", size="sm", variant="secondary")
                
                with gr.Column(scale=1):
                     unload_btn = gr.Button("🧹 Unload", size="sm", variant="stop")

            status_display = gr.Textbox(
                label="Assistant Status", 
                value="Ready", 
                interactive=False, 
                lines=2,
                max_lines=4
            )
            
            enhance_btn.click(
                self.enhance_prompt, 
                inputs=[target_textbox], 
                outputs=[target_textbox, status_display]
            )

            clear_btn.click(lambda: "", outputs=[target_textbox])

            if image_input:
                describe_btn.click(
                    self.describe_image, 
                    inputs=[image_input, target_textbox], 
                    outputs=[target_textbox, status_display]
                )
                image_input.change(lambda: "", outputs=[target_textbox])

            unload_btn.click(
                self.unload_llms, 
                outputs=[status_display]
            )