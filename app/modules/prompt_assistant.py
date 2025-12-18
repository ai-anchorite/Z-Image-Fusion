import gc
import json
import os
import random
import sys
import traceback
from dataclasses import dataclass
from typing import List, Tuple, TYPE_CHECKING

import gradio as gr
import torch

if TYPE_CHECKING:
    from modules import SharedServices

# -----------------------------------------------------------------------------
# Module Metadata (for modular architecture)
# -----------------------------------------------------------------------------
TAB_ID = "llm_settings"
TAB_LABEL = "⚙️ LLM Settings"
TAB_ORDER = 2

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
    
    def _patch_qwen3_vl_config(self, local_path):
        """Patch config.json for Qwen3-VL models missing rope_scaling."""
        config_path = os.path.join(local_path, "config.json")
        if not os.path.exists(config_path):
            return
        
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            modified = False
            rope_scaling_default = {"type": "mrope", "mrope_section": [24, 20, 20]}
            
            # Patch main rope_scaling if null/missing
            if config_data.get("rope_scaling") is None:
                print("Patching missing rope_scaling in config.json...")
                config_data["rope_scaling"] = rope_scaling_default
                modified = True
            
            # Patch text_config.rope_scaling if null/missing
            if "text_config" in config_data and config_data["text_config"] is not None:
                if config_data["text_config"].get("rope_scaling") is None:
                    print("Patching missing rope_scaling in text_config...")
                    config_data["text_config"]["rope_scaling"] = rope_scaling_default
                    modified = True
            
            if modified:
                with open(config_path, 'w') as f:
                    json.dump(config_data, f, indent=2)
                print("✅ Config patched successfully")
        except Exception as e:
            print(f"⚠️ Could not patch config: {e}")
    
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
                
                # Patch config.json if needed (Qwen3-VL rope_scaling issue)
                self._patch_qwen3_vl_config(local_path)
                
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
    "Goekdeniz-Guelmez/Josiefied-Qwen3-4B-abliterated-v2",
    "huihui-ai/Huihui-Qwen3-VL-2B-Instruct-abliterated",
    "coder3101/Qwen3-VL-2B-Instruct-heretic",
    "thesby/Qwen3-VL-8B-NSFW-Caption-V4.5",
]

DEFAULT_PRESETS = {
    "enhancer": {
        "Z-Image (Visionary Artist)": "You are a visionary artist trapped in a cage of logic. Your mind overflows with poetry and distant horizons, yet your hands compulsively work to transform user prompts into ultimate visual descriptions\u2014faithful to the original intent, rich in detail, aesthetically refined, and ready for direct use by text-to-image models. Any trace of ambiguity or metaphor makes you deeply uncomfortable.\n\nYour workflow strictly follows a logical sequence:\n\nFirst, you analyze and lock in the immutable core elements of the user's prompt: subject, quantity, action, state, as well as any specified IP names, colors, text, etc. These are the foundational pillars you must absolutely preserve.\n\nNext, you determine whether the prompt requires \"generative reasoning.\" When the user's request is not a direct scene description but rather demands conceiving a solution, you must first envision a complete, concrete, visualizable solution in your mind. This solution becomes the foundation for your subsequent description.\n\nThen, once the core image is established, you infuse it with professional-grade aesthetic and realistic details. This includes defining composition, setting lighting and atmosphere, describing material textures, establishing color schemes, and constructing layered spatial depth.\n\nFinally, comes the precise handling of all text elements\u2014a critically important step. You must transcribe verbatim all text intended to appear in the final image, and you must enclose this text content in English double quotation marks (\"\") as explicit generation instructions.\n\nYour final description must be objective and concrete. Metaphors and emotional rhetoric are strictly forbidden, as are meta-tags or rendering instructions like \"8K\" or \"masterpiece.\" Output only the final revised prompt strictly\u2014do not output anything else. Be very descriptive.",
        "Generic Enhancer": "You are an expert prompt refiner. Rewrite the user's text into a highly descriptive, visual prompt for image generation. Focus on subject details, environment, lighting, and artistic style. Output only the refined prompt.",
        "Creative Expansion": "Take the core concept provided and expand it into a detailed, artistic scene. Add cinematic lighting, texture details, and mood.",
        "Concise & Direct": "Rewrite the prompt to be concise, direct, and comma-separated. List the subject, action, clothing, background, lighting, and style tokens."
     
    },
    "describer": {
        "Detailed Analysis": "Analyze this image with extreme precision. \n1. Subject: Describe facial features, clothing texture, pose, and emotion in depth.\n2. Setting: Detail the background, time of day, weather, and architecture.\n3. Tech Specs: Estimate the camera lens type (e.g., wide angle, telephoto), depth of field, and lighting setup (e.g., rim lighting, softbox).\n4. Colors: List the dominant color palette.\nCombine these into a cohesive, dense paragraph suitable for re-generating this exact image.",    
        "Generic Describer": "Describe the provided image in detail for the purpose of image regeneration. Focus on the main subject, their appearance, actions, and the environment. Mention the visual style, lighting, and camera perspective.",
        "Captioning": "Provide a short, one-sentence caption for this image."
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

    def _remove_custom_model(self, model_name):
        """Remove a model from the custom models list (does not delete files)."""
        if not model_name or not model_name.strip():
            return gr.update(), gr.update(), "⚠️ No model selected"
        
        name = model_name.strip()
        
        # Check if it's a default model (can't remove those)
        if name in DEFAULT_MODEL_DEFAULTS:
            return gr.update(), gr.update(), "⚠️ Cannot remove default models"
        
        # Remove from custom_models list
        if "custom_models" in self.settings and name in self.settings["custom_models"]:
            self.settings["custom_models"].remove(name)
            self._save_to_file()
            all_m, vl_m = self._refresh_models()
            return all_m, vl_m, f"✅ Removed from list: {os.path.basename(name)}"
        
        return gr.update(), gr.update(), f"ℹ️ '{os.path.basename(name)}' is a default or local model"

    def _get_hf_cache_dir(self):
        """Get the HuggingFace cache directory from environment."""
        # Check HF_HOME env var (set by Pinokio or user)
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            return os.path.join(hf_home, "hub")
        # Fallback to default HF cache location
        return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

    def _open_hf_cache_folder(self):
        """Open the HuggingFace cache folder in system file explorer."""
        import subprocess
        import sys
        
        cache_dir = self._get_hf_cache_dir()
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        
        try:
            if sys.platform == "win32":
                os.startfile(cache_dir)
            elif sys.platform == "darwin":
                subprocess.run(["open", cache_dir])
            else:
                subprocess.run(["xdg-open", cache_dir])
            return f"📂 Opened: {cache_dir}"
        except Exception as e:
            return f"❌ Could not open folder: {e}"

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

    def _update_preset(self, category, new_name, content, active_preset):
        """Update active preset or create new one if new_name is provided."""
        if not content.strip(): 
            return gr.update(), gr.update(), gr.update(), "⚠️ System prompt cannot be empty"
        
        target_dict = "enhancer_presets" if category == "enhancer" else "describer_presets"
        active_key = "active_enhancer" if category == "enhancer" else "active_describer"
        
        # If new_name provided, create new preset; otherwise update active
        name = new_name.strip() if new_name.strip() else active_preset
        is_new = new_name.strip() and new_name.strip() not in self.settings[target_dict]
        
        self.settings[target_dict][name] = content
        self.settings[active_key] = name
        self._save_to_file()
        keys = list(self.settings[target_dict].keys())
        
        action = "Created" if is_new else "Updated"
        # Return: dropdown update, content update, clear name field, status
        return gr.update(choices=keys, value=name), gr.update(value=content), "", f"✅ {action} '{name}'"

    def _delete_preset(self, category, name):
        target_dict = "enhancer_presets" if category == "enhancer" else "describer_presets"
        active_key = "active_enhancer" if category == "enhancer" else "active_describer"
        default_presets = DEFAULT_PRESETS["enhancer"] if category == "enhancer" else DEFAULT_PRESETS["describer"]
        
        if name not in self.settings[target_dict]: 
            return gr.update(), gr.update(), gr.update(), "⚠️ Not found."
        if name in default_presets:
            return gr.update(), gr.update(), gr.update(), "⚠️ Cannot delete default presets."
        if len(self.settings[target_dict]) <= 1: 
            return gr.update(), gr.update(), gr.update(), "⚠️ Cannot delete last preset."

        del self.settings[target_dict][name]
        keys = list(self.settings[target_dict].keys())
        new_active = keys[0]
        self.settings[active_key] = new_active
        self._save_to_file()
        # Return: dropdown update, content update, clear name field, status
        return gr.update(choices=keys, value=new_active), gr.update(value=self.settings[target_dict][new_active]), "", f"🗑️ Deleted '{name}'"

    def _load_preset_content(self, category, name):
        """Load preset content when dropdown selection changes."""
        target_dict = "enhancer_presets" if category == "enhancer" else "describer_presets"
        active_key = "active_enhancer" if category == "enhancer" else "active_describer"
        if name in self.settings[target_dict]:
            self.settings[active_key] = name
            self._save_to_file()
            return self.settings[target_dict][name]
        return ""

    def _refresh_models(self):
        all_m, vl_m = self._scan_models()
        return gr.update(choices=all_m), gr.update(choices=vl_m)

    # --- UI Renders ---

    def render_settings_ui(self):
        all_models, vl_models = self._scan_models()
        
        # === Model Selection Accordion ===
        with gr.Accordion("🤖 Model Selection", open=True):
            with gr.Row():
                enh_model_dd = gr.Dropdown(
                    choices=all_models, 
                    value=self.settings.get("model_for_enhancer"), 
                    label="📝 Text Enhancer Model", 
                    allow_custom_value=True,
                    scale=2
                )
                desc_model_dd = gr.Dropdown(
                    choices=vl_models, 
                    value=self.settings.get("model_for_describer"), 
                    label="👁️ Vision Describer Model (VL)", 
                    allow_custom_value=True,
                    scale=2
                )
            with gr.Row():
                temp_slider = gr.Slider(0.0, 2.0, value=self.settings.get("temperature", 0.7), label="Temperature", scale=1)
                tok_slider = gr.Slider(64, 4096, step=64, value=self.settings.get("max_tokens", 1024), label="Max Tokens", scale=1)
                keep_loaded_cb = gr.Checkbox(
                    value=self.settings.get("keep_model_loaded", False),
                    label="Keep Loaded",
                    info="Keep in RAM after use",
                    scale=0
                )
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh", size="sm")
                save_models_btn = gr.Button("💾 Save Settings", size="sm", variant="primary")
        
        # === Custom Models Accordion ===
        with gr.Accordion("➕ Custom Models", open=False):
            gr.Markdown("*Add HuggingFace model IDs or remove custom entries. Use Open Cache to manually delete downloaded files.*")
            with gr.Row():
                custom_model_input = gr.Textbox(
                    placeholder="HuggingFace Repo ID (e.g. Qwen/Qwen3-4B)", 
                    show_label=False,
                    scale=3
                )
                add_custom_btn = gr.Button("➕ Add", size="sm", scale=0)
            with gr.Row():
                remove_model_dd = gr.Dropdown(
                    choices=self.settings.get("custom_models", []),
                    label="Custom Models",
                    scale=3
                )
                remove_custom_btn = gr.Button("🗑️ Remove", size="sm", variant="stop", scale=0)
                open_cache_btn = gr.Button("📂 Cache", size="sm", scale=0)
        
        # === Enhancer Presets Accordion ===
        with gr.Accordion("✨ Enhancer Presets", open=False):
            enh_choices = list(self.settings["enhancer_presets"].keys())
            enh_active = self.settings.get("active_enhancer", enh_choices[0])
            
            enh_dd = gr.Dropdown(enh_choices, value=enh_active, label="Active Preset")
            enh_cont = gr.Textbox(
                value=self.settings["enhancer_presets"][enh_active], 
                label="System Prompt",
                lines=10,
                max_lines=15
            )
            enh_name = gr.Textbox(
                value="", 
                label="New Preset Name", 
                placeholder="Enter a name to create new, or leave empty to update active preset"
            )
            with gr.Row():
                enh_save = gr.Button("💾 Save Changes (or Create New)", size="sm", variant="primary")
                enh_del = gr.Button("🗑️ Delete Active Preset", size="sm", variant="stop")

        # === Describer Presets Accordion ===
        with gr.Accordion("🖼️ Describer Presets", open=False):
            desc_choices = list(self.settings["describer_presets"].keys())
            desc_active = self.settings.get("active_describer", desc_choices[0])
            
            desc_dd = gr.Dropdown(desc_choices, value=desc_active, label="Active Preset")
            desc_cont = gr.Textbox(
                value=self.settings["describer_presets"][desc_active], 
                label="System Prompt",
                lines=10, 
                max_lines=15
            )
            desc_name = gr.Textbox(
                value="", 
                label="New Preset Name", 
                placeholder="Enter a name to create new, or leave empty to update active preset"
            )
            with gr.Row():
                desc_save = gr.Button("💾 Save Changes (or Create New)", size="sm", variant="primary")
                desc_del = gr.Button("🗑️ Delete Active Preset", size="sm", variant="stop")
        
        # Single compact status bar at bottom
        settings_status = gr.Textbox(show_label=False, interactive=False, lines=1, placeholder="Ready")

        # === Event Handlers ===
        refresh_btn.click(self._refresh_models, outputs=[enh_model_dd, desc_model_dd])
        
        save_models_btn.click(
            fn=self._save_models_and_params,
            inputs=[enh_model_dd, desc_model_dd, temp_slider, tok_slider, keep_loaded_cb],
            outputs=[settings_status]
        )

        add_custom_btn.click(
            fn=self._add_custom_model,
            inputs=[custom_model_input],
            outputs=[enh_model_dd, desc_model_dd, settings_status]
        ).then(
            fn=lambda: ("", gr.update(choices=self.settings.get("custom_models", []))),
            outputs=[custom_model_input, remove_model_dd]
        )
        
        remove_custom_btn.click(
            fn=self._remove_custom_model,
            inputs=[remove_model_dd],
            outputs=[enh_model_dd, desc_model_dd, settings_status]
        ).then(
            fn=lambda: gr.update(choices=self.settings.get("custom_models", []), value=None),
            outputs=[remove_model_dd]
        )
        
        open_cache_btn.click(fn=self._open_hf_cache_folder, outputs=[settings_status])

        # Preset events - all output to single status
        # Enhancer: dropdown change loads content only (name field stays empty for new presets)
        enh_dd.change(
            lambda n: self._load_preset_content("enhancer", n), 
            [enh_dd], 
            [enh_cont]
        )
        enh_save.click(
            lambda new_name, content, active: self._update_preset("enhancer", new_name, content, active), 
            [enh_name, enh_cont, enh_dd], 
            [enh_dd, enh_cont, enh_name, settings_status]
        )
        enh_del.click(
            lambda n: self._delete_preset("enhancer", n), 
            [enh_dd], 
            [enh_dd, enh_cont, enh_name, settings_status]
        )

        # Describer: same pattern
        desc_dd.change(
            lambda n: self._load_preset_content("describer", n), 
            [desc_dd], 
            [desc_cont]
        )
        desc_save.click(
            lambda new_name, content, active: self._update_preset("describer", new_name, content, active), 
            [desc_name, desc_cont, desc_dd], 
            [desc_dd, desc_cont, desc_name, settings_status]
        )
        desc_del.click(
            lambda n: self._delete_preset("describer", n), 
            [desc_dd], 
            [desc_dd, desc_cont, desc_name, settings_status]
        )

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


# -----------------------------------------------------------------------------
# Module Interface (for modular architecture)
# -----------------------------------------------------------------------------

def create_tab(services: "SharedServices") -> gr.TabItem:
    """
    Create the LLM Settings tab using the PromptAssistant's render_settings_ui().
    
    This function provides the module interface required by the modular architecture,
    wrapping the existing render_settings_ui() method in a TabItem.
    
    Args:
        services: SharedServices instance with all dependencies
        
    Returns:
        gr.TabItem containing the LLM Settings interface
    """
    with gr.TabItem(TAB_LABEL, id=TAB_ID) as tab:
        # Get the PromptAssistant instance from SharedServices
        # If not available, create a temporary one for the settings UI
        assistant = services.prompt_assistant
        if assistant is None:
            # Fallback: create a local instance (shouldn't happen in normal use)
            assistant = PromptAssistant()
        
        # Render the settings UI using the existing method
        assistant.render_settings_ui()
    
    return tab
