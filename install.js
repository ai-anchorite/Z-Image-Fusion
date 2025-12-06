module.exports = {
  run: [
    // Clone ComfyUI into app/comfyui (skip if already exists)
    {
      method: "shell.run",
      params: {
        message: [
          "git clone https://github.com/comfyanonymous/ComfyUI app/comfyui",
        ],
      }
    },
    // Clone ComfyUI-Manager for custom node management
    {
      method: "shell.run",
      params: {
        message: [
          "git clone https://github.com/ltdrdata/ComfyUI-Manager app/comfyui/custom_nodes/ComfyUI-Manager",
        ],
      }
    },
    // Clone ComfyUI-GGUF for quantized model support
    {
      method: "shell.run",
      params: {
        message: [
          "git clone https://github.com/city96/ComfyUI-GGUF app/comfyui/custom_nodes/ComfyUI-GGUF",
        ],
      }
    },
    // Clone ComfyUI-SeedVR2_VideoUpscaler for upscaling
    {
      method: "shell.run",
      params: {
        message: [
          "git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler app/comfyui/custom_nodes/ComfyUI-SeedVR2_VideoUpscaler",
        ],
      }
    },
    // Install ComfyUI requirements
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -r comfyui/requirements.txt",
        ],
      }
    },
    // Install ComfyUI-GGUF requirements
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install gguf>=0.13.0 sentencepiece protobuf",
        ],
      }
    },
    // Install ComfyUI-SeedVR2_VideoUpscaler requirements
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install einops omegaconf>=2.3.0 diffusers>=0.33.1 peft>=0.17.0 rotary_embedding_torch>=0.5.3 opencv-python matplotlib",
        ],
      }
    },
    // Install Gradio app requirements (includes transformers, qwen-vl-utils, accelerate for LLM prompter)
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -r requirements.txt",
        ],
      }
    },
    // Install PyTorch (handles CUDA/ROCm/CPU automatically)
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app",
        }
      }
    },
    
    // Link models to Pinokio shared drive (cross-app model sharing)
    {
      method: "fs.link",
      params: {
        drive: {
          "checkpoints": "app/comfyui/models/checkpoints",
          "clip": "app/comfyui/models/clip",
          "clip_vision": "app/comfyui/models/clip_vision",
          "configs": "app/comfyui/models/configs",
          "controlnet": "app/comfyui/models/controlnet",
          "diffusers": "app/comfyui/models/diffusers",
          "diffusion_models": "app/comfyui/models/diffusion_models",
          "embeddings": "app/comfyui/models/embeddings",
          "gligen": "app/comfyui/models/gligen",
          "hypernetworks": "app/comfyui/models/hypernetworks",
          "ipadapter": "app/comfyui/models/ipadapter",
          "loras": "app/comfyui/models/loras",
          "photomaker": "app/comfyui/models/photomaker",
          "style_models": "app/comfyui/models/style_models",
          "text_encoders": "app/comfyui/models/text_encoders",
          "unet": "app/comfyui/models/unet",
          "upscale_models": "app/comfyui/models/upscale_models",
          "vae": "app/comfyui/models/vae",
          "vae_approx": "app/comfyui/models/VAE-approx",
        },
        peers: [
          "https://github.com/cocktailpeanut/fluxgym.git",
          "https://github.com/cocktailpeanutlabs/automatic1111.git",
          "https://github.com/cocktailpeanutlabs/fooocus.git",
          "https://github.com/cocktailpeanutlabs/comfyui.git",
          "https://github.com/pinokiofactory/comfy.git",
          "https://github.com/pinokiofactory/stable-diffusion-webui-forge.git",
        ]
      }
    },
    // Link outputs folder
    {
      method: "fs.link",
      params: {
        drive: {
          "output": "app/comfyui/output"
        }
      }
    },
  ]
}
