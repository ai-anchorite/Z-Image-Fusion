module.exports = {
  run: [
    // Pull latest changes for launcher repository
    {
      method: "shell.run",
      params: {
        message: "git pull"
      }
    },
    // Pull latest changes for ComfyUI
    {
      method: "shell.run",
      params: {
        path: "app/comfyui",
        message: "git pull"
      }
    },
    // Pull latest changes for ComfyUI-Manager
    {
      method: "shell.run",
      params: {
        path: "app/comfyui/custom_nodes/ComfyUI-Manager",
        message: "git pull"
      }
    },
    // Pull latest changes for ComfyUI-GGUF
    {
      method: "shell.run",
      params: {
        path: "app/comfyui/custom_nodes/ComfyUI-GGUF",
        message: "git pull"
      }
    },
    // Pull latest changes for ComfyUI-SeedVR2_VideoUpscaler
    {
      method: "shell.run",
      params: {
        path: "app/comfyui/custom_nodes/ComfyUI-SeedVR2_VideoUpscaler",
        message: "git pull"
      }
    },
    // Clone ComfyUI-VideoHelperSuite if not exists (for existing installs)
    {
      method: "shell.run",
      params: {
        message: "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git app/comfyui/custom_nodes/ComfyUI-VideoHelperSuite"
      }
    },
    // Pull latest changes for ComfyUI-VideoHelperSuite
    {
      method: "shell.run",
      params: {
        path: "app/comfyui/custom_nodes/ComfyUI-VideoHelperSuite",
        message: "git pull"
      }
    },
    // Clone SeedVarianceEnhancer if not exists (for existing installs)
    {
      method: "shell.run",
      params: {
        message: "git clone https://github.com/ChangeTheConstants/SeedVarianceEnhancer.git app/comfyui/custom_nodes/SeedVarianceEnhancer"
      }
    },
    // Pull latest changes for SeedVarianceEnhancer
    {
      method: "shell.run",
      params: {
        path: "app/comfyui/custom_nodes/SeedVarianceEnhancer",
        message: "git pull"
      }
    },
    // Clone CameraPromptsGenerator if not exists (for existing installs)
    {
      method: "shell.run",
      params: {
        message: "git clone https://github.com/demon4932/CameraPromptsGenerator.git app/CameraPromptsGenerator"
      }
    },
    // Update CameraPromptsGenerator
    {
      method: "shell.run",
      params: {
        path: "app/CameraPromptsGenerator",
        message: "git pull"
      }
    },       
    // Reinstall ComfyUI requirements (in case of changes)
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: "uv pip install -r comfyui/requirements.txt",
      }
    },
    // Reinstall ComfyUI-GGUF requirements
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: "uv pip install gguf>=0.13.0 sentencepiece protobuf",
      }
    },
    // Reinstall ComfyUI-SeedVR2_VideoUpscaler requirements
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: "uv pip install einops omegaconf>=2.3.0 diffusers>=0.33.1 peft>=0.17.0 rotary_embedding_torch>=0.5.3 opencv-python matplotlib",
      }
    },
    // Reinstall ComfyUI-VideoHelperSuite requirements
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: "uv pip install opencv-python imageio-ffmpeg",
      }
    },
    // Reinstall Gradio app requirements
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: "uv pip install -r requirements.txt",
      }
    },
    // Reinstall PyTorch
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
  ]
}
