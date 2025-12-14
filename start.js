module.exports = {
  daemon: true,
  run: [
    // Start ComfyUI backend first
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          PYTORCH_ENABLE_MPS_FALLBACK: "1",
          TOKENIZERS_PARALLELISM: "false"
        },
        path: "app",
        message: [
          "python comfyui/main.py {{platform === 'win32' && gpu === 'amd' ? '--directml' : args.sage ? '--use-sage-attention' : args.flash ? '--use-flash-attention' : ''}}"          
        ],
        on: [{
          // Wait for ComfyUI to be ready
          event: "/To see the GUI go to:\\s+(http:\\/\\/\\S+)/",
          done: true
        }, {
          event: "/errno/i",
          break: false
        }, {
          event: "/error:/i",
          break: false
        }]
      }
    },
    // Start Gradio app (connects to ComfyUI backend via ComfyKit default URL)
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "python app.py {{port}}"
        ],
        on: [{
          // Match Gradio's specific "Running on local URL:" message
          event: "/Running on local URL:\\s+(http:\\/\\/[0-9.:]+)/",
          done: true
        }]
      }
    },
    // Set the Gradio URL for the "Open Web UI" button
    {
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"
      }
    }
  ]
}
