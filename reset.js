module.exports = {
  run: [
    // Remove ComfyUI directory (includes custom_nodes)
    {
      method: "fs.rm",
      params: {
        path: "app/comfyui"
      }
    },
    // Remove shared virtual environment
    {
      method: "fs.rm",
      params: {
        path: "app/env"
      }
    }
  ]
}
