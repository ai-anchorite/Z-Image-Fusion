# Z-Image Fusion

An enhanced Gradio interface for fast image generation using the Z-Image 6B turbo model, powered by a ComfyUI backend.

This project is designed for 1-click installing via [Pinokio](https://pinokio.co/)

---

## Screenshots

<table>
<tr>
<td align="center"><b>⚡ Z-Image Turbo</b></td>
<td align="center"><b>🔍 Upscale</b></td>
<td align="center"><b>⚙️ LLM Settings</b></td>
</tr>
<tr>
<td><img src="./app/assets/z-image-tab.png" width="300"/></td>
<td><img src="./app/assets/upscale-tab.png" width="300"/></td>
<td><img src="./app/assets/llm-tab.png" width="300"/></td>
</tr>
</table>

<details>
<summary>📸 Click to view full-size screenshots</summary>

### ⚡ Z-Image Turbo Tab
<img src="./assets/z-image-tab.png" width="800"/>

### 🔍 Upscale Tab
<img src="./assets/upscale-tab.png" width="800"/>

### ⚙️ LLM Settings Tab
<img src="./assets/llm-tab.png" width="800"/>

</details>

## Features

- ⚡ **Z-Image Turbo** - Fast, low-step image generation with the Z-Image 6B distilled model
- 🖼️ **Txt2Img & Img2Img** - with comfyui node parameters exposed to the UI
- 🔍 **SeedVR2 4K Upscaler** - High-quality image and video upscaling
- 🤖 **LLM Prompt Assistant** - AI-powered prompt enhancement and image description
- 🎨 **LoRA Support** - Apply style/character Z-Image LoRAs
- ⚙️ **GGUF friendly** - Quantized models for low VRAM

## License

MIT License - See [LICENSE](LICENSE) file

### Third-Party
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - GPL-3.0
- [ComfyKit](https://github.com/puke3615/ComfyKit) - MIT
- [Gradio](https://gradio.app/) - Apache-2.0

## Credits

- Built with [Gradio](https://gradio.app/) and [ComfyKit](https://puke3615.github.io/ComfyKit/)
- Powered by [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- GGUF support via [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) (Apache-2.0)
- Upscaling via [ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) (Apache-2.0)
- Video export via [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) (GPL-3.0)
- Seed variance via [SeedVarianceEnhancer](https://github.com/ChangeTheConstants/SeedVarianceEnhancer) (MIT)
- Camera prompts reference by [CameraPromptsGenerator](https://github.com/demon4932/CameraPromptsGenerator) (MIT)
