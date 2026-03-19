# ComfyUI-SDCPP

A custom node for **ComfyUI** that integrates the powerful [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) inference engine. This node brings high-performance, `gguf` quantized model support to ComfyUI, with massive optimization specifically for Apple Silicon (Metal) devices to efficiently run huge models like Flux!

## 🌟 Features

- **GGUF Support In ComfyUI**: Automatically patches ComfyUI's model loaders to recognize `.gguf` extensions. You can effortlessly load `unet`, `vae`, and `clip` files in GGUF format!
- **Apple Silicon / Metal Optimization**: 
  - Forces GGML to correctly locate the `ggml-metal.metal` shader files, preventing silent CPU fallbacks.
  - Implements `enable_mmap` and `offload_params_to_cpu` for native zero-copy streaming directly from ssd to Unified Memory. Massively reduces swapping out RAM, preventing OOMs (Out Of Memory) entirely.
- **Flash Attention**: Greatly speeds up generation and reduces VRAM/RAM load footprint.
- **ComfyUI Compatibility**: Fully integrates with ComfyUI's step progress bar, interruption triggers (Cancel button), and robustly translates the output directly into perfectly contiguous standard `IMAGE` tensors, seamlessly compatible with downstream ComfyUI PyTorch nodes.

## 📦 Installation
1. Ensure your Python environment has the `stable_diffusion_cpp` bindings installed:
   ```bash
   pip install stable-diffusion-cpp-python
   ```
2. Navigate to your ComfyUI `custom_nodes` folder and clone this repository (or copy the `ComfyUI-SDCPP` folder inside):
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/your-username/ComfyUI-SDCPP.git
   ```
3. Boot up ComfyUI!

## 🚀 Usage

Inside ComfyUI:
1. Double click the workspace (or search the node list) and add: **`SD.cpp Flux (UNet/VAE/CLIP)`**.
2. Place your downloaded `.gguf` files into the standard ComfyUI directories:
   - `ComfyUI/models/unet/`
   - `ComfyUI/models/clip/`
   - `ComfyUI/models/vae/`
   *(Node specifically ensures any `qwen` / `mistral` / `llama` named text encoders route to the `llm_path` for correct Flux handling).*

3. Select your UNet, CLIP, and VAE.
4. Set your prompt, resolution, and steps, then click **Queue Prompt**.

## 🔧 Acknowledgements
Powered by [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) and [ggml](https://github.com/ggerganov/ggml).
