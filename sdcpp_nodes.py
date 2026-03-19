import torch
import numpy as np
import folder_paths
import comfy.utils
from stable_diffusion_cpp import StableDiffusion

# --- Apple Silicon Metal Fix ---
# Force GGML to look inside the stable_diffusion_cpp installation directory
# for the ggml-metal.metal shader file, otherwise it silently falls back to CPU!
import os
import stable_diffusion_cpp
metal_path = os.path.dirname(stable_diffusion_cpp.__file__)
os.environ["GGML_METAL_PATH_RESOURCES"] = metal_path
# -------------------------------

# --- THE FIX: Tell ComfyUI to look for .gguf files ---
# We inject .gguf into the allowed extensions for these specific folders
if "unet" in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["unet"][1].add(".gguf")
    folder_paths.folder_names_and_paths["unet"][1].add(".GGUF")
if "clip" in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["clip"][1].add(".gguf")
    folder_paths.folder_names_and_paths["clip"][1].add(".GGUF")
if "vae" in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["vae"][1].add(".gguf")
    folder_paths.folder_names_and_paths["vae"][1].add(".GGUF")

# Purge ComfyUI's internal model cache so it immediately rescans the folders for our injected .GGUF files
if hasattr(folder_paths, "filename_list_cache"):
    folder_paths.filename_list_cache.clear()
if hasattr(folder_paths, "cache_helper") and hasattr(folder_paths.cache_helper, "clear"):
    folder_paths.cache_helper.clear()
# -----------------------------------------------------

class SDCppCustomFlux:
    def __init__(self):
        self.model = None
        self.current_unet = ""
        self.current_vae = ""
        self.current_clip = ""

    @classmethod
    def INPUT_TYPES(cls):
        # Because we added .gguf above, these lists will now include your GGUF files!
        clip_list = folder_paths.get_filename_list("clip")
        
        # Sort so Qwen3 (or any Qwen) automatically bubbles up and becomes the default selection!
        clip_list = sorted(clip_list, key=lambda x: 0 if "qwen" in x.lower() else 1)
        
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("unet"), ),
                "vae_name": (folder_paths.get_filename_list("vae"), ),
                "clip_name": (clip_list, ),
                
                "prompt": ("STRING", {"multiline": True, "default": "A highly detailed portrait..."}),
                
                "width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 30.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "n_threads": ("INT", {"default": -1, "min": -1, "max": 64, "step": 1}), 
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "sampling/sdcpp"

    def generate(self, unet_name, vae_name, clip_name, prompt, width, height, steps, cfg_scale, seed, n_threads):
        
        # Resolve the actual file paths on your Mac
        unet_path = folder_paths.get_full_path("unet", unet_name)
        vae_path = folder_paths.get_full_path("vae", vae_name)
        clip_path = folder_paths.get_full_path("clip", clip_name)

        # Check if you changed models in the dropdown
        paths_changed = (
            self.current_unet != unet_path or
            self.current_vae != vae_path or
            self.current_clip != clip_path
        )

        # Load the models into memory
        if self.model is None or paths_changed:
            if self.model is not None:
                try:
                    # Free the C++ memory on Apple Silicon to prevent OOM
                    import gc
                    if hasattr(self.model, "close"):
                        self.model.close()
                    self.model = None
                    gc.collect()
                except Exception:
                    pass

            print(f"Loading Flux UNet from {unet_path}...")
            
            # Route the text encoder to the correct sd.cpp argument
            kwargs = {
                "diffusion_model_path": unet_path,
                "vae_path": vae_path,
                "n_threads": n_threads,
                # Apple Silicon Optimization: Massively reduce unified memory footprint
                "flash_attn": True,
                # Pure Metal Memory Optimization: Native zero-copy streaming from SSD mapping to avoid RAM duplication
                "enable_mmap": True,
                "offload_params_to_cpu": True
            }
            
            if "qwen" in clip_name.lower() or "mistral" in clip_name.lower() or "llama" in clip_name.lower():
                kwargs["llm_path"] = clip_path
            else:
                kwargs["t5xxl_path"] = clip_path
                
            self.model = StableDiffusion(**kwargs)
            
            # Cache the paths so it doesn't reload on every generation
            self.current_unet = unet_path
            self.current_vae = vae_path
            self.current_clip = clip_path

        print("Generating image on Apple Silicon...")
        
        # Initialize ComfyUI progress bar and interruption checks
        import time
        import comfy.model_management
        
        start_time = time.time()
        pbar = comfy.utils.ProgressBar(steps)
        def progress_callback(step: int, steps_total: int, ttime: float):
            # Check if user clicked Cancel correctly
            comfy.model_management.throw_exception_if_processing_interrupted()
            
            now = time.time()
            elapsed = now - start_time
            print(f"Step {step}/{steps_total} completed in {elapsed:.1f}s total (this step approx {ttime:.2f}s internally if tracked)")
            if hasattr(pbar, "update_absolute"):
                pbar.update_absolute(step, steps_total)
            else:
                pbar.update(1)
        
        # FIX: The method is generate_image, not txt2img
        result = self.model.generate_image(
            prompt=prompt,
            negative_prompt="", # Flux ignores negative prompts
            width=width,
            height=height,
            sample_steps=steps,
            cfg_scale=cfg_scale,
            seed=seed if seed != -1 else 42,
            progress_callback=progress_callback
        )
        
        # Final pass check if cancelled exactly as execution returned
        comfy.model_management.throw_exception_if_processing_interrupted()

        # Convert back to ComfyUI Image format
        pil_image = result[0].convert("RGB")
        img_array = np.array(pil_image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).contiguous()

        return (img_tensor,)