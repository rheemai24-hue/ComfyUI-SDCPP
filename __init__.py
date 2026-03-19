from .sdcpp_nodes import SDCppCustomFlux

NODE_CLASS_MAPPINGS = {
    "SDCppCustomFlux": SDCppCustomFlux
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDCppCustomFlux": "SD.cpp Flux (UNet/VAE/CLIP)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']