import torch
import os
from PIL import Image
import numpy as np
import gc
from transformers import AutoProcessor
import comfy.utils
import comfy.sd
import folder_paths

from .longcat.pipeline_longcat_image import LongCatImagePipeline
from .longcat.pipeline_longcat_image_edit import LongCatImageEditPipeline
from .longcat.transformer_longcat_image import LongCatImageTransformer2DModel
from comfy.comfy_types.node_typing import IO
import re
from diffusers.utils import (
    convert_unet_state_dict_to_peft,
)
from peft.utils import set_peft_model_state_dict
from safetensors.torch import save_file
from diffusers import DiffusionPipeline
import inspect

global_transformer = None

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.autoencoders import AutoencoderKL
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor
from .longcat.transformer_longcat_image import LongCatImageTransformer2DModel


# Map pipeline names to their classes with their available components
pipeline_registry = {
    'LongCatImagePipeline': {
        'class': LongCatImagePipeline,
        'components': {
            'scheduler': FlowMatchEulerDiscreteScheduler,
            'vae': AutoencoderKL, 
            'text_encoder': Qwen2_5_VLForConditionalGeneration,
            'tokenizer': Qwen2Tokenizer,
            'text_processor': Qwen2VLProcessor,
            'transformer': LongCatImageTransformer2DModel
        }
    },
    'LongCatImageEditPipeline': {
        'class': LongCatImageEditPipeline,
        'components': {
            'scheduler': FlowMatchEulerDiscreteScheduler,
            'vae': AutoencoderKL,
            'text_encoder': Qwen2_5_VLForConditionalGeneration,
            'tokenizer': Qwen2Tokenizer,
            'text_processor': Qwen2VLProcessor,
            'transformer': LongCatImageTransformer2DModel
        }
    }
}

EDIT_PIPELINES = [
    LongCatImageEditPipeline
]

def swap_components(piepeline, available_components):
    # remove all components
    for component in available_components.keys():
        if hasattr(piepeline, component):
            c = getattr(piepeline, component)
            if c is not None and hasattr(c, "to"):
                c = c.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

class DiffusersTextEncode:
    """
    Abstract node to encode prompts for various pipeline types.

    This node can handle both text-only and text+image encoding depending on the pipeline type.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("PIPELINE",),
                "prompt": ("STRING", {"multiline": True, "default": "Masterpiece, best quality, 8k uhd, photo realistic,"})
            },
            "optional": {
                "image": ("IMAGE",),  # Optional image input for pipelines that support text+image encoding
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64})
            }
        }

    RETURN_TYPES = ("DIFFUSERS_COND", )  # positive, negative
    RETURN_NAMES = ("diffusers_cond", )
    FUNCTION = "encode_prompt"
    CATEGORY = "Diffusers/Encode"

    def encode_prompt(self, pipeline, prompt, image=None, batch_size=1):
        """
        Encodes the prompt using the provided pipeline.

        Args:
            pipeline: The loaded pipeline object
            prompt: Text prompt to encode
            negative_prompt: Negative text prompt to encode
            image: Optional image for pipelines that support text+image encoding
            batch_size: Number of images to generate

        Returns:
            tuple: (positive_conditioning, negative_conditioning)
        """
        # Determine pipeline type and use appropriate encoding method
        if hasattr(pipeline, 'encode_prompt'):
            if image is not None:
                # Already a PIL image
                encoded_output = pipeline.encode_prompt(
                    prompt=[prompt],
                    image=image,
                )
            else:
                # Handle text-only encoding
                encoded_output = pipeline.encode_prompt([prompt])

            # Pass the output directly without assuming its structure
            cond = {
                "encoded_output": encoded_output,
                "pipeline_type": type(pipeline).__name__,
                "batch_size": batch_size,
                "prompt": prompt,
            }

        else:
            # Fallback for pipelines that don't have encode_prompt method
            raise NotImplementedError(f"The pipeline {type(pipeline)} does not have an encode_prompt method implemented")

        return (cond, )


class DiffusersPipeline:
    """
    A flexible node that can initialize different LongCat diffusion pipelines with dynamic components.

    This node accepts a pipeline class name and specifies which components to load,
    allowing for flexible initialization of various LongCat models.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline_class_name": (["LongCatImagePipeline", "LongCatImageEditPipeline"], {"default": "LongCatImagePipeline"}),
                "model_path": ("STRING", {"default": ""}),  # Updated default for LongCat
            },
            "optional": {
                "any": (IO.ANY, {}),
                "pipeline": ("PIPELINE",),
                "components": ("STRING", {"default": ""}),  # Comma-separated list of components to load
                "presets": (["text_encoder, tokenizer, text_processor", "scheduler, vae, text_processor, transformer"], {"default": "text_encoder, tokenizer, text_processor"}),
                "torch_dtype": (["float16", "float32", "bfloat16"], {"default": "bfloat16"}),  # Updated default for LongCat
                # "device_map": ("STRING", {"default": "auto"}),  # e.g., "auto", "balanced", "balanced_low_0", "sequential"
                # "variant": ("STRING", {"default": "None"}),  # e.g., "fp16" for models with fp16 weights
                # "low_cpu_mem_usage": ("BOOLEAN", {"default": True}),
                # "use_safetensors": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load_pipeline"
    CATEGORY = "Diffusers/LongCat"

    def load_pipeline(self, pipeline_class_name, model_path, 
                      any=None,
                      pipeline=None, 
                      torch_dtype="bfloat16",
                    #   device_map="auto",
                    #   variant="None",
                    #   low_cpu_mem_usage=True,
                    #   use_safetensors=True,
                      components="",
                      presets=""):
        if components == "":
            components = presets
        
        # Get the pipeline class and component info
        if pipeline_class_name not in pipeline_registry:
            raise ValueError(f"Pipeline class '{pipeline_class_name}' not found in LongCat pipeline registry")

        pipeline_info = pipeline_registry[pipeline_class_name]
        pipeline_class = pipeline_info['class']
        available_components = pipeline_info['components']

        # swap components to cpu
        if pipeline is not None:
            swap_components(pipeline, available_components)

        # Convert dtype string to actual dtype
        dtype_mapping = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16
        }
        torch_dtype = dtype_mapping.get(torch_dtype, torch.bfloat16)  # Updated default for LongCat

        # Parse components string - comma separated values
        if components and components.lower() != "none" and components.strip():
            component_list = [comp.strip().lower() for comp in components.split(",") if comp.strip()]
        else:
            component_list = []  # Default components

        # Prepare kwargs for pipeline initialization
        kwargs = {
            "torch_dtype": torch_dtype,
            # "device_map": "auto"
            # "low_cpu_mem_usage": low_cpu_mem_usage,
            # "use_safetensors": use_safetensors
        }

        # if device_map and device_map.lower() != "none":
        #     kwargs["device_map"] = device_map

        # if variant and variant.lower() != "none":
        #     kwargs["variant"] = variant

        # Initialize components dictionary with all available components, but set to None if not requested
        components_dict = {}

        # Iterate over available components and set them based on whether they were requested
        for comp_name, comp_class in available_components.items():
            if comp_name in component_list:
                # Load the requested component
                if comp_name == 'transformer':
                    components_dict[comp_name] = comp_class.from_pretrained(
                        model_path, subfolder=comp_name, **kwargs
                    )
                elif comp_name in ['vae', 'text_encoder']:
                    components_dict[comp_name] = comp_class.from_pretrained(
                        model_path, subfolder=comp_name, **kwargs
                    )
                elif comp_name == 'tokenizer':
                    components_dict[comp_name] = comp_class.from_pretrained(
                        model_path, subfolder=comp_name
                    )
                elif comp_name == 'text_processor':
                    components_dict[comp_name] = comp_class.from_pretrained(
                        model_path, subfolder=comp_name
                    )
                elif comp_name == 'scheduler':
                    components_dict[comp_name] = comp_class.from_pretrained(
                        model_path, subfolder=comp_name
                    )
            else:
                # Set to None if not in requested components
                components_dict[comp_name] = None

        if pipeline is None:
            pipeline = pipeline_class(**components_dict)
        else:
            # Update pipeline with loaded components
            for comp_name, comp_class in available_components.items():
                if comp_name in component_list:
                    if hasattr(pipeline, comp_name):
                        setattr(pipeline, comp_name, components_dict[comp_name])

        # Set attributes to track which components were actually loaded (not None)
        loaded_components = [name for name, comp in components_dict.items() if comp is not None]
        pipeline._loaded_components = loaded_components
        pipeline._requested_components = component_list

        # Move to appropriate device based on device_map
        # if device_map and device_map.lower() == "auto":
        #     # Let accelerate handle the device mapping
        #     pass
        # else:
        #     # Move to specific device if not using auto mapping
        # if hasattr(pipeline, 'to'):
        #     pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

        pipeline.enable_model_cpu_offload()
        return (pipeline,)

def get_prompt_embeds(diffusers_cond):
    # Extract conditioning information from diffusers_cond
    # The encode_prompt method returns an encoded_output that could be various formats
    encoded_output = diffusers_cond["encoded_output"]

    # Extract the actual prompt embeddings from the encoded output
    # Based on how encode_prompt is typically implemented, it might return different structures
    if isinstance(encoded_output, tuple):
        # If it's a tuple like (prompt_embeds, text_ids), extract the first element which is prompt_embeds
        prompt_embeds = encoded_output[0] if len(encoded_output) > 0 else None
    elif isinstance(encoded_output, dict):
        # If it's a dict, try common keys used for embeddings
        if 'prompt_embeds' in encoded_output:
            prompt_embeds = encoded_output['prompt_embeds']
    elif isinstance(encoded_output, torch.Tensor):
        # If it's directly a tensor, use it
        prompt_embeds = encoded_output
    else:
        # Fallback: assume it's something we can use directly or convert to tensor
        prompt_embeds = encoded_output
    
    return prompt_embeds

# def is_edit_pipeline(pipeline):
#     # get pipeline class
#     pipeline_class = type(pipeline)
#     print("pipeline_class", pipeline_class)
#     is_edit = pipeline_class in EDIT_PIPELINES
#     print("is_edit", is_edit)
    
#     return is_edit

class DiffusersSampling:
    """
    A flexible sampling node that performs the generation loop using a pipeline and conditioning.

    This node accepts a loaded pipeline, diffusers conditioning, and sampling parameters
    to generate images using the diffusion process.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("PIPELINE",),
                "diffusers_cond": ("DIFFUSERS_COND",),
                "steps": ("INT", {"default": 26, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {
                "negative_diffusers_cond": ("DIFFUSERS_COND",),
                "num_images_per_prompt": ("INT", {"default": 1, "min": 1, "max": 8}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "image": ("IMAGE",),  # Optional image input for image editing
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),  # Optional width for regular generation
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),  # Optional height for regular generation
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "Diffusers/Sampling"

    def generate(self, pipeline, diffusers_cond, steps, cfg,
                 negative_diffusers_cond=None, num_images_per_prompt=1, seed=42, image=None, width=1024, height=1024):
        """
        Performs the generation loop using the provided pipeline and conditioning.

        Args:
            pipeline: Loaded diffusion pipeline
            diffusers_cond: Conditioning from DiffusersTextEncode node
            steps: Number of inference steps
            cfg: Guidance scale
            negative_diffusers_cond: Negative conditioning (optional)
            num_images_per_prompt: Number of images to generate per prompt
            seed: Random seed for generation
            image: Input image for image editing (optional)
            width: Output image width (for regular generation, or override for editing)
            height: Output image height (for regular generation, or override for editing)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = torch.Generator(device=device).manual_seed(seed)

        prompt_embeds = get_prompt_embeds(diffusers_cond)
        prompt_embeds = prompt_embeds.to(device)

        if negative_diffusers_cond is not None:
            negative_prompt_embeds = get_prompt_embeds(negative_diffusers_cond)
        else:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)

        # Check if this is an image editing pipeline by looking for image input
        # is_image_edit = is_edit_pipeline(pipeline)

        # Prepare kwargs for pipeline call
        pipeline_kwargs = {
            "prompt_embeds": prompt_embeds,
            "num_inference_steps": steps,
            "guidance_scale": cfg,
            "generator": generator,
            "num_images_per_prompt": num_images_per_prompt,
        }

        # Add negative prompt only if it's available
        if negative_prompt_embeds is not None:
            pipeline_kwargs["negative_prompt_embeds"] = negative_prompt_embeds

        # Handle image editing vs regular generation based on pipeline type and input
        if image is not None:
            # Get input image dimensions
            if len(image.shape) == 4:
                # [B, H, W, C] format
                _, img_height, img_width, _ = image.shape
            elif len(image.shape) == 3:
                # [H, W, C] format
                img_height, img_width, _ = image.shape
            else:
                raise ValueError(f"Unexpected image tensor shape: {image.shape}")

            # Convert image tensor to PIL for image editing
            # Assuming image is in [B, H, W, C] format with values in [0, 1]
            if len(image.shape) == 4:
                # Get first image in batch if multiple are provided
                image_tensor = image[0] if image.shape[0] > 1 else image.squeeze(0)
            else:
                image_tensor = image

            # Convert from [0,1] to [0,255] and to numpy
            image_np = (image_tensor * 255).byte().numpy()
            image_pil = Image.fromarray(image_np.astype(np.uint8))

            # Use input image dimensions for editing, or use custom dimensions if provided
            if width == 1024 and height == 1024:  # Default values, meaning use input image size
                # Use the input image's actual dimensions, maintaining its exact aspect ratio
                target_w, target_h = img_width, img_height
            else:
                # User specified custom dimensions - maintain the input image's aspect ratio
                # but scale to fit within the specified bounds
                original_ratio = img_width / img_height
                target_ratio = width / height

                if original_ratio > target_ratio:
                    # Input image is wider than target - fit to target width
                    target_w = width
                    target_h = int(width / original_ratio)
                else:
                    # Input image is taller than target - fit to target height
                    target_h = height
                    target_w = int(height * original_ratio)

            # Resize the image to the calculated dimensions that maintain aspect ratio
            image_pil = image_pil.resize((target_w, target_h), Image.LANCZOS)

            # Add image for image editing pipeline
            # The LongCat pipeline expects a list of PIL images for the 'image' parameter
            pipeline_kwargs["image"] = [image_pil]
            result = pipeline(**pipeline_kwargs)
        else:
            # For regular generation, use the provided width and height
            pipeline_kwargs["height"] = height
            pipeline_kwargs["width"] = width
            result = pipeline(**pipeline_kwargs)

        # Convert PIL images to tensors in ComfyUI format
        images = []
        for img in result.images:
            # Convert PIL image to numpy array in [H, W, C] format with values in [0, 1]
            img_array = np.array(img).astype(np.float32) / 255.0
            # Convert to torch tensor with shape [H, W, C]
            img_tensor = torch.from_numpy(img_array)
            # Ensure it's in the right format [H, W, C]
            if img_tensor.dim() == 3:
                # Add batch dimension if not already present
                img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension at index 0 to make [B, H, W, C]
            images.append(img_tensor)

        # Stack images if there are multiple
        if len(images) > 1:
            final_image = torch.cat(images, dim=0)  # Concatenate along batch dimension
        else:
            final_image = images[0]

        return (final_image,)


def get_lora_dir_filename(lora_path):
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA file does not exist: {lora_path}")

    # Extract directory and filename from the full path
    lora_dir = os.path.dirname(lora_path)
    lora_filename = os.path.basename(lora_path)

    # Get the filename without extension for adapter_name
    lora_name_without_ext = os.path.splitext(lora_filename)[0]
    lora_filename_with_ext = lora_filename
    # Check if the original lora_filename ends with .safetensors
    if not lora_filename.lower().endswith('.safetensors'):
        # If it doesn't have the extension, add it for the full path check
        lora_filename_with_ext = lora_filename + '.safetensors'

    # Construct full path again to ensure correct format for checking
    full_lora_path = os.path.join(lora_dir, lora_filename_with_ext)
    if not os.path.exists(full_lora_path):
        raise FileNotFoundError(f"LoRA file does not exist: {full_lora_path}")

    
    return (lora_dir, lora_filename_with_ext, lora_name_without_ext)


class DiffusersLoadLoraOnly:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("LORA",)
    RETURN_NAMES = ("lora",)
    FUNCTION = "load_lora"

    CATEGORY = "Diffusers/Lora"
    DESCRIPTION = "Load a LoRA file without applying it to any models. Use with MergeLoraToModel or other nodes to apply the LoRA later."

    def load_lora(self, lora_path):
        # lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        return (lora,)


class DiffusersLoraLayersOperation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora": ("LORA", {"tooltip": "The LoRA state dictionary to modify."}),
                "layer_pattern": ("STRING", {"default": ".*transformer_blocks\\.(\\d+)\\.", "multiline": False, "tooltip": "Regex pattern to match layer names. Use groups to extract layer indices."}),
                "layer_indices": ("STRING", {"default": "59", "multiline": False, "tooltip": "Comma-separated list of layer indices to operate on, with support for ranges (e.g., '59', '10,11,12', or '50-53')."}),
                "scale_factor": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "Scale factor to apply. Use 0 to zero out layers."}),
            }
        }

    RETURN_TYPES = ("LORA",)
    RETURN_NAMES = ("modified_lora",)
    FUNCTION = "modify_lora"

    CATEGORY = "Diffusers/Lora"
    DESCRIPTION = "Modify specific layers in a LoRA by zeroing them out (when scale=0) or scaling them (otherwise) based on pattern matching."

    def modify_lora(self, lora, layer_pattern, layer_indices, scale_factor):
        # Parse layer indices from string, supporting ranges like "50-53" and comma-separated list
        def parse_indices(indices_str):
            indices = []
            if indices_str.strip() == "":
                return []
            parts = indices_str.split(",")
            for part in parts:
                part = part.strip()
                
                # skip empty parts
                if not part:
                    continue
                
                if "-" in part:
                    # Handle range notation like "50-53"
                    try:
                        start, end = part.split("-")
                        start_idx = int(start.strip())
                        end_idx = int(end.strip())
                        if start_idx > end_idx:
                            raise ValueError(f"Invalid range: {part} (start index greater than end index)")
                        indices.extend(range(start_idx, end_idx + 1))
                    except ValueError:
                        raise ValueError(f"Invalid range format: {part}. Use format like '10-15'.")
                else:
                    # Handle single integer
                    try:
                        indices.append(int(part))
                    except ValueError:
                        raise ValueError(f"Invalid layer index: {part}")
            return indices
        
        try:
            layer_indices_list = parse_indices(layer_indices)
        except ValueError as e:
            raise ValueError(f"Error parsing layer indices: {str(e)}")
        
        layer_set = set(layer_indices_list)
        modified_keys = []
        
        # Compile the pattern
        pattern = re.compile(layer_pattern)
        
        # Work on a copy of the state dict to avoid modifying original
        modified_lora = {}
        for key, value in lora.items():
            modified_lora[key] = value.clone()  # Clone tensors to avoid modifying original

        # check if layer index is empty, return original lora
        if not layer_indices_list:
            return (modified_lora,)
        
        for key in modified_lora:
            match = pattern.search(key)
            if match:
                # Try to extract layer index - looking for first captured group
                layer_id = None
                if len(match.groups()) > 0:
                    try:
                        layer_id = int(match.group(1))
                    except ValueError:
                        # If grouping didn't extract a number, try looking for digits in the full match
                        digit_match = re.search(r'\d+', match.group(0))
                        if digit_match:
                            layer_id = int(digit_match.group())
                
                if layer_id is not None and layer_id in layer_set:
                    if scale_factor == 0:
                        # Zero out the layer when scale factor is 0
                        modified_lora[key] = torch.zeros_like(modified_lora[key])
                        modified_keys.append(key)
                    else:
                        # Scale the layer by the scale factor
                        modified_lora[key] = modified_lora[key] * scale_factor
                        modified_keys.append(key)

        print(f"Modified {len(modified_keys)} parameters in layers: {sorted(layer_set)} using pattern '{layer_pattern}'")
        return (modified_lora,)


class DiffusersSaveLora:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora": ("LORA", {"tooltip": "The modified LoRA state dictionary to save."}),
                "filename": ("STRING", {"default": "my_lora.safetensors", "tooltip": "Filename to save the LoRA as (e.g. my_lora.safetensors)."}),                
            },
            "optional": {
                "output_dir": ("STRING", {"default": folder_paths.get_output_directory(), "tooltip": "Directory to save the LoRA to. Defaults to ComfyUI output directory if not provided."}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_lora"
    OUTPUT_NODE = True
    CATEGORY = "Diffusers/Lora"

    def save_lora(self, lora, filename, output_dir=None):
        if output_dir is None:
            output_dir = self.output_dir
            
        # Ensure the directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create the full path
        full_output_path = os.path.join(output_dir, filename)
        # Add .safetensors extension if not present
        if not full_output_path.lower().endswith('.safetensors'):
            full_output_path += '.safetensors'

        # Filter out layers where all values are zero
        filtered_lora = {}
        zero_layers = []
        for key, tensor in lora.items():
            if isinstance(tensor, torch.Tensor):
                if not torch.allclose(tensor, torch.zeros_like(tensor), atol=1e-12):
                    filtered_lora[key] = tensor
                else:
                    zero_layers.append(key)
            else:
                # Skip non-tensor items or optionally warn/log
                filtered_lora[key] = tensor  # or skip if undesired

        if zero_layers:
            print(f"[SaveLora] Removed {len(zero_layers)} zero-only layers: {zero_layers}")

        # Save the filtered lora state dict
        save_file(filtered_lora, full_output_path)

        print(f"LoRA saved to: {full_output_path} (original {len(lora)} layers → {len(filtered_lora)} layers)")
        return {}

class DiffusersLoraStatViewer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora": ("LORA", {"tooltip": "The loaded LoRA to analyze."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_info",)
    FUNCTION = "view_lora_stats"

    CATEGORY = "Diffusers/Lora"
    DESCRIPTION = "View information about LoRA layers to help define layer patterns for LoraLayersOperation."

    def view_lora_stats(self, lora):
        result = []
        result.append("=== LoRA Statistics ===")
        result.append(f"Total number of keys: {len(lora.keys())}")
        
        # Group keys by pattern
        key_types = {}
        for key in lora.keys():
            # Extract the layer type (like 'lora.down.weight', 'lora.up.weight', etc.)
            # Common pattern: prefix.layer_type
            key_parts = key.split('.')
            if len(key_parts) >= 3:
                layer_type = '.'.join(key_parts[-3:])  # Get last 3 parts as layer type
            else:
                layer_type = key  # If not enough parts, use the full key
            
            if layer_type not in key_types:
                key_types[layer_type] = []
            key_types[layer_type].append(key)
        
        result.append("\nLayer types found in LoRA:")
        for layer_type, keys in key_types.items():
            result.append(f"  {layer_type}: {len(keys)} keys")

        # Show some example keys to help with pattern matching
        result.append("\nFirst 10 keys (helpful for pattern creation):")
        for i, key in enumerate(list(lora.keys())[:10]):
            result.append(f"  [{i}] {key}")
        
        # Look for transformer blocks if present
        transformer_keys = [key for key in lora.keys() if 'transformer_blocks' in key]
        if transformer_keys:
            result.append(f"\nFound {len(transformer_keys)} transformer block related keys")
            unique_blocks = set()
            for key in transformer_keys:
                # Find pattern like transformer_blocks.59. or similar
                matches = re.findall(r'transformer_blocks\.(\d+)', key)
                for match in matches:
                    unique_blocks.add(int(match))
            
            if unique_blocks:
                sorted_blocks = sorted(list(unique_blocks))
                result.append(f"Transformer block indices present: {sorted_blocks[:10]}{'...' if len(sorted_blocks) > 10 else ''}")
                if len(sorted_blocks) <= 10:
                    result.append(f"All transformer block indices: {sorted_blocks}")
        
        result.append("========================")
        
        # Add the full state dictionary
        result.append("\nFull LoRA State Dictionary Keys:")
        for i, key in enumerate(lora.keys()):
            result.append(f"  [{i}] {key}")
        
        # Join all results into a single string
        output_string = "\n".join(result)
        
        return (output_string,)

class DiffusersMergeLoraToPipeline:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("PIPELINE",),
                "lora": ("LORA", {"tooltip": "The loaded LoRA to apply."}),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                "adapter_name": ("STRING", {"default": "default", "multiline": False, "tooltip": "The name of the adapter to use."}),
            }
        }

    RETURN_TYPES = ("PIPELINE", )
    OUTPUT_TOOLTIPS = ("The modified pipeline.", )
    FUNCTION = "apply_lora"

    CATEGORY = "Diffusers/Lora"
    DESCRIPTION = "Apply a pre-loaded LoRA to transformer. This allows separation of loading and applying LoRAs."

    def apply_lora(self, pipeline, lora, strength, adapter_name):
        # Both model and clip are provided
        if strength == 0:
            return (pipeline, )

        keys = list(lora.keys())
        network_alphas = {}
        for k in keys:
            if "alpha" in k:
                alpha_value = lora.get(k)
                if (torch.is_tensor(alpha_value) and torch.is_floating_point(alpha_value)) or isinstance(
                    alpha_value, float
                ):
                    network_alphas[k] = lora.pop(k)
                else:
                    raise ValueError(
                        f"The alpha key ({k}) seems to be incorrect. If you think this error is unexpected, please open as issue."
                    )
        
        pipeline.transformer.load_lora_adapter(
            lora,
            network_alphas=network_alphas,
            adapter_name=adapter_name
        )

        return (pipeline, )

# Register the nodes
NODE_CLASS_MAPPINGS = {
    "DiffusersPipeline": DiffusersPipeline,
    "DiffusersTextEncode": DiffusersTextEncode,
    "DiffusersSampling": DiffusersSampling,
    "DiffusersLoadLoraOnly": DiffusersLoadLoraOnly,
    "DiffusersLoraLayersOperation": DiffusersLoraLayersOperation,
    "DiffusersSaveLora": DiffusersSaveLora,
    "DiffusersLoraStatViewer": DiffusersLoraStatViewer,
    "DiffusersMergeLoraToPipeline": DiffusersMergeLoraToPipeline,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusersPipeline": "Diffusers Pipeline Loader",
    "DiffusersTextEncode": "Diffusers Text Encode",
    "DiffusersSampling": "Diffusers Sampling",
    "DiffusersLoadLoraOnly": "Diffusers Load LoRA Only",
    "DiffusersLoraLayersOperation": "Diffusers LoRA Layers Operation",
    "DiffusersSaveLora": "Diffusers Save LoRA",
    "DiffusersLoraStatViewer": "Diffusers LoRA Stat Viewer",
    "DiffusersMergeLoraToPipeline": "Diffusers Merge LoRA to Pipeline",
}
