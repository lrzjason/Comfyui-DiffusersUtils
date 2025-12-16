import torch
import os
from PIL import Image
import numpy as np
import gc
from transformers import AutoProcessor
import comfy.utils
import comfy.sd
import folder_paths

from ..longcat.pipeline_longcat_image import LongCatImagePipeline
from ..longcat.pipeline_longcat_image_edit import LongCatImageEditPipeline
from ..longcat.transformer_longcat_image import LongCatImageTransformer2DModel
from comfy.comfy_types.node_typing import IO
import re
from diffusers.utils import (
    convert_unet_state_dict_to_peft,
)
from peft.utils import set_peft_model_state_dict
from safetensors.torch import save_file

global_transformer = None
def clear_pipeline(piepeline):
    # remove all components
    for component in [
        "text_encoder",
        "transformer",
        "vae",
        "tokenizer",
        "text_processor",
        "image_encoder",
        "feature_extractor",
    ]:
        if hasattr(piepeline, component):
            c = getattr(piepeline, component)
            if c is not None and hasattr(c, "to"):
                c = c.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
def get_preprocessor(model_path):
    processor = None
    processor = AutoProcessor.from_pretrained(model_path, subfolder='tokenizer')
    return processor

def get_tokenizer(model_path):
    tokenizer = None
    try:
        pipe = LongCatImagePipeline.from_pretrained(
            model_path,
            text_encoder=None,
            vae=None,
            transformer=None,
        )
        tokenizer = pipe.tokenizer
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
    except:
        raise ValueError(f"Could not load tokenizer from {model_path}")
    return tokenizer
def get_vae(model_path, torch_dtype):
    vae = None
    try:
        pipe = LongCatImagePipeline.from_pretrained(
            model_path,
            text_encoder=None,
            tokenizer=None,
            transformer=None,
            torch_dtype=torch_dtype,
        )
        vae = pipe.vae
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
    except:
        raise ValueError(f"Could not load VAE from {model_path}")
    return vae
def get_transformer(model_path, torch_dtype):
    if global_transformer is not None:
        print("Using global transformer")
        return global_transformer
    else:
        print("Renew transformer")
    
    transformer = None
    # First, try to load transformer from subfolder named "transformer"
    transformer_path = os.path.join(model_path, "transformer")
    if os.path.exists(transformer_path):
        model_path = transformer_path
    try:
        transformer = LongCatImageTransformer2DModel.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
        )
        
    except Exception as e:
        print(f"Failed to load traansformer from {model_path}: {e}")
        raise ValueError(f"Could not load transformer from {model_path}")
    return transformer
def get_text_encoder(model_path, torch_dtype):
    text_encoder = None
    # Try loading text encoder from different pipeline types
    try:
        pipe = LongCatImagePipeline.from_pretrained(
            model_path,
            transformer=None,
            vae=None,
            torch_dtype=torch_dtype,
        )
        text_encoder = pipe.text_encoder
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
    except:
        raise ValueError(f"Could not load text encoder from {model_path}")
    
    return text_encoder

# def create_components(model_path, pipeline_info):
#     text_encoder = None
#     transformer = None
#     vae = None
#     tokenizer = None
#     preprocessor = None
    
#     # load components
#     if pipeline_info["text_encoder"]:
#         text_encoder = get_text_encoder(model_path, pipeline_info["dtype"])
#     if pipeline_info["transformer"]:
#         transformer = get_text_encoder(model_path, pipeline_info["dtype"])
#     if pipeline_info["vae"]:
#         vae = get_vae(model_path, pipeline_info["dtype"])
#     if pipeline_info["tokenizer"]:
#         tokenizer = get_tokenizer(model_path)
#     if pipeline_info["preprocessor"]:
#         preprocessor = get_preprocessor(model_path)
    
#     return text_encoder, transformer, vae, tokenizer, preprocessor
        

def create_pipeline(pipeline_type, model_path, vae, pipeline, text_encoder, transformer, tokenizer, preprocessor, 
                    torch_dtype):
    if pipeline is None:
        loading_kwargs = {}
        loading_kwargs["torch_dtype"] = torch_dtype if torch_dtype is not None else torch.bfloat16
        loading_kwargs["vae"] = vae if vae is not None else None
        loading_kwargs["text_encoder"] = text_encoder if text_encoder is not None else None
        loading_kwargs["transformer"] = transformer if transformer is not None else None
        loading_kwargs["tokenizer"] = tokenizer if tokenizer is not None else None
        # longcat is called text_processor for preprocessor
        loading_kwargs["text_processor"] = preprocessor if preprocessor is not None else None
        loading_kwargs["image_encoder"] = None
        loading_kwargs["feature_extractor"] = None

        # Build pipeline based on user selection
        if pipeline_type == "image":
            # Build regular LongCat image pipeline
            try:
                pipe = LongCatImagePipeline.from_pretrained(
                    model_path,
                    **loading_kwargs
                )
                print(f"Built LongCatImagePipeline from {model_path}")
            except Exception as e:
                raise ValueError(f"Could not build LongCatImagePipeline: {e}")
        elif pipeline_type == "image_edit":
            # Build LongCat image edit pipeline
            try:
                pipe = LongCatImageEditPipeline.from_pretrained(
                    model_path,
                    **loading_kwargs
                )
                print(f"Built LongCatImageEditPipeline from {model_path}")
            except Exception as e:
                raise ValueError(f"Could not build LongCatImageEditPipeline: {e}")
        else:  # pipeline_type == "auto"
            # Try to determine pipeline type and build accordingly
            try:
                # Try building a LongCat image pipeline first
                pipe = LongCatImagePipeline.from_pretrained(
                    model_path,
                    **loading_kwargs
                )
                print(f"Auto-detected and built LongCatImagePipeline from {model_path}")
            except:
                try:
                    # Try building a LongCat image edit pipeline
                    pipe = LongCatImageEditPipeline.from_pretrained(
                        model_path,
                        **loading_kwargs
                    )
                    print(f"Auto-detected and built LongCatImageEditPipeline from {model_path}")
                except:
                    raise ValueError(f"Could not build pipeline from components - tried both image and image_edit types")
    else:
        pipe = pipeline
        setattr(pipe, "vae", vae if vae is not None else None)
        setattr(pipe, "text_encoder", text_encoder if text_encoder is not None else None)
        setattr(pipe, "transformer", transformer if transformer is not None else None)
        setattr(pipe, "tokenizer", tokenizer if tokenizer is not None else None)
        setattr(pipe, "text_processor", preprocessor if preprocessor is not None else None)
        setattr(pipe, "image_encoder", None)
        setattr(pipe, "feature_extractor", None)
    return pipe

def manage_pipeline_for_text_encoding(
                                        pipeline, 
                                        image_pil=None, 
                                        prompt="", 
                                        prompt_embeds=None,
                                        num_images_per_prompt=1
                                    ):
    with torch.no_grad():
        if hasattr(pipeline, 'encode_prompt'):
            if hasattr(pipeline, 'image_processor_vl'):  # Image edit pipeline
                # Image edit pipeline has signature: encode_prompt(image, prompts, device, dtype)
                prompt_embeds, _ = pipeline.encode_prompt(
                    image=[image_pil],  # Pass the image for image edit pipelines
                    prompt=[prompt],
                    prompt_embeds=prompt_embeds,
                    num_images_per_prompt=num_images_per_prompt
                )
            else:  # Regular pipeline
                # Regular pipeline has signature: encode_prompt(prompts, device, dtype)
                prompt_embeds, _ = pipeline.encode_prompt(
                    prompt=[prompt],
                    prompt_embeds=prompt_embeds,
                    num_images_per_prompt=num_images_per_prompt
                )
    return prompt_embeds


class DiffusersModelLoader:
    """Load a diffusers model from a directory path."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING", {"default": "", "multiline": False}),
                "dtype": (["float32", "float16", "bfloat16"], {"default": "float16"}),
            },
            "optional": {
                "load_vae": ("BOOLEAN", {"default": True}),
                "load_text_encoder": ("BOOLEAN", {"default": True}),
                "load_transformer": ("BOOLEAN", {"default": True}),
                "load_tokenizer": ("BOOLEAN", {"default": True}),
                "load_image_encoder": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("PIPELINE",)
    FUNCTION = "load_model"
    CATEGORY = "Diffusers"

    def load_model(self, model_path, dtype, load_vae=True, load_text_encoder=True,
                   load_transformer=True, load_tokenizer=True, load_image_encoder=True):
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        # Set torch dtype
        torch_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }[dtype]

        # Prepare component loading arguments
        loading_kwargs = {}

        # Only load components that are requested
        if not load_vae:
            loading_kwargs["vae"] = None
        if not load_text_encoder:
            loading_kwargs["text_encoder"] = None
        if not load_transformer:
            loading_kwargs["transformer"] = None
        if not load_tokenizer:
            loading_kwargs["tokenizer"] = None
        if not load_image_encoder:
            loading_kwargs["image_encoder"] = None

        loading_kwargs["torch_dtype"] = torch_dtype

        # Determine pipeline type based on config
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            # Try to detect pipeline type from config
            # For now, assume LongCat pipeline
            try:
                pipe = LongCatImagePipeline.from_pretrained(
                    model_path,
                    **loading_kwargs
                )
            except:
                # Try alternative pipeline
                pipe = LongCatImageEditPipeline.from_pretrained(
                    model_path,
                    **loading_kwargs
                )
        else:
            raise ValueError(f"Could not determine pipeline type for {model_path}")

        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipe = pipe.to(device)

        return (pipe,)


class DiffusersTextEncoderLoader:
    """Load only the text encoder from a diffusers model."""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING", {"default": "", "multiline": False}),
                "dtype": (["float32", "float16", "bfloat16"], {"default": "float16"}),
            }
        }
    
    RETURN_TYPES = ("TEXT_ENCODER",)
    FUNCTION = "load_text_encoder"
    CATEGORY = "Diffusers"

    def load_text_encoder(self, model_path, dtype):
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        torch_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }[dtype]
        
        text_encoder = get_text_encoder(model_path, torch_dtype)
        
        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        text_encoder = text_encoder.to(device)
        
        return (text_encoder,)


class DiffusersTransformerLoader:
    """Load only the transformer from a diffusers model."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (IO.ANY, {}),
                "model_path": ("STRING", {"default": "", "multiline": False}),
                "dtype": (["float32", "float16", "bfloat16"], {"default": "float16"}),
            }
        }

    RETURN_TYPES = ("TRANSFORMER",)
    FUNCTION = "load_transformer"
    CATEGORY = "Diffusers"

    def load_transformer(self, any, model_path, dtype):
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        torch_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }[dtype]

        global_transformer = get_transformer(model_path, torch_dtype)

        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        global_transformer = global_transformer.to(device)

        return (global_transformer,)


class DiffusersVAELoader:
    """Load only the VAE from a diffusers model."""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING", {"default": "", "multiline": False}),
                "dtype": (["float32", "float16", "bfloat16"], {"default": "float16"}),
            }
        }
    
    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "Diffusers"

    def load_vae(self, model_path, dtype):
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        torch_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }[dtype]
        
        vae = get_vae(model_path, torch_dtype)
        
        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vae = vae.to(device)
        
        return (vae,)


class DiffusersTokenizerLoader:
    """Load only the tokenizer from a diffusers model."""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING", {"default": "", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("TOKENIZER",)
    FUNCTION = "load_tokenizer"
    CATEGORY = "Diffusers"

    def load_tokenizer(self, model_path):
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        tokenizer = get_tokenizer(model_path)
        
        return (tokenizer,)


class DiffusersPreprocessorLoader:
    """Load only the preprocessor from a diffusers model."""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING", {"default": "", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("PREPROCESSOR",)
    FUNCTION = "load_preprocessor"
    CATEGORY = "Diffusers"
    def load_preprocessor(self, model_path):
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        try:
            preprocessor = get_preprocessor(model_path)
        except:
            raise ValueError(f"Could not load text processor from {model_path}")
        
        return (preprocessor,)


class DiffusersPipelineBuilder:
    """Build a pipeline from individual components."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING", {"default": "", "multiline": False}),
                "pipeline_type": (["auto", "image", "image_edit"], {"default": "auto"}),
                "dtype": (["float32", "float16", "bfloat16"], {"default": "float16"}),
            },
            "optional": {
                "pipeline": ("PIPELINE", ),
                "text_encoder": ("TEXT_ENCODER",),
                "transformer": ("TRANSFORMER",),
                "vae": ("VAE",),
                "tokenizer": ("TOKENIZER",),
                "preprocessor": ("PREPROCESSOR",),
                # "image_encoder": ("IMAGE_ENCODER",),
                # "feature_extractor": ("FEATURE_EXTRACTOR",),
                # "load_remaining_components": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("PIPELINE", 
                    # "PIPELINE_INFO"
                    )
    FUNCTION = "build_pipeline"
    CATEGORY = "Diffusers"

    def build_pipeline(self, model_path, pipeline_type, dtype, pipeline=None, text_encoder=None, transformer=None,
                       vae=None, tokenizer=None, preprocessor=None,
                    #    image_encoder=None, 
                    #    feature_extractor=None
                       ):
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        torch_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }[dtype]

        pipe = create_pipeline(pipeline_type, model_path, vae, pipeline, text_encoder, transformer, tokenizer, preprocessor, torch_dtype)
        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipe = pipe.to(device)
        
        # pipeline_info = {
        #     "model_path": model_path,
        #     "pipeline_type": pipeline_type,
        #     "dtype": dtype,
        #     "text_encoder": text_encoder is not None,
        #     "transformer": transformer is not None,
        #     "vae": vae is not None,
        #     "tokenizer": tokenizer is not None,
        #     "preprocessor": preprocessor is not None,
        #     # "image_encoder": image_encoder is not None,
        # }
        pipe.enable_model_cpu_offload()
        return (pipe, 
                # pipeline_info
                )


# Keep the existing text encode nodes for compatibility
class TextEncodeDiffusersLongCat:
    """Encode text using the pipeline's text encoder. Can optionally offload the text encoder after encoding."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("PIPELINE",),
                # "pipeline_info": ("PIPELINE_INFO",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": {
                "num_images_per_prompt": ("INT", {"default": 1}),
                # "offload_after_encode": ("BOOLEAN", {"default": False}),
                "cache_embeddings": ("BOOLEAN", {"default": False}),
                "cache_file": ("STRING", {"default": "diffusers_embedding_cache.pt"}),
                "recreate_cache": ("BOOLEAN", {"default": False}),
            }
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ("EMBEDDING", )
    FUNCTION = "encode"
    CATEGORY = "Diffusers"

    def encode(
                self, 
                pipeline, 
                prompt, 
                num_images_per_prompt=1,
                cache_embeddings=False, 
                cache_file="diffusers_embedding_cache.pt",
                recreate_cache=False
            ):
        load_cache = False
        if recreate_cache and os.path.exists(cache_file):
            os.remove(cache_file)
        # Check if embeddings are cached and if the prompt matches
        if cache_embeddings and os.path.exists(cache_file):
            print(f"Loading cached prompt embeddings from {cache_file}...")
            try:
                cached_data = torch.load(cache_file)
                # Check if the cached data includes prompt hash for verification
                if "prompt_hash" in cached_data and "prompt" in cached_data:
                    cached_prompt = cached_data["prompt"]
                    if cached_prompt == prompt:
                        print("Prompt matches cached embeddings, using cache...")
                        prompt_embeds = cached_data["prompt_embeds"]
                        # text_ids = cached_data["text_ids"]
                        load_cache = True
            except Exception as e:
                print(f"Error reading cache file, regenerating embeddings: {e}")
        
        if not load_cache:
            print("Prompt changed, regenerating embeddings...")
            # Use the helper method to manage pipeline components for text encoding
            
            prompt_embeds = manage_pipeline_for_text_encoding(
                pipeline, 
                prompt=prompt,
                prompt_embeds=prompt_embeds,
                num_images_per_prompt=num_images_per_prompt
            )

            if cache_embeddings:
                # Update cache with new prompt
                cache_data = {
                    "prompt_embeds": prompt_embeds,
                    # "text_ids": text_ids,
                    "prompt": prompt,  # Store the prompt for verification
                    "prompt_hash": hash(prompt)  # Also store hash as additional verification
                }
                torch.save(cache_data, cache_file)
                print(f"New prompt embeddings cached to {cache_file}")
                
        clear_pipeline(pipeline)
        return (prompt_embeds, )


class TextEncodeDiffusersLongCatCached:
    """Load cached prompt embeddings from a file."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cache_file": ("STRING", {"default": "diffusers_embedding_cache.pt"}),
            }
        }

    RETURN_TYPES = ("EMBEDDING", "STRING", )  # Added STRING output for the cached prompt
    FUNCTION = "load_cached_embeddings"
    CATEGORY = "Diffusers"

    def load_cached_embeddings(self, cache_file):
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Cache file does not exist: {cache_file}")

        print(f"Loading cached prompt embeddings from {cache_file}...")
        try:
            cached_data = torch.load(cache_file)
            prompt_embeds = cached_data["prompt_embeds"]
            # text_ids = cached_data["text_ids"] if "text_ids" in cached_data else None

            # Get the cached prompt if available for verification
            cached_prompt = cached_data.get("prompt", "Prompt not available in cache")

        except Exception as e:
            raise Exception(f"Error loading cache file: {e}")

        return (prompt_embeds, cached_prompt)


class TextEncodeDiffusersLongCatImageEdit:
    """Encode text and image for image edit pipelines."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("PIPELINE",),
                # "pipeline_info": ("PIPELINE_INFO",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "image": ("IMAGE",),
            },
            "optional": {
                "num_images_per_prompt": ("INT", {"default": 1}),
                "resolution": ("INT", {"default": 512, "min": 256, "max": 2048}),
                # "offload_after_encode": ("BOOLEAN", {"default": False}),
                "cache_embeddings": ("BOOLEAN", {"default": False}),
                "cache_file": ("STRING", {"default": "diffusers_image_edit_embedding_cache.pt"}),
                "recreate_cache": ("BOOLEAN", {"default": False}),
            }
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ("EMBEDDING", "IMAGE", )
    FUNCTION = "encode"
    CATEGORY = "Diffusers"

    def encode(
                self, 
                pipeline, 
                prompt, 
                image, 
                num_images_per_prompt=1,
                resolution=512, 
                cache_embeddings=False, 
                cache_file="diffusers_image_edit_embedding_cache.pt",
                recreate_cache=False
               ):

        if recreate_cache and os.path.exists(cache_file):
            os.remove(cache_file)
        # Ensure we have batch dimension
        if len(image.shape) == 3:  # [H, W, C]
            image_batch = image.unsqueeze(0)  # Add batch dimension -> [B, H, W, C]
        elif len(image.shape) == 4:  # [B, H, W, C]
            image_batch = image
        else:
            raise ValueError(f"Unexpected image tensor shape: {image.shape}")

        # Process first image in batch for the pipeline (which expects single image)
        image_to_process = image_batch[0]  # Get first image [H, W, C]

        # Convert from [0,1] to [0,255] and to numpy
        image_np = (image_to_process * 255).byte().numpy()
        image_pil = Image.fromarray(image_np.astype(np.uint8))

        # Resize image to resolution
        image_pil = image_pil.resize((resolution, resolution), Image.LANCZOS)
        
        load_cache = False
        # Check if embeddings are cached and if the prompt matches
        if cache_embeddings and os.path.exists(cache_file):
            print(f"Loading cached prompt embeddings from {cache_file}...")
            try:
                cached_data = torch.load(cache_file)
                # Check if the cached data includes prompt hash for verification
                if "prompt_hash" in cached_data and "prompt" in cached_data:
                    cached_prompt = cached_data["prompt"]
                    # Also check if image shape matches
                    cached_image_shape = cached_data.get("image_shape", None)
                    image_shape_matches = cached_image_shape == tuple(image.shape) if cached_image_shape else True

                    if cached_prompt == prompt and image_shape_matches:
                        print("Prompt and image shape match cached embeddings, using cache...")
                        prompt_embeds = cached_data["prompt_embeds"]
                        # text_ids = cached_data["text_ids"]
                        load_cache = True
            except Exception as e:
                print(f"Error reading cache file, regenerating embeddings: {e}")
        
        if not load_cache:
            # If there's an error reading the cache, regenerate embeddings
            # Use the helper method to manage pipeline components for text encoding
            prompt_embeds = manage_pipeline_for_text_encoding(
                pipeline, 
                image_pil=image_pil, 
                prompt=prompt, 
                prompt_embeds=None,
                num_images_per_prompt=num_images_per_prompt
            )

            # Cache the embeddings if requested
            if cache_embeddings:
                cache_data = {
                    "prompt_embeds": prompt_embeds,
                    # "text_ids": text_ids,
                    "prompt": prompt,  # Store the prompt for verification
                    "prompt_hash": hash(prompt),  # Also store hash as additional verification
                    "image_shape": tuple(image.shape)  # Include image shape to detect image changes
                }
                torch.save(cache_data, cache_file)
                print(f"Prompt embeddings cached to {cache_file}")
            
        # Return the original image in proper ComfyUI format [B, H, W, C]
        # The processed image_tensor is only used for the pipeline encoding
        # Return the original input image, but ensure it's in the proper [B, H, W, C] format
        if len(image.shape) == 3:  # [H, W, C] - add batch dimension
            output_image = image.unsqueeze(0)
        elif len(image.shape) == 4:  # [B, H, W, C] - already correct format
            output_image = image
        else:
            raise ValueError(f"Unexpected original image tensor shape: {image.shape}")

        # Ensure image values are in the expected [0, 1] range
        if output_image.max() > 1.0:
            output_image = output_image / 255.0 if output_image.max() > 255.0 else output_image

        # detach
        # prompt_embeds = prompt_embeds.detach()
        # text_ids = text_ids.detach()
        
        clear_pipeline(pipeline)
        return (prompt_embeds, output_image)


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


class LoadLoraOnly:
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

    CATEGORY = "LoraUtils"
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


class LoraLayersOperation:
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

    CATEGORY = "LoraUtils"
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


class SaveLora:
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
    CATEGORY = "LoraUtils"

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

class LoraStatViewer:
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

    CATEGORY = "LoraUtils"
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

class MergeLoraToTransformer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "transformer": ("TRANSFORMER", {"tooltip": "The transformer model to apply the LoRA to."}),
                "lora": ("LORA", {"tooltip": "The loaded LoRA to apply."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                "adapter_name": ("STRING", {"default": "default", "multiline": False, "tooltip": "The name of the adapter to use."}),
            }
        }

    RETURN_TYPES = ("TRANSFORMER", )
    OUTPUT_TOOLTIPS = ("The modified transformer model.", )
    FUNCTION = "apply_lora"

    CATEGORY = "LoraUtils"
    DESCRIPTION = "Apply a pre-loaded LoRA to transformer. This allows separation of loading and applying LoRAs."

    def apply_lora(self, transformer, lora, strength_model, adapter_name):
        # Both model and clip are provided
        if strength_model == 0:
            return (transformer, )

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
        
        transformer.load_lora_adapter(
            lora,
            network_alphas=network_alphas,
            adapter_name=adapter_name
        )

        return (transformer, )

class DiffusersLoraLoader:
    """Load and apply LoRA weights to a pipeline."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("PIPELINE",),
                "lora_path": ("STRING", {"default": "", "multiline": False}),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("PIPELINE",)
    FUNCTION = "load_lora"
    CATEGORY = "Diffusers"

    def load_lora(self, pipeline, lora_path, strength=1.0):
        lora_dir, lora_filename_with_ext, lora_name_without_ext = get_lora_dir_filename(lora_path)

        # Load LoRA weights into the pipeline
        try:
            pipeline.load_lora_weights(
                lora_dir,
                prefix=None,
                weight_name=lora_filename_with_ext,  # Use filename with extension for weight_name
                adapter_name=lora_name_without_ext  # Use filename without extension for adapter name
            )

            # Set the adapter with the given strength
            pipeline.set_adapters([lora_name_without_ext], adapter_weights=[strength])
        except Exception as e:
            print(f"LoRA loading failed even with prefix=None: {e2}")
            raise e  # Re-raise original exception

        return (pipeline,)


class DiffusersLoraUnloader:
    """Unload LoRA weights from a pipeline."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("PIPELINE",),
            }
        }

    RETURN_TYPES = ("PIPELINE",)
    FUNCTION = "unload_lora"
    CATEGORY = "Diffusers"

    def unload_lora(self, pipeline):
        if hasattr(pipeline, 'unload_lora_weights'):
            pipeline.unload_lora_weights()
        return (pipeline,)

def complie_transformer(pipeline):
    print(f"Compiling transformer")
    pipeline.transformer = torch.compile(pipeline.transformer, mode="max-autotune", fullgraph=True)
    print(f"Compiled transformer")

class DiffusersImageGenerator:
    """Generate images using a pipeline with embeddings and LoRAs."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("PIPELINE",),
                "prompt_embeds": ("EMBEDDING",),
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "num_inference_steps": ("INT", {"default": 26, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 10.0}),
            },
            "optional": {
                # "text_ids": ("TEXT_IDS",),
                "negative_prompt_embeds": ("EMBEDDING",),
                # "negative_text_ids": ("TEXT_IDS",),
                "num_images_per_prompt": ("INT", {"default": 1, "max": 10}),
                "seed": ("INT", {"default": 42}),
                "auto_unload_lora": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "Diffusers"

    def generate(self, pipeline, prompt_embeds, width, height, num_inference_steps,
                 guidance_scale, negative_prompt_embeds=None, 
                #  text_ids=None,
                #  negative_text_ids=None, 
                 num_images_per_prompt=1, seed=42, auto_unload_lora=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        generator = torch.Generator(device=device).manual_seed(seed)

        # Prepare text IDs if available
        kwargs = {}
        # if text_ids is not None:
        #     kwargs["text_ids"] = text_ids
        # if negative_text_ids is not None:
        #     kwargs["negative_text_ids"] = negative_text_ids

        # Handle different pipeline types
        if negative_prompt_embeds is None:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        # complie_transformer(pipeline)
        # LongCat pipeline
        result = pipeline(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=num_images_per_prompt,
            **kwargs
        )

        # Convert PIL images to tensors
        images = []
        for img in result.images:
            img_array = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            images.append(img_tensor)

        # Stack images if there are multiple
        if len(images) > 1:
            final_image = torch.cat(images, dim=0)
        else:
            final_image = images[0]

        # Auto-unload LoRA weights after generation to free memory
        if auto_unload_lora and hasattr(pipeline, 'unload_lora_weights'):
            pipeline.unload_lora_weights()

        return (final_image,)


class DiffusersImageEditGenerator:
    """Generate edited images using a pipeline with embeddings and LoRAs."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("PIPELINE",),
                "image": ("IMAGE",),
                "prompt_embeds": ("EMBEDDING",),
                # "text_ids": ("TEXT_IDS",),
                "num_inference_steps": ("INT", {"default": 26, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 10.0}),
            },
            "optional": {
                "negative_prompt_embeds": ("EMBEDDING",),
                # "negative_text_ids": ("TEXT_IDS",),
                "num_images_per_prompt": ("INT", {"default": 1, "max": 10}),
                "seed": ("INT", {"default": 42}),
                "auto_unload_lora": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "Diffusers"

    def generate(self, pipeline, image, prompt_embeds, 
                #  text_ids, 
                 num_inference_steps,
                 guidance_scale, negative_prompt_embeds=None, 
                #  negative_text_ids=None, 
                 num_images_per_prompt=1, seed=42, auto_unload_lora=True):
        print("prompt_embeds", prompt_embeds)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        generator = torch.Generator(device=device).manual_seed(seed)

        # Convert image tensor to PIL
        image_np = image.squeeze().numpy()
        image_pil = Image.fromarray((image_np * 255).astype(np.uint8))

        # Prepare text IDs if available
        kwargs = {}
        # if text_ids is not None:
        #     kwargs["text_ids"] = text_ids
        # if negative_text_ids is not None:
        #     kwargs["negative_text_ids"] = negative_text_ids

        # Handle image editing pipeline - check for specific attributes to identify image edit pipeline
        # complie_transformer(pipeline)
        # LongCat pipeline
        # LongCat image edit pipeline - accepts image parameter
        
        if negative_prompt_embeds is None:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        result = pipeline(
            image=image_pil,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=num_images_per_prompt,
            **kwargs
        )

        # Convert PIL images to tensors
        images = []
        for img in result.images:
            img_array = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            images.append(img_tensor)

        # Stack images if there are multiple
        if len(images) > 1:
            final_image = torch.cat(images, dim=0)
        else:
            final_image = images[0]

        # Auto-unload LoRA weights after generation to free memory
        if auto_unload_lora and hasattr(pipeline, 'unload_lora_weights'):
            pipeline.unload_lora_weights()

        return (final_image,)


NODE_CLASS_MAPPINGS = {
    "LoadLoraOnly": LoadLoraOnly,
    "LoraLayersOperation": LoraLayersOperation,
    "MergeLoraToTransformer": MergeLoraToTransformer,
    "LoraStatViewer": LoraStatViewer,
    "SaveLora": SaveLora,
    
    "DiffusersModelLoader": DiffusersModelLoader,
    "DiffusersTextEncoderLoader": DiffusersTextEncoderLoader,
    "DiffusersTransformerLoader": DiffusersTransformerLoader,
    "DiffusersVAELoader": DiffusersVAELoader,
    "DiffusersTokenizerLoader": DiffusersTokenizerLoader,
    "DiffusersPreprocessorLoader": DiffusersPreprocessorLoader,
    "DiffusersPipelineBuilder": DiffusersPipelineBuilder,
    "DiffusersLoraLoader": DiffusersLoraLoader,
    "DiffusersLoraUnloader": DiffusersLoraUnloader,
    "TextEncodeDiffusersLongCat": TextEncodeDiffusersLongCat,
    "TextEncodeDiffusersLongCatCached": TextEncodeDiffusersLongCatCached,
    "TextEncodeDiffusersLongCatImageEdit": TextEncodeDiffusersLongCatImageEdit,
    "DiffusersImageGenerator": DiffusersImageGenerator,
    "DiffusersImageEditGenerator": DiffusersImageEditGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadLoraOnly": "Load LoRA Only",
    "LoraLayersOperation": "LoRA Layers Operation",
    "MergeLoraToTransformer": "Merge LoRA to Transformer",
    "LoraStatViewer": "LoRA Stat Viewer",
    "SaveLora": "Save LoRA",
    
    "DiffusersModelLoader": "Load Diffusers Model",
    "DiffusersTextEncoderLoader": "Load Diffusers Text Encoder",
    "DiffusersTransformerLoader": "Load Diffusers Transformer",
    "DiffusersVAELoader": "Load Diffusers VAE",
    "DiffusersTokenizerLoader": "Load Diffusers Tokenizer",
    "DiffusersPreprocessorLoader": "Load Diffusers Preprocessor",
    "DiffusersPipelineBuilder": "Build Diffusers Pipeline from Components",
    "DiffusersLoraLoader": "Load Diffusers LoRA",
    "DiffusersLoraUnloader": "Unload Diffusers LoRA",
    "TextEncodeDiffusersLongCat": "Encode Prompt (Diffusers LongCat)",
    "TextEncodeDiffusersLongCatCached": "Load Cached Embeddings",
    "TextEncodeDiffusersLongCatImageEdit": "Encode Prompt + Image (Diffusers LongCat Image Edit)",
    "DiffusersImageGenerator": "Generate Image (Diffusers)",
    "DiffusersImageEditGenerator": "Edit Image (Diffusers)",
}