import torch
import folder_paths
import comfy.utils
import os
import hashlib
from PIL import Image
import numpy as np
import gc
from transformers import AutoProcessor
from diffusers import StableDiffusionPipeline, StableDiffusionXLImg2ImgPipeline

from .longcat.pipeline_longcat_image import LongCatImagePipeline
from .longcat.pipeline_longcat_image_edit import LongCatImageEditPipeline
from .longcat.longcat_image_dit import LongCatImageTransformer2DModel
from comfy.comfy_types.node_typing import IO

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
    except:
        try:
            pipe = LongCatImageEditPipeline.from_pretrained(
                model_path,
                text_encoder=None,
                vae=None,
                transformer=None,
            )
            tokenizer = pipe.tokenizer
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
    except:
        try:
            pipe = LongCatImageEditPipeline.from_pretrained(
                model_path,
                text_encoder=None,
                tokenizer=None,
                transformer=None,
                torch_dtype=torch_dtype,
            )
            vae = pipe.vae
        except:
            raise ValueError(f"Could not load VAE from {model_path}")
    return vae
def get_transformer(model_path, torch_dtype):
    transformer = None
    # First, try to load transformer from subfolder named "transformer"
    transformer_path = os.path.join(model_path, "transformer")
    if os.path.exists(transformer_path):
        try:
            transformer = LongCatImageTransformer2DModel.from_pretrained(
                transformer_path,
                torch_dtype=torch_dtype,
            )
            print(f"Loaded transformer from subfolder: {transformer_path}")
        except Exception as e:
            print(f"Failed to load transformer from subfolder {transformer_path}: {e}")
            raise ValueError(f"Could not load transformer from {transformer_path}")
    else:
        # If transformer subfolder doesn't exist, try loading from the main model path directly
        try:
            transformer = LongCatImageTransformer2DModel.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
            )
            print(f"Loaded transformer from main model path: {model_path}")
        except Exception as e:
            print(f"Failed to load transformer from main path {model_path}: {e}")
            raise ValueError(f"Could not load transformer from {model_path}. "
                            f"Tried both main path and 'transformer' subfolder.")
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
    except:
        try:
            pipe = LongCatImageEditPipeline.from_pretrained(
                model_path,
                transformer=None,
                vae=None,
                torch_dtype=torch_dtype,
            )
            text_encoder = pipe.text_encoder
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

def manage_pipeline_for_text_encoding(pipeline, 
                                    #   pipeline_info, 
                                      device, 
                                    #   offload_after_encode=False, 
                                      is_image_edit=False, image_pil=None, prompt=""):
    
    
    dtype = torch.bfloat16

    with torch.no_grad():
        if hasattr(pipeline, 'encode_prompt'):
            if is_image_edit and hasattr(pipeline, 'image_processor_vl'):  # Image edit pipeline
                # Image edit pipeline has signature: encode_prompt(image, prompts, device, dtype)
                prompt_embeds, text_ids = pipeline.encode_prompt(
                    [image_pil],  # Pass the image for image edit pipelines
                    [prompt],
                    device,
                    dtype
                )
            else:  # Regular pipeline
                # Regular pipeline has signature: encode_prompt(prompts, device, dtype)
                prompt_embeds, text_ids = pipeline.encode_prompt(
                    [prompt],
                    device,
                    dtype
                )
        else:
            # Standard diffusers pipeline
            tokenized = pipeline.tokenizer(
                [prompt],
                padding="max_length",
                truncation=True,
                max_length=pipeline.tokenizer.model_max_length,
                return_tensors="pt"
            ).input_ids.to(device)

            prompt_embeds = pipeline.text_encoder(tokenized)[0]
            text_ids = None  # Standard pipeline doesn't have text_ids
    
    return prompt_embeds, text_ids


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

        transformer = get_transformer(model_path, torch_dtype)

        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transformer = transformer.to(device)

        return (transformer,)


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
                # "offload_after_encode": ("BOOLEAN", {"default": False}),
                "cache_embeddings": ("BOOLEAN", {"default": False}),
                "cache_file": ("STRING", {"default": "diffusers_embedding_cache.pt"}),
            }
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ("EMBEDDING", "TEXT_IDS")
    FUNCTION = "encode"
    CATEGORY = "Diffusers"

    def encode(self, pipeline, 
            #    pipeline_info, 
               prompt, 
            #    offload_after_encode=False, 
               cache_embeddings=False, cache_file="diffusers_embedding_cache.pt"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        load_cache = False
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
                        text_ids = cached_data["text_ids"]
                        load_cache = True
            except Exception as e:
                print(f"Error reading cache file, regenerating embeddings: {e}")
        
        if not load_cache:
            print("Prompt changed, regenerating embeddings...")
            # Use the helper method to manage pipeline components for text encoding
            prompt_embeds, text_ids = manage_pipeline_for_text_encoding(
                pipeline, 
                # pipeline_info, 
                device, 
                # offload_after_encode=offload_after_encode,
                is_image_edit=False, prompt=prompt
            )

            if cache_embeddings:
                # Update cache with new prompt
                cache_data = {
                    "prompt_embeds": prompt_embeds,
                    "text_ids": text_ids,
                    "prompt": prompt,  # Store the prompt for verification
                    "prompt_hash": hash(prompt)  # Also store hash as additional verification
                }
                torch.save(cache_data, cache_file)
                print(f"New prompt embeddings cached to {cache_file}")
                
                
        # detach
        # prompt_embeds = prompt_embeds.detach()
        # text_ids = text_ids.detach()
        
        clear_pipeline(pipeline)
        return (prompt_embeds, text_ids)


class TextEncodeDiffusersLongCatCached:
    """Load cached prompt embeddings from a file."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cache_file": ("STRING", {"default": "diffusers_embedding_cache.pt"}),
            }
        }

    RETURN_TYPES = ("EMBEDDING", "TEXT_IDS", "STRING")  # Added STRING output for the cached prompt
    FUNCTION = "load_cached_embeddings"
    CATEGORY = "Diffusers"

    def load_cached_embeddings(self, cache_file):
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Cache file does not exist: {cache_file}")

        print(f"Loading cached prompt embeddings from {cache_file}...")
        try:
            cached_data = torch.load(cache_file)
            prompt_embeds = cached_data["prompt_embeds"]
            text_ids = cached_data["text_ids"] if "text_ids" in cached_data else None

            # Get the cached prompt if available for verification
            cached_prompt = cached_data.get("prompt", "Prompt not available in cache")

        except Exception as e:
            raise Exception(f"Error loading cache file: {e}")

        return (prompt_embeds, text_ids, cached_prompt)


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
                "resolution": ("INT", {"default": 512, "min": 256, "max": 2048}),
                # "offload_after_encode": ("BOOLEAN", {"default": False}),
                "cache_embeddings": ("BOOLEAN", {"default": False}),
                "cache_file": ("STRING", {"default": "diffusers_image_edit_embedding_cache.pt"}),
            }
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ("EMBEDDING", "TEXT_IDS", "IMAGE")
    FUNCTION = "encode"
    CATEGORY = "Diffusers"

    def encode(self, pipeline, 
            #    pipeline_info, 
               prompt, image, resolution=512, 
            #    offload_after_encode=False,
               cache_embeddings=False, cache_file="diffusers_image_edit_embedding_cache.pt"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                        text_ids = cached_data["text_ids"]
                        load_cache = True
            except Exception as e:
                print(f"Error reading cache file, regenerating embeddings: {e}")
        
        if not load_cache:
            # If there's an error reading the cache, regenerate embeddings
            # Use the helper method to manage pipeline components for text encoding
            prompt_embeds, text_ids = manage_pipeline_for_text_encoding(
                pipeline, 
                # pipeline_info, 
                device, 
                # offload_after_encode=offload_after_encode,
                is_image_edit=True, image_pil=image_pil, prompt=prompt
            )

            # Cache the embeddings if requested
            if cache_embeddings:
                cache_data = {
                    "prompt_embeds": prompt_embeds,
                    "text_ids": text_ids,
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
        return (prompt_embeds, text_ids, output_image)


class DiffusersLoraLoader:
    """Load and apply LoRA weights to a pipeline."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("PIPELINE",),
                "lora_path": ("STRING", {"default": "", "multiline": False}),
                "lora_name": ("STRING", {"default": ""}),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("PIPELINE",)
    FUNCTION = "load_lora"
    CATEGORY = "Diffusers"

    def load_lora(self, pipeline, lora_path, lora_name, strength=1.0):
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA path does not exist: {lora_path}")

        # Check if lora_name ends with .safetensors and add it if not
        if not lora_name.endswith('.safetensors'):
            lora_name += '.safetensors'

        full_lora_path = os.path.join(lora_path, lora_name)

        if not os.path.exists(full_lora_path):
            raise FileNotFoundError(f"LoRA file does not exist: {full_lora_path}")

        # Load LoRA weights into the pipeline
        try:
            pipeline.load_lora_weights(
                lora_path,
                weight_name=lora_name,
                adapter_name=lora_name.replace('.safetensors', '')
            )

            # Set the adapter with the given strength
            pipeline.set_adapters([lora_name.replace('.safetensors', '')], adapter_weights=[strength])
        except Exception as e:
            print(f"Warning: Error during LoRA loading: {e}")
            print("This may be due to components being set to None. Attempting to load with prefix=None...")
            try:
                # Try loading with prefix=None to handle missing components
                pipeline.load_lora_weights(
                    lora_path,
                    weight_name=lora_name,
                    adapter_name=lora_name.replace('.safetensors', ''),
                    prefix=None  # Use None prefix to handle missing components
                )
                pipeline.set_adapters([lora_name.replace('.safetensors', '')], adapter_weights=[strength])
            except Exception as e2:
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
                "num_inference_steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0}),
            },
            "optional": {
                "text_ids": ("TEXT_IDS",),
                "negative_prompt_embeds": ("EMBEDDING",),
                "negative_text_ids": ("TEXT_IDS",),
                "num_images_per_prompt": ("INT", {"default": 1, "max": 10}),
                "seed": ("INT", {"default": 42}),
                "auto_unload_lora": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "Diffusers"

    def generate(self, pipeline, prompt_embeds, width, height, num_inference_steps,
                 guidance_scale, negative_prompt_embeds=None, text_ids=None,
                 negative_text_ids=None, num_images_per_prompt=1, seed=42, auto_unload_lora=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        generator = torch.Generator(device=device).manual_seed(seed)

        # Prepare text IDs if available
        kwargs = {}
        if text_ids is not None:
            kwargs["text_ids"] = text_ids
        if negative_text_ids is not None:
            kwargs["negative_text_ids"] = negative_text_ids

        # Handle different pipeline types
        try:
            if hasattr(pipeline, 'transformer') and hasattr(pipeline, 'scheduler'):
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
            else:
                # Standard diffusers pipeline
                # This is a fallback implementation - adjust based on actual pipeline requirements
                raise NotImplementedError("Standard diffusers generation not implemented for this pipeline")

        except Exception as e:
            print(f"Error during generation: {e}")
            raise e

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
                "text_ids": ("TEXT_IDS",),
                "num_inference_steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0}),
            },
            "optional": {
                "negative_prompt_embeds": ("EMBEDDING",),
                "negative_text_ids": ("TEXT_IDS",),
                "num_images_per_prompt": ("INT", {"default": 1, "max": 10}),
                "seed": ("INT", {"default": 42}),
                "auto_unload_lora": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "Diffusers"

    def generate(self, pipeline, image, prompt_embeds, text_ids, num_inference_steps,
                 guidance_scale, negative_prompt_embeds=None, 
                 negative_text_ids=None, num_images_per_prompt=1, seed=42, auto_unload_lora=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        generator = torch.Generator(device=device).manual_seed(seed)

        # Convert image tensor to PIL
        image_np = image.squeeze().numpy()
        image_pil = Image.fromarray((image_np * 255).astype(np.uint8))

        # Prepare text IDs if available
        kwargs = {}
        if text_ids is not None:
            kwargs["text_ids"] = text_ids
        if negative_text_ids is not None:
            kwargs["negative_text_ids"] = negative_text_ids

        # Handle image editing pipeline - check for specific attributes to identify image edit pipeline
        try:
            if hasattr(pipeline, 'image_processor_vl'):  # Image edit pipeline
                # LongCat image edit pipeline - accepts image parameter
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
            elif hasattr(pipeline, 'transformer') and hasattr(pipeline, 'scheduler'):
                # This appears to be a regular LongCat image pipeline, not an image edit pipeline
                raise NotImplementedError("DiffusersImageEditGenerator requires an image editing pipeline. "
                                        "Please use a LongCat image editing model with image_processor_vl attribute, "
                                        "or use the regular DiffusersImageGenerator node instead.")
            else:
                raise NotImplementedError("Image editing not implemented for this pipeline type. "
                                        "Make sure you're using a LongCat image editing pipeline.")

        except Exception as e:
            print(f"Error during image editing: {e}")
            raise e

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