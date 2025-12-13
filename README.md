
# ComfyUI Diffusers Utils

⚠️ **EXPERIMENTAL NODES - UNDER ACTIVE DEVELOPMENT** ⚠️

**IMPORTANT NOTICE:** This repository contains experimental nodes that are actively under development. There is currently a known bug that occurs when changing prompts after a generation. Use with caution and expect potential instability.

Feel free to open issues and submit pull requests to help improve this extension!

A set of nodes which provide flexible inference using diffusers in comfyui env.

## Current Supported Models:
- LongCat 6B Dev
- LongCat 6B Image
- LongCat 6B Edit

## Installation

1. Clone this repository into your ComfyUI custom_nodes directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/Comfyui-DiffusersUtils.git
```

2. Install the required dependencies:
```bash
cd Comfyui-DiffusersUtils
pip install -r requirements.txt
```

3. Restart ComfyUI

## Available Nodes

### Model Loading Nodes

#### Load Diffusers Model (`DiffusersModelLoader`)
Loads a complete diffusers model from a directory path.
- **Inputs:**
  - `model_path`: Path to the model directory
  - `dtype`: Data type (float32, float16, bfloat16)
  - `load_vae`: Whether to load VAE (default: True)
  - `load_text_encoder`: Whether to load text encoder (default: True)
  - `load_transformer`: Whether to load transformer (default: True)
  - `load_tokenizer`: Whether to load tokenizer (default: True)
  - `load_image_encoder`: Whether to load image encoder (default: True)
- **Outputs:**
  - `PIPELINE`: The loaded pipeline object

#### Load Diffusers Text Encoder (`DiffusersTextEncoderLoader`)
Loads only the text encoder component from a diffusers model.
- **Inputs:**
  - `model_path`: Path to the model directory
  - `dtype`: Data type (float32, float16, bfloat16)
- **Outputs:**
  - `TEXT_ENCODER`: The loaded text encoder

#### Load Diffusers Transformer (`DiffusersTransformerLoader`)
Loads only the transformer component from a diffusers model.
- **Inputs:**
  - `any`: Any input (workaround for ComfyUI)
  - `model_path`: Path to the model directory
  - `dtype`: Data type (float32, float16, bfloat16)
- **Outputs:**
  - `TRANSFORMER`: The loaded transformer

#### Load Diffusers VAE (`DiffusersVAELoader`)
Loads only the VAE component from a diffusers model.
- **Inputs:**
  - `model_path`: Path to the model directory
  - `dtype`: Data type (float32, float16, bfloat16)
- **Outputs:**
  - `VAE`: The loaded VAE

#### Load Diffusers Tokenizer (`DiffusersTokenizerLoader`)
Loads only the tokenizer component from a diffusers model.
- **Inputs:**
  - `model_path`: Path to the model directory
- **Outputs:**
  - `TOKENIZER`: The loaded tokenizer

#### Load Diffusers Preprocessor (`DiffusersPreprocessorLoader`)
Loads only the preprocessor component from a diffusers model.
- **Inputs:**
  - `model_path`: Path to the model directory
- **Outputs:**
  - `PREPROCESSOR`: The loaded preprocessor

### Pipeline Building Nodes

#### Build Diffusers Pipeline from Components (`DiffusersPipelineBuilder`)
Builds a pipeline from individual components, allowing for flexible model construction.
- **Inputs:**
  - `model_path`: Path to the model directory
  - `pipeline_type`: Type of pipeline (auto, image, image_edit)
  - `dtype`: Data type (float32, float16, bfloat16)
  - `pipeline`: Existing pipeline (optional)
  - `text_encoder`: Text encoder component (optional)
  - `transformer`: Transformer component (optional)
  - `vae`: VAE component (optional)
  - `tokenizer`: Tokenizer component (optional)
  - `preprocessor`: Preprocessor component (optional)
- **Outputs:**
  - `PIPELINE`: The constructed pipeline

### Text Encoding Nodes

#### Encode Prompt (Diffusers LongCat) (`TextEncodeDiffusersLongCat`)
Encodes text using the pipeline's text encoder with optional caching.
- **Inputs:**
  - `pipeline`: The pipeline to use for encoding
  - `prompt`: Input text prompt
  - `cache_embeddings`: Whether to cache embeddings (default: False)
  - `cache_file`: Path to cache file (default: diffusers_embedding_cache.pt)
- **Outputs:**
  - `EMBEDDING`: The encoded prompt embeddings
  - `TEXT_IDS`: Text IDs for the prompt (if applicable)

#### Encode Prompt + Image (Diffusers LongCat Image Edit) (`TextEncodeDiffusersLongCatImageEdit`)
Encodes text and image together for image edit pipelines.
- **Inputs:**
  - `pipeline`: The pipeline to use for encoding
  - `prompt`: Input text prompt
  - `image`: Input image for editing
  - `resolution`: Resolution to resize image to (default: 512)
  - `cache_embeddings`: Whether to cache embeddings (default: False)
  - `cache_file`: Path to cache file (default: diffusers_image_edit_embedding_cache.pt)
- **Outputs:**
  - `EMBEDDING`: The encoded prompt embeddings
  - `TEXT_IDS`: Text IDs for the prompt
  - `IMAGE`: The processed image in proper format

#### Load Cached Embeddings (`TextEncodeDiffusersLongCatCached`)
Loads cached prompt embeddings from a file.
- **Inputs:**
  - `cache_file`: Path to cache file
- **Outputs:**
  - `EMBEDDING`: The cached prompt embeddings
  - `TEXT_IDS`: Cached text IDs (if available)
  - `STRING`: The cached prompt text

### LoRA Utility Nodes

#### Load LoRA Only (`LoadLoraOnly`)
Load a LoRA file without applying it to any models. Use with MergeLoraToModel or other nodes to apply the LoRA later.
- **Inputs:**
  - `lora_path`: Path to the LoRA file
- **Outputs:**
  - `LORA`: The loaded LoRA state dictionary

#### LoRA Layers Operation (`LoraLayersOperation`)
Modify specific layers in a LoRA by zeroing them out (when scale=0) or scaling them (otherwise) based on pattern matching.
- **Inputs:**
  - `lora`: The LoRA state dictionary to modify
  - `layer_pattern`: Regex pattern to match layer names. Use groups to extract layer indices.
  - `layer_indices`: Comma-separated list of layer indices to operate on, with support for ranges (e.g., '59', '10,11,12', or '50-53')
  - `scale_factor`: Scale factor to apply. Use 0 to zero out layers.
- **Outputs:**
  - `modified_lora`: The modified LoRA state dictionary

#### LoRA Stat Viewer (`LoraStatViewer`)
View information about LoRA layers to help define layer patterns for LoraLayersOperation.
- **Inputs:**
  - `lora`: The loaded LoRA to analyze
- **Outputs:**
  - `lora_info`: Information about the LoRA layers

#### Save LoRA (`SaveLora`)
Save a modified LoRA state dictionary to a file.
- **Inputs:**
  - `lora`: The modified LoRA state dictionary to save
  - `filename`: Filename to save the LoRA as (e.g. my_lora.safetensors)
  - `output_dir`: Directory to save the LoRA to (optional, defaults to ComfyUI output directory)
- **Outputs:**
  - None (saves file to specified location)

#### Merge LoRA to Transformer (`MergeLoraToTransformer`)
Apply a pre-loaded LoRA to transformer. This allows separation of loading and applying LoRAs.
- **Inputs:**
  - `transformer`: The transformer model to apply the LoRA to
  - `lora`: The loaded LoRA to apply
  - `strength_model`: How strongly to modify the diffusion model. This value can be negative.
  - `adapter_name`: The name of the adapter to use
- **Outputs:**
  - `TRANSFORMER`: The modified transformer model

### LoRA Handling Nodes

#### Load Diffusers LoRA (`DiffusersLoraLoader`)
Loads and applies LoRA weights to a pipeline.
- **Inputs:**
  - `pipeline`: The pipeline to apply LoRA to
  - `lora_path`: Path to the LoRA file (full path including filename)
  - `strength`: Strength of the LoRA effect (default: 1.0)
- **Outputs:**
  - `PIPELINE`: The pipeline with LoRA applied

#### Unload Diffusers LoRA (`DiffusersLoraUnloader`)
Removes LoRA weights from a pipeline.
- **Inputs:**
  - `pipeline`: The pipeline to unload LoRA from
- **Outputs:**
  - `PIPELINE`: The pipeline with LoRA removed

### Generation Nodes

#### Generate Image (Diffusers) (`DiffusersImageGenerator`)
Generates images using a pipeline with embeddings and optional LoRAs.
- **Inputs:**
  - `pipeline`: The pipeline to use for generation
  - `prompt_embeds`: Prompt embeddings to use
  - `width`: Width of generated image (default: 1024)
  - `height`: Height of generated image (default: 1024)
  - `num_inference_steps`: Number of inference steps (default: 30)
  - `guidance_scale`: Guidance scale (default: 2.5)
  - `text_ids`: Text IDs (optional)
  - `negative_prompt_embeds`: Negative prompt embeddings (optional)
  - `negative_text_ids`: Negative text IDs (optional)
  - `num_images_per_prompt`: Number of images to generate (default: 1)
  - `seed`: Random seed (default: 42)
  - `auto_unload_lora`: Automatically unload LoRA after generation (default: True)
- **Outputs:**
  - `IMAGE`: The generated image(s)

#### Edit Image (Diffusers) (`DiffusersImageEditGenerator`)
Edits images using a pipeline with embeddings and optional LoRAs.
- **Inputs:**
  - `pipeline`: The pipeline to use for editing
  - `image`: Input image to edit
  - `prompt_embeds`: Prompt embeddings to use
  - `text_ids`: Text IDs
  - `num_inference_steps`: Number of inference steps (default: 30)
  - `guidance_scale`: Guidance scale (default: 2.5)
  - `negative_prompt_embeds`: Negative prompt embeddings (optional)
  - `negative_text_ids`: Negative text IDs (optional)
  - `num_images_per_prompt`: Number of images to generate (default: 1)
  - `seed`: Random seed (default: 42)
  - `auto_unload_lora`: Automatically unload LoRA after editing (default: True)
- **Outputs:**
  - `IMAGE`: The edited image(s)

## Usage Examples

The nodes are designed to work together in a pipeline:

1. Use `DiffusersModelLoader` or individual component loaders to load your model
2. Combine components using `DiffusersPipelineBuilder` if needed
3. Encode your prompts using the appropriate text encoding node
4. Apply LoRAs if desired using `DiffusersLoraLoader` or the LoRA utility nodes
5. Generate images using `DiffusersImageGenerator` or edit images using `DiffusersImageEditGenerator`

For image editing workflows, use the dedicated image editing nodes which handle both image and text processing.

## Contact
- **Twitter**: [@Lrzjason](https://twitter.com/Lrzjason)
- **Email**: lrzjason@gmail.com
- **QQ Group**: 866612947
- **Wechatid**: fkdeai
- **Civitai**: [xiaozhijason](https://civitai.com/user/xiaozhijason)

## Sponsors me for more open source projects:
<div align="center">
  <table>
    <tr>
      <td align="center">
        <p>Buy me a coffee:</p>
        <img src="https://github.com/lrzjason/Comfyui-In-Context-Lora-Utils/blob/main/image/bmc_qr.png" alt="Buy Me a Coffee QR" width="200" />
      </td>
      <td align="center">
        <p>WeChat:</p>
        <img src="https://github.com/lrzjason/Comfyui-In-Context-Lora-Utils/blob/main/image/wechat.jpg" alt="WeChat QR" width="200" />
      </td>
    </tr>
  </table>
</div>