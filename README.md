# ComfyUI Diffusers Utils

⚠️ **EXPERIMENTAL NODES - UNDER ACTIVE DEVELOPMENT** ⚠️

A set of nodes which provide flexible inference using diffusers in ComfyUI environment.

**Latest Update (Jan 16, 2026):** Added support for GLM Image models (both text-to-image and image-to-image) with dedicated installation instructions.

## Current Supported Models:
- LongCat 6B Image
- LongCat 6B Image Edit
- GLM Image (Text-to-Image)
- GLM Image (Image-to-Image)

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

3. For GLM Image support, also install the following packages:
```bash
pip install git+https://github.com/huggingface/transformers.git
pip install git+https://github.com/huggingface/diffusers.git
pip install git+https://github.com/huggingface/peft.git
```

4. Restart ComfyUI

## Available Nodes

### Pipeline Loading Node

#### Diffusers Pipeline Loader (`DiffusersPipeline`)
Initializes different LongCat diffusion pipelines with dynamic components.

- **Inputs:**
  - `pipeline_class_name`: Select pipeline type ("LongCatImagePipeline" or "LongCatImageEditPipeline")
  - `model_path`: Path to the model directory  
  - `pipeline`: Existing pipeline (optional, for component swapping)
  - `torch_dtype`: Data type (float16, float32, bfloat16) - Default: bfloat16
  - `components`: Comma-separated list of components to load (optional)
  - `presets`: Predefined component sets to load - Default: "text_encoder, tokenizer, text_processor"
- **Outputs:**
  - `pipeline`: The loaded pipeline object
- **Category:** Diffusers/LongCat

### Text Encoding Node

#### Diffusers Text Encode (`DiffusersTextEncode`)
Encodes text prompts using the pipeline's text encoder. Supports both text-only and text+image encoding depending on the pipeline type.

- **Inputs:**
  - `pipeline`: The loaded pipeline object
  - `prompt`: Text prompt to encode (multiline, default: "Masterpiece, best quality, 8k uhd, photo realistic,")
  - `image`: Optional image input for pipelines that support text+image encoding
  - `batch_size`: Number of images to generate (default: 1, min: 1, max: 64)
- **Outputs:**
  - `diffusers_cond`: Conditioning information for sampling
- **Category:** Diffusers/Encode

### Sampling Node

#### Diffusers Sampling (`DiffusersSampling`)
Performs the generation loop using a pipeline and conditioning to generate images using the diffusion process.

- **Inputs:**
  - `pipeline`: Loaded diffusion pipeline
  - `diffusers_cond`: Conditioning from DiffusersTextEncode node
  - `steps`: Number of inference steps (default: 26, min: 1, max: 100)
  - `cfg`: Guidance scale (default: 1.5, min: 0.0, max: 20.0)
  - `negative_diffusers_cond`: Negative conditioning (optional)
  - `num_images_per_prompt`: Number of images to generate per prompt (default: 1, min: 1, max: 8)
  - `seed`: Random seed for generation (default: 42, range: 0 to 0xffffffffffffffff)
  - `image`: Input image for image editing (optional)
  - `width`: Output image width (for regular generation, or override for editing) (default: 1024, min: 256, max: 4096, step: 64)
  - `height`: Output image height (for regular generation, or override for editing) (default: 1024, min: 256, max: 4096, step: 64)
- **Outputs:**
  - `image`: Generated image(s)
- **Category:** Diffusers/Sampling

### LoRA Utility Nodes

#### Diffusers Load LoRA Only (`DiffusersLoadLoraOnly`)
Load a LoRA file without applying it to any models. Use with MergeLoraToModel or other nodes to apply the LoRA later.

- **Inputs:**
  - `lora_path`: Path to the LoRA file
- **Outputs:**
  - `lora`: The loaded LoRA state dictionary
- **Category:** Diffusers/Lora

#### Diffusers LoRA Layers Operation (`DiffusersLoraLayersOperation`)
Modify specific layers in a LoRA by zeroing them out (when scale=0) or scaling them (otherwise) based on pattern matching.

- **Inputs:**
  - `lora`: The LoRA state dictionary to modify
  - `layer_pattern`: Regex pattern to match layer names (default: `.*transformer_blocks\.(\d+)\.`)
  - `layer_indices`: Comma-separated list of layer indices to operate on, with support for ranges (e.g., '59', '10,11,12', or '50-53') (default: '59')
  - `scale_factor`: Scale factor to apply. Use 0 to zero out layers (default: 1.0, range: -10.0 to 10.0)
- **Outputs:**
  - `modified_lora`: The modified LoRA state dictionary
- **Category:** Diffusers/Lora

#### Diffusers Save LoRA (`DiffusersSaveLora`)
Save a modified LoRA state dictionary to a file.

- **Inputs:**
  - `lora`: The modified LoRA state dictionary to save
  - `filename`: Filename to save the LoRA as (e.g. my_lora.safetensors) (default: "my_lora.safetensors")
  - `output_dir`: Directory to save the LoRA to (optional, defaults to ComfyUI output directory)
- **Outputs:**
  - None (saves file to specified location)
- **Category:** Diffusers/Lora

#### Diffusers LoRA Stat Viewer (`DiffusersLoraStatViewer`)
View information about LoRA layers to help define layer patterns for LoraLayersOperation.

- **Inputs:**
  - `lora`: The loaded LoRA to analyze
- **Outputs:**
  - `lora_info`: Information about the LoRA layers
- **Category:** Diffusers/Lora

#### Diffusers Merge LoRA to Pipeline (`DiffusersMergeLoraToPipeline`)
Apply a pre-loaded LoRA to transformer. This allows separation of loading and applying LoRAs.

- **Inputs:**
  - `pipeline`: The pipeline to apply the LoRA to
  - `lora`: The loaded LoRA to apply
  - `strength`: How strongly to modify the diffusion model. This value can be negative (default: 1.0, range: -10.0 to 10.0)
  - `adapter_name`: The name of the adapter to use (default: "default")
- **Outputs:**
  - `pipeline`: The modified pipeline
- **Category:** Diffusers/Lora

## Usage Examples

The nodes are designed to work together in a pipeline:

1. Use `DiffusersPipeline` to load your model (select appropriate pipeline type)
2. Encode your prompts using `DiffusersTextEncode` (with optional image for edit pipelines)
3. Apply LoRAs if desired using the LoRA utility nodes:
   - Load with `DiffusersLoadLoraOnly`
   - Modify with `DiffusersLoraLayersOperation` (optional)
   - Apply to pipeline with `DiffusersMergeLoraToPipeline`
4. Generate images using `DiffusersSampling` with your conditioning and parameters

For image editing workflows, use `LongCatImageEditPipeline` with an input image to the sampling node.

## Contact
- **Twitter**: [@Lrzjason](https://twitter.com/Lrzjason)
- **Email**: lrzjason@gmail.com
- **QQ Group**: 866612947
- **Wechatid**: fkdeai
- **Civitai**: [xiaozhijason](https://civitai.com/user/xiaozhijason)

## Changelog

- **Jan 16, 2026**: Added support for GLM Image models (both text-to-image and image-to-image) with dedicated installation instructions
- **Earlier**: Initial release with LongCat 6B Image and LongCat 6B Image Edit support

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