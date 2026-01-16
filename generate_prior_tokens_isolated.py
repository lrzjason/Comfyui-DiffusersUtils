import sys
import torch
import argparse
import json
from pathlib import Path
import os

# Add the project root to the path to import modules
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from glm_image.pipeline_glm_image import GlmImagePipeline


def generate_prior_tokens_isolated(model_path, prompt, height, width, output_path, device="cuda", dtype_str="bfloat16"):
    """
    Generate prior tokens in an isolated environment with controlled settings
    """
    # Set consistent environment settings
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Convert dtype string to actual dtype
    if dtype_str == "bfloat16":
        dtype = torch.bfloat16
    elif dtype_str == "float16":
        dtype = torch.float16
    elif dtype_str == "float32":
        dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    
    # Set device
    device = torch.device(device)
    
    # Set random seeds for consistency
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    try:
        # Load a minimal pipeline for prior token generation
        text_pipeline = GlmImagePipeline.from_pretrained(
            model_path,
            text_encoder=None,
            vae=None,
            transformer=None,
            torch_dtype=dtype,
        )
        
        text_pipeline = text_pipeline.to(device)
        
        # Generate prior tokens
        with torch.no_grad():
            prior_token_ids, prior_image_token_ids = text_pipeline.generate_prior_tokens(
                prompt=prompt,
                image=None,  # No conditional image for this use case
                height=height,
                width=width,
                device=device,
            )
        
        # Prepare data for saving
        prior_tokens_data = {
            "prior_token_ids": prior_token_ids.cpu(),
            "prior_image_token_ids": prior_image_token_ids.cpu() if prior_image_token_ids is not None else None,
        }
        
        # Save the prior tokens
        torch.save(prior_tokens_data, output_path)
        
        # Return success indicator and output path
        result = {
            "success": True,
            "output_path": output_path,
            "prior_token_ids_shape": list(prior_token_ids.shape) if prior_token_ids is not None else None,
            "prior_image_token_ids_shape": list(prior_image_token_ids.shape) if prior_image_token_ids is not None else None,
        }
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Generate prior tokens in isolated environment")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the GLM model")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for prior token generation")
    parser.add_argument("--height", type=int, default=1024, help="Height for generation")
    parser.add_argument("--width", type=int, default=1024, help="Width for generation")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for prior tokens")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type to use (bfloat16/float16/float32)")
    
    args = parser.parse_args()
    
    result = generate_prior_tokens_isolated(
        model_path=args.model_path,
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        output_path=args.output_path,
        device=args.device,
        dtype_str=args.dtype
    )
    
    # Print result as JSON to stdout
    print(json.dumps(result))


if __name__ == "__main__":
    main()