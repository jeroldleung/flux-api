import argparse
import io

import torch
import uvicorn
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers import FluxPipeline, FluxTransformer2DModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from scalar_fastapi import get_scalar_api_reference
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from transformers import T5EncoderModel


class Request(BaseModel):
    prompt: str = Field("A cat holding a sign that says hello world", description="Text prompt for image generation.")
    width: int = Field(32, ge=16, le=1024, description="Width of the generated image in pixels. Must can be devided by 16.")
    height: int = Field(32, ge=16, le=1024, description="Height of the generated image in pixels. Must can be devided by 16.")
    num_inference_steps: int | None = Field(4, ge=1, le=50, description="Number of steps for the image generation process.")
    guidance_scale: int | None = Field(0.0, description="High guidance scales improve prompt adherence but reduce realism.")


api = FastAPI(title="flux-api", docs_url=None, redoc_url=None)

api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@api.get("/docs", include_in_schema=False)
async def scalar_html():
    return get_scalar_api_reference(openapi_url=api.openapi_url, title=api.title)


@api.post("/v1/inference")
async def generate_image(req: Request):
    out = pipe(**req.model_validate_json()).images[0]
    imageb = io.BytesIO()
    out.save(imageb, format="PNG")
    imageb.seek(0)
    return StreamingResponse(imageb, media_type="image/png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=50001)
    parser.add_argument("--path", type=str, default="pretrained_models/FLUX.1-dev")
    args = parser.parse_args()

    # quantizing the flux in 4-bit

    quant_config = TransformersBitsAndBytesConfig(load_in_4bit=True)
    text_encoder_2_4bit = T5EncoderModel.from_pretrained(
        args.path,
        subfolder="text_encoder_2",
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
    )

    quant_config = DiffusersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    transformer_4bit = FluxTransformer2DModel.from_pretrained(
        args.path,
        subfolder="transformer",
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
    )

    pipe = FluxPipeline.from_pretrained(
        args.path,
        transformer=transformer_4bit,
        text_encoder_2=text_encoder_2_4bit,
        torch_dtype=torch.bfloat16,
    )

    pipe.to("cuda")  # load model to gpu

    uvicorn.run(api, host="0.0.0.0", port=args.port)
