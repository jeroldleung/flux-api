import argparse

import torch
import uvicorn
from diffusers import FluxPipeline
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from scalar_fastapi import get_scalar_api_reference


class Request(BaseModel):
    prompt: str = Field("A cat holding a sign that says hello world", description="Text prompt for image generation.")
    width: int = Field(32, description="Width of the generated image in pixels.")
    height: int = Field(32, description="Height of the generated image in pixels.")
    steps: int | None = Field(4, ge=1, le=50, description="Number of steps for the image generation process.")
    guidance: int | None = Field(0.0, description="High guidance scales improve prompt adherence but reduce realism.")


api = FastAPI(title="flux-dev", docs_url=None, redoc_url=None)

api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@api.get("/docs", include_in_schema=False)
async def scalar_html():
    return get_scalar_api_reference(openapi_url=api.openapi_url, title=api.title)


@api.post("/v1/inference")
async def generate_image(req: Request):
    out = pipe(
        prompt=req.prompt,
        width=req.width,
        height=req.height,
        num_inference_steps=req.steps,
        guidance_scale=req.guidance,
        # generator=torch.Generator("cuda").manual_seed(0),
    ).images[0]

    return FileResponse(out.save("result.png"), media_type="image/png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=50001)
    parser.add_argument("--path", type=str, default="pretrained_models/FLUX.1-dev")
    args = parser.parse_args()

    pipe = FluxPipeline.from_pretrained(args.path, torch_dtype=torch.bfloat16)

    uvicorn.run(api, host="0.0.0.0", port=args.port)
