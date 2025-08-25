#!/usr/bin/env python3
"""
FastAPI server for Nomic Embed Vision v1.5 with OpenAI-compatible API
Supports both text and image embeddings in a shared embedding space
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Union, Optional, Dict, Any
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from PIL import Image
import base64
import io
import uvicorn
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models
vision_model = None
text_model = None
processor = None
tokenizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup and cleanup on shutdown"""
    global vision_model, text_model, processor, tokenizer

    logger.info("Loading Nomic Embed Vision v1.5 models...")

    try:
        # Load vision components
        processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")
        vision_model = AutoModel.from_pretrained(
            "nomic-ai/nomic-embed-vision-v1.5",
            trust_remote_code=True
        )

        # Load text components
        tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5")
        text_model = AutoModel.from_pretrained(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True
        )

        # Set models to evaluation mode
        vision_model.eval()
        text_model.eval()

        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vision_model.to(device)
        text_model.to(device)

        logger.info(f"Models loaded successfully on device: {device}")

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

    yield

    # Cleanup
    logger.info("Shutting down...")

app = FastAPI(
    title="Nomic Embed Vision API",
    description="OpenAI-compatible API for Nomic Embed Vision v1.5",
    version="1.0.0",
    lifespan=lifespan
)

# Request/Response Models
class EmbeddingInput(BaseModel):
    """Individual input item - can be text or base64 image"""
    type: str = Field(description="Type of input: 'text' or 'image_url'")
    text: Optional[str] = Field(default=None, description="Text content")
    image_url: Optional[Dict[str, str]] = Field(default=None, description="Image URL object with 'url' field")

class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request"""
    input: Union[str, List[str], List[EmbeddingInput]] = Field(
        description="Input text, image, or list of inputs to embed"
    )
    model: str = Field(default="nomic-embed-vision-v1.5", description="Model to use")
    encoding_format: str = Field(default="float", description="Encoding format")
    dimensions: Optional[int] = Field(default=None, description="Number of dimensions (not supported)")

class EmbeddingData(BaseModel):
    """Individual embedding result"""
    object: str = "embedding"
    embedding: List[float]
    index: int

class Usage(BaseModel):
    """Token usage information"""
    prompt_tokens: int
    total_tokens: int

class EmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response"""
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: Usage

def mean_pooling(model_output, attention_mask):
    """Mean pooling for text embeddings"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )

def decode_base64_image(base64_string: str) -> Image.Image:
    """Decode base64 image string to PIL Image"""
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:'):
            base64_string = base64_string.split(',', 1)[1]

        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image.convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")

def process_text_input(texts: List[str]) -> torch.Tensor:
    """Process text inputs and return embeddings"""
    device = next(text_model.parameters()).device

    # Add search_query prefix as recommended by Nomic
    prefixed_texts = []
    for text in texts:
        if not text.startswith('search_query:') and not text.startswith('search_document:'):
            text = f'search_query: {text}'
        prefixed_texts.append(text)

    # Tokenize
    encoded_input = tokenizer(
        prefixed_texts,
        padding=True,
        truncation=True,
        return_tensors='pt'
    ).to(device)

    # Generate embeddings
    with torch.no_grad():
        model_output = text_model(**encoded_input)
        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
        embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings

def process_image_input(images: List[Image.Image]) -> torch.Tensor:
    """Process image inputs and return embeddings"""
    device = next(vision_model.parameters()).device

    embeddings_list = []

    for image in images:
        # Process image
        inputs = processor(image, return_tensors="pt").to(device)

        # Generate embedding
        with torch.no_grad():
            img_emb = vision_model(**inputs).last_hidden_state
            img_embedding = F.normalize(img_emb[:, 0], p=2, dim=1)
            embeddings_list.append(img_embedding)

    return torch.cat(embeddings_list, dim=0)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Nomic Embed Vision v1.5 API Server",
        "status": "running",
        "model": "nomic-embed-vision-v1.5"
    }

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)"""
    return {
        "object": "list",
        "data": [
            {
                "id": "nomic-embed-vision-v1.5",
                "object": "model",
                "created": 1677610602,
                "owned_by": "nomic-ai"
            }
        ]
    }

@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """Create embeddings for text and/or images"""

    if vision_model is None or text_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        embeddings_list = []
        total_tokens = 0

        # Handle different input formats
        if isinstance(request.input, str):
            # Single text input
            embeddings = process_text_input([request.input])
            embeddings_list.extend(embeddings.cpu().numpy().tolist())
            total_tokens += len(request.input.split())

        elif isinstance(request.input, list):
            if len(request.input) == 0:
                raise HTTPException(status_code=400, detail="Empty input list")

            # Check if list contains strings or EmbeddingInput objects
            if isinstance(request.input[0], str):
                # List of strings
                embeddings = process_text_input(request.input)
                embeddings_list.extend(embeddings.cpu().numpy().tolist())
                total_tokens += sum(len(text.split()) for text in request.input)

            else:
                # List of EmbeddingInput objects (mixed text/image)
                texts = []
                images = []
                input_types = []

                for item in request.input:
                    if item.type == "text":
                        texts.append(item.text)
                        input_types.append("text")
                        total_tokens += len(item.text.split()) if item.text else 0

                    elif item.type == "image_url":
                        if not item.image_url or "url" not in item.image_url:
                            raise HTTPException(
                                status_code=400,
                                detail="image_url must contain 'url' field"
                            )

                        url = item.image_url["url"]
                        if url.startswith('data:'):
                            # Base64 encoded image
                            image = decode_base64_image(url)
                            images.append(image)
                            input_types.append("image")
                            total_tokens += 1  # Count images as 1 token
                        else:
                            raise HTTPException(
                                status_code=400,
                                detail="Only base64 data URLs are supported for images"
                            )
                    else:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Unsupported input type: {item.type}"
                        )

                # Process texts and images separately
                all_embeddings = []

                if texts:
                    text_embeddings = process_text_input(texts)
                    all_embeddings.append(text_embeddings)

                if images:
                    image_embeddings = process_image_input(images)
                    all_embeddings.append(image_embeddings)

                # Combine embeddings in the original order
                if all_embeddings:
                    combined_embeddings = torch.cat(all_embeddings, dim=0)
                    embeddings_list.extend(combined_embeddings.cpu().numpy().tolist())

        # Create response data
        data = [
            EmbeddingData(
                embedding=embedding,
                index=i
            )
            for i, embedding in enumerate(embeddings_list)
        ]

        return EmbeddingResponse(
            data=data,
            model=request.model,
            usage=Usage(
                prompt_tokens=total_tokens,
                total_tokens=total_tokens
            )
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/embeddings")
async def create_embeddings_alt(request: EmbeddingRequest) -> EmbeddingResponse:
    """Alternative endpoint without /v1 prefix"""
    return await create_embeddings(request)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Nomic Embed Vision FastAPI Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=11439, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")

    args = parser.parse_args()

    uvicorn.run(
        app,  # Pass the app directly instead of module string
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )