import os
import time
import gc
import logging
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# Configuration
MODEL_ID = "mlx-community/Llama-3.2-3B-Instruct-4bit"
ADAPTERS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../adapters"))

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MultiLoRaServer")

app = FastAPI(title="MultiLoRa MLX Server")

class EfficientMultiLoRA:
    def __init__(self, base_model_path, adapters_dir):
        logger.info(f"Initializing EfficientMultiLoRA with base: {base_model_path}")
        self.base_model_path = base_model_path
        self.adapters_dir = adapters_dir
        self.model = None
        self.tokenizer = None
        self.current_adapter = None
        
        # Load base model initially
        self.load_persona("base")

    def load_persona(self, adapter_name: str):
        # Normalize name (remove 'default' or handled elsewhere)
        if adapter_name == self.current_adapter:
            return

        logger.info(f"Switching adapter from '{self.current_adapter}' to '{adapter_name}'")
        start_time = time.time()
        
        # 1. Clear Memory
        if self.model is not None:
            del self.model
            del self.tokenizer
            mx.metal.clear_cache()
            gc.collect()
            
        # 2. Determine Path
        adapter_path = None
        if adapter_name not in ["base", "default"]:
            # Check if adapter exists
            candidate_path = os.path.join(self.adapters_dir, adapter_name)
            if os.path.exists(candidate_path):
                adapter_path = candidate_path
            else:
                logger.warning(f"Adapter '{adapter_name}' not found. Falling back to base.")
                adapter_name = "base"

        # 3. Load
        self.model, self.tokenizer = load(
            self.base_model_path,
            adapter_path=adapter_path
        )
        
        self.current_adapter = adapter_name
        elapsed = time.time() - start_time
        logger.info(f"Loaded '{adapter_name}' in {elapsed:.2f}s")

    def generate(self, messages: List[Dict[str, str]], max_tokens: int = 512, temp: float = 0.7):
        # Apply chat template
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # Fallback for models without template (simple concat)
            prompt = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
            prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

        sampler = make_sampler(temp=temp, top_p=0.9)
        
        response = generate(
            self.model, 
            self.tokenizer, 
            prompt=prompt, 
            max_tokens=max_tokens, 
            verbose=False, 
            sampler=sampler
        )
        return response

# Global Engine
engine = EfficientMultiLoRA(MODEL_ID, ADAPTERS_DIR)

# API Models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str = "chatcmpl-default"
    object: str = "chat.completion"
    created: int = int(time.time())
    model: str
    choices: List[ChatCompletionResponseChoice]

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = int(time.time())
    owned_by: str = "mlx-local"

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]

# Endpoints
@app.get("/v1/models", response_model=ModelList)
async def list_models():
    # Scan adapters directory
    adapters = ["base"]
    if os.path.exists(ADAPTERS_DIR):
        for name in os.listdir(ADAPTERS_DIR):
            if os.path.isdir(os.path.join(ADAPTERS_DIR, name)):
                adapters.append(name)
    
    return ModelList(data=[ModelCard(id=name) for name in adapters])

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    # 1. Switch Model if needed
    engine.load_persona(request.model)
    
    # 2. Generate
    output_text = engine.generate(
        [m.dict() for m in request.messages],
        max_tokens=request.max_tokens,
        temp=request.temperature
    )

    # 3. Form Response
    return ChatCompletionResponse(
        model=request.model,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=output_text),
                finish_reason="stop"
            )
        ]
    )

if __name__ == "__main__":
    import uvicorn
    # Listen on all interfaces so Docker can reach it via host.docker.internal (or if using host net)
    uvicorn.run(app, host="0.0.0.0", port=8000)
