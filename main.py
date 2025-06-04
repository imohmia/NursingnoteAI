from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch

app = FastAPI()

# Load your model from local files (already uploaded in this repo)
tokenizer = AutoTokenizer.from_pretrained("./", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./", torch_dtype=torch.float16)
model.to("cuda" if torch.cuda.is_available() else "cpu")

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    def stream_response():
        streamer = TextStreamer(tokenizer)
        _ = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer
        )

    return StreamingResponse(stream_response(), media_type="text/plain")
