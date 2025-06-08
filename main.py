import torch
import threading
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList
)

# === Step 1: Load Model ===
model_path = r"C:\Users\mmmaf\OneDrive\Desktop\i\NursingNotes\mistral-merged-full"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=quant_config
)

print(f" ^|^e Model loaded on: {model.device}")

# === Step 2: Define StopOnPhrase ===
class StopOnPhrase(StoppingCriteria):
    def __init__(self, tokenizer, stop_phrases):
        self.tokenizer = tokenizer
        self.stop_phrases = stop_phrases
        self.triggered = False
        self.previous_text = ""

    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True).lower().strip()
        if self.triggered:
            if text == self.previous_text or text.endswith("\n") or text.endswith("\n\n"):
                return True
            self.previous_text = text
            return False

        for phrase in self.stop_phrases:
            if phrase in text:
                self.triggered = True
                self.previous_text = text
                break

        return False

# === Step 3: FastAPI App ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "AI Streaming Server Ready"}

@app.get("/stream")
def stream(prompt: str):
    def generate():
        inputs = tokenizer(f"<s>[INST] {prompt.strip()} [/INST]", return_tensors="pt").to(model.device)

        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)

        stop_phrases = [
            "note concluded.",
            "discharged in stable condition",
            "patient discharged",
            "discharged home",
            "discharged with",
            "handover given",
            "transferred to labor room",
            "received in good condition"
        ]

        generation_kwargs = {
    **inputs,
    "max_new_tokens": 1024,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.2,
    "no_repeat_ngram_size": 6,
    "pad_token_id": tokenizer.eos_token_id,
    "stopping_criteria": StoppingCriteriaList([StopOnPhrase(tokenizer, stop_phrases)]),
    "streamer": streamer
}



        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        for token in streamer:
            if token.strip() == "":
                continue  # skip empty tokens that cause blank lines
            print(token, end="", flush=True)
            yield f"data: {token}\n\n"

        print("\n=== [End of Note] ===")
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
