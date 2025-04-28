from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
import torch
import gradio as gr

def load_r_llama(model_dir="r_model"):
    # tokenizer + pad_token
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token

    # 4-bit quant config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # load base model in 4-bit
    base = AutoModelForCausalLM.from_pretrained(
        "codellama/CodeLlama-7b-hf",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # apply LoRA adapters
    model = PeftModel.from_pretrained(base, model_dir)
    model.eval()
    return model, tokenizer

# load once
MODEL, TOKENIZER = load_r_llama()

def r_chat_fn(message, history):
    system = "You are an expert R programmer. Only output valid R code. Do not include explanations, examples, or any extra text. Output R code only."
    instr  = f"Task: {message}\n\nOnly output valid R code, and nothing else."
    prompt = f"{system}\n\n### Instruction:\n{instr}\n\n### R Code:\n"

    inputs = TOKENIZER(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    ).to(MODEL.device)

    outputs = MODEL.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        eos_token_id=TOKENIZER.eos_token_id,
        pad_token_id=TOKENIZER.eos_token_id,
    )

    gen = TOKENIZER.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    ).strip()

    return gen




def create_interface():
    interface = gr.ChatInterface(
        fn=r_chat_fn,
        type="messages",
        title="R-Programming LLaMA Assistant",
        description="Ask your fine-tuned R-LLaMA to write or explain R code. Each prompt is independent of the previous prompt.",
    )

    # See chatbot at http://127.0.0.1:7860
    print("Launching chatbot…")
    interface.launch()
