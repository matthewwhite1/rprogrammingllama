import sys, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

sys.stdout.reconfigure(encoding="utf-8")

def train():
    ##### 1. Load local JSON as HF Dataset #####
    raw = load_dataset(
        "json", 
        data_files="data/statcodesearch.json",
        split=None
    )
    # If the dataset is returned as a DatasetDict, extract the dataset
    if isinstance(raw, dict):
        raw = raw[list(raw.keys())[0]]

    ##### 2. Reformat examples into instruction–response pairs #####
    def make_instruction(ex):
        instr = f"### Instruction:\nWrite ONLY R code to {ex['Comment'].strip()}. Respond with valid, efficient R code only. Do not explain the code; just provide the code.\n### Response:"
        return {
            "prompt": instr,
            "code": ex["Code"].strip()
        }
    inst = raw.map(make_instruction)

    ##### 3. Tokenize prompts + code separately #####
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    tokenizer.pad_token = tokenizer.eos_token

    def preprocess(ex):
        # Concatenate prompt and code
        full_text = ex["prompt"] + "\n" + ex["code"]
        enc = tokenizer(full_text, truncation=True, padding="max_length", max_length=512)
        
        # Find the split point between prompt and code
        prompt_ids = tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]
        split_idx = len(prompt_ids)
        
        # Mask out the prompt part for loss (-100)
        labels = [-100] * split_idx + enc["input_ids"][split_idx:]
        labels = labels[:512]  # Make sure labels match input length
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels
        }

    tok = inst.map(preprocess, remove_columns=inst.column_names)

    ##### 4. Prepare 4-bit quantization + LoRA configs #####
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj","v_proj"],
        lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.config.pad_token_id = model.config.eos_token_id

    ##### 5. Setup Trainer & Train #####
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=100,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
    )
    response_template = "### Response:\n"
    data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tok,
        processing_class=tokenizer,
        data_collator=data_collator
    )
    trainer.train()

    ##### 6. Save LoRA adapters & tokenizer #####
    model.save_pretrained("./r_model")
    tokenizer.save_pretrained("./r_model")
    print("Training complete; adapters saved to ./r_model")

