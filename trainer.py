from zenml import step
import sys, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

sys.stdout.reconfigure(encoding="utf-8")

@step
def train():
    ##### 1. Load local JSON as HF Dataset #####
    raw = load_dataset(
        "json", 
        data_files="data/statcodesearch.json", 
        split="train",
    )

    ##### 2. Tokenize into input_ids & attention_mask #####
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    def preprocess(ex):
        text = ex["Comment"].strip() + "\n" + ex["Code"].strip()
        enc = tokenizer(text, truncation=True, padding="max_length", max_length=512)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    tok = raw.map(preprocess, remove_columns=raw.column_names)

    ##### 3. Prepare 4-bit quant + LoRA configs #####
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B",
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    ###### 4. Setup Trainer & train #####
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
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tok,
        tokenizer=tokenizer,
        data_collator=lambda batch: {
            "input_ids": torch.stack([f["input_ids"] for f in batch]),
            "attention_mask": torch.stack([f["attention_mask"] for f in batch]),
            "labels": torch.stack([f["input_ids"] for f in batch]),
        },
    )
    trainer.train()  # 

    ###### 5. Save only LoRA adapters & tokenizer #####
    model.save_pretrained("./r_model")
    tokenizer.save_pretrained("./r_model")
    print("Training complete; adapters saved to ./r_model")
