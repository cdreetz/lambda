# Script 2: train_model.py

from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    AutoConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from datasets import DatasetDict, load_dataset
import wandb

def tokenize(element):
    outputs = tokenizer(
        element["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


if __name__ == "__main__":
    context_length = 128
    tokenizer = AutoTokenizer.from_pretrained(
        'huggingface-course/code-search-net-tokenizer'
    )
    ds_train = load_dataset(
        "huggingface-course/codeparrot-ds-train", split="train"
    )
    ds_valid = load_dataset(
        "huggingface-course/codeparrot-ds-valid", split="validation"
    )
    raw_datasets = DatasetDict(
        {
            "train": ds_train,
            "valid": ds_valid
        }
    )
    tokenized_datasets = raw_datasets.map(
        tokenize, batched=True,
        remove_columns=raw_datasets["train"].column_names
    )
    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        n_ctx=context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(config)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    out = data_collator([tokenized_datasets["train"][i] for i in range(5)])
    for key in out:
        print(f"{key} shape: {out[key].shape}")

    # Initialize Wandb
    wandb.init(
        project="my-awesome-project",
        config={
            "learning_rate": 5e-4,
            "architecture": "GPT2",
            "dataset": "codeparrot-ds",
            "epochs": 1,
        }
    )
        
    args = TrainingArguments(
        output_dir="codeparrot-ds2",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        evaluation_strategy="steps",
        eval_steps=500,
        logging_steps=500,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        weight_decay=0.1,
        warmup_steps=200,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=500,
        fp16=True,
        push_to_hub=True,
        report_to="wandb"  # ensure the metrics are reported to wandb
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"]
    )
    
    # Start training
    trainer.train()
