import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# offline should be since hpc has no net

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
# output dirs
# Base model: loaded from a local HuggingFace cache on the HPC cluster; on a
# networked machine use the hub id directly ("openai/gpt-oss-20b").
MODEL_PATH = os.environ.get("GPTOSS_MODEL_PATH", "openai/gpt-oss-20b")
# Balanced 18k training set (Stage 2; regenerable; not shipped). Default points at
# <repo>/data/; override with FINETUNE_TRAIN if staged elsewhere.
DATA_PATH = os.environ.get(
    "FINETUNE_TRAIN",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data",
        "data_balanced_18k.jsonl",
    ),
)
OUTPUT_DIR = "./2026_consis_output_fixed"

SAVE_DIR = "./2026_consis_trained_fixed"
MAX_LENGTH = 256


dataset = load_dataset("json", data_files=DATA_PATH, split="train")
print(f"Loaded {len(dataset)} samples")


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


config = AutoConfig.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True,
)
if hasattr(config, "quantization_config"):
    delattr(config, "quantization_config")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    config=config,
    local_files_only=True,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    attn_implementation="eager",
    low_cpu_mem_usage=True,
)

# lora config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


def tokenize_function(examples):

    all_input_ids = []
    all_attention_mask = []
    all_labels = []

    for instruction, input_text, output in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):

        prompt = f"{instruction}\n\n{input_text}\n\n"

        full_text = prompt + output

        prompt_tokens = tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        prompt_length = len(prompt_tokens["input_ids"])

        full_tokens = tokenizer(
            full_text,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )

        input_ids = full_tokens["input_ids"]
        attention_mask = full_tokens["attention_mask"]

        labels = [-100] * len(input_ids)

        for i in range(prompt_length, len(input_ids)):
            if attention_mask[i] == 1:

                labels[i] = input_ids[i]

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_labels.append(labels)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    }


print("\n" + "=" * 60)
print("VERIFYING TOKENIZATION")
print("=" * 60)


test_example = {
    "instruction": ["Classify the following text as 'toxic' or 'non-toxic'."],
    "input": ["You are an idiot!"],
    "output": ["toxic"],
}
test_result = tokenize_function(test_example)


print("\nExample tokenization:")

print(f"Full text tokens: {len(test_result['input_ids'][0])}")


labels = test_result["labels"][0]
first_label_idx = next((i for i, l in enumerate(labels) if l != -100), -1)
print(f"First output token index: {first_label_idx}")


if first_label_idx != -1:
    output_tokens = [l for l in labels if l != -100]
    decoded_output = tokenizer.decode(output_tokens)
    print(f"Output tokens the model will learn: '{decoded_output}'")


masked_tokens = test_result["input_ids"][0][:first_label_idx]
decoded_masked = tokenizer.decode(masked_tokens)
print(f"Masked (prompt) tokens: '{decoded_masked[:100]}...'")

print("=" * 60 + "\n")


tokenized_dataset = dataset.map(
    tokenize_function, batched=True, remove_columns=dataset.column_names, desc="Tokenizing dataset"
)

print(f"Tokenized {len(tokenized_dataset)} samples")


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    gradient_checkpointing=False,
    logging_steps=50,
    num_train_epochs=3,
    learning_rate=3e-5,
    fp16=True,
    save_total_limit=2,
    save_strategy="steps",
    save_steps=200,
    optim="adamw_torch",
    max_grad_norm=1.0,
    warmup_ratio=0.1,
    push_to_hub=False,
    report_to="none",
    dataloader_num_workers=0,
    lr_scheduler_type="cosine",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)
print(f"Dataset size: {len(tokenized_dataset)}")

print(f"Batch size: {training_args.per_device_train_batch_size}")

print(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")

print(
    f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}"
)


print(f"Epochs: {training_args.num_train_epochs}")
print(f"Learning rate: {training_args.learning_rate}")
print("=" * 60 + "\n")

trainer.train()


model.save_pretrained(SAVE_DIR)

tokenizer.save_pretrained(SAVE_DIR)

print("\n" + "=" * 60)
print(f"Training complete! Model saved to {SAVE_DIR}")
print("=" * 60)
