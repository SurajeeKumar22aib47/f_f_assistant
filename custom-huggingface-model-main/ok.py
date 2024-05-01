from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Load pre-trained model and tokenizer
model_name = "gpt2-large"  # Using a larger model variant
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Load your domain-specific dataset
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=r"C:\Users\Success\Desktop\custom-huggingface-model-main\custom-huggingface-model-main\boyQA.txt",
    block_size=128,
)

# Fine-tune the model
training_args = TrainingArguments(
    output_dir="output_directory",
    overwrite_output_dir=True,
    num_train_epochs=10,  # Train for more epochs
    per_device_train_batch_size=2,  # Decrease batch size
    save_steps=5_000,  # Save more frequently
    save_total_limit=5,  # Save more checkpoints
    disable_tqdm=True,  # Disable progress bars for faster training
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # MLM is set to False for language modeling
    ),
    train_dataset=dataset,
)

trainer.train()

# Save the fine-tuned model
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")
