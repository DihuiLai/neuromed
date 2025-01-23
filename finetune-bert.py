import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
import pandas as pd

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Example Chinese Dataset (Replace this with your dataset)
data = {
    "text": [
        "这个产品非常好，我很喜欢！",
        "非常差劲，不推荐购买。",
        "质量还可以，但是服务不好。",
        "超级棒，物超所值！",
        "不满意，完全不符合预期。"
    ],
    "label": [1, 0, 1, 1, 0],  # 1: Positive, 0: Negative
}
df = pd.DataFrame(data)

# Split the data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# Load the Chinese BERT tokenizer
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)

# Tokenize the data
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,  # Max sequence length
    )

# Convert data into Hugging Face Dataset format
train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})

# Apply tokenization
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Remove the "text" column and set format for PyTorch
train_dataset = train_dataset.remove_columns(["text"]).with_format("torch")
val_dataset = val_dataset.remove_columns(["text"]).with_format("torch")

# Load the pre-trained BERT model for classification
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",             # Directory for model checkpoints
    evaluation_strategy="epoch",       # Evaluate at the end of each epoch
    save_strategy="epoch",             # Save model checkpoints at the end of each epoch
    learning_rate=5e-5,                # Learning rate
    per_device_train_batch_size=8,     # Batch size for training
    per_device_eval_batch_size=8,      # Batch size for evaluation
    num_train_epochs=3,                # Number of epochs
    weight_decay=0.01,                 # Weight decay for AdamW optimizer
    logging_dir="./logs",              # Directory for logs
    logging_steps=10,                  # Log every 10 steps
    load_best_model_at_end=True,       # Load the best model at the end of training
    metric_for_best_model="accuracy",  # Metric for selecting the best model
    save_total_limit=2                 # Limit the number of checkpoints
)

# Define accuracy as a metric
from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(f"Validation Results: {results}")

# Save the fine-tuned model
model.save_pretrained("./finetuned-bert-chinese")
tokenizer.save_pretrained("./finetuned-bert-chinese")
