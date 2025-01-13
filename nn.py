import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 1. Prepare your dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        text_indices = [self.vocab.get(word, self.vocab["<UNK>"]) for word in text.split()]
        padded_text = text_indices[:self.max_len] + [0] * (self.max_len - len(text_indices))
        return torch.tensor(padded_text, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# Example texts and labels
texts = ["this is a document", "this is another document", "classification example text"]
labels = ["class1", "class2", "class1"]

# Label encoding
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Vocabulary and dataset
vocab = {word: idx for idx, word in enumerate({"<PAD>", "<UNK>"} | set(" ".join(texts).split()))}
max_len = 10
dataset = TextDataset(texts, labels, vocab, max_len)

# Split dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(dataset.texts, dataset.labels, test_size=0.2)
train_dataset = TextDataset(train_texts, train_labels, vocab, max_len)
val_dataset = TextDataset(val_texts, val_labels, vocab, max_len)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# 2. Define the model
class DocumentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(DocumentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)  # [batch_size, max_len, embed_dim]
        pooled = embedded.mean(dim=1)  # Average over the sequence
        output = self.fc(pooled)  # [batch_size, num_classes]
        return output

# Model parameters
vocab_size = len(vocab)
embed_dim = 50
num_classes = len(set(labels))
model = DocumentClassifier(vocab_size, embed_dim, num_classes)

# 3. Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Train the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Validation loop
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Accuracy: {correct/total:.4f}")

# 5. Save the model
torch.save(model.state_dict(), "document_classifier.pth")
