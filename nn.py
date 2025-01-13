import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

# Example texts and labels (multi-label format)
texts = ["this is a document", "this is another document", "classification example text"]
labels = [["class1"], ["class2"], ["class1", "class3"]]

# Multi-label binarizer
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

# Vocabulary and dataset
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
        return torch.tensor(padded_text, dtype=torch.long), torch.tensor(label, dtype=torch.float)

vocab = {word: idx for idx, word in enumerate({"<PAD>", "<UNK>"} | set(" ".join(texts).split()))}
max_len = 10
dataset = TextDataset(texts, labels, vocab, max_len)
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Define the model
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
num_classes = labels.shape[1]
model = DocumentClassifier(vocab_size, embed_dim, num_classes)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)  # Raw logits
        loss = criterion(outputs, targets)  # Multi-label loss
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Save the model
torch.save(model.state_dict(), "multi_label_document_classifier.pth")
