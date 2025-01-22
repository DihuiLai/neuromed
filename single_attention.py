import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Define Dataset
class DummyDataset(Dataset):
    def __init__(self, num_samples, seq_len, vocab_size):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        data = torch.randint(0, self.vocab_size, (self.seq_len,))
        label = torch.randint(0, 2, (1,)).item()  # Binary labels: 0 or 1
        return data, label

# Self-Attention Layer
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = torch.sqrt(torch.FloatTensor([embed_dim]))

    def forward(self, x):
        # x: [batch_size, seq_len, embed_dim]
        Q = self.query(x)  # [batch_size, seq_len, embed_dim]
        K = self.key(x)    # [batch_size, seq_len, embed_dim]
        V = self.value(x)  # [batch_size, seq_len, embed_dim]

        # Scaled Dot-Product Attention
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale  # [batch_size, seq_len, seq_len]
        attention_weights = torch.softmax(attention_scores, dim=-1)           # [batch_size, seq_len, seq_len]
        context = torch.matmul(attention_weights, V)                          # [batch_size, seq_len, embed_dim]

        return context, attention_weights

# Binary Classifier with Self-Attention
class SelfAttentionBinaryClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_classes):
        super(SelfAttentionBinaryClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.self_attention = SelfAttention(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Embedding Layer
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]

        # Self-Attention Layer
        context, attention_weights = self.self_attention(embedded)  # [batch_size, seq_len, embed_dim]

        # Aggregate context (mean pooling)
        aggregated_context = context.mean(dim=1)  # [batch_size, embed_dim]

        # Fully Connected Layer
        output = self.fc(aggregated_context)  # [batch_size, num_classes]
        return output, attention_weights

# Hyperparameters
vocab_size = 1000
embed_dim = 64
num_classes = 2
seq_len = 20
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# Dataset and DataLoader
dataset = DummyDataset(num_samples=1000, seq_len=seq_len, vocab_size=vocab_size)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model, Loss, Optimizer
model = SelfAttentionBinaryClassifier(vocab_size, embed_dim, num_heads=1, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs, _ = model(inputs)  # [batch_size, num_classes]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

# Inference Example
model.eval()
sample = torch.randint(0, vocab_size, (1, seq_len))  # Random sample input
output, attention_weights = model(sample)
predicted = torch.argmax(output, dim=1)
print(f"Predicted Class: {predicted.item()}, Attention Weights Shape: {attention_weights.shape}")
