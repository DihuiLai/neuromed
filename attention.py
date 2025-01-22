import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math

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

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # [1, max_len, embed_dim]

    def forward(self, x):
        # Add positional encoding to embeddings
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)

# Multi-Head Self-Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads."
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        # Linear transformations
        Q = self.query(x)  # [batch_size, seq_len, embed_dim]
        K = self.key(x)    # [batch_size, seq_len, embed_dim]
        V = self.value(x)  # [batch_size, seq_len, embed_dim]

        # Reshape into [batch_size, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale  # [batch_size, num_heads, seq_len, seq_len]
        attention_weights = torch.softmax(attention_scores, dim=-1)           # [batch_size, num_heads, seq_len, seq_len]
        attention_output = torch.matmul(attention_weights, V)                # [batch_size, num_heads, seq_len, head_dim]

        # Concatenate heads and pass through final linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.fc_out(attention_output)  # [batch_size, seq_len, embed_dim]
        return output, attention_weights

# Binary Classifier with Multi-Head Attention
class MultiHeadAttentionBinaryClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_classes, max_len):
        super(MultiHeadAttentionBinaryClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len)
        self.multi_head_attention = MultiHeadAttention(embed_dim, num_heads)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Embedding + Positional Encoding
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        embedded = self.positional_encoding(embedded)

        # Multi-Head Attention
        attention_output, attention_weights = self.multi_head_attention(embedded)

        # Aggregate attention output (mean pooling)
        aggregated_context = attention_output.mean(dim=1)  # [batch_size, embed_dim]

        # Fully Connected Layer
        output = self.fc(aggregated_context)  # [batch_size, num_classes]
        return output, attention_weights

# Hyperparameters
vocab_size = 1000
embed_dim = 64
num_heads = 4
num_classes = 2
seq_len = 50
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# Dataset and DataLoader
dataset = DummyDataset(num_samples=1000, seq_len=seq_len, vocab_size=vocab_size)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model, Loss, Optimizer
model = MultiHeadAttentionBinaryClassifier(vocab_size, embed_dim, num_heads, num_classes, seq_len)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

# Inference Example
model.eval()
sample = torch.randint(0, vocab_size, (1, seq_len))
output, attention_weights = model(sample)
predicted = torch.argmax(output, dim=1)
print(f"Predicted Class: {predicted.item()}, Attention Weights Shape: {attention_weights.shape}")
