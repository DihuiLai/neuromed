import torch
import torch.nn as nn
import torch.optim as optim

# Custom Attention Encoder from earlier
class AttentionEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, ff_hidden_dim, max_seq_len, dropout=0.1):
        super(AttentionEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.position_encoding = self._generate_position_encoding(max_seq_len, embed_dim)

        self.multihead_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim)
        )

        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _generate_position_encoding(self, max_seq_len, embed_dim):
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Shape: (1, max_seq_len, embed_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len = x.size()
        x = self.embedding(x) * math.sqrt(self.embed_dim)
        x = x + self.position_encoding[:, :seq_len, :].to(x.device)

        x_norm = self.layernorm1(x)
        x_attention, _ = self.multihead_attention(x_norm.transpose(0, 1), x_norm.transpose(0, 1), x_norm.transpose(0, 1), attn_mask=mask)
        x_attention = x_attention.transpose(0, 1)
        x = x + self.dropout(x_attention)

        x_norm = self.layernorm2(x)
        x_ff = self.feed_forward(x_norm)
        x = x + self.dropout(x_ff)

        return x

# Binary Classification Model
class AttentionBinaryClassifier(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, ff_hidden_dim, max_seq_len, dropout=0.1):
        super(AttentionBinaryClassifier, self).__init__()
        self.encoder = AttentionEncoder(input_dim, embed_dim, num_heads, ff_hidden_dim, max_seq_len, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()  # Binary classification output
        )

    def forward(self, x, mask=None):
        # Encoder output
        encoded = self.encoder(x, mask)  # Shape: (batch_size, seq_len, embed_dim)
        # Pooling: Take the [CLS] token (first token) representation
        cls_representation = encoded[:, 0, :]  # Shape: (batch_size, embed_dim)
        # Classification
        logits = self.classifier(cls_representation)
        return logits

# Example Usage
if __name__ == "__main__":
    # Parameters
    input_dim = 1000  # Vocabulary size
    embed_dim = 64    # Embedding dimension
    num_heads = 4     # Number of attention heads
    ff_hidden_dim = 128  # Feed-forward hidden layer dimension
    max_seq_len = 50  # Maximum sequence length
    batch_size = 32   # Batch size

    # Dummy Data
    x_data = torch.randint(0, input_dim, (batch_size, max_seq_len))  # Random input sequences
    y_data = torch.randint(0, 2, (batch_size, 1)).float()  # Random binary labels

    # Model
    model = AttentionBinaryClassifier(input_dim, embed_dim, num_heads, ff_hidden_dim, max_seq_len)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Loss and Optimizer
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    model.train()
    for epoch in range(5):  # Train for 5 epochs
        optimizer.zero_grad()
        outputs = model(x_data).squeeze()  # Shape: (batch_size,)
        loss = criterion(outputs, y_data.squeeze())
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # Testing
    model.eval()
    with torch.no_grad():
        test_data = torch.randint(0, input_dim, (5, max_seq_len))  # 5 random test sequences
        predictions = model(test_data)
        print("Predictions:", predictions)
