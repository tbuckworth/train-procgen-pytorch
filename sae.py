import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Hyperparameters
latent_dim = 256   # Dimension of the latent representation from the RL policy net
hidden_dim = 64     # Dimension of the hidden layer in the SAE
rho = 0.05          # Target sparsity
beta = 1e-3         # Weight of the sparsity penalty term
batch_size = 64
num_epochs = 50
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Suppose X is a NumPy array of shape (N, latent_dim) containing your extracted latent representations
# Replace this with loading your actual data
X = np.random.randn(10000, latent_dim).astype(np.float32)  # Dummy data

dataset = TensorDataset(torch.from_numpy(X))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the Sparse Autoencoder model
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SparseAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            # For an autoencoder, often no activation at the output
            # or something like sigmoid if inputs are normalized between [0,1].
        )

    def forward(self, x):
        hidden = self.encoder(x)
        reconstructed = self.decoder(hidden)
        return reconstructed, hidden

    def kl_divergence(self, rho, rho_hat):
        # KL divergence for sparsity: sum over hidden units
        # rho_hat: average activation of each hidden unit
        # ensure numerical stability with a small epsilon
        eps = 1e-10
        return torch.sum(rho * torch.log((rho + eps) / (rho_hat + eps)) +
                         (1 - rho) * torch.log((1 - rho + eps) / (1 - rho_hat + eps)))

model = SparseAutoencoder(latent_dim, hidden_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop with sparsity penalty
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_data in dataloader:
        data = batch_data[0].to(device)

        optimizer.zero_grad()
        reconstructed, encoded = model(data)
        mse_loss = criterion(reconstructed, data)

        # Compute average activation per neuron
        # hidden shape: (batch_size, hidden_dim)
        rho_hat = torch.mean(encoded, dim=0)  # average over batch
        sparse_loss = kl_divergence(rho, rho_hat)

        loss = mse_loss + beta * sparse_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)

    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# After training, you can use the encoder part for interpretability analysis
model.eval()
encoder = model.encoder

# Encode all X into sparse representations
X_encoded = []
with torch.no_grad():
    for batch_data in dataloader:
        data = batch_data[0].to(device)
        encoded = encoder(data)
        X_encoded.append(encoded.cpu().numpy())

X_encoded = np.concatenate(X_encoded, axis=0)  # shape: (N, hidden_dim)

# Next, you can use these sparse encodings for linear probes or other interpretability tasks.
# For example, if you have labels like actions or presence of a coin in the original state:
Y = np.random.randint(0, 15, size=(X_encoded.shape[0],)) # dummy action labels

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X_encoded, Y, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, Y_train)
print("Validation Accuracy:", clf.score(X_val, Y_val))
