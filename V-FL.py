"""
APPROACH 4: VERTICAL FEDERATED LEARNING WITH TRANSFER LEARNING
Purpose: Handle vertically partitioned data (e.g., images vs. metadata) with pre-trained models
Complexity: Medium | Privacy: Medium | Accuracy: High
Key Feature: Split learning + transfer from pre-trained ResNet
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet50
import matplotlib.pyplot as plt

# ===========================
# 1. MODEL DEFINITION (SPLIT FOR VERTICAL FL)
# ===========================

class VerticalResNet(nn.Module):
    """Split ResNet: Bottom for features, top for classification. Pre-trained base."""
    def __init__(self, num_classes=14, pretrained=True):
        super(VerticalResNet, self).__init__()
        base = resnet50(pretrained=pretrained)
        self.bottom = nn.Sequential(*list(base.children())[:-2])  # Feature extractor
        self.top = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(base.fc.in_features, num_classes)
        )
    
    def forward_bottom(self, x):
        return self.bottom(x)
    
    def forward_top(self, features):
        return self.top(features)
    
    def forward(self, x):
        features = self.forward_bottom(x)
        return self.forward_top(features)
    
    def get_bottom_weights(self):
        return [p.cpu().detach().numpy() for p in self.bottom.parameters()]
    
    def get_top_weights(self):
        return [p.cpu().detach().numpy() for p in self.top.parameters()]
    
    def set_bottom_weights(self, weights):
        with torch.no_grad():
            for p, w in zip(self.bottom.parameters(), weights):
                p.copy_(torch.from_numpy(w))
    
    def set_top_weights(self, weights):
        with torch.no_grad():
            for p, w in zip(self.top.parameters(), weights):
                p.copy_(torch.from_numpy(w))


# ===========================
# 2. VERTICAL CLIENT (E.G., HOSPITAL WITH PARTIAL DATA)
# ===========================

class VerticalClient:
    def __init__(self, client_id, data_loader, device, is_bottom=True):
        self.client_id = client_id
        self.data_loader = data_loader
        self.device = device
        self.model = VerticalResNet(num_classes=14).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCEWithLogitsLoss()
        self.is_bottom = is_bottom  # Bottom: features, Top: labels/metadata
    
    def train_local(self, epochs=1, remote_features=None):
        self.model.train()
        total_loss = 0
        for epoch in range(epochs):
            for batch in self.data_loader:
                if self.is_bottom:
                    X, _ = batch  # Only images
                    features = self.model.forward_bottom(X.to(self.device))
                    # Simulate sending features to top client
                    # In practice, use secure channel
                    outputs = remote_features if remote_features else features  # Placeholder
                    loss = torch.tensor(0.0)  # Bottom doesn't compute loss
                else:
                    _, y = batch  # Only labels/metadata
                    features = remote_features  # Received from bottom
                    outputs = self.model.forward_top(features.to(self.device))
                    loss = self.criterion(outputs, y.float().to(self.device))
                
                if loss.requires_grad:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                total_loss += loss.item()
        
        return total_loss / len(self.data_loader)


# ===========================
# 3. CENTRAL SERVER FOR VERTICAL AGGREGATION
# ===========================

class VerticalServer:
    def __init__(self, num_classes=14, device='cpu'):
        self.device = device
        self.global_model = VerticalResNet(num_classes).to(device)
        self.clients = []
        self.num_rounds = 0
        self.loss_history = []
    
    def register_client(self, client):
        self.clients.append(client)
    
    def vertical_round(self, epochs=1):
        print(f"\n--- Vertical FL Round {self.num_rounds + 1} ---")
        
        # Assume 2 clients: bottom (images), top (labels)
        bottom_client, top_client = self.clients[0], self.clients[1]
        
        # Distribute global weights
        bottom_weights = self.global_model.get_bottom_weights()
        top_weights = self.global_model.get_top_weights()
        bottom_client.model.set_bottom_weights(bottom_weights)
        top_client.model.set_top_weights(top_weights)
        
        # Local training: Bottom computes features, sends to top
        local_losses = []
        for _ in range(epochs):
            # Simulate data alignment (in practice, use sample IDs)
            for X_batch, y_batch in zip(bottom_client.data_loader, top_client.data_loader):
                features = bottom_client.model.forward_bottom(X_batch[0].to(bottom_client.device))
                # Secure transfer (simulated)
                features = features.detach().requires_grad_(True)  # For backprop
                outputs = top_client.model.forward_top(features.to(top_client.device))
                loss = top_client.criterion(outputs, y_batch[0].float().to(top_client.device))
                loss.backward()
                # Backprop to bottom via gradients
                bottom_client.optimizer.zero_grad()
                top_client.optimizer.zero_grad()
                features_grad = features.grad.clone().to(bottom_client.device)
                bottom_client.model.bottom.backward(features_grad)
                bottom_client.optimizer.step()
                top_client.optimizer.step()
                local_losses.append(loss.item())
        
        # Aggregate (average bottom and top separately)
        agg_bottom = np.mean([c.model.get_bottom_weights() for c in [bottom_client]], axis=0) if bottom_client.is_bottom else None
        agg_top = np.mean([c.model.get_top_weights() for c in [top_client]], axis=0)
        self.global_model.set_bottom_weights(agg_bottom)
        self.global_model.set_top_weights(agg_top)
        
        avg_loss = np.mean(local_losses)
        self.loss_history.append(avg_loss)
        self.num_rounds += 1
        print(f"  Aggregated Loss: {avg_loss:.4f}")
        return avg_loss
    
    def evaluate_global_model(self, test_loader):
        self.global_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = self.global_model(X_batch.to(self.device))
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == y_batch.to(self.device)).sum().item()
                total += y_batch.numel()
        return correct / total


# ===========================
# 4. DATA SIMULATION (VERTICAL SPLIT)
# ===========================

def create_vertical_data(num_clients=2, samples=100):
    """Simulate vertical split: Client1 has images, Client2 has labels"""
    X = np.random.randn(samples, 1, 224, 224).astype(np.float32)  # ResNet input size
    y = np.random.randint(0, 2, size=(samples, 14))
    clients_data = [(torch.FloatTensor(X), None), (None, torch.LongTensor(y))]
    return clients_data


# ===========================
# 5. MAIN EXECUTION
# ===========================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLIENTS = 2
    SAMPLES = 100
    ROUNDS = 5
    EPOCHS = 2
    BATCH_SIZE = 32
    
    server = VerticalServer(device=device)
    data = create_vertical_data(NUM_CLIENTS, SAMPLES)
    
    for i in range(NUM_CLIENTS):
        loader = DataLoader(TensorDataset(data[i][0] if data[i][0] is not None else torch.empty(SAMPLES),
                                          data[i][1] if data[i][1] is not None else torch.empty(SAMPLES, 14)),
                            batch_size=BATCH_SIZE)
        client = VerticalClient(f"Client_{i+1}", loader, device, is_bottom=(i==0))
        server.register_client(client)
    
    for r in range(ROUNDS):
        server.vertical_round(EPOCHS)
    
    # Test
    X_test = np.random.randn(200, 1, 224, 224).astype(np.float32)
    y_test = np.random.randint(0, 2, size=(200, 14))
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)), batch_size=BATCH_SIZE)
    acc = server.evaluate_global_model(test_loader)
    print(f"Test Accuracy: {acc:.4f}")
    
    plt.plot(server.loss_history)
    plt.title('Vertical FL Loss')
    plt.show()

if __name__ == "__main__":
    main()