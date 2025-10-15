"""
APPROACH 3: FEDERATED LEARNING WITH DIFFERENTIAL PRIVACY & SECURE AGGREGATION
Purpose: Privacy-preserving federated learning for HIPAA/GDPR compliance
Complexity: High | Privacy: Excellent | Accuracy: Good
Key Feature: Differential privacy + gradient clipping + noise addition
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
import matplotlib.pyplot as plt

# ===========================
# 1. DIFFERENTIAL PRIVACY UTILITIES
# ===========================

class DifferentialPrivacyEngine:
    """
    Implements ε-δ differential privacy
    epsilon: privacy budget (smaller = more private)
    delta: failure probability
    """
    
    def __init__(self, epsilon=1.0, delta=1e-5, max_grad_norm=1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.sigma = self._calculate_noise_scale()
    
    def _calculate_noise_scale(self):
        """
        Calculate noise scale using moment accountant method
        σ = sqrt(2 * ln(1.25/δ)) / ε
        """
        return np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
    
    def clip_gradients(self, model, max_norm=1.0):
        """Clip gradients to prevent information leakage"""
        total_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        
        total_norm = np.sqrt(total_norm)
        clip_coef = max_norm / (total_norm + 1e-8)
        
        if clip_coef < 1:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
        
        return total_norm
    
    def add_noise_to_gradients(self, model):
        """Add Gaussian noise to gradients"""
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * self.sigma
                param.grad.data.add_(noise)
    
    def add_noise_to_weights(self, weights, scale=None):
        """Add noise to model weights during aggregation"""
        if scale is None:
            scale = self.sigma
        
        noisy_weights = []
        for weight in weights:
            noise = np.random.normal(0, scale, size=weight.shape)
            noisy_weights.append(weight + noise)
        
        return noisy_weights


# ===========================
# 2. MODEL DEFINITION
# ===========================

class ResNet50_DP(nn.Module):
    """ResNet50 with DP-compatible design"""
    def __init__(self, num_classes=14):
        super(ResNet50_DP, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def get_weights(self):
        return [param.cpu().detach().numpy().copy() for param in self.parameters()]
    
    def set_weights(self, weights):
        with torch.no_grad():
            for param, weight in zip(self.parameters(), weights):
                param.copy_(torch.from_numpy(weight).float())


# ===========================
# 3. SECURE AGGREGATION
# ===========================

class SecureAggregator:
    """
    Simulates secure multi-party computation (SMPC)
    In production: use PySyft, TensorFlow Privacy, or CrypTen
    """
    
    def __init__(self, dp_engine):
        self.dp_engine = dp_engine
        self.aggregation_count = 0
    
    def aggregate_with_dp(self, weights_list):
        """
        Secure aggregation with differential privacy:
        1. Add noise to each client's weights
        2. Average noisy weights
        3. Apply secure aggregation (simulated)
        """
        noisy_weights_list = []
        for weights in weights_list:
            noisy = self.dp_engine.add_noise_to_weights(weights)
            noisy_weights_list.append(noisy)
        
        # Standard average (in production: use cryptographic protocols)
        aggregated = []
        num_params = len(noisy_weights_list[0])
        
        for param_idx in range(num_params):
            param_values = np.array([w[param_idx] for w in noisy_weights_list])
            aggregated.append(np.mean(param_values, axis=0))
        
        self.aggregation_count += 1
        return aggregated
    
    def get_privacy_budget_spent(self):
        """Compute cumulative privacy cost"""
        # Simple accounting: epsilon spent per round
        return self.aggregation_count * self.dp_engine.epsilon


# ===========================
# 4. HOSPITAL CLIENT WITH DP
# ===========================

class HospitalClient_DP:
    """Hospital with differential privacy"""
    
    def __init__(self, client_id, train_loader, device, dp_engine):
        self.client_id = client_id
        self.train_loader = train_loader
        self.device = device
        self.dp_engine = dp_engine
        
        self.model = ResNet50_DP(num_classes=14).to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.BCEWithLogitsLoss()
        
        self.privacy_log = []
    
    def train_epoch_with_dp(self, epochs=1):
        """Train with gradient clipping and noise"""
        self.model.train()
        total_loss = 0
        batches = 0
        
        for epoch in range(epochs):
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch.float())
                loss.backward()
                
                # DP Step 1: Clip gradients
                clipped_norm = self.dp_engine.clip_gradients(
                    self.model, 
                    max_norm=self.dp_engine.max_grad_norm
                )
                
                # DP Step 2: Add noise to gradients
                self.dp_engine.add_noise_to_gradients(self.model)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                batches += 1
                
                self.privacy_log.append({
                    'epoch': epoch,
                    'batch': batches,
                    'loss': loss.item(),
                    'clipped_norm': clipped_norm
                })
        
        return total_loss / batches if batches > 0 else 0
    
    def get_weights(self):
        return self.model.get_weights()
    
    def set_weights(self, weights):
        self.model.set_weights(weights)


# ===========================
# 5. CENTRAL SERVER WITH DP
# ===========================

class CentralServer_DP:
    """Privacy-preserving federated server"""
    
    def __init__(self, num_classes=14, device='cpu', epsilon=1.0, delta=1e-5):
        self.device = device
        self.global_model = ResNet50_DP(num_classes=num_classes).to(device)
        self.clients = []
        
        # Privacy configuration
        self.dp_engine = DifferentialPrivacyEngine(epsilon=epsilon, delta=delta)
        self.secure_aggregator = SecureAggregator(self.dp_engine)
        
        self.num_rounds = 0
        self.loss_history = []
        self.privacy_budget_history = []
    
    def register_client(self, client):
        self.clients.append(client)
    
    def dp_federated_round(self, epochs=1):
        """Execute one DP-FL round"""
        print(f"\n--- DP-FL Round {self.num_rounds + 1} ---")
        print(f"   Privacy Budget (ε, δ): ({self.dp_engine.epsilon}, {self.dp_engine.delta})")
        print(f"   Noise Scale (σ): {self.dp_engine.sigma:.6f}")
        
        # Distribute global model
        global_weights = self.global_model.get_weights()
        for client in self.clients:
            client.set_weights(global_weights)
        
        # Local training with DP
        local_losses = []
        for client in self.clients:
            loss = client.train_epoch_with_dp(epochs=epochs)
            local_losses.append(loss)
            print(f"  {client.client_id}: Loss = {loss:.4f} (with DP)")
        
        # Secure aggregation
        clients_weights = [client.get_weights() for client in self.clients]
        aggregated_weights = self.secure_aggregator.aggregate_with_dp(clients_weights)
        self.global_model.set_weights(aggregated_weights)
        
        avg_loss = np.mean(local_losses)
        privacy_spent = self.secure_aggregator.get_privacy_budget_spent()
        
        self.loss_history.append(avg_loss)
        self.privacy_budget_history.append(privacy_spent)
        self.num_rounds += 1
        
        print(f"  Aggregated Loss: {avg_loss:.4f}")
        print(f"  Total Privacy Spent: ε = {privacy_spent:.4f}")
        
        return avg_loss
    
    def evaluate_global_model(self, test_loader):
        """Evaluate without noise"""
        self.global_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.global_model(X_batch)
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predictions == y_batch.to(self.device)).sum().item()
                total += y_batch.numel()
        
        return correct / total if total > 0 else 0


# ===========================
# 6. MAIN EXECUTION
# ===========================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    NUM_CLIENTS = 3
    SAMPLES_PER_CLIENT = 100
    FEDERATED_ROUNDS = 4
    LOCAL_EPOCHS = 2
    BATCH_SIZE = 32
    EPSILON = 2.0  # Privacy budget
    DELTA = 1e-5
    
    print("\n[SETUP] Creating DP-FL environment...")
    print(f"Privacy Configuration: ε={EPSILON}, δ={DELTA}")
    
    server = CentralServer_DP(num_classes=14, device=device, 
                              epsilon=EPSILON, delta=DELTA)
    
    # Create clients with DP
    for hospital_id in range(NUM_CLIENTS):
        X = np.random.randn(SAMPLES_PER_CLIENT, 1, 128, 128).astype(np.float32)
        y = np.random.randint(0, 2, size=(SAMPLES_PER_CLIENT, 14))
        
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        dp_client = DifferentialPrivacyEngine(epsilon=EPSILON, delta=DELTA)
        client = HospitalClient_DP(
            client_id=f"Hospital_{hospital_id+1}",
            train_loader=loader,
            device=device,
            dp_engine=dp_client
        )
        server.register_client(client)
        print(f"  Registered {client.client_id} (DP-enabled)")
    
    # Test data
    X_test = np.random.randn(200, 1, 128, 128).astype(np.float32)
    y_test = np.random.randint(0, 2, size=(200, 14))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    print("\n[TRAINING] Starting Differentially Private Federated Learning...")
    
    for round_num in range(FEDERATED_ROUNDS):
        server.dp_federated_round(epochs=LOCAL_EPOCHS)
    
    print("\n[EVALUATION] Final Results...")
    accuracy = server.evaluate_global_model(test_loader)
    print(f"  Test Accuracy: {accuracy:.4f}")
    print(f"  Total Privacy Budget Used: ε={server.privacy_budget_history[-1]:.4f}")
    from fl_model_saving import save_dpfl_model
    save_dpfl_model(server)
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(server.loss_history, marker='o', linewidth=2, label='DP-FL Loss')
    axes[0].set_xlabel('Federated Round')
    axes[0].set_ylabel('Average Loss')
    axes[0].set_title('DP-FL: Training Loss Over Rounds')
    axes[0].grid(True)
    axes[0].legend()
    
    axes[1].plot(server.privacy_budget_history, marker='s', linewidth=2, 
                 color='red', label='Privacy Spent (ε)')
    axes[1].axhline(y=EPSILON, color='orange', linestyle='--', 
                    label=f'Initial Budget (ε={EPSILON})')
    axes[1].set_xlabel('Federated Round')
    axes[1].set_ylabel('Cumulative Privacy Budget')
    axes[1].set_title('Privacy Budget Consumption')
    axes[1].grid(True)
    axes[1].legend()
    
    plt.tight_layout()
