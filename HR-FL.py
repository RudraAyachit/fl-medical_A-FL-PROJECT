"""
APPROACH 1: HORIZONTAL FEDERATED LEARNING WITH FEDAVG (BASIC)
Purpose: Classic federated averaging across hospital clients
Complexity: Low | Privacy: Basic | Accuracy: Medium
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
import matplotlib.pyplot as plt

# ===========================
# 1. MODEL DEFINITION
# ===========================

class ResNet50_FedAvg(nn.Module):
    """ResNet50 simplified for chest X-ray classification"""
    def __init__(self, num_classes=14):
        super(ResNet50_FedAvg, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Simplified ResNet blocks
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
        """Extract model weights as numpy arrays"""
        return [param.cpu().detach().numpy() for param in self.parameters()]
    
    def set_weights(self, weights):
        """Set model weights from numpy arrays"""
        with torch.no_grad():
            for param, weight in zip(self.parameters(), weights):
                param.copy_(torch.from_numpy(weight).float())


# ===========================
# 2. LOCAL HOSPITAL CLIENT
# ===========================

class HospitalClient:
    """Simulates a single hospital with local data"""
    
    def __init__(self, client_id, train_loader, device):
        self.client_id = client_id
        self.train_loader = train_loader
        self.device = device
        self.model = ResNet50_FedAvg(num_classes=14).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCEWithLogitsLoss()
    
    def train_epoch(self, epochs=1):
        """Train local model"""
        self.model.train()
        total_loss = 0
        for epoch in range(epochs):
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch.float())
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def get_weights(self):
        return self.model.get_weights()
    
    def set_weights(self, global_weights):
        self.model.set_weights(global_weights)


# ===========================
# 3. CENTRAL SERVER (ORCHESTRATOR)
# ===========================

class CentralServer:
    """Central server for FedAvg aggregation"""
    
    def __init__(self, num_classes=14, device='cpu'):
        self.device = device
        self.global_model = ResNet50_FedAvg(num_classes=num_classes).to(device)
        self.clients = []
        self.num_rounds = 0
        self.loss_history = []
    
    def register_client(self, client):
        """Register a hospital client"""
        self.clients.append(client)
    
    def aggregate_weights(self, weights_list):
        """FedAvg aggregation: average weights from all clients"""
        # Stack weights
        aggregated_weights = []
        num_params = len(weights_list[0])
        
        for param_idx in range(num_params):
            # Average across all clients
            param_values = np.array([weights[param_idx] for weights in weights_list])
            aggregated_param = np.mean(param_values, axis=0)
            aggregated_weights.append(aggregated_param)
        
        return aggregated_weights
    
    def federated_round(self, epochs=1):
        """Execute one round of federated learning"""
        print(f"\n--- Federated Round {self.num_rounds + 1} ---")
        
        # Step 1: Distribute global weights to all clients
        global_weights = self.global_model.get_weights()
        for client in self.clients:
            client.set_weights(global_weights)
        
        # Step 2: Local training on each hospital
        local_losses = []
        for client in self.clients:
            loss = client.train_epoch(epochs=epochs)
            local_losses.append(loss)
            print(f"  Hospital {client.client_id}: Loss = {loss:.4f}")
        
        # Step 3: Collect weights from all clients
        clients_weights = [client.get_weights() for client in self.clients]
        
        # Step 4: Aggregate weights on central server
        aggregated_weights = self.aggregate_weights(clients_weights)
        self.global_model.set_weights(aggregated_weights)
        
        avg_loss = np.mean(local_losses)
        self.loss_history.append(avg_loss)
        self.num_rounds += 1
        
        print(f"  Aggregated Loss: {avg_loss:.4f}")
        return avg_loss
    
    def evaluate_global_model(self, test_loader):
        """Evaluate global model on test set"""
        self.global_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.global_model(X_batch)
                # For multilabel, use threshold of 0.5
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predictions == y_batch.to(self.device)).sum().item()
                total += y_batch.numel()
        
        accuracy = correct / total if total > 0 else 0
        return accuracy


# ===========================
# 4. DATA SIMULATION
# ===========================

def create_federated_data(num_clients=3, samples_per_client=100, 
                         num_features=8192, num_classes=14):
    """
    Simulate distributed hospital data
    In practice, this would come from multiple hospital systems
    """
    clients_data = []
    
    for client_id in range(num_clients):
        # Simulate X-ray images (flattened to 8192 features)
        # In practice, use CheXpert or NIH Chest X-ray dataset
        X = np.random.randn(samples_per_client, 1, 128, 128).astype(np.float32)
        y = np.random.randint(0, 2, size=(samples_per_client, num_classes))
        
        # Create some non-IID (heterogeneous) distribution
        # Each hospital specializes in different diseases
        y[:, client_id * 4:(client_id + 1) * 4] = np.random.randint(0, 2, 
                                                   (samples_per_client, 4))
        
        clients_data.append((X, y))
    
    return clients_data


# ===========================
# 5. MAIN EXECUTION
# ===========================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    NUM_CLIENTS = 3  # 3 hospitals
    SAMPLES_PER_CLIENT = 100
    FEDERATED_ROUNDS = 5
    LOCAL_EPOCHS = 2
    BATCH_SIZE = 32
    
    # ============ SETUP ============
    print("\n[SETUP] Creating federated learning environment...")
    
    # Create server
    server = CentralServer(num_classes=14, device=device)
    
    # Create hospital clients with local data
    clients_data = create_federated_data(num_clients=NUM_CLIENTS, 
                                         samples_per_client=SAMPLES_PER_CLIENT)
    
    hospitals = []
    for hospital_id, (X, y) in enumerate(clients_data):
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        client = HospitalClient(
            client_id=f"Hospital_{hospital_id+1}",
            train_loader=loader,
            device=device
        )
        server.register_client(client)
        hospitals.append(client)
        print(f"  Registered Hospital {hospital_id+1} with {len(X)} samples")
    
    # Create test data (simulated)
    X_test = np.random.randn(200, 1, 128, 128).astype(np.float32)
    y_test = np.random.randint(0, 2, size=(200, 14))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), 
                                 torch.LongTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # ============ TRAINING ============
    print("\n[TRAINING] Starting Federated Learning with FedAvg...")
    print(f"  Rounds: {FEDERATED_ROUNDS}")
    print(f"  Local Epochs: {LOCAL_EPOCHS}")
    print(f"  Number of Hospitals: {NUM_CLIENTS}\n")
    
    for round_num in range(FEDERATED_ROUNDS):
        server.federated_round(epochs=LOCAL_EPOCHS)
    
    # ============ EVALUATION ============
    print("\n[EVALUATION] Evaluating Global Model...")
    accuracy = server.evaluate_global_model(test_loader)
    print(f"  Test Accuracy: {accuracy:.4f}")
    
    # ============ VISUALIZATION ============
    plt.figure(figsize=(10, 5))
    plt.plot(server.loss_history, marker='o', linewidth=2)
    plt.xlabel('Federated Round')
    plt.ylabel('Average Loss')
    plt.title('FedAvg: Training Loss Over Rounds')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('fedavg_training_loss.png', dpi=150)
    print("\n[OUTPUT] Training loss plot saved as 'fedavg_training_loss.png'")
    
    # ============ SUMMARY ============
    print("\n" + "="*50)
    print("APPROACH 1: FEDAVG - SUMMARY")
    print("="*50)
    print(f"Total Federated Rounds: {server.num_rounds}")
    print(f"Total Hospitals: {NUM_CLIENTS}")
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print(f"Final Training Loss: {server.loss_history[-1]:.4f}")
    print("\nStrengths:")
    print("  ✓ Simple implementation")
    print("  ✓ Low communication overhead")
    print("  ✓ Baseline for federated learning")
    print("\nLimitations:")
    print("  ✗ Struggles with non-IID data")
    print("  ✗ No uncertainty quantification")
    print("  ✗ Synchronous communication bottleneck")


if __name__ == "__main__":
    main()