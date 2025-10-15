"""
APPROACH 2: FEDPROX - ENHANCED FEDERATED LEARNING WITH HETEROGENEOUS DATA
Purpose: Handle non-IID data across hospitals with regularization
Complexity: Low-Medium | Privacy: Basic | Accuracy: Excellent
Key Feature: Proximity term to handle statistical heterogeneity
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

class ResNet50_FedProx(nn.Module):
    """ResNet50 for chest X-ray with FedProx support"""
    def __init__(self, num_classes=14):
        super(ResNet50_FedProx, self).__init__()
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
# 2. FEDPROX LOSS WITH PROXIMITY TERM
# ===========================

class FedProxLoss(nn.Module):
    """
    FedProx loss = Original Loss + (mu/2) * ||w - w_global||^2
    Penalty term keeps local model close to global model
    """
    def __init__(self, global_model, mu=0.01):
        super(FedProxLoss, self).__init__()
        self.global_model = global_model
        self.mu = mu
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, model, output, target):
        loss = self.criterion(output, target.float())
        
        prox_term = 0
        for param, global_param in zip(model.parameters(), 
                                       self.global_model.parameters()):
            prox_term += torch.sum(torch.pow(param - global_param, 2))
        
        total_loss = loss + (self.mu / 2) * prox_term
        return total_loss


# ===========================
# 3. HOSPITAL CLIENT WITH FEDPROX
# ===========================

class HospitalClient_FedProx:
    """Hospital with FedProx local training"""
    
    def __init__(self, client_id, train_loader, device, global_model, mu=0.01):
        self.client_id = client_id
        self.train_loader = train_loader
        self.device = device
        self.mu = mu
        self.model = ResNet50_FedProx(num_classes=14).to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.criterion = FedProxLoss(global_model=global_model, mu=mu)
    
    def train_epoch(self, global_model, epochs=1):
        """Train with FedProx: minimize (loss + proximity term)"""
        self.model.train()
        self.criterion.global_model = global_model
        
        total_loss = 0
        batches = 0
        
        for epoch in range(epochs):
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(self.model, outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                batches += 1
        
        return total_loss / batches if batches > 0 else 0
    
    def get_weights(self):
        return self.model.get_weights()
    
    def set_weights(self, weights):
        self.model.set_weights(weights)


# ===========================
# 4. CENTRAL SERVER WITH FEDPROX
# ===========================

class CentralServer_FedProx:
    """Central server orchestrating FedProx"""
    
    def __init__(self, num_classes=14, device='cpu', mu=0.01):
        self.device = device
        self.global_model = ResNet50_FedProx(num_classes=num_classes).to(device)
        self.clients = []
        self.mu = mu
        self.num_rounds = 0
        self.loss_history = []
        self.convergence_metrics = {
            'avg_client_loss': [],
            'weight_divergence': []
        }
    
    def register_client(self, client):
        self.clients.append(client)
    
    def aggregate_weights_weighted(self, weights_list, sample_counts=None):
        """Weighted average based on number of samples"""
        if sample_counts is None:
            sample_counts = np.ones(len(weights_list))
        
        total_samples = np.sum(sample_counts)
        aggregated_weights = []
        
        num_params = len(weights_list[0])
        for param_idx in range(num_params):
            weighted_param = np.zeros_like(weights_list[0][param_idx])
            
            for client_idx, weights in enumerate(weights_list):
                weight = sample_counts[client_idx] / total_samples
                weighted_param += weight * weights[param_idx]
            
            aggregated_weights.append(weighted_param)
        
        return aggregated_weights
    
    def compute_weight_divergence(self, weights_list):
        """Measure heterogeneity of data across hospitals"""
        global_weights = self.global_model.get_weights()
        divergences = []
        
        for local_weights in weights_list:
            divergence = 0
            for g_w, l_w in zip(global_weights, local_weights):
                divergence += np.sum(np.power(g_w - l_w, 2))
            divergences.append(np.sqrt(divergence))
        
        return np.mean(divergences)
    
    def fedprox_round(self, epochs=1, sample_counts=None):
        """Execute one FedProx round"""
        print(f"\n--- FedProx Round {self.num_rounds + 1} ---")
        
        global_weights = self.global_model.get_weights()
        for client in self.clients:
            client.set_weights(global_weights)
        
        local_losses = []
        for client in self.clients:
            loss = client.train_epoch(global_model=self.global_model, epochs=epochs)
            local_losses.append(loss)
            print(f"  {client.client_id}: Loss = {loss:.4f} (with proximity term μ={self.mu})")
        
        clients_weights = [client.get_weights() for client in self.clients]
        aggregated_weights = self.aggregate_weights_weighted(clients_weights, sample_counts)
        self.global_model.set_weights(aggregated_weights)
        
        avg_loss = np.mean(local_losses)
        divergence = self.compute_weight_divergence(clients_weights)
        
        self.loss_history.append(avg_loss)
        self.convergence_metrics['avg_client_loss'].append(avg_loss)
        self.convergence_metrics['weight_divergence'].append(divergence)
        self.num_rounds += 1
        
        print(f"  Aggregated Loss: {avg_loss:.4f} | Data Heterogeneity: {divergence:.4f}")
        return avg_loss
    
    def evaluate_global_model(self, test_loader):
        """Evaluate global model"""
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
# 5. DATA SIMULATION
# ===========================

def create_heterogeneous_data(num_clients=3, samples_per_client=100):
    """Create highly non-IID data to show FedProx advantage"""
    clients_data = []
    sample_counts = []
    
    for client_id in range(num_clients):
        X = np.random.randn(samples_per_client, 1, 128, 128).astype(np.float32)
        y = np.zeros((samples_per_client, 14))
        
        # Each hospital specializes in different diseases (extreme heterogeneity)
        specialty_start = (client_id * 14) // num_clients
        specialty_end = ((client_id + 1) * 14) // num_clients
        
        y[:, specialty_start:specialty_end] = np.random.randint(0, 2, 
                            (samples_per_client, specialty_end - specialty_start))
        
        clients_data.append((X, y))
        sample_counts.append(samples_per_client)
    
    return clients_data, np.array(sample_counts)


# ===========================
# 6. MAIN EXECUTION
# ===========================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    NUM_CLIENTS = 4
    SAMPLES_PER_CLIENT = 80
    FEDERATED_ROUNDS = 5
    LOCAL_EPOCHS = 2
    BATCH_SIZE = 32
    MU = 0.01  # FedProx regularization parameter
    
    print("\n[SETUP] Creating FedProx environment...")
    
    server = CentralServer_FedProx(num_classes=14, device=device, mu=MU)
    clients_data, sample_counts = create_heterogeneous_data(num_clients=NUM_CLIENTS)
    
    for hospital_id, (X, y) in enumerate(clients_data):
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        client = HospitalClient_FedProx(
            client_id=f"Hospital_{hospital_id+1}",
            train_loader=loader,
            device=device,
            global_model=server.global_model,
            mu=MU
        )
        server.register_client(client)
        print(f"  Registered Hospital {hospital_id+1} with {len(X)} samples")
    
    X_test = np.random.randn(200, 1, 128, 128).astype(np.float32)
    y_test = np.random.randint(0, 2, size=(200, 14))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    print("\n[TRAINING] Starting FedProx with heterogeneous data...")
    print(f"  Rounds: {FEDERATED_ROUNDS}")
    print(f"  FedProx μ (mu): {MU}")
    print(f"  Hospitals: {NUM_CLIENTS}\n")
    
    for round_num in range(FEDERATED_ROUNDS):
        server.fedprox_round(epochs=LOCAL_EPOCHS, sample_counts=sample_counts)
    
    print("\n[EVALUATION] Final Results...")
    accuracy = server.evaluate_global_model(test_loader)
    print(f"  Test Accuracy: {accuracy:.4f}")
    
    from fl_model_saving import save_fedprox_model
    save_fedprox_model(server)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(server.convergence_metrics['avg_client_loss'], marker='o', linewidth=2)
    axes[0].set_xlabel('Federated Round')
    axes[0].set_ylabel('Average Loss')
    axes[0].set_title('FedProx: Training Loss Over Rounds')
    axes[0].grid(True)
    
    axes[1].plot(server.convergence_metrics['weight_divergence'], marker='s', 
                 linewidth=2, color='orange')
    axes[1].set_xlabel('Federated Round')
    axes[1].set_ylabel('Weight Divergence (Heterogeneity)')
    axes[1].set_title('FedProx: Data Heterogeneity Measure')
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig('fedprox_analysis.png', dpi=150)
    print("  Plot saved as 'fedprox_analysis.png'")
    
    print("\n" + "="*60)
    print("APPROACH 2: FEDPROX - SUMMARY")
    print("="*60)
    print(f"Total Federated Rounds: {server.num_rounds}")
    print(f"Total Hospitals: {NUM_CLIENTS}")
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print(f"FedProx μ (proximity term weight): {MU}")



if __name__ == "__main__":
    main()
