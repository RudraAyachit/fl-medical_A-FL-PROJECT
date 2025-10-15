"""
COMPLETE FEDERATED LEARNING COMPARATIVE ANALYSIS
Loads all 5 FL approach models and performs comprehensive evaluation
Run this after training all 5 approaches with model saving
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            hamming_loss, classification_report)
import os
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. SYNTHETIC MEDICAL DATASET (Small version for testing)
# ============================================================================

class SyntheticMedicalDataset(Dataset):
    """Generate synthetic chest X-ray-like images with 14 disease labels"""
    
    def __init__(self, num_samples=5000, num_classes=14, seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.num_samples = num_samples
        self.num_classes = num_classes
        
        print(f"\n{'='*70}")
        print(f"GENERATING SYNTHETIC MEDICAL DATASET")
        print(f"{'='*70}")
        print(f"Samples: {num_samples:,}")
        print(f"Diseases: {num_classes}")
        print(f"Image Size: 224x224x3")
        print(f"Estimated Size: ~{(num_samples * 224 * 224 * 3 * 4) / (1024**3):.2f} GB")
        
        # Generate images
        self.images = self._generate_synthetic_xrays(num_samples)
        
        # Generate labels (multi-label: each sample can have multiple diseases)
        self.labels = self._generate_disease_labels(num_samples, num_classes)
        
        print(f"✓ Dataset created successfully!")
        print(f"  Images shape: {self.images.shape}")
        print(f"  Labels shape: {self.labels.shape}")
        
    def _generate_synthetic_xrays(self, num_samples):
        """Generate synthetic X-ray-like images"""
        print("\nGenerating synthetic X-ray images...")
        images = np.zeros((num_samples, 3, 224, 224), dtype=np.float32)
        
        for i in range(num_samples):
            if i % 1000 == 0:
                print(f"  Progress: {i}/{num_samples}")
            
            # Base image (grayscale-like medical image)
            base = np.random.normal(0.5, 0.15, (3, 224, 224))
            
            # Add anatomical structures (simulate ribs, lungs, heart)
            y, x = np.ogrid[-112:112, -112:112]
            
            # Simulate lungs (two large regions)
            lung_left = ((x + 40)**2 + y**2 < 50**2).astype(float) * 0.3
            lung_right = ((x - 40)**2 + y**2 < 50**2).astype(float) * 0.3
            
            # Simulate heart (center region)
            heart = ((x - 5)**2 + (y + 20)**2 < 30**2).astype(float) * 0.4
            
            # Add structures to all channels
            for c in range(3):
                base[c] += lung_left + lung_right + heart
                
                # Add random lesions/abnormalities (simulate diseases)
                num_lesions = np.random.randint(0, 4)
                for _ in range(num_lesions):
                    lx = np.random.randint(50, 174)
                    ly = np.random.randint(50, 174)
                    radius = np.random.randint(10, 25)
                    lesion_mask = ((x - (lx - 112))**2 + (y - (ly - 112))**2 < radius**2)
                    base[c, lesion_mask] += np.random.uniform(0.2, 0.4)
            
            # Add noise (imaging artifacts)
            noise = np.random.normal(0, 0.05, (3, 224, 224))
            images[i] = np.clip(base + noise, 0, 1)
        
        print(f"✓ Generated {num_samples} synthetic X-ray images")
        return images
    
    def _generate_disease_labels(self, num_samples, num_classes):
        """Generate multi-label disease annotations"""
        print("\nGenerating disease labels...")
        labels = np.zeros((num_samples, num_classes), dtype=np.float32)
        
        # Disease names for reference
        disease_names = [
            'Pneumonia', 'Nodule', 'Mass', 'Cardiomegaly', 'Infiltrate',
            'Atelectasis', 'Effusion', 'Consolidation', 'Pneumothorax', 
            'Fibrosis', 'Edema', 'Emphysema', 'Hernia', 'Tuberculosis'
        ]
        
        # Realistic disease prevalence (based on medical literature)
        disease_prevalence = np.array([
            0.25,  # Pneumonia
            0.15,  # Nodule
            0.12,  # Mass
            0.20,  # Cardiomegaly
            0.18,  # Infiltrate
            0.22,  # Atelectasis
            0.14,  # Effusion
            0.10,  # Consolidation
            0.08,  # Pneumothorax
            0.09,  # Fibrosis
            0.11,  # Edema
            0.13,  # Emphysema
            0.03,  # Hernia
            0.05   # Tuberculosis
        ])
        
        for i in range(num_samples):
            for j in range(num_classes):
                # Add disease correlation (comorbidity)
                base_prob = disease_prevalence[j]
                
                # If patient already has certain diseases, increase probability of others
                if j > 0 and labels[i, :j].sum() > 0:
                    base_prob = min(base_prob + 0.15, 0.8)
                
                labels[i, j] = 1 if np.random.random() < base_prob else 0
        
        # Statistics
        print(f"\nDisease Prevalence in Dataset:")
        for idx, name in enumerate(disease_names):
            prevalence = labels[:, idx].sum() / num_samples * 100
            print(f"  {name:18} - {prevalence:5.1f}%")
        
        avg_diseases = labels.sum(axis=1).mean()
        print(f"\nAverage diseases per patient: {avg_diseases:.2f}")
        
        return labels
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        image = torch.FloatTensor(self.images[idx])
        label = torch.FloatTensor(self.labels[idx])
        return image, label


def create_test_dataset(num_samples=1000, num_classes=14):
    """Create test dataset for evaluation"""
    print(f"\n{'='*70}")
    print("CREATING TEST DATASET")
    print(f"{'='*70}")
    
    dataset = SyntheticMedicalDataset(num_samples=num_samples, num_classes=num_classes)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    print(f"\n✓ Test DataLoader created")
    print(f"  Batch size: 32")
    print(f"  Total batches: {len(test_loader)}")
    
    return test_loader


# ============================================================================
# 2. MODEL ARCHITECTURE (Compatible with all 5 approaches)
# ============================================================================

class UnifiedMedicalModel(nn.Module):
    """Unified ResNet-based model architecture"""
    
    def __init__(self, num_classes=14):
        super(UnifiedMedicalModel, self).__init__()
        
        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ============================================================================
# 3. MODEL LOADER
# ============================================================================

class ModelLoader:
    """Load FL models from saved files"""
    
    def __init__(self, models_dir='./fl_saved_models', device='cpu'):
        self.models_dir = Path(models_dir)
        self.device = device
        self.loaded_models = {}
        self.metadata = {}
    
    def discover_models(self):
        """Auto-discover all saved models"""
        print(f"\n{'='*70}")
        print("DISCOVERING SAVED MODELS")
        print(f"{'='*70}")
        print(f"Searching in: {self.models_dir}")
        
        if not self.models_dir.exists():
            print(f"✗ Directory not found: {self.models_dir}")
            return []
        
        model_files = {
            'FedAvg': list(self.models_dir.glob('fedavg_model_*.pkl')),
            'FedProx': list(self.models_dir.glob('fedprox_model_*.pkl')),
            'DP-FL': list(self.models_dir.glob('dpfl_model_*.pkl')),
            'Vertical-FL': list(self.models_dir.glob('vfl_full_model_*.pkl')),
            'Hierarchical-FL': list(self.models_dir.glob('hfl_model_*.pkl')),
        }
        
        found_models = {}
        for approach, files in model_files.items():
            if files:
                latest = max(files, key=lambda p: p.stat().st_mtime)
                found_models[approach] = latest
                print(f"✓ Found {approach}: {latest.name}")
        
        if not found_models:
            print("✗ No models found!")
            print("\nExpected files:")
            for approach, pattern in model_files.items():
                print(f"  - {approach}: {pattern}")
        
        return found_models
    
    def load_model(self, model_path, approach_name):
        """Load a single model"""
        try:
            print(f"\nLoading {approach_name}...")
            
            if str(model_path).endswith('.pkl'):
                model = joblib.load(model_path)
            elif str(model_path).endswith('.pth'):
                model = UnifiedMedicalModel(num_classes=14)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
            else:
                print(f"  ✗ Unknown format: {model_path}")
                return None
            
            model = model.to(self.device)
            model.eval()
            
            # Load metadata if exists
            metadata_path = str(model_path).replace('_model_', '_metadata_').replace('.pkl', '.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata[approach_name] = json.load(f)
                print(f"  ✓ Loaded with metadata")
            else:
                print(f"  ✓ Loaded (no metadata)")
            
            self.loaded_models[approach_name] = model
            return model
        
        except Exception as e:
            print(f"  ✗ Failed to load: {str(e)}")
            return None
    
    def load_all(self):
        """Load all discovered models"""
        discovered = self.discover_models()
        
        for approach, path in discovered.items():
            self.load_model(path, approach)
        
        print(f"\n{'='*70}")
        print(f"LOADED {len(self.loaded_models)} MODELS SUCCESSFULLY")
        print(f"{'='*70}\n")
        
        return self.loaded_models


# ============================================================================
# 4. COMPREHENSIVE EVALUATOR
# ============================================================================

class MedicalModelEvaluator:
    """Evaluate models with medical-specific metrics"""
    
    def __init__(self, model, test_loader, device='cpu', model_name='Model'):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.model_name = model_name
        self.results = {}
    
    def evaluate(self):
        """Run comprehensive evaluation"""
        print(f"\n{'='*70}")
        print(f"EVALUATING: {self.model_name}")
        print(f"{'='*70}")
        
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        inference_times = []
        
        print("Running inference...")
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.test_loader):
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{len(self.test_loader)}")
                
                images = images.to(self.device)
                labels_np = labels.cpu().numpy()
                
                # Measure inference time
                start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                
                if torch.cuda.is_available():
                    start_time.record()
                
                outputs = self.model(images)
                
                if torch.cuda.is_available():
                    end_time.record()
                    torch.cuda.synchronize()
                    inference_times.append(start_time.elapsed_time(end_time))
                
                # Convert to probabilities
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                all_preds.extend(preds)
                all_labels.extend(labels_np)
                all_probs.extend(probs)
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        print("\nCalculating metrics...")
        
        # Overall metrics
        self.results = {
            'accuracy': accuracy_score(all_labels.flatten(), all_preds.flatten()),
            'precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
            'precision_weighted': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0),
            'recall_weighted': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
            'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
            'f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
            'hamming_loss': hamming_loss(all_labels, all_preds),
            'avg_inference_time_ms': np.mean(inference_times) if inference_times else 0,
        }
        
        # Per-class metrics
        disease_names = [
            'Pneumonia', 'Nodule', 'Mass', 'Cardiomegaly', 'Infiltrate',
            'Atelectasis', 'Effusion', 'Consolidation', 'Pneumothorax', 
            'Fibrosis', 'Edema', 'Emphysema', 'Hernia', 'Tuberculosis'
        ]
        
        self.results['per_disease'] = {}
        for i, disease in enumerate(disease_names):
            self.results['per_disease'][disease] = {
                'accuracy': accuracy_score(all_labels[:, i], all_preds[:, i]),
                'precision': precision_score(all_labels[:, i], all_preds[:, i], zero_division=0),
                'recall': recall_score(all_labels[:, i], all_preds[:, i], zero_division=0),
                'f1': f1_score(all_labels[:, i], all_preds[:, i], zero_division=0),
            }
        
        self._print_results()
        return self.results
    
    def _print_results(self):
        """Print formatted results"""
        print(f"\n{'='*70}")
        print(f"RESULTS: {self.model_name}")
        print(f"{'='*70}")
        
        print("\nOverall Metrics:")
        print(f"  Accuracy:             {self.results['accuracy']:.4f}")
        print(f"  Precision (macro):    {self.results['precision_macro']:.4f}")
        print(f"  Precision (weighted): {self.results['precision_weighted']:.4f}")
        print(f"  Recall (macro):       {self.results['recall_macro']:.4f}")
        print(f"  Recall (weighted):    {self.results['recall_weighted']:.4f}")
        print(f"  F1-Score (macro):     {self.results['f1_macro']:.4f}")
        print(f"  F1-Score (weighted):  {self.results['f1_weighted']:.4f}")
        print(f"  Hamming Loss:         {self.results['hamming_loss']:.4f}")
        print(f"  Inference Time:       {self.results['avg_inference_time_ms']:.2f} ms")
        
        print("\nTop 5 Diseases Performance:")
        sorted_diseases = sorted(self.results['per_disease'].items(), 
                                key=lambda x: x[1]['f1'], reverse=True)[:5]
        for disease, metrics in sorted_diseases:
            print(f"  {disease:18} - F1: {metrics['f1']:.4f}, "
                  f"Acc: {metrics['accuracy']:.4f}")


# ============================================================================
# 5. COMPARATIVE ANALYZER
# ============================================================================

class ComparativeAnalyzer:
    """Compare multiple FL models"""
    
    def __init__(self, models, test_loader, device='cpu'):
        self.models = models
        self.test_loader = test_loader
        self.device = device
        self.all_results = {}
    
    def run_comparison(self):
        """Evaluate all models"""
        print(f"\n{'='*70}")
        print("RUNNING COMPARATIVE ANALYSIS")
        print(f"{'='*70}")
        
        for model_name, model in self.models.items():
            evaluator = MedicalModelEvaluator(model, self.test_loader, 
                                             self.device, model_name)
            results = evaluator.evaluate()
            self.all_results[model_name] = results
        
        return self.all_results
    
    def print_comparison_table(self):
        """Print side-by-side comparison"""
        print(f"\n{'='*70}")
        print("COMPARATIVE RESULTS TABLE")
        print(f"{'='*70}\n")
        
        # Header
        models = list(self.all_results.keys())
        print(f"{'Metric':<25}", end='')
        for model in models:
            print(f"{model:<18}", end='')
        print()
        print("-" * (25 + 18 * len(models)))
        
        # Metrics
        metrics_to_show = [
            ('accuracy', 'Accuracy'),
            ('f1_weighted', 'F1-Score (weighted)'),
            ('precision_weighted', 'Precision (weighted)'),
            ('recall_weighted', 'Recall (weighted)'),
            ('hamming_loss', 'Hamming Loss'),
            ('avg_inference_time_ms', 'Inference Time (ms)'),
        ]
        
        for key, label in metrics_to_show:
            print(f"{label:<25}", end='')
            for model in models:
                value = self.all_results[model][key]
                print(f"{value:<18.4f}", end='')
            print()
        
        print("\n" + "="*70)
    
    def generate_visualizations(self, output_dir='./fl_comparison_results'):
        """Create comparison plots"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*70}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*70}")
        
        models = list(self.all_results.keys())
        
        # 1. Accuracy Comparison
        accuracies = [self.all_results[m]['accuracy'] for m in models]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(models)])
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Federated Learning Approaches - Accuracy Comparison', fontsize=14, fontweight='bold')
        plt.ylim([0, 1])
        plt.grid(axis='y', alpha=0.3)
        
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, acc + 0.02, 
                    f'{acc:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: accuracy_comparison.png")
        plt.close()
        
        # 2. F1-Score Comparison
        f1_scores = [self.all_results[m]['f1_weighted'] for m in models]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(models, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(models)])
        plt.ylabel('F1-Score (Weighted)', fontsize=12)
        plt.title('Federated Learning Approaches - F1-Score Comparison', fontsize=14, fontweight='bold')
        plt.ylim([0, 1])
        plt.grid(axis='y', alpha=0.3)
        
        for bar, f1 in zip(bars, f1_scores):
            plt.text(bar.get_x() + bar.get_width()/2, f1 + 0.02, 
                    f'{f1:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'f1_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: f1_comparison.png")
        plt.close()
        
        # 3. Multi-Metric Radar Chart
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for idx, model in enumerate(models):
            values = [
                self.all_results[model]['accuracy'],
                self.all_results[model]['precision_weighted'],
                self.all_results[model]['recall_weighted'],
                self.all_results[model]['f1_weighted']
            ]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax.set_title('Multi-Metric Performance Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'multi_metric_radar.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: multi_metric_radar.png")
        plt.close()
        
        # 4. Inference Time Comparison
        inference_times = [self.all_results[m]['avg_inference_time_ms'] for m in models]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(models, inference_times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(models)])
        plt.ylabel('Inference Time (ms)', fontsize=12)
        plt.title('Federated Learning Approaches - Inference Speed', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        for bar, time in zip(bars, inference_times):
            plt.text(bar.get_x() + bar.get_width()/2, time + 0.5, 
                    f'{time:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'inference_time_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: inference_time_comparison.png")
        plt.close()
        
        print(f"\nAll visualizations saved in: {output_dir}")
    
    def save_report(self, output_dir='./fl_comparison_results'):
        """Save comprehensive JSON report"""
        os.makedirs(output_dir, exist_ok=True)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'num_models': len(self.models),
            'test_samples': len(self.test_loader.dataset),
            'results': self.all_results
        }
        
        report_path = os.path.join(output_dir, f'comparative_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Comprehensive report saved: {report_path}")
        
        return report_path


# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("FEDERATED LEARNING - COMPLETE COMPARATIVE ANALYSIS")
    print("="*70)
    print("Comparing: FedAvg, FedProx, DP-FL, Vertical-FL, Hierarchical-FL")
    print("="*70)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # 1. Create test dataset
    test_loader = create_test_dataset(num_samples=1000, num_classes=14)
    
    # 2. Load models
    loader = ModelLoader(models_dir='./fl_saved_models', device=device)
    models = loader.load_all()
    
    if not models:
        print("\n" + "="*70)
        print("WARNING: NO MODELS FOUND!")
        print("="*70)
        print("\nPlease ensure you have:")
        print("1. Trained all 5 FL approaches (HR-FL.py, FEDPROX.py, etc.)")
        print("2. Added model saving code to each approach")
        print("3. Models are saved in './fl_saved_models/' directory")
        print("\nExpected files:")
        print("  - fedavg_model_*.pkl")
        print("  - fedprox_model_*.pkl")
        print("  - dpfl_model_*.pkl")
        print("  - vfl_full_model_*.pkl")
        print("  - hfl_model_*.pkl")
        print("\nCreating dummy models for demonstration...")
        
        # Create dummy models for demonstration
        os.makedirs('./fl_saved_models', exist_ok=True)
        for approach in ['FedAvg', 'FedProx', 'DP-FL', 'Vertical-FL', 'Hierarchical-FL']:
            dummy_model = UnifiedMedicalModel(num_classes=14)
            dummy_path = f'./fl_saved_models/{approach.lower().replace("-", "")}_model_demo.pkl'
            joblib.dump(dummy_model, dummy_path, compress=3)
            models[approach] = dummy_model.to(device)
        
        print("✓ Created dummy models for demonstration")
    
    # 3. Run comparative analysis
    analyzer = ComparativeAnalyzer(models, test_loader, device)
    results = analyzer.run_comparison()
    
    # 4. Print comparison table
    analyzer.print_comparison_table()
    
    # 5. Generate visualizations
    analyzer.generate_visualizations(output_dir='./fl_comparison_results')
    
    # 6. Save comprehensive report
    report_path = analyzer.save_report(output_dir='./fl_comparison_results')
    
    # 7. Print summary and recommendations
    print_final_summary(results, loader.metadata)
    
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved in: ./fl_comparison_results/")
    print("Files generated:")
    print("  - accuracy_comparison.png")
    print("  - f1_comparison.png")
    print("  - multi_metric_radar.png")
    print("  - inference_time_comparison.png")
    print(f"  - {os.path.basename(report_path)}")
    print("\n" + "="*70)


def print_final_summary(results, metadata):
    """Print final analysis summary with recommendations"""
    print("\n" + "="*70)
    print("FINAL ANALYSIS SUMMARY")
    print("="*70)
    
    # Find best performers
    best_accuracy = max(results.items(), key=lambda x: x[1]['accuracy'])
    best_f1 = max(results.items(), key=lambda x: x[1]['f1_weighted'])
    best_speed = min(results.items(), key=lambda x: x[1]['avg_inference_time_ms'])
    
    print("\n🏆 BEST PERFORMERS:")
    print(f"  Best Accuracy:   {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.4f})")
    print(f"  Best F1-Score:   {best_f1[0]} ({best_f1[1]['f1_weighted']:.4f})")
    print(f"  Fastest:         {best_speed[0]} ({best_speed[1]['avg_inference_time_ms']:.2f} ms)")
    
    # Calculate overall rankings
    print("\n📊 OVERALL RANKINGS (based on F1-Score):")
    ranked = sorted(results.items(), key=lambda x: x[1]['f1_weighted'], reverse=True)
    for idx, (model, metrics) in enumerate(ranked, 1):
        print(f"  {idx}. {model:20} - F1: {metrics['f1_weighted']:.4f}, Acc: {metrics['accuracy']:.4f}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("\n  Use Case Recommendations:")
    print("  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    print("\n  1️⃣  FedAvg (Baseline)")
    print("      ✓ Simple deployment, low complexity")
    print("      ✓ Best for: Proof of concept, equal data distribution")
    print("      ✗ Struggles with non-IID data")
    
    print("\n  2️⃣  FedProx (Non-IID Handling)")
    print("      ✓ Handles heterogeneous hospital data")
    print("      ✓ Best for: Hospitals with different disease prevalence")
    print("      ✓ Good balance of accuracy and convergence")
    
    print("\n  3️⃣  DP-FL (Privacy-Preserving)")
    print("      ✓ HIPAA/GDPR compliant with differential privacy")
    print("      ✓ Best for: Highly regulated environments")
    print("      ✗ Slight accuracy trade-off for privacy")
    print("      ✓ Recommended when: Privacy is critical priority")
    
    print("\n  4️⃣  Vertical-FL (Split Features)")
    print("      ✓ Handles vertically partitioned data")
    print("      ✓ Best for: When features split across institutions")
    print("      ✓ Example: One hospital has images, another has metadata")
    
    print("\n  5️⃣  Hierarchical-FL (Scalable)")
    print("      ✓ Scalable to large hospital networks")
    print("      ✓ Provides uncertainty quantification (Monte Carlo Dropout)")
    print("      ✓ Best for: Multi-region deployments (e.g., national networks)")
    print("      ✓ Recommended when: Need confidence scores for predictions")
    
    print("\n  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # Deployment recommendations
    print("\n  📋 DEPLOYMENT RECOMMENDATIONS:")
    
    if 'FedProx' in results and 'FedAvg' in results:
        if results['FedProx']['f1_weighted'] > results['FedAvg']['f1_weighted']:
            print("\n  ✅ RECOMMENDED: FedProx")
            print("     Reason: Better performance than FedAvg, handles heterogeneity")
    
    if 'DP-FL' in results:
        print("\n  🔒 For Privacy-Critical Applications: DP-FL")
        print("     Trade-off: ~5-10% accuracy loss for strong privacy guarantees")
    
    if 'Hierarchical-FL' in results:
        print("\n  🌐 For Large-Scale Deployments: Hierarchical-FL")
        print("     Benefit: Scalable + uncertainty quantification for clinical use")
    
    # Print metadata summary
    if metadata:
        print("\n  📝 TRAINING DETAILS:")
        for model_name, meta in metadata.items():
            if meta:
                print(f"\n  {model_name}:")
                print(f"    Rounds: {meta.get('training_rounds', 'N/A')}")
                print(f"    Final Loss: {meta.get('final_loss', 'N/A')}")
                if 'epsilon' in meta:
                    print(f"    Privacy (ε): {meta.get('epsilon', 'N/A')}")
                if 'mu_regularization' in meta:
                    print(f"    FedProx μ: {meta.get('mu_regularization', 'N/A')}")


# ============================================================================
# 7. STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Analysis interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nPlease check:")
        print("1. Models are saved in './fl_saved_models/'")
        print("2. PyTorch and required libraries are installed")
        print("3. Sufficient memory available")
