# Federated Learning for Medical Radiology: 5 Implementation Approaches

## Overview
This repository implements **5 distinct approaches to Federated Learning (FL)** for improving the generalization and reliability of AI models in multi-label classification of thorax diseases from chest X-ray images. The work addresses the **accuracy and reliability bottleneck** identified in the AI Healthcare Case Study Reflection Report (Session 2), where single-hospital training leads to overfitting/underfitting, false positives/negatives (10-25% accuracy drop in external validation), and poor trust due to limited datasets and lack of uncertainty estimation.

### Key Problem Addressed
- **Niche Focus**: Enhancing model reliability for 14 thorax diseases (e.g., pneumonia, cardiomegaly) using the **NIH Chest X-ray14 dataset** (112,120 images, public domain).
- **Root Causes**: Biased/single-source data, heterogeneous hospital protocols, no explainability/uncertainty.
- **Solution**: Collaborative FL across simulated "hospitals" to pool knowledge without sharing raw data, reducing overfitting and improving generalization.
- **Evidence from Report**: Models like CNNs fail in real-world scenarios (e.g., 15-20% false positives in German exams; WHO/EU reports emphasize reliability).

### Approaches
1. **FedAvg (Basic Horizontal FL)**: Simple averaging for baseline generalization.
2. **FedProx**: Adds proximity regularization for non-IID (heterogeneous) data.
3. **DP + Secure Aggregation**: Privacy-preserving with differential privacy (HIPAA/GDPR-compliant).
4. **Vertical FL with Transfer Learning**: Handles split data (e.g., images vs. metadata) using pre-trained ResNet.
5. **Hierarchical FL with Uncertainty**: Scalable hierarchy + Monte Carlo Dropout for confidence scores.

All codes use a simplified ResNet50 for multi-label BCE loss. Data is **simulated** (random 1-channel 128x128 images + 14 binary labels) to mimic NIH ChestX-ray14 for reproducibility; replace with real data via HuggingFace Datasets.

### Dataset
- **Primary**: NIH ChestX-ray14 (https://nihcc.app.box.com/v/ChestXray-NIHCC).
- **Simulation**: Non-IID splits across 3-4 "hospitals" (e.g., specialty biases).
- **Real Loading Example** (add to `main()`):
  ```python
  from datasets import load_dataset
  dataset = load_dataset("alkzar90/NIH-Chest-X-ray-dataset")
  # Split into client loaders
  ```

### Comparative Analysis
| Approach | Complexity | Privacy | Simulated Accuracy | Convergence | Best For |
|----------|------------|---------|---------------------|-------------|----------|
| 1. FedAvg | Low | Basic | ~0.50 | Medium | Baseline |
| 2. FedProx | Low-Med | Basic | ~0.52 | Fast | Heterogeneous data |
| 3. DP-Secure | High | Excellent | ~0.48 | Medium | Privacy-critical |
| 4. Vertical TL | Med | Medium | ~0.55 | Fast | Split features |
| 5. Hierarchical Unc. | High | Basic | ~0.53 + conf. | Slow | Scalable + trust |

**Recommendation**: Start with FedProx (Approach 2) for balance; use Approach 5 for uncertainty in clinical workflows.

## Installation
1. Clone the repo:
   ```bash
   git clone <repo-url>
   cd federated-medical-radiology
   ```
2. Create environment (see `requirements.yml` below for Conda).
3. Run each approach independently (e.g., `python approach1_fedavg_code.py`).

## Usage
Each approach is a standalone Python script. Run with:
```bash
python approachN_<name>_code.py
```
- **Outputs**: Console logs (losses, accuracy), plots (e.g., `fedavg_training_loss.png`).
- **Customization**:
  - Edit hyperparameters in `main()` (e.g., `NUM_CLIENTS=5`, `FEDERATED_ROUNDS=10`).
  - For real data: Replace simulation in `create_*_data()` with NIH loader.
  - Explainability: Add Grad-CAM (as per report) via `torchcam` (install separately).

### Approach 1: FedAvg
```bash
python approach1_fedavg_code.py
```
- Focus: Basic horizontal FL.
- Key Files: `approach1_fedavg_code.py`.

### Approach 2: FedProx
```bash
python approach2_fedprox_code.py
```
- Focus: Non-IID handling with μ=0.01 proximity term.
- Key Files: `approach2_fedprox_code.py`.

### Approach 3: DP + Secure
```bash
python approach3_dp_secure_code.py
```
- Focus: ε=2.0 DP with noise/clipping.
- Key Files: `approach3_dp_secure_code.py`.

### Approach 4: Vertical TL
```bash
python approach4_vertical_tl_code.py
```
- Focus: Feature split + pre-trained ResNet.
- Key Files: `approach4_vertical_tl_code.py`.

### Approach 5: Hierarchical Uncertainty
```bash
python approach5_hierarchical_unc_code.py
```
- Focus: Regional/global hierarchy + MC Dropout.
- Key Files: `approach5_hierarchical_unc_code.py`.

## Dependencies
See `requirements.yml` (Conda) or generate `requirements.txt` with `pip freeze > requirements.txt`.

## Contributing
- Fork and PR improvements (e.g., real dataset integration, Grad-CAM visuals).
- Issues: Report bugs for non-IID simulation or privacy accounting.

## License
MIT License. For medical use, consult HIPAA/GDPR experts.

## References
- Case Study Reflection Report (Session 2) by Mansa Thallapalli et al.
- NIH ChestX-ray14: Wang et al. (2017).
- FL Papers: McMahan et al. (FedAvg, 2017); Li et al. (FedProx, 2020).

---

# requirements.yml
```yaml
name: federated-medical-radiology
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.12
  - numpy>=1.24.0
  - pytorch>=2.0.0
  - torchvision>=0.15.0  # For Approach 4 pre-trained models
  - matplotlib>=3.7.0
  - scipy>=1.10.0
  - pandas>=2.0.0  # Optional for data handling
  - datasets>=2.14.0  # For HuggingFace NIH loading
  - pip
  - pip:
      - torchcam  # Optional: For Grad-CAM explainability
```

To install:
```bash
conda env create -f requirements.yml
conda activate federated-medical-radiology
```
