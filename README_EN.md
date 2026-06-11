<div align="center">

# 🤖 RGB Dynamic Attention Localization

### Indoor Robot Localization via Feature-level Dynamic Attention from RGB Images

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-orange?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Branch](https://img.shields.io/badge/Branch-main-purple)](https://github.com)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76b900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)

**[🇻🇳 Tiếng Việt](README.md)** | **🇬🇧 English**

</div>

---

## 📌 Project Overview

This project implements a **hierarchical indoor mobile robot localization system** from single RGB images using **Feature-level Dynamic Attention** — automatically filtering out the influence of dynamic obstacles (pedestrians) during the feature extraction process.

The system addresses a **multi-task learning** problem:
- **Task 1 — Topological Area Classification** (`Classification`): Identifies the geographical region (Area A–E).
- **Task 2 — Positioning Regression** (`Regression`): Predicts the exact physical $(x, y)$ coordinates in meters.

> Inspired by: *"Indoor Robot Localization Based on Vision"* — Applied Sciences (2021).

---

## 🏆 Test Set Evaluation Results

### Topological Area Classification Head
| Metric | Value |
|:---|:---:|
| **Accuracy** | **100.00%** |
| **Precision (Macro)** | **100.00%** |
| **Recall (Macro)** | **100.00%** |
| **F1-Score (Macro)** | **100.00%** |

### Positioning Regression Head (Euclidean Tolerance @ Test)
| Threshold | Accuracy (Success Rate) | F1-Score |
|:---:|:---:|:---:|
| 0.5 m | 45.37% | 62.42% |
| **1.0 m** | **83.90%** | **91.25%** |
| 1.5 m | 97.32% | 98.64% |
| 2.0 m | 99.51% | 99.76% |

### Detailed Localization Error Statistics
| Area | Mean Euclidean Error | Max Error | Std Dev |
|:---:|:---:|:---:|:---:|
| A (Room) | 0.5726 m | 1.6650 m | 0.3335 m |
| B (Corridor B) | 0.6062 m | 1.7130 m | 0.3809 m |
| C (Intersection) | 0.5271 m | 1.4386 m | 0.3238 m |
| D (Corridor D) | 0.6477 m | 1.9510 m | 0.3345 m |
| E (Elevator) | 0.7837 m | 2.3749 m | 0.5492 m |
| **AVERAGE** | **0.6187 m** | **2.3749 m** | **0.3896 m** |

---

## 🏗️ System Architecture

```
Input RGB Image (H×W×3)
        │
        ▼
 ┌─────────────────┐
 │ YOLOv8 (Offline)│  ← Detect & filter dynamic objects (humans)
 └────────┬────────┘
          │ Attention Map A_t = 1 - α·M_t
          ▼
 ┌─────────────────┐
 │ ResNet50 Layer4 │  ← Extract deep features
 │  (2048 channels)│
 └────────┬────────┘
          │ Feature-level Fusion: F'_t = F_t ⊙ A_t'
          │ (Bilinear interpolation + Element-wise mul)
          ▼
 ┌─────────────────┐
 │ Global Avg Pool │  → z_t ∈ ℝ²⁰⁴⁸
 └────────┬────────┘
          │
    ┌─────┴──────┐
    ▼            ▼
 ┌────────┐  ┌─────────┐
 │  MLP   │  │   MLP   │
 │ Topo   │  │  Coord  │
 │ Head   │  │  Head   │
 └───┬────┘  └────┬────┘
     │             │
     ▼             ▼
  Area Label   (x̂, ŷ) meters
 (A, B, C, D, E)
```

**Multi-task Loss Function:**
```
L = L_CE + γ · L_SmoothL1     (γ = 0.5)
```

---

## 📁 Repository Structure

```
rgb_dynamic_attention_localization/
│
├── 📄 train.py                    # Main training coordinator (Multi-task, Resume, Colab)
├── 📄 config.json                 # Complete hyperparameter configuration
├── 📄 requirements.txt            # Repository Python dependencies
├── 📄 COLAB_TRAINING_GUIDE.md     # Step-by-step Google Colab guide
│
├── 📂 models/
│   └── architecture.py            # FeatureAttentionHierarchicalNet (ResNet50 backbone)
│
├── 📂 utils/
│   └── dataset.py                 # RobotLocalizationDataset, Min-Max normalizer
│
├── 📂 scripts/
│   ├── preprocess_offline.py      # YOLOv8 offline detector → generates Attention Maps
│   └── inference.py               # Production inference pipeline
│
├── 📂 evaluation/
│   └── evaluate.py                # Model evaluation, charts, and report generator
│
├── 📂 results/                    # Evaluation artifacts & figures
│   ├── SUMARY.md                  # Comprehensive results report
│   ├── figure1/                   # Training Loss & Accuracy curves
│   ├── figure2/                   # Confusion Matrix (Topo Classification)
│   ├── figure3/                   # Trajectory colored by Area (Predicted)
│   ├── figure5/                   # Trajectory Comparison (GT vs CNN)
│   └── figure6/                   # Error Bar Charts & Metrics Tables
│
└── 📂 scratch/                    # Utility scripts (quick tests & visualization)
    ├── inference_custom.py
    ├── evaluate_directory_trajectory.py
    ├── test_individual_samples.py
    └── visualize_predictions.py
```

---

## ⚙️ Environment Setup

### Prerequisites
- Python ≥ 3.8
- CUDA ≥ 12.1 (Highly recommended for training)
- GPU ≥ 4GB VRAM (≥ 8GB recommended)

### Installation
```bash
# Clone the repository
git clone https://github.com/<your-username>/rgb_dynamic_attention_localization.git
cd rgb_dynamic_attention_localization

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage Guide

### 1. Data Preprocessing (Generate Attention Maps)

```bash
# Run YOLOv8 offline to generate attention maps for the entire dataset
python3 scripts/preprocess_offline.py --config config.json
```

This step performs:
- Detection and binary mask creation for dynamic objects (humans).
- Attention map formulation: `A_t = 1 - α·M_t`.
- Linear interpolation to map Encoder coordinates -> image frames.
- Output file saving at `data/processed/total_poses.csv`.

---

### 2. Model Training

#### Train from scratch (Local GPU):
```bash
python3 train.py --config config.json
```

#### Resume training from a checkpoint:
```bash
python3 train.py --config config.json --resume checkpoints/best_model.pth
```

#### Train on Google Colab:
```bash
python3 train.py --config config.json --colab
```
> Read detailed instructions: [COLAB_TRAINING_GUIDE.md](COLAB_TRAINING_GUIDE.md)

---

### 3. Model Evaluation

```bash
# Run full evaluation, save all plots and reports to results/
python3 evaluation/evaluate.py \
    --model checkpoints/best_model.pth \
    --config config.json \
    --device cuda \
    --history checkpoints/training_history.txt
```

This script automatically outputs:
| Output | Path |
|:---|:---|
| Confusion Matrix | `results/figure2/confusion_matrix.png` |
| Trajectory by Area | `results/figure3/trajectory_by_area.png` |
| Trajectory Comparison | `results/figure5/trajectory_comparison.png` |
| Error Bar Chart | `results/figure6/errors_bar_chart.png` |
| Training Curves | `results/figure1/training_loss_curves.png` |
| Table 3 Metrics | `results/figure6/table3_metrics.txt` |
| Performance Summary | `results/figure6/model_performance_summary.txt` |

---

### 4. Inference on Custom Images

```bash
# Evaluate a custom target image (generates real-time Attention Map)
PYTHONPATH=. python3 scratch/inference_custom.py \
    --paths <image-path> \
    --output_dir results/
```

---

## 📊 Visualized Results

### Figure 1 — Training Loss and Accuracy Curves
![Training Loss and Accuracy Curves](results/figure1/training_loss_curves.png)

### Figure 2 — Topological Classification Confusion Matrix (100% Accuracy)
![Confusion Matrix](results/figure2/confusion_matrix.png)

### Figure 3 — Predicted Trajectory Colored by Area
![Trajectory by Area](results/figure3/trajectory_by_area.png)

### Figure 5 — Trajectory Comparison: CNN vs Encoder Ground Truth
![Trajectory Comparison](results/figure5/trajectory_comparison.png)

### Figure 6 — Localization Error by Area
![Error Bar Chart](results/figure6/errors_bar_chart.png)

---

## 🔬 Training Configuration (`config.json`)

| Parameter | Value | Description |
|:---|:---:|:---|
| `backbone` | ResNet50 | Pre-trained on ImageNet |
| `image_size` | 240×320 | Input image dimensions |
| `batch_size` | 64 | Training batch size |
| `epochs` | 100 | Maximum training epochs |
| `learning_rate` | 0.002 | Initial learning rate |
| `weight_decay` | 0.0001 | L2 regularization weight decay |
| `dropout_p` | 0.3 | Dropout probability rate |
| `attention_alpha` | 0.7 | Attention mask scale factor (α) |
| `early_stopping` | 15 | Early stopping patience epochs |
| `train_ratio` | 70% | Training split ratio |
| `val_ratio` | 20% | Validation split ratio |
| `test_ratio` | 10% | Test split ratio |
| `seed` | 42 | Random seed value |

---

## 📄 Reference

> T. T. Mac et al., *"Regression-based Real-Time Robot Localization System Using Convolutional Neural Networks"*, Applied Sciences, 2021.

---

## 📬 Contact

This project is developed as part of research on indoor mobile robot localization using monocular RGB images.

---

<div align="center">
<sub>Built with ❤️ using PyTorch · ResNet50 · YOLOv8</sub>
</div>
