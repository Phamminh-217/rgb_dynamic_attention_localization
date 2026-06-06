# Hierarchical RGB Robot Localization with Feature-level Attention

[👉 Click here for Vietnamese version (README.md)](README.md)

This project implements a hierarchical indoor mobile robot localization system using raw RGB images and dynamic-aware Feature-level Attention. The network concurrently predicts the topological region class (Room/Corridor) and the continuous physical $(x, y)$ coordinate coordinates of the robot, especially in environments with dynamic obstacles (moving humans).

---

## 🚀 Test Set Evaluation Results

Below are the detailed performance metrics evaluated on the independent **Test Set**:

### 1. Global Metrics

| Metric | Value | Description |
| :--- | :---: | :--- |
| **Topological Area Accuracy** | **100.00%** | 100% accuracy class classifying Room vs. Corridor zones |
| **Mean Localization Error (MAE)** | **3.20 cm** | Mean physical distance localization error |
| **Root Mean Squared Error (RMSE)** | **3.93 cm** | Root mean square error measuring trajectory variance |
| **MAE - X axis** | **1.98 cm** | Average absolute error on the X coordinate axis |
| **MAE - Y axis** | **1.97 cm** | Average absolute error on the Y coordinate axis |
| **Maximum Localization Error** | **15.48 cm** | Peak error registered at sharp 90-degree corridor turns |

*Room ground truth vs predicted trajectory comparison plot is saved at: [results/room_trajectory_comparison.png](results/room_trajectory_comparison.png)*
*Training history curves (loss & accuracy): [checkpoints/training_curves.png](checkpoints/training_curves.png)* (automatically backed up to Drive when training on Colab)

---

## 🛠️ System Pipeline Architecture

The system strictly adheres to the **Separation of Concerns** software design principle:

```text
       Raw RGB Image ──► [ YOLOv8 (Offline) ] ──► Grayscale Attention Map
            │                                           │
            ▼                                           ▼
    [ ResNet50 Layer 4 ]                         [ Bilinear Interpolation ]
    (Feature Map 2048)                                  │
            │                                           │
            ▼                                           ▼
            └──────────────► [ Element-wise ⊙ ] ◄───────┘
                           (Dynamic Noise Filtering)
                                     │
                                     ▼
                            [ Global Avg Pool ]
                                     │
                                     ▼
                          ┌──────────┴──────────┐
                          ▼                     ▼
                  [ Topological Classifier ]  [ Pose Regressor ]
                    (Room/Corridor Classifier) (x,y coordinates)
```

---

## 📁 Repository Directory Structure

```text
├── data/                      # Local dataset folder (Ignored by Git)
│   ├── raw/                   # Contains raw images and coordinates CSV files
│   └── processed/             # Holds merged CSV dataset and offline Attention Maps
├── models/
│   └── architecture.py        # Neural Network architecture definitions
├── utils/
│   └── dataset.py             # DataLoader pipeline & Min-Max normalization
├── COLAB_TRAINING_GUIDE.md    # Guide to training on Google Colab GPU runtimes
├── checkpoints/               # Directory containing saved model weights (.pth) & plots
├── config.json                # Main configuration parameter scheme
├── train.py                   # Master Training Coordinator script
├── results/                   # Destination folder for trajectory plots & report images
├── scripts/                   # Core preprocessing automation scripts
│   └── preprocess_offline.py  # Generates Attention maps using YOLOv8 offline
└── scratch/                   # Auxiliary testing & trajectory plotting tools
    ├── inference_custom.py    # Test single/multiple images -> save 3-panel report image to results/
    ├── evaluate_directory_trajectory.py # Trajectory plotting tool for custom folders
    ├── evaluate_room.py       # Plots path for Room coordinates only
    └── evaluate_corridor.py   # Plots path for Corridor coordinates only
```

---

## 💻 Quick Start & Detailed Instructions

### 1. Training the Model

#### On local GPU workstation:
```bash
python3 train.py --config config.json
```

#### On Google Colab (with Google Drive Backup):
*Read [COLAB_TRAINING_GUIDE.md](COLAB_TRAINING_GUIDE.md) for full instructions.*
```bash
python3 train.py --config config.json --colab
```

---

### 2. Single Image Custom Inference
Use `inference_custom.py` to evaluate custom target images. It runs live YOLOv8 to detect humans on-the-fly, constructs attention masks, and generates a **3-panel visual report (original image, attention map, and predicted 2D coordinates vs ground truth)** saved directly inside the `results/` folder.

```bash
# Run on a single image (e.g., Room frame)
PYTHONPATH=. python3 scratch/inference_custom.py --paths data/raw/room/images/20260514_202335_361623_000100.png --output_dir results

# Run on multiple images simultaneously
PYTHONPATH=. python3 scratch/inference_custom.py --paths <image-path-1> <image-path-2> --output_dir results
```

---

### 3. Folder Trajectory Evaluation
Use `evaluate_directory_trajectory.py` to process an entire folder of sequence images. It sorts frames chronologically, reconstructs the predicted robot trajectory, plots it over the Ground Truth path, highlights the **START** and **END** points, and calculates localization error statistics.

```bash
# Reconstruct and compare Room trajectory
PYTHONPATH=. python3 scratch/evaluate_directory_trajectory.py --images_dir data/raw/room/images/ --output_plot results/room_trajectory_comparison.png

# Reconstruct and compare Corridor trajectory
PYTHONPATH=. python3 scratch/evaluate_directory_trajectory.py --images_dir data/raw/corridor/images/ --output_plot results/corridor_trajectory_comparison.png
```
