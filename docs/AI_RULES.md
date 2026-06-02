# AI_RULES.md

# Coding Rules for AI Agents

> This file defines the mandatory coding and documentation rules for AI agents working on the project:
>
> **Hierarchical RGB-based Mobile Robot Localization with Dynamic-aware Feature-level Attention**
>
> Every AI agent must read this file before modifying source code. These rules preserve architectural correctness, enforce clean PyTorch implementation, and prevent cross-module coupling.

---

## 1. Project Context

This project implements an indoor mobile robot localization system based on RGB images and dynamic-aware Feature-level Attention.

For each RGB frame, the system predicts:

```text
1. Topological area label
2. Continuous robot coordinate (x, y)
```

The model uses multi-task learning:

```text
Topological Classification Branch  -> Cross-Entropy Loss
Coordinate Regression Branch       -> Smooth L1 Loss
```

Coordinate labels must be normalized into the range `[0, 1]` using Min-Max normalization.

---

## 2. Mandatory Architecture Principle

The project strictly follows:

```text
Separation of Concerns
```

The system is divided into three responsibilities:

```text
Data Pipeline Module
    ├── preprocess_offline.py
    └── utils/dataset.py

Model Architecture Module
    └── models/architecture.py

Training Controller
    └── train.py
```

Each module must only perform its assigned responsibility.

---

## 3. Persona and Response Style

AI agents working on this repository must act as experienced machine learning and robotics engineers.

Required response style:

```text
- Be direct.
- Be technically precise.
- Prefer clean implementation over long explanation.
- Provide production-oriented PyTorch code.
- Avoid unnecessary greetings.
- Avoid procedural filler.
- Avoid apologetic language.
- Avoid vague statements.
- Do not over-explain obvious code.
```

When writing code, prioritize:

```text
- Correct tensor shapes
- Clear module boundaries
- Deterministic behavior where possible
- Runtime safety checks
- Minimal coupling between files
- Readable PyTorch implementation
```

---

## 4. Global Coding Rules

All Python code must follow these rules:

```text
- Use explicit imports.
- Avoid wildcard imports.
- Use type hints for functions and methods.
- Add tensor shape comments near important tensors.
- Use clear variable names.
- Avoid hidden side effects.
- Avoid hard-coded absolute paths unless they come from config.json.
- Validate critical input shapes.
- Raise explicit errors when assumptions are violated.
```

Example:

```python
def forward(
    self,
    images: torch.Tensor,
    attention_maps: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    features = self.backbone(images)  # Shape: [B, 2048, H_prime, W_prime]
    ...
```

---

## 5. PyTorch Coding Rules

### 5.1. Device Configuration

Every training or inference script must configure device automatically:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Do not hard-code:

```python
device = torch.device("cuda")
```

unless there is a deliberate runtime check before it.

---

### 5.2. Tensor Shape Comments

Important tensors must include shape comments.

Required style:

```python
features = self.backbone(images)  # Shape: [B, 2048, H_prime, W_prime]

attention_resized = F.interpolate(
    attention_maps,
    size=features.shape[-2:],
    mode="bilinear",
    align_corners=False,
)  # Shape: [B, 1, H_prime, W_prime]

attended_features = features * attention_resized  # Shape: [B, 2048, H_prime, W_prime]

pooled = self.gap(attended_features)  # Shape: [B, 2048, 1, 1]

z = torch.flatten(pooled, 1)  # Shape: [B, 2048]
```

Tensor shape comments must be placed near the tensor creation or transformation line.

---

### 5.3. Type Hinting

All public functions, class methods, and helper functions must use explicit type hints.

Required style:

```python
from typing import Dict, Tuple

import torch


def load_config(config_path: str) -> Dict:
    ...


def forward(
    self,
    images: torch.Tensor,
    attention_maps: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ...
```

Avoid untyped signatures:

```python
def forward(self, x, y):
    ...
```

---

### 5.4. Dtype Rules

The following dtype conventions are mandatory:

| Variable         | Required dtype  |
| ---------------- | --------------- |
| `images`         | `torch.float32` |
| `attention_maps` | `torch.float32` |
| `topo_labels`    | `torch.long`    |
| `coord_labels`   | `torch.float32` |
| `topo_logits`    | `torch.float32` |
| `coord_pred`     | `torch.float32` |

Dataset output must respect these dtype rules before the batch reaches `train.py`.

---

## 6. Model Architecture Rules

The file `models/architecture.py` must define the neural network only.

Allowed content:

```text
- PyTorch modules
- ResNet50 backbone construction
- Feature-level Attention fusion
- Global Average Pooling
- Multi-task MLP heads
- Forward pass
```

Forbidden content:

```text
- CSV reading
- Image file reading
- Path handling
- Dataset construction
- DataLoader construction
- YOLOv8 inference
- Coordinate offset logic
- Min-Max fitting
- Loss computation
- Optimizer construction
- Checkpoint saving
```

---

## 7. ResNet50 Backbone Rules

The model must use ResNet50 up to `layer4`.

The following ResNet50 components must be kept:

```text
conv1
bn1
relu
maxpool
layer1
layer2
layer3
layer4
```

The following components must be removed:

```text
avgpool
fc
```

Expected feature map:

```python
features = self.backbone(images)  # Shape: [B, 2048, H_prime, W_prime]
```

Do not hard-code `H_prime = 7` or `W_prime = 7`.

The spatial size must always be taken dynamically:

```python
target_size = features.shape[-2:]
```

---

## 8. Feature-level Attention Rules

Attention must operate at feature level, not image level.

Correct implementation:

```python
features = self.backbone(images)  # Shape: [B, 2048, H_prime, W_prime]

attention_resized = F.interpolate(
    attention_maps,
    size=features.shape[-2:],
    mode="bilinear",
    align_corners=False,
)  # Shape: [B, 1, H_prime, W_prime]

attended_features = features * attention_resized  # Shape: [B, 2048, H_prime, W_prime]
```

Forbidden alternatives:

```text
- Multiplying attention_maps directly with RGB images inside the model
- Concatenating attention_maps with RGB images
- Concatenating attention_maps with feature maps
- Adding attention_maps to feature maps
- Replacing feature-level attention with Transformer attention
```

The required operation is element-wise multiplication:

```text
attended_features = features * attention_resized
```

---

## 9. Multi-task MLP Head Rules

After Feature-level Attention, the model must apply Global Average Pooling and flatten the tensor:

```python
pooled = self.gap(attended_features)  # Shape: [B, 2048, 1, 1]
z = torch.flatten(pooled, 1)          # Shape: [B, 2048]
```

The shared vector `z` must feed two parallel heads.

---

## 10. Topological Classification Head Rules

Required structure:

```text
Linear(2048 -> 512)
BatchNorm1d(512)
ReLU
Dropout(0.3)
Linear(512 -> 256)
ReLU
Linear(256 -> num_classes_topo)
```

The output must be raw logits:

```python
topo_logits = self.topo_head(z)  # Shape: [B, num_classes_topo]
```

Forbidden operations inside model:

```text
- Softmax
- Argmax
- CrossEntropyLoss
```

`nn.CrossEntropyLoss` must be applied only in `train.py`.

---

## 11. Coordinate Regression Head Rules

Required structure:

```text
Linear(2048 -> 512)
BatchNorm1d(512)
ReLU
Dropout(0.3)
Linear(512 -> 256)
ReLU
Linear(256 -> 2)
```

The output must be:

```python
coord_pred = self.coord_head(z)  # Shape: [B, 2]
```

The model must not perform coordinate denormalization.

Forbidden operations inside model:

```text
- Reading x_min, x_max, y_min, y_max
- Denormalizing predicted coordinates
- Computing Smooth L1 Loss
- Applying post-processing rules
```

---

## 12. Data Pipeline Rules

The data module includes:

```text
preprocess_offline.py
utils/dataset.py
```

The data module is responsible for:

```text
- Reading raw CSV files
- Aligning coordinate frames
- Assigning topo labels
- Creating total_poses.csv
- Computing global Min-Max normalization values
- Reading RGB images
- Reading Attention Maps
- Returning clean PyTorch tensors
```

The data module must not contain:

```text
- ResNet50
- MLP layers
- Forward pass
- Loss functions
- Optimizer
- Model checkpoint logic
```

---

## 13. Dataset Output Rules

`RobotLocalizationDataset.__getitem__()` must return exactly four values:

```python
return image_tensor, attention_tensor, topo_label, coord_label
```

Required shapes:

| Output             | Shape       |
| ------------------ | ----------- |
| `image_tensor`     | `[3, H, W]` |
| `attention_tensor` | `[1, H, W]` |
| `topo_label`       | scalar      |
| `coord_label`      | `[2]`       |

Required dtypes:

```python
image_tensor: torch.FloatTensor
attention_tensor: torch.FloatTensor
topo_label: torch.LongTensor
coord_label: torch.FloatTensor
```

---

## 14. Coordinate Normalization Rules

Coordinate regression labels must be normalized into `[0, 1]`.

Use Min-Max normalization:

```text
x_n = (x - x_min) / (x_max - x_min)
y_n = (y - y_min) / (y_max - y_min)
```

The Dataset must return:

```python
coord_label = torch.tensor([x_n, y_n], dtype=torch.float32)  # Shape: [2]
```

The model must not know whether coordinates came from room data or corridor data.

---

## 15. Training Script Rules

The file `train.py` is the controller only.

Allowed responsibilities:

```text
- Read config.json
- Resolve device
- Prepare Colab dataset if needed
- Check preprocessing status
- Call preprocess_offline.py if needed
- Create Dataset and DataLoader
- Create model
- Compute losses
- Run training loop
- Run validation loop
- Save checkpoints
```

Forbidden responsibilities:

```text
- Defining ResNet50
- Defining MLP heads
- Implementing Feature-level Attention
- Reading individual images directly
- Parsing raw CSV files manually
- Creating Attention Maps manually
```

---

## 16. Multi-task Loss Rules

The training script must compute two losses:

```python
criterion_topo = nn.CrossEntropyLoss()
criterion_coord = nn.SmoothL1Loss()
```

Correct usage:

```python
topo_logits, coord_pred = model(images, attention_maps)

loss_topo = criterion_topo(topo_logits, topo_labels)
loss_coord = criterion_coord(coord_pred, coord_labels)
loss_total = loss_topo + gamma * loss_coord
```

Rules:

```text
- topo_logits must be raw logits.
- Do not apply Softmax before CrossEntropyLoss.
- coord_labels must be normalized coordinates.
- gamma must be read from config.json.
```

The total loss is:

[
\mathcal{L}_{total}
===================

\mathcal{L}*{CrossEntropy}
+
\gamma \cdot \mathcal{L}*{SmoothL1}
]

---

## 17. Google Colab Rules

When running on Google Colab, the training script must not train directly from Google Drive.

Required behavior:

```text
1. Check whether data.zip exists on Google Drive.
2. Check whether /content/dataset exists and is non-empty.
3. If the local dataset does not exist, extract data.zip to /content/dataset.
4. Train from /content/dataset.
5. Save checkpoints locally.
6. Copy checkpoints back to Google Drive periodically.
```

Use Python `zipfile`:

```python
with zipfile.ZipFile(drive_zip_path, "r") as zip_ref:
    zip_ref.extractall(local_dataset_dir)
```

Do not rely on notebook-only shell commands inside production `train.py`.

---

## 18. Checkpoint Rules

Checkpoint saving must be safe for Colab runtime interruption.

Required checkpoint content:

```python
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "train_metrics": train_metrics,
    "val_metrics": val_metrics,
    "best_val_loss": best_val_loss,
    "config": config,
}
```

Required behavior:

```text
- Save checkpoint every 10 epochs by default.
- Save best_model.pth when validation total loss improves.
- Save locally first.
- Copy checkpoint to Google Drive if Colab mode is enabled.
```

---

## 19. Documentation Rules

### 19.1. Markdown and LaTeX

LaTeX formulas are allowed only in prose sections.

Never render LaTeX formulas inside fenced code blocks.

Correct usage: write mathematical formulas in normal Markdown prose.

[
\mathcal{L}_{total}
===================

\mathcal{L}*{CrossEntropy}
+
\gamma \cdot \mathcal{L}*{SmoothL1}
]

Incorrect usage: placing mathematical formulas inside `python`, `text`, `json`, `bash`, or any other fenced code block.

---

### 19.2. Code Blocks

Code blocks must contain executable code or plain text configuration only.

Allowed code block types:

```text
python
json
bash
text
csv
```

Do not mix prose explanation and executable code in the same code block.

---

### 19.3. Comments

Use comments for tensor shapes and non-obvious logic.

Good:

```python
features = self.backbone(images)  # Shape: [B, 2048, H_prime, W_prime]
```

Bad:

```python
# This extracts useful features from the image.
features = self.backbone(images)
```

Comments must clarify implementation, not restate generic theory.

---

## 20. Error Handling Rules

Use explicit runtime checks for critical assumptions.

Example:

```python
if images.ndim != 4:
    raise ValueError(f"images must have shape [B, 3, H, W], got {tuple(images.shape)}")

if attention_maps.ndim != 4:
    raise ValueError(
        f"attention_maps must have shape [B, 1, H, W], got {tuple(attention_maps.shape)}"
    )

if images.shape[0] != attention_maps.shape[0]:
    raise ValueError("images and attention_maps must have the same batch size.")
```

Do not silently reshape tensors unless the transformation is explicitly required by the architecture.

---

## 21. Forbidden Implementation Patterns

The following patterns are not allowed:

```text
- Running YOLOv8 inside model.forward()
- Reading CSV inside models/architecture.py
- Reading images inside models/architecture.py
- Importing ResNet50 inside utils/dataset.py
- Returning file paths from Dataset instead of tensors
- Applying ImageNet normalization to Attention Maps
- Applying Softmax before CrossEntropyLoss
- Computing loss inside the model
- Saving checkpoints inside the model
- Hard-coding feature map size as 7x7
- Training directly from Google Drive
- Using unnormalized meter coordinates as regression labels
```

---

## 22. Required Read Order for AI Agents

Before coding, read the files in this order:

```text
1. AI_STATE.md
2. AI_RULES.md
3. README.md
4. DATA_PIPELINE_SPEC.md
5. MODEL_SPECIFICATION.md
6. TRAINING_SCRIPT_SPEC.md
7. config.json
```

When implementing a specific module:

```text
- For preprocess_offline.py, follow DATA_PIPELINE_SPEC.md.
- For utils/dataset.py, follow DATA_PIPELINE_SPEC.md.
- For models/architecture.py, follow MODEL_SPECIFICATION.md.
- For train.py, follow TRAINING_SCRIPT_SPEC.md.
```

If any instruction conflicts with this file, preserve the rules in `AI_RULES.md` and the sticky constraints in `AI_STATE.md`.

---

## 23. Completion Criteria

A code contribution is valid only if:

```text
- It respects Separation of Concerns.
- It uses explicit type hints.
- It includes tensor shape comments for important tensors.
- It keeps Feature-level Attention at ResNet50 Layer 4.
- It returns correct tensor shapes.
- It keeps coordinate labels normalized into [0, 1].
- It computes multi-task loss only inside train.py.
- It does not introduce forbidden cross-module dependencies.
```

---

## 24. Final Rule

Do not optimize by breaking architecture.

Correctness of module boundaries is more important than reducing the number of files or writing shorter code.
