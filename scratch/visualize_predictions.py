import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision import transforms

from models.architecture import FeatureAttentionHierarchicalNet
from scripts.inference import denormalize_coordinates, build_attention_map_live

def main():
    config_path = "config.json"
    model_path = "checkpoints/best_model.pth"
    output_dir = Path("checkpoints/inference_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = FeatureAttentionHierarchicalNet(
        num_classes_topo=int(config["model"]["num_classes_topo"]),
        pretrained=False,
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Load total poses CSV
    total_csv = Path(config["data"]["total_csv"])
    df = pd.read_csv(total_csv)
    
    # Select the exact same 6 samples as the verification run for consistency
    np.random.seed(42)
    room_samples = df[df["topo_label"] == 0].sample(3)
    corridor_samples = df[df["topo_label"] == 1].sample(3)
    test_samples = pd.concat([room_samples, corridor_samples])
    
    image_size = config["data"]["image_size"]
    h, w = int(image_size[0]), int(image_size[1])
    mean = config["image_normalization"]["mean"]
    std = config["image_normalization"]["std"]
    
    image_transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    attention_transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
    ])
    
    alpha = float(config.get("attention", {}).get("alpha", 0.7))
    norm_bounds = config["global_normalization"]
    x_min, x_max = norm_bounds["x_min"], norm_bounds["x_max"]
    y_min, y_max = norm_bounds["y_min"], norm_bounds["y_max"]
    
    print(f"\nGenerating visual output images to: {output_dir}/")
    
    for idx, row in test_samples.iterrows():
        img_path = Path(row["image_path"])
        true_topo = "Room" if row["topo_label"] == 0 else "Corridor"
        true_x, true_y = float(row["x"]), float(row["y"])
        
        # Build attention map
        attention_map = build_attention_map_live(img_path, alpha)
        
        # Transform inputs
        image = Image.open(img_path).convert("RGB")
        image_tensor = image_transform(image).unsqueeze(0).to(device)
        attention_tensor = attention_transform(attention_map).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            topo_logits, coord_pred = model(image_tensor, attention_tensor)
            
        # Parse outputs
        topo_prob = torch.softmax(topo_logits, dim=1).squeeze(0).cpu().numpy()
        pred_class = int(np.argmax(topo_prob))
        pred_topo = "Corridor" if pred_class == 1 else "Room"
        confidence = topo_prob[pred_class] * 100
        
        coord_normalized = coord_pred.squeeze(0).cpu().numpy()
        pred_x, pred_y = denormalize_coordinates(
            coord_normalized[0], coord_normalized[1],
            x_min, x_max, y_min, y_max
        )
        
        distance_error_cm = np.sqrt((true_x - pred_x)**2 + (true_y - pred_y)**2) * 100
        
        # Plotting side-by-side using Matplotlib
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
        
        # Left Panel: Original Image
        axes[0].imshow(image)
        axes[0].set_title(f"Original Image: {img_path.name}")
        axes[0].axis("off")
        
        # Middle Panel: Attention Map (Grayscale)
        axes[1].imshow(attention_map, cmap="gray")
        axes[1].set_title("Generated Attention Map (YOLOv8)")
        axes[1].axis("off")
        
        # Right Panel: 2D Coordinates comparison
        # Add a local boundary around the true point for context
        pad_val = 0.5
        axes[2].plot(true_x, true_y, "go", markersize=12, label=f"GT ({true_x:+.2f}, {true_y:+.2f})")
        axes[2].plot(pred_x, pred_y, "rx", markersize=12, markeredgewidth=3, label=f"Pred ({pred_x:+.2f}, {pred_y:+.2f})")
        axes[2].plot([true_x, pred_x], [true_y, pred_y], "k--", alpha=0.6, label=f"Error: {distance_error_cm:.2f} cm")
        
        axes[2].set_xlim(min(true_x, pred_x) - pad_val, max(true_x, pred_x) + pad_val)
        axes[2].set_ylim(min(true_y, pred_y) - pad_val, max(true_y, pred_y) + pad_val)
        axes[2].set_xlabel("X Coordinate (m)")
        axes[2].set_ylabel("Y Coordinate (m)")
        axes[2].set_title(f"Position Compare ({true_topo} -> {pred_topo})")
        axes[2].legend(loc="upper right", frameon=True, edgecolor="black")
        axes[2].grid(True, linestyle=":", alpha=0.6)
        
        plt.suptitle(
            f"Visual Localization Report - {img_path.name}\n"
            f"Area Accuracy: {true_topo} == {pred_topo} ({confidence:.1f}% Conf) | Distance Error: {distance_error_cm:.2f} cm",
            fontsize=12, fontweight="bold", y=0.98
        )
        
        plt.tight_layout()
        out_path = output_dir / f"{img_path.stem}_result.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"  - Saved visualization to: {out_path}")
        
    print("\nVisual verification images generation finished successfully!")

if __name__ == "__main__":
    main()
