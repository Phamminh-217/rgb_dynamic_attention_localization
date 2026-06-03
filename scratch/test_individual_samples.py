import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models.architecture import FeatureAttentionHierarchicalNet
from scripts.inference import denormalize_coordinates, build_attention_map_live

def main():
    config_path = "config.json"
    model_path = "checkpoints/best_model.pth"
    
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
    
    # Select 3 random frames from Room (0) and 3 from Corridor (1)
    np.random.seed(42)  # For reproducible sample selection
    room_samples = df[df["topo_label"] == 0].sample(3)
    corridor_samples = df[df["topo_label"] == 1].sample(3)
    test_samples = pd.concat([room_samples, corridor_samples])
    
    print("\n==========================================================================================")
    print("INDIVIDUAL SAMPLES VERIFICATION RUN (Comparing Prediction vs. Ground Truth)")
    print("==========================================================================================")
    
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
    
    results = []
    
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
        
        # Compute errors
        distance_error_cm = np.sqrt((true_x - pred_x)**2 + (true_y - pred_y)**2) * 100
        
        print(f"\n[Test Frame: {img_path.name}]")
        print(f"  - Topological Area : TRUE = {true_topo:<8} | PRED = {pred_topo:<8} (Confidence: {confidence:.2f}%)")
        print(f"  - Coordinate X     : TRUE = {true_x:+.4f}m   | PRED = {pred_x:+.4f}m   (Error: {abs(true_x - pred_x)*100:.2f} cm)")
        print(f"  - Coordinate Y     : TRUE = {true_y:+.4f}m   | PRED = {pred_y:+.4f}m   (Error: {abs(true_y - pred_y)*100:.2f} cm)")
        print(f"  - Spatial Distance Error: {distance_error_cm:.2f} cm")
        
        results.append({
            "File": img_path.name,
            "True_Topo": true_topo,
            "Pred_Topo": pred_topo,
            "Error_cm": distance_error_cm
        })
        
    print("\n==========================================================================================")
    mean_sample_err = np.mean([r["Error_cm"] for r in results])
    print(f"Verification completed. Average Distance Error on these 6 random samples: {mean_sample_err:.2f} cm")
    print("==========================================================================================\n")

if __name__ == "__main__":
    main()
