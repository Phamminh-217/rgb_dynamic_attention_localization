import os
import sys
import json
from pathlib import Path

# Add project root to sys.path to allow correct package imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, Tuple

# Import architecture dynamically to follow system design
from models.architecture import FeatureAttentionHierarchicalNet
from utils.dataset import RobotLocalizationDataset

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate_model(
    model_path: str,
    config_path: str,
    device_name: str = "cpu"
) -> None:
    """Evaluate localization model, compute metrics, and generate PDF report visualizations.
    
    Generates:
      1. Confusion Matrix (classification errors visualization).
      2. Trajectory Maps (Ground Truth vs Predictions comparison).
      3. Statistical Error Report (Average, Max, Min errors per Area).
    """
    config = load_config(config_path)
    device = torch.device(device_name if (device_name == "cpu" or torch.cuda.is_available()) else "cpu")
    
    # Create output directories conforming to paper figures
    res_dir = Path("results")
    fig2_dir = res_dir / "figure2"
    fig3_dir = res_dir / "figure3"
    fig5_dir = res_dir / "figure5"
    fig6_dir = res_dir / "figure6"
    
    fig2_dir.mkdir(parents=True, exist_ok=True)
    fig3_dir.mkdir(parents=True, exist_ok=True)
    fig5_dir.mkdir(parents=True, exist_ok=True)
    fig6_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    total_csv = config["data"]["total_csv"]
    dataset = RobotLocalizationDataset(total_csv=total_csv, config_path=config_path)
    
    # We will split data deterministically using the seed configured in training
    train_ratio = float(config["data"]["train_ratio"])
    val_ratio = float(config["data"]["val_ratio"])
    test_ratio = float(config["data"].get("test_ratio", 0.1))

    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    _, _, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config["project"]["seed"])
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=False
    )
    
    # Load model architecture
    model = FeatureAttentionHierarchicalNet(
        num_classes_topo=int(config["model"]["num_classes_topo"]),
        pretrained=False
    )
    
    print(f"Loading checkpoint weights from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    all_preds_topo = []
    all_gts_topo = []
    all_preds_coord = []
    all_gts_coord = []
    
    # Bounds for denormalization
    x_min = float(config["global_normalization"]["x_min"])
    x_max = float(config["global_normalization"]["x_max"])
    y_min = float(config["global_normalization"]["y_min"])
    y_max = float(config["global_normalization"]["y_max"])
    
    print("Evaluating on Test Set...")
    with torch.no_grad():
        for images, attention_maps, topo_labels, coord_labels in test_loader:
            images = images.to(device)
            attention_maps = attention_maps.to(device)
            
            topo_logits, coord_pred = model(images, attention_maps)
            
            preds_topo = torch.argmax(topo_logits, dim=1).cpu().numpy()
            all_preds_topo.extend(preds_topo)
            all_gts_topo.extend(topo_labels.numpy())
            
            all_preds_coord.extend(coord_pred.cpu().numpy())
            all_gts_coord.extend(coord_labels.numpy())

    all_preds_topo = np.array(all_preds_topo)
    all_gts_topo = np.array(all_gts_topo)
    all_preds_coord = np.array(all_preds_coord)
    all_gts_coord = np.array(all_gts_coord)

    # Denormalize predictions and ground truths to meters scale
    pred_x_m = all_preds_coord[:, 0] * (x_max - x_min) + x_min
    pred_y_m = all_preds_coord[:, 1] * (y_max - y_min) + y_min
    gt_x_m = all_gts_coord[:, 0] * (x_max - x_min) + x_min
    gt_y_m = all_gts_coord[:, 1] * (y_max - y_min) + y_min

    # Calculate Euclidean distance errors
    errors_m = np.sqrt((pred_x_m - gt_x_m)**2 + (pred_y_m - gt_y_m)**2)
    
    # -----------------------------------------------------------------------
    # 1. Generate Confusion Matrix Plot (Figure 2 equivalent)
    # -----------------------------------------------------------------------
    labels = sorted(list(config["labels"].keys())) # ['A', 'B', 'C', 'D', 'E']
    cm = confusion_matrix(all_gts_topo, all_preds_topo)
    
    plt.figure(figsize=(8, 6), dpi=150)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Figure 2: Topological Classification Confusion Matrix", fontsize=12, pad=10)
    plt.xlabel("Predicted Area", fontsize=10)
    plt.ylabel("Actual Area", fontsize=10)
    plt.tight_layout()
    cm_plot_path = fig2_dir / "confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {cm_plot_path}")

    # Export Figure 2 data to txt
    cm_txt_path = fig2_dir / "confusion_matrix.txt"
    with open(cm_txt_path, "w", encoding="utf-8") as f:
        f.write("Topological Area Confusion Matrix\n")
        f.write("Rows: Actual Area, Columns: Predicted Area\n\n")
        header = "      " + "     ".join(labels)
        f.write(header + "\n")
        for i, row in enumerate(cm):
            row_str = f"{labels[i]:<4} " + "  ".join(f"{val:>4d}" for val in row)
            f.write(row_str + "\n")
    print(f"Saved: {cm_txt_path}")

    # -----------------------------------------------------------------------
    # 2. Generate Trajectory Representation per Area (Figure 3 equivalent)
    # -----------------------------------------------------------------------
    # Color mapping for areas
    area_colors = {"A": "red", "B": "green", "C": "blue", "D": "orange", "E": "purple"}
    
    plt.figure(figsize=(10, 8), dpi=150)
    # Draw predictions colored by predicted room class
    for label_name, color in area_colors.items():
        label_val = config["labels"][label_name]
        mask = (all_preds_topo == label_val)
        if np.any(mask):
            plt.scatter(
                pred_x_m[mask], pred_y_m[mask], 
                c=color, s=15, 
                label=f"Predicted Area {label_name}", 
                alpha=0.6, edgecolors='none'
            )
            
    plt.title("Figure 3: Predicted Trajectory Points Colored by Area Label", fontsize=12)
    plt.xlabel("X (meters)", fontsize=10)
    plt.ylabel("Y (meters)", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc="best", frameon=True)
    plt.axis("equal")
    plt.tight_layout()
    fig3_plot_path = fig3_dir / "trajectory_by_area.png"
    plt.savefig(fig3_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {fig3_plot_path}")

    # Export Figure 3 data to txt (tab-separated, sorted by true area A-E)
    fig3_txt_path = fig3_dir / "trajectory_by_area.txt"
    df_fig3 = pd.DataFrame({
        "pred_x_m": pred_x_m,
        "pred_y_m": pred_y_m,
        "predicted_area": [labels[val] for val in all_preds_topo],
        "true_area": [labels[val] for val in all_gts_topo]
    })
    df_fig3 = df_fig3.sort_values(by="true_area")
    df_fig3.to_csv(fig3_txt_path, sep="\t", index=False)
    print(f"Saved: {fig3_txt_path}")

    plt.figure(figsize=(12, 10), dpi=150)
    # Plot Ground Truth Path as scatter (only points)
    plt.scatter(gt_x_m, gt_y_m, color="#FFA500", s=15, label="Ground Truth (Encoder)", alpha=0.8)
    # Plot Predictions
    plt.scatter(pred_x_m, pred_y_m, c="#1E90FF", s=15, label="Estimated Pose (CNN)", alpha=0.6, zorder=5)

    plt.title("Figure 5: Global Representation of Predicted Trajectory vs Ground Truth", fontsize=13)
    plt.xlabel("X (meters)", fontsize=11)
    plt.ylabel("Y (meters)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best", frameon=True)
    plt.axis("equal")
    plt.tight_layout()
    traj_plot_path = fig5_dir / "trajectory_comparison.png"
    plt.savefig(traj_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {traj_plot_path}")

    # Export Figure 5 data to txt (tab-separated, sorted by true area A-E)
    fig5_txt_path = fig5_dir / "trajectory_comparison.txt"
    df_fig5 = pd.DataFrame({
        "gt_x_m": gt_x_m,
        "gt_y_m": gt_y_m,
        "pred_x_m": pred_x_m,
        "pred_y_m": pred_y_m,
        "true_area": [labels[val] for val in all_gts_topo]
    })
    df_fig5 = df_fig5.sort_values(by="true_area")
    df_fig5.to_csv(fig5_txt_path, sep="\t", index=False)
    print(f"Saved: {fig5_txt_path}")

    # -----------------------------------------------------------------------
    # 4. Generate Classification report & Regression Error stats (Figure 6 / Table 3 equivalent)
    # -----------------------------------------------------------------------
    report_dict = classification_report(all_gts_topo, all_preds_topo, target_names=labels, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.to_csv(fig6_dir / "classification_report.csv", index=True)
    
    area_metrics = []
    for val, label in enumerate(labels):
        mask = (all_gts_topo == val)
        if not np.any(mask):
            continue
        area_errs = errors_m[mask]
        area_metrics.append({
            "Area": label,
            "Mean Error (m)": np.mean(area_errs),
            "Max Error (m)": np.max(area_errs),
            "Min Error (m)": np.min(area_errs),
            "Std Dev (m)": np.std(area_errs)
        })
    
    # Overall metrics
    area_metrics.append({
        "Area": "AVERAGE / GLOBAL",
        "Mean Error (m)": np.mean(errors_m),
        "Max Error (m)": np.max(errors_m),
        "Min Error (m)": np.min(errors_m),
        "Std Dev (m)": np.std(errors_m)
    })
    
    df_metrics = pd.DataFrame(area_metrics)
    df_metrics.to_csv(fig6_dir / "localization_errors_report.csv", index=False)
    
    # -----------------------------------------------------------------------
    # Table 3 calculation (as defined in TABLE.md)
    # -----------------------------------------------------------------------
    table3_rows = [
        "1. Average error X (m)",
        "2. Maximum error X (m)",
        "3. Minimum error X (m)",
        "4. Error deviation in X (m)",
        "5. Average error Y (m)",
        "6. Maximum error Y (m)",
        "7. Minimum error Y (m)",
        "8. Error deviation in Y (m)",
        "9. Average error Euclidean (m)",
        "10. Maximum error Euclidean (m)",
        "11. Minimum error Euclidean (m)",
        "12. Error deviation Euclidean (m)"
    ]
    
    table3_data = {"Metrics": table3_rows}
    errors_x = np.abs(pred_x_m - gt_x_m)
    errors_y = np.abs(pred_y_m - gt_y_m)
    errors_euc = errors_m
    
    for val, label in enumerate(labels):
        mask = (all_gts_topo == val)
        if np.any(mask):
            x_m = errors_x[mask]
            y_m = errors_y[mask]
            euc_m = errors_euc[mask]
            
            table3_data[label] = [
                np.mean(x_m), np.max(x_m), np.min(x_m), np.std(x_m),
                np.mean(y_m), np.max(y_m), np.min(y_m), np.std(y_m),
                np.mean(euc_m), np.max(euc_m), np.min(euc_m), np.std(euc_m)
            ]
            
    table3_data["Average"] = [
        np.mean(errors_x), np.max(errors_x), np.min(errors_x), np.std(errors_x),
        np.mean(errors_y), np.max(errors_y), np.min(errors_y), np.std(errors_y),
        np.mean(errors_euc), np.max(errors_euc), np.min(errors_euc), np.std(errors_euc)
    ]
    
    df_table3 = pd.DataFrame(table3_data)
    df_table3.to_csv(fig6_dir / "table3_metrics.csv", index=False)
    
    # Save formatted markdown table to txt
    table3_txt_path = fig6_dir / "table3_metrics.txt"
    with open(table3_txt_path, "w", encoding="utf-8") as f:
        headers = ["Metrics"] + labels + ["Average"]
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| :--- | " + " | ".join([":---:" for _ in range(len(headers) - 1)]) + " |"
        f.write(header_line + "\n")
        f.write(sep_line + "\n")
        for idx, row_name in enumerate(table3_rows):
            row_vals = []
            for col in labels + ["Average"]:
                row_vals.append(f"{table3_data[col][idx]:.6f}")
            f.write(f"| {row_name} | " + " | ".join(row_vals) + " |\n")
            
    print(f"Saved: {fig6_dir / 'table3_metrics.csv'}")
    print(f"Saved: {table3_txt_path}")
    print(f"Saved reports under: {fig6_dir}")
    
    # Generate Error Comparison Bar Chart (Figure 6 equivalent)
    plt.figure(figsize=(10, 6), dpi=150)
    # Filter out total average for plotting individual area performance
    plot_df = df_metrics[df_metrics["Area"] != "AVERAGE / GLOBAL"]
    
    x_indices = np.arange(len(plot_df))
    width = 0.35
    
    plt.bar(x_indices - width/2, plot_df["Mean Error (m)"], width, label="Mean Error", color="#1E90FF")
    plt.bar(x_indices + width/2, plot_df["Std Dev (m)"], width, label="Std Dev", color="#FFA500")
    
    plt.title("Figure 6: Localization Translation Errors per Topological Area", fontsize=12)
    plt.xlabel("Topological Area", fontsize=10)
    plt.ylabel("Error magnitude (meters)", fontsize=10)
    plt.xticks(x_indices, plot_df["Area"])
    plt.grid(True, linestyle=":", alpha=0.5, axis='y')
    plt.legend(frameon=True)
    plt.tight_layout()
    fig6_plot_path = fig6_dir / "errors_bar_chart.png"
    plt.savefig(fig6_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {fig6_plot_path}")
    
    # -----------------------------------------------------------------------
    # 5. Export Model Performance Summary (Topo and Regression metrics)
    # -----------------------------------------------------------------------
    summary_path = fig6_dir / "model_performance_summary.txt"
    
    topo_acc = np.mean(all_preds_topo == all_gts_topo)
    topo_precision = report_dict["macro avg"]["precision"]
    topo_recall = report_dict["macro avg"]["recall"]
    topo_f1 = report_dict["macro avg"]["f1-score"]
    
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=========================================================\n")
        f.write("             MODEL PERFORMANCE SUMMARY REPORT            \n")
        f.write("=========================================================\n\n")
        
        f.write("1. TOPOLOGICAL CLASSIFICATION BRANCH PERFORMANCE:\n")
        f.write(f"  - Accuracy:                  {topo_acc * 100:.4f}%\n")
        f.write(f"  - Precision (Macro Average): {topo_precision * 100:.4f}%\n")
        f.write(f"  - Recall (Macro Average):    {topo_recall * 100:.4f}%\n")
        f.write(f"  - F1-Score (Macro Average):  {topo_f1 * 100:.4f}%\n\n")
        
        f.write("  Detailed Topo Branch Report:\n")
        f.write(df_report.to_string() + "\n\n")
        
        f.write("2. REGRESSION POSITIONING BRANCH PERFORMANCE:\n")
        f.write(f"  - Mean Absolute Error X (MAE X): {np.mean(errors_x):.6f} m\n")
        f.write(f"  - Mean Absolute Error Y (MAE Y): {np.mean(errors_y):.6f} m\n")
        f.write(f"  - Mean Euclidean Distance Error: {np.mean(errors_euc):.6f} m\n")
        f.write(f"  - RMSE X: {np.sqrt(np.mean(errors_x**2)):.6f} m\n")
        f.write(f"  - RMSE Y: {np.sqrt(np.mean(errors_y**2)):.6f} m\n")
        f.write(f"  - RMSE Euclidean: {np.sqrt(np.mean(errors_euc**2)):.6f} m\n\n")
        
        f.write("  Localization Success Metrics (at various tolerance thresholds):\n")
        f.write(f"  {'Threshold':<12} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}\n")
        f.write("  " + "-"*66 + "\n")
        for th in [0.5, 1.0, 1.5, 2.0]:
            success = (errors_euc <= th)
            acc = np.mean(success)
            tp = np.sum(success)
            fn = np.sum(~success)
            fp = 0
            prec = 1.0 if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            
            f.write(f"  {f'{th}m':<12} | {acc*100:>9.2f}% | {prec*100:>9.2f}% | {rec*100:>9.2f}% | {f1*100:>9.2f}%\n")
            
        f.write("\n=========================================================\n")
        
    print(f"Saved: {summary_path}")
    
    print("\n==========================================")
    print("EVALUATION METRICS SUMMARY:")
    print(df_metrics.to_string(index=False))
    print("==========================================\n")

def plot_training_history(history_path: str, output_dir: Path) -> None:
    """Plot training and validation curves (Loss and Accuracy) from training_history.txt."""
    if not os.path.exists(history_path):
        print(f"Warning: History file not found at {history_path}")
        return
        
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(history_path, sep="\t")
    
    plt.figure(figsize=(15, 10), dpi=150)
    
    # 1. Total Loss
    plt.subplot(2, 2, 1)
    plt.plot(df['Epoch'], df['Train_Total_Loss'], label='Train Total Loss', color='blue', linewidth=1.5)
    plt.plot(df['Epoch'], df['Val_Total_Loss'], label='Val Total Loss', color='red', linestyle='--', linewidth=1.5)
    plt.title('Total Loss Curve', fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    
    # Set y-limit to avoid outlier skewing the scale (specifically epoch 78)
    clean_val = df['Val_Total_Loss'][df['Val_Total_Loss'] < 5.0]
    if len(clean_val) > 0:
        plt.ylim(0, max(clean_val.max() * 1.2, df['Train_Total_Loss'].max() * 1.2))
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True)
    
    # 2. Topo Loss
    plt.subplot(2, 2, 2)
    plt.plot(df['Epoch'], df['Train_Topo_Loss'], label='Train Topo Loss', color='blue', linewidth=1.5)
    plt.plot(df['Epoch'], df['Val_Topo_Loss'], label='Val Topo Loss', color='red', linestyle='--', linewidth=1.5)
    plt.title('Topological Classification Loss', fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    
    clean_topo_val = df['Val_Topo_Loss'][df['Val_Topo_Loss'] < 5.0]
    if len(clean_topo_val) > 0:
        plt.ylim(0, max(clean_topo_val.max() * 1.2, df['Train_Topo_Loss'].max() * 1.2))
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True)
    
    # 3. Coord Loss
    plt.subplot(2, 2, 3)
    plt.plot(df['Epoch'], df['Train_Coord_Loss'], label='Train Coord Loss', color='blue', linewidth=1.5)
    plt.plot(df['Epoch'], df['Val_Coord_Loss'], label='Val Coord Loss', color='red', linestyle='--', linewidth=1.5)
    plt.title('Coordinate Regression Loss', fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    
    clean_coord_val = df['Val_Coord_Loss'][df['Val_Coord_Loss'] < 1.0]
    if len(clean_coord_val) > 0:
        plt.ylim(0, max(clean_coord_val.max() * 1.2, df['Train_Coord_Loss'].max() * 1.2))
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True)
    
    # 4. Topo Accuracy
    plt.subplot(2, 2, 4)
    plt.plot(df['Epoch'], df['Train_Topo_Acc'], label='Train Topo Acc', color='blue', linewidth=1.5)
    plt.plot(df['Epoch'], df['Val_Topo_Acc'], label='Val Topo Acc', color='red', linestyle='--', linewidth=1.5)
    plt.title('Topological Classification Accuracy', fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True)
    
    plt.tight_layout()
    plot_path = output_dir / "training_loss_curves.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved training curves: {plot_path}")

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Model Evaluator conforming to Paper specifications")
    parser.add_argument("--model", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run evaluation ('cpu' or 'cuda')")
    parser.add_argument("--history", type=str, default="checkpoints/training_history.txt", help="Path to training_history.txt")
    
    args = parser.parse_args()
    evaluate_model(args.model, args.config, args.device)
    
    # Generate training curves under results/figure1/
    plot_training_history(args.history, Path("results/figure1"))
