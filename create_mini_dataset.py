import json
import pandas as pd
from pathlib import Path

def main():
    # Define paths
    original_csv_path = Path("data/processed/total_poses.csv")
    mini_csv_path = Path("data/processed/total_poses_mini.csv")
    original_config_path = Path("config.json")
    mini_config_path = Path("config_mini.json")

    if not original_csv_path.exists():
        print(f"Error: Unified poses CSV not found at '{original_csv_path}'.")
        print("Please ensure you have completed preprocessing first.")
        return

    # Load original dataset
    print(f"Loading unified dataset from {original_csv_path}...")
    df = pd.read_csv(original_csv_path)
    total_samples = len(df)
    print(f"Total samples available: {total_samples}")

    # Set sample size (e.g., ~5% of the dataset, around 190 samples)
    sample_size = 200
    
    # Stratified sampling to maintain class balance between Room (0) and Corridor (1)
    print(f"Sampling {sample_size} records while maintaining class distribution...")
    mini_df = df.groupby("topo_label", group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), sample_size // 2), random_state=42)
    )

    # Save the sampled dataframe
    mini_csv_path.parent.mkdir(parents=True, exist_ok=True)
    mini_df.to_csv(mini_csv_path, index=False)
    print(f"Successfully saved mini dataset ({len(mini_df)} samples) to: {mini_csv_path}")
    print(f"Class breakdown in mini dataset:\n{mini_df['topo_label'].value_counts()}")

    # Create config_mini.json from config.json
    if original_config_path.exists():
        print(f"\nGenerating '{mini_config_path}' from '{original_config_path}'...")
        with open(original_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Point to the mini dataset
        config["data"]["total_csv"] = str(mini_csv_path)
        
        # Optimize hyperparameters for local rapid training
        config["training"]["epochs"] = 10                  # Low epoch limit for quick local test runs
        config["data"]["batch_size"] = 8                    # Smaller batch size for CPU/low-end GPU local run
        config["data"]["num_workers"] = 0                   # Disable multi-processing overhead for small dataset
        
        # Disable Colab drive integration for local training runs
        config["colab"]["enabled"] = False

        with open(mini_config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"Successfully created '{mini_config_path}' for fast local debugging!")
        print("\nTo train the mini local model, simply run:")
        print(f"  python3 train.py --config {mini_config_path}")
    else:
        print(f"Warning: '{original_config_path}' not found. Skipping config generation.")

if __name__ == "__main__":
    main()
