import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple


class FeatureAttentionHierarchicalNet(nn.Module):
    def __init__(
        self,
        num_classes_topo: int,
        pretrained: bool = True,
        dropout_p: float = 0.3,
    ) -> None:
        """Initialize FeatureAttentionHierarchicalNet with ResNet50 backbone, 

        feature-level attention, and parallel multi-task heads.
        """
        super().__init__()

        if num_classes_topo <= 0:
            raise ValueError("num_classes_topo must be a positive integer.")

        # Load ResNet50 backbone with or without pre-trained weights
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
        else:
            weights = None

        resnet = models.resnet50(weights=weights)

        # Slice ResNet50 up to end of layer4, discarding avgpool and fc layers
        self.backbone = nn.Sequential(
            resnet.conv1,       # Shape: [B, 64, H/2, W/2]
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,     # Shape: [B, 64, H/4, W/4]
            resnet.layer1,      # Shape: [B, 256, H/4, W/4]
            resnet.layer2,      # Shape: [B, 512, H/8, W/8]
            resnet.layer3,      # Shape: [B, 1024, H/16, W/16]
            resnet.layer4,      # Shape: [B, 2048, H/32, W/32]
        )

        # Global Average Pooling to reduce spatial dimensions to 1x1
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Topological Classification MLP Head
        self.topo_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes_topo),
        )

        # Coordinate Regression MLP Head
        self.coord_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform forward pass using dynamic-aware Feature-level Attention fusion.

        Args:
            x: Input RGB image batch of shape [B, 3, H, W]
            attention_map: Grayscale attention map batch of shape [B, 1, H, W]

        Returns:
            topo_logits: Raw classification logits of shape [B, num_classes_topo]
            coord_pred: Predicted spatial coordinates of shape [B, 2]
        """
        # Validate inputs according to project runtime checks
        if x.ndim != 4:
            raise ValueError(
                f"images must have shape [B, 3, H, W], got {tuple(x.shape)}"
            )

        if attention_map.ndim != 4:
            raise ValueError(
                f"attention_maps must have shape [B, 1, H, W], got {tuple(attention_map.shape)}"
            )

        if x.shape[0] != attention_map.shape[0]:
            raise ValueError("images and attention_maps must have the same batch size.")

        if x.shape[1] != 3:
            raise ValueError(
                f"images channel dimension must be 3, got {x.shape[1]}"
            )

        if attention_map.shape[1] != 1:
            raise ValueError(
                f"attention_maps channel dimension must be 1, got {attention_map.shape[1]}"
            )

        # Extract deep features using Backbone ResNet50 Layer 4
        features = self.backbone(x)  # Shape: [B, 2048, H_prime, W_prime]

        # Resample attention map to match deep feature map resolution dynamically
        attention_resized = F.interpolate(
            attention_map,
            size=features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )  # Shape: [B, 1, H_prime, W_prime]

        # Element-wise multiplication to apply dynamic-aware attention at feature-level
        attended_features = features * attention_resized  # Shape: [B, 2048, H_prime, W_prime]

        # Apply Global Average Pooling
        pooled = self.gap(attended_features)  # Shape: [B, 2048, 1, 1]

        # Flatten into shared latent vector
        z_t = torch.flatten(pooled, 1)  # Shape: [B, 2048]

        # Run independent parallel MLP heads
        topo_logits = self.topo_head(z_t)  # Shape: [B, num_classes_topo]
        coord_pred = self.coord_head(z_t)  # Shape: [B, 2]

        return topo_logits, coord_pred


if __name__ == "__main__":
    # Configure auto device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize mock batch parameters
    B = 2
    H, W = 224, 224
    num_classes_topo = 2

    # Instantiate model
    model = FeatureAttentionHierarchicalNet(
        num_classes_topo=num_classes_topo,
        pretrained=False,
    ).to(device)
    
    # Generate dummy input tensors representing raw RGB images and offline masks
    x_dummy = torch.randn(B, 3, H, W, dtype=torch.float32).to(device)         # Shape: [B, 3, H, W]
    attention_dummy = torch.randn(B, 1, H, W, dtype=torch.float32).to(device) # Shape: [B, 1, H, W]

    print("\n--- Running Forward Pass Verification ---")
    try:
        # Run forward pass through the model
        topo_logits, coord_pred = model(x_dummy, attention_dummy)
        
        print("Forward pass completed successfully without errors!")
        print(f"Topological Logits Shape   : {list(topo_logits.shape)} (Expected: [{B}, {num_classes_topo}])")
        print(f"Coordinate Regression Shape: {list(coord_pred.shape)} (Expected: [{B}, 2])")
        
        # Additional assertion tests
        assert topo_logits.shape == (B, num_classes_topo)
        assert coord_pred.shape == (B, 2)
        print("Tensor shape assertions passed successfully!")
    except Exception as e:
        print(f"An error occurred during verification: {e}")
