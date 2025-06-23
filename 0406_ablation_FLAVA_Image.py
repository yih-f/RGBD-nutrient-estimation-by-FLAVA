import os
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import torch.optim as optim
from datetime import datetime
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import swin_v2_t, Swin_V2_T_Weights
from transformers import FlavaProcessor, FlavaTextModel,FlavaImageModel
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Suppress Warning

# Define paths
rgbd_dir = '/home/yif22003/generic_food/nutrition5k_rgbd'
metadata_file = '/home/yif22003/generic_food/nutrition5k_metadata/N5K_rgbd_dish_updated.txt'
train_ids_file = '/home/yif22003/generic_food/nutrition5k_metadata/0404-train_split_ratio4to1.txt'
test_ids_file = '/home/yif22003/generic_food/nutrition5k_metadata/0404-test_split_ratio4to1.txt'
results_dir = f'/home/yif22003/generic_food/code_results/Ablation_studies/040625_FLAVA_Image'
os.makedirs(results_dir, exist_ok=True) # Create the results directory
print("use updated ingredient text metadata-N5K_rgbd_dish_updated.txt")
print("use randow 4:1 splits")

# Define parameters
initial_lr = 5e-4
eta_min=1e-7 # End point learning rate for the scheduler
batch_size=64
weight_decay=1e-3
num_epochs =100
vfw = 200
tvw = 0 # Set tvw=0 to study the ablation of text-visual contrastive loss
similarity_temperature=0.4
contrastive_weights = {
    'text_visual_weight': tvw,  # Weight for text-visual contrastive loss
    'visual_flava_weight': vfw,  # Weight for visual-flava contrastive loss
    'temperature': similarity_temperature  # Temperature for similarity scaling
}   
depth_clipping_flag=1 #Flag to clip >4000 depth pixels to 4000
image_size_before_CenterCrop=260
image_size=256
random_seed=42 
# Record parameters in bash out file
print(f"contrastive_weights_tvw_{tvw}_vfw_{vfw},similarity_temperature={similarity_temperature},initial_lr={initial_lr},image_size={image_size},num_epochs={num_epochs},batch_size={batch_size},weight_decay={weight_decay},eta_min={eta_min},random_seed={random_seed},image_size_before_CenterCrop={image_size_before_CenterCrop},depth_clipping_flag={depth_clipping_flag}")

# Set random seed for reproducibility
torch.manual_seed(random_seed)
np.random.seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class Nutrition5kDataset(Dataset):
    def __init__(self, dish_ids_file, metadata_file, rgbd_dir, transform=None):
        """
        Nutrition5k dataset with ingredient information

        Args:
            dish_ids_file: Path to file containing dish IDs
            metadata_file: Path to nutrition metadata file
            rgbd_dir: Directory containing RGB-D images
            transform: Transformations to apply to images
        """
        # Load dish IDs from file
        self.dish_ids = []
        with open(dish_ids_file, 'r') as f:
            self.dish_ids = [line.strip() for line in f]
        print(f"Loading metadata for {len(self.dish_ids)} dishes...")

        # Load nutrition metadata and ingredient information
        self.nutrition_data = {}
        self.ingredient_data = {}

        with open(metadata_file, 'r') as f:
            for line in tqdm(f, desc="Processing metadata"):
                parts = line.strip().split(',')
                dish_id = parts[0]
                if dish_id in self.dish_ids:
                    # Extract nutritional values
                    self.nutrition_data[dish_id] = {
                        'calories': float(parts[1]),
                        'mass': float(parts[2]),
                        'fat': float(parts[3]),
                        'carb': float(parts[4]),
                        'protein': float(parts[5])
                    }

                    # Extract ingredient information
                    ingredients = []
                    # Each ingredient has 7 fields, starting from index 6
                    for i in range(6, len(parts), 7):
                        if i+2 < len(parts):  # Ensure we have the ingredient name and gram weight
                            ingredient_name = parts[i+1]
                            ingredient_grams = float(parts[i+2])  # Convert grams to float
                            ingredients.append(f"{ingredient_grams:.1f}g {ingredient_name}") 

                    # Create ingredient text description
                    if ingredients:
                        ingredient_text = "A plate of dish contains "
                        ingredient_text += ", ".join(ingredients[:])
                        ingredient_text +="."
                    else:
                        ingredient_text = "A dish."

                    self.ingredient_data[dish_id] = ingredient_text

        # Keep only dish IDs that have both nutrition data and ingredient data
        self.dish_ids = [dish_id for dish_id in self.dish_ids
                        if dish_id in self.nutrition_data and dish_id in self.ingredient_data]
        print(f"Found {len(self.dish_ids)} dishes with nutrition and ingredient data")

        # Store parameters
        self.rgbd_dir = rgbd_dir
        self.transform = transform

    def __len__(self):
        return len(self.dish_ids)

    def __getitem__(self, idx):
        dish_id = self.dish_ids[idx]
        rgb_path = os.path.join(self.rgbd_dir, dish_id, 'rgb.png')
        depth_path = os.path.join(self.rgbd_dir, dish_id, 'depth_raw.png')

        try:
            # Load RGB image
            rgb_img = Image.open(rgb_path).convert('RGB')
            
            # Load depth image (uint16 format)
            depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)  # uint16 format
            if depth_clipping_flag==1:
                # Trim values ≥ 4000 to 4000
                depth_img = np.clip(depth_img, 0, 4000)
                # Normalize depth to 0-255 for easier processing
                depth_img = (depth_img / 4000.0 * 255).astype(np.uint8)
            else:
                max_depth = np.max(depth_img)
                if max_depth == 0:  # Avoid division by zero
                    max_depth = 1
                depth_img = (depth_img / max_depth * 255).astype(np.uint8)
            # Convert single-channel depth to 3-channel by repeating
            depth_img_3channel = np.stack([depth_img, depth_img, depth_img], axis=2)

            # Convert depth to PIL Image for transformation
            depth_img = Image.fromarray(depth_img_3channel)

            # Apply transforms
            if self.transform:
                rgb_img = self.transform(rgb_img)
                depth_img = self.transform(depth_img)

            # Get nutritional values
            nutritional_values = torch.tensor([
                self.nutrition_data[dish_id]['calories'],
                self.nutrition_data[dish_id]['mass'],
                self.nutrition_data[dish_id]['fat'],
                self.nutrition_data[dish_id]['carb'],
                self.nutrition_data[dish_id]['protein']
            ], dtype=torch.float32)

            # Get ingredient text
            ingredient_text = self.ingredient_data[dish_id]

            return rgb_img, depth_img, nutritional_values, ingredient_text, dish_id

        except Exception as e:
            print(f"Error loading {dish_id}: {e}")
            # Return default values
            input_size = image_size  # This should match the expected input size after transforms
            return torch.zeros(3, input_size, input_size), torch.zeros(3, input_size, input_size), torch.zeros(5), "", dish_id
        
# Define transformations - for Swin V2 Tiny pretrained weights
train_transform = transforms.Compose([
    transforms.Resize(image_size_before_CenterCrop, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(image_size),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize(image_size_before_CenterCrop, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = Nutrition5kDataset(
    train_ids_file,
    metadata_file,
    rgbd_dir,
    train_transform
)
test_dataset = Nutrition5kDataset(
    test_ids_file,
    metadata_file,
    rgbd_dir,
    test_transform
)
print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# Function to preload images by running an empty loop
def preload_images():
    print("Preloading training images...")
    for rgb_imgs, depth_imgs, targets, ingredient_texts, dish_ids in tqdm(train_loader, desc="Training data"):
        # This is an empty loop, just iterate through loader to cache images
        pass

    print("Preloading test images...")
    for rgb_imgs, depth_imgs, targets, ingredient_texts, dish_ids in tqdm(test_loader, desc="Test data"):
        # This is an empty loop, just iterate through loader to cache images
        pass

    print("\nAll images preloaded!")

class VisualProcessor(nn.Module):
    def __init__(self, num_classes=5):
        super(VisualProcessor, self).__init__()

        # Load Swin V2 Tiny models for RGB and Depth
        weights = Swin_V2_T_Weights.IMAGENET1K_V1

        # Create RGB model
        self.rgb_model = swin_v2_t(weights=weights)
        # Create Depth model
        self.depth_model = swin_v2_t(weights=weights)

        # Remove the classification heads
        feature_dim = self.rgb_model.head.in_features  # This is 768 for Swin V2 Tiny
        self.rgb_model.head = nn.Identity()
        self.depth_model.head = nn.Identity()

        # Define feature dimensions for each stage
        # Swin V2 Tiny has 4 stages with dimensions: [96, 192, 384, 768]
        self.stage_dims = [96, 192, 384, 768]

        # Create feature extraction dictionaries for each stage
        self.rgb_features = {}
        self.depth_features = {}

        # Feature mixer modules for each stage (RGB + Depth only)
        self.stage_mixers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim*2, dim),  # RGB + Depth
                nn.LayerNorm(dim),
                nn.GELU()
            ) for dim in self.stage_dims
        ])

        # Feature projectors to common dimension
        self.stage_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 128),
                nn.GELU()
            ) for dim in self.stage_dims
        ])

        # Final prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(128*4, 256),  # 4 stages with 128-dim projections
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        # Register hooks to get intermediate features
        self._register_hooks()

    def _register_hooks(self):
        """Register hooks to extract features from each stage"""

        def get_hook(stage_name, feature_dict):
            def hook(module, input, output):
                # For Swin, we perform global average pooling on the output
                # Output shape is [B, H, W, C] -> Pooled shape is [B, C]
                # First permute to [B, C, H, W] for pooling
                pooled = output.permute(0, 3, 1, 2)
                pooled = F.adaptive_avg_pool2d(pooled, 1).flatten(1)
                feature_dict[stage_name] = pooled
            return hook

        # Register hooks for RGB backbone
        self.rgb_model.features[1].register_forward_hook(get_hook("stage0", self.rgb_features))  # First stage
        self.rgb_model.features[3].register_forward_hook(get_hook("stage1", self.rgb_features))  # Second stage
        self.rgb_model.features[5].register_forward_hook(get_hook("stage2", self.rgb_features))  # Third stage
        self.rgb_model.features[7].register_forward_hook(get_hook("stage3", self.rgb_features))  # Fourth stage

        # Register hooks for Depth backbone
        self.depth_model.features[1].register_forward_hook(get_hook("stage0", self.depth_features))  # First stage
        self.depth_model.features[3].register_forward_hook(get_hook("stage1", self.depth_features))  # Second stage
        self.depth_model.features[5].register_forward_hook(get_hook("stage2", self.depth_features))  # Third stage
        self.depth_model.features[7].register_forward_hook(get_hook("stage3", self.depth_features))  # Fourth stage

    def forward(self, rgb, depth):
        """
        Forward pass through the visual processing
        
        Args:
            rgb: RGB images, tensor of shape [B, 3, H, W]
            depth: Depth images, tensor of shape [B, 3, H, W]
            
        Returns:
            predictions: Nutritional predictions, tensor of shape [B, 5]
            final_visual_features: Final stage visual features for alignment, tensor of shape [B, 768]
        """
        # Forward pass through RGB model
        _ = self.rgb_model(rgb)

        # Forward pass through Depth model
        _ = self.depth_model(depth)

        # Collect and mix features from each stage
        mixed_features = []

        for i in range(4):  # 4 stages
            stage_name = f"stage{i}"

            # Get RGB and Depth features
            rgb_feat = self.rgb_features[stage_name]
            depth_feat = self.depth_features[stage_name]

            # Concatenate RGB and Depth features
            concat_feat = torch.cat([rgb_feat, depth_feat], dim=1)

            # Mix features
            mixed_feat = self.stage_mixers[i](concat_feat)

            # Project to common dimension
            projected_feat = self.stage_projectors[i](mixed_feat)

            mixed_features.append(projected_feat)

        # Concatenate all mixed features for final prediction
        all_features = torch.cat(mixed_features, dim=1)

        # Final prediction
        predictions = self.prediction_head(all_features)

        # Return predictions and final visual features (for alignment, if needed)
        final_visual_features = self.rgb_features["stage3"]  # Use final stage RGB features
        
        return predictions, final_visual_features
    
class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        
        # Load FLAVA text model
        self.text_model = FlavaTextModel.from_pretrained("facebook/flava-full")
        
        # Freeze text model parameters
        for param in self.text_model.parameters():
            param.requires_grad = False
            
        # Initialize text processor
        self.processor = FlavaProcessor.from_pretrained("facebook/flava-full")
        
    def forward(self, text_list):
        """
        Encode text using FLAVA
        
        Args:
            text_list: List of text strings to encode
            
        Returns:
            text_features: Text features, tensor of shape [B, 768]
        """
        if not text_list:
            return None
            
        # Move to the same device as the model
        device = next(self.parameters()).device
        
        # Process text with FLAVA processor
        # Max token size is 512 for FlavaProcessor. 245 token size is enough for the longest ingredient text.
        inputs = self.processor(text=text_list, return_tensors="pt", padding='max_length', 
                                truncation=True, max_length=245)
                                
        # Move inputs to the device
        inputs = {k: v.to(device) for k, v in inputs.items() if k != 'pixel_values'}
        
        # Get text features from FLAVA text model
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            # Use the CLS token embedding as text representation
            text_features = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
            
        return text_features
    
class FlavaImageEncoder(nn.Module):
    def __init__(self):
        super(FlavaImageEncoder, self).__init__()
        
        # Load FLAVA image model
        self.image_model = FlavaImageModel.from_pretrained("facebook/flava-full")
        
        # Freeze image model parameters
        for param in self.image_model.parameters():
            param.requires_grad = False
            
        # Initialize image processor
        self.processor = FlavaProcessor.from_pretrained("facebook/flava-full")
        
    def forward(self, images):
        """
        Encode images using FLAVA
        
        Args:
            images: RGB images, tensor of shape [B, 3, H, W]
            
        Returns:
            image_features: Image features, tensor of shape [B, 768]
        """
        # Move to the same device as the model
        device = next(self.parameters()).device
        
        # Convert from normalized tensors back to PIL Images for FLAVA processor
        # This is necessary because FLAVA has its own normalization
        # First denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        images_denorm = images * std + mean
        
        # Convert to PIL images
        images_pil = []
        for img in images_denorm.cpu():
            img = img.permute(1, 2, 0).numpy()
            img = (img * 255).astype(np.uint8)
            images_pil.append(Image.fromarray(img))
        
        # Process images with FLAVA processor
        inputs = self.processor(images=images_pil, return_tensors="pt")
        
        # Move inputs to the device
        inputs = {k: v.to(device) for k, v in inputs.items() if k == 'pixel_values'}
        
        # Get image features from FLAVA image model
        with torch.no_grad():
            outputs = self.image_model(**inputs)
            # Use the CLS token embedding as image representation
            image_features = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
            
        return image_features
    
class FeatureAlignmentProjectors(nn.Module):
    def __init__(self, swin_dim=768, flava_dim=768):
        super(FeatureAlignmentProjectors, self).__init__()
        
        # Projection from Swin features to FLAVA text space
        self.swin_to_text_projector = nn.Sequential(
            nn.Linear(swin_dim, flava_dim),
            nn.LayerNorm(flava_dim)
        )
        
        # Projection from Swin features to FLAVA image space
        self.swin_to_flava_img_projector = nn.Sequential(
            nn.Linear(swin_dim, flava_dim),
            nn.LayerNorm(flava_dim)
        )
        
    def swin_to_text(self, swin_features):
        """Project Swin visual features to FLAVA text space"""
        return self.swin_to_text_projector(swin_features)
    
    def swin_to_flava_img(self, swin_features):
        """Project Swin visual features to FLAVA image space"""
        return self.swin_to_flava_img_projector(swin_features)

class EnhancedNutritionPredictionModel(nn.Module):
    def __init__(self, num_classes=5):
        super(EnhancedNutritionPredictionModel, self).__init__()
        
        # Initialize components
        self.visual_processor = VisualProcessor(num_classes)
        self.text_encoder = TextEncoder()
        self.flava_image_encoder = FlavaImageEncoder()
        self.feature_projectors = FeatureAlignmentProjectors()
        
    def forward(self, rgb, depth, text_list=None):
        """
        Forward pass through the model
        
        Args:
            rgb: RGB images, tensor of shape [B, 3, H, W]
            depth: Depth images, tensor of shape [B, 3, H, W]
            text_list: List of ingredient text descriptions (optional)
            
        Returns:
            If text_list is None (inference mode):
                predictions: Nutrition predictions, tensor of shape [B, 5]
            If text_list is provided (training mode):
                (predictions, visual_text_embeddings, text_embeddings, visual_image_embeddings, flava_image_embeddings)
        """
        # Get predictions and visual features from visual processor
        predictions, swin_visual_features = self.visual_processor(rgb, depth)
        
        # If in inference mode (no text provided), just return predictions
        if text_list is None:
            return predictions
            
        # If in training mode, prepare all features for the various contrastive losses
        
        # Project Swin visual features to text space for text-visual contrastive loss
        visual_text_embeddings = self.feature_projectors.swin_to_text(swin_visual_features)
        
        # Project Swin visual features to FLAVA image space for visual-visual contrastive loss
        visual_image_embeddings = self.feature_projectors.swin_to_flava_img(swin_visual_features)
        
        # Encode text with FLAVA text encoder
        text_embeddings = self.text_encoder(text_list)
        
        # Encode RGB images with FLAVA image encoder
        flava_image_embeddings = self.flava_image_encoder(rgb)
        
        # Return all embeddings for contrastive learning
        return predictions, visual_text_embeddings, text_embeddings, visual_image_embeddings, flava_image_embeddings
    
    def get_inference_model(self):
        """
        Return a deployment-ready model containing only the visual processing components
        """
        return self.visual_processor

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
        #Values for WeightedMSELoss based on training set means:
        nutrient_scales = torch.tensor([257.40, 217.18, 12.86, 19.24, 18.44])
        print("use values for WeightedMSELoss based on training set means")

        # Weights inversely proportional to square of scales
        # Makes smaller nutrients (fat, carb, protein) more important
        self.weights = 1.0 / (nutrient_scales ** 2)

        # Normalize weights to sum to number of nutrients (5)
        self.weights = self.weights * (len(nutrient_scales) / self.weights.sum())

    def forward(self, predictions, targets):
        weights = self.weights.to(predictions.device)
        squared_diff = (predictions - targets) ** 2
        weighted_squared_diff = squared_diff * weights.unsqueeze(0)
        return torch.mean(weighted_squared_diff)
    
def calculate_contrastive_losses(visual_text_embeddings, text_embeddings, 
                               visual_image_embeddings, flava_image_embeddings,
                               temperature=0.4):
    """
    Calculate multiple contrastive losses to align different feature spaces
    
    Args:
        visual_text_embeddings: Swin features projected to text space [B, 768]
        text_embeddings: FLAVA text embeddings [B, 768]
        visual_image_embeddings: Swin features projected to FLAVA image space [B, 768]
        flava_image_embeddings: FLAVA image embeddings [B, 768]
        temperature: Temperature parameter for scaling similarity scores
    
    Returns:
        dict of contrastive losses:
            - text_visual_loss: Swin-to-Text alignment loss
            - visual_flava_loss: Swin-to-FLAVA-Image alignment loss
    """
    # Normalize all feature vectors to unit length
    visual_text_embeddings = F.normalize(visual_text_embeddings, dim=1)
    text_embeddings = F.normalize(text_embeddings, dim=1)
    visual_image_embeddings = F.normalize(visual_image_embeddings, dim=1)
    flava_image_embeddings = F.normalize(flava_image_embeddings, dim=1)
    
    # Create identity matrix as ground truth (batch_size x batch_size)
    batch_size = visual_text_embeddings.size(0)
    labels = torch.arange(batch_size, device=visual_text_embeddings.device)
    
    # Calculate text-visual contrastive loss
    text_visual_sim = torch.matmul(visual_text_embeddings, text_embeddings.t())
    text_visual_loss = F.cross_entropy(text_visual_sim / temperature, labels)
    
    # Calculate visual-flava contrastive loss
    visual_flava_sim = torch.matmul(visual_image_embeddings, flava_image_embeddings.t())
    visual_flava_loss = F.cross_entropy(visual_flava_sim / temperature, labels)
    
    return {
        'text_visual_loss': text_visual_loss,
        'visual_flava_loss': visual_flava_loss
    }

def train_enhanced_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device, 
                       num_epochs, contrastive_weights):
    """
    Train the enhanced nutrition prediction model with multiple contrastive losses
    
    Args:
        model: EnhancedNutritionPredictionModel
        train_loader: DataLoader for training data
        test_loader: DataLoader for testing data
        criterion: Loss function for nutritional prediction (e.g., WeightedMSELoss)
        optimizer: Optimizer (e.g., AdamW)
        scheduler: Learning rate scheduler
        device: Device to run training on (cuda or cpu)
        num_epochs: Number of training epochs
        contrastive_weights: Dict with weights for different contrastive losses
            - text_visual_weight: Weight for Swin-to-Text alignment
            - visual_flava_weight: Weight for Swin-to-FLAVA-Image alignment
    """
    # Lists to track metrics
    train_losses = []
    test_losses = []
    lr_history = []
    mse_losses = []
    contrastive_losses = {
        'text_visual': [],
        'visual_flava': []
    }
    
    # Get contrastive weights
    text_visual_weight = contrastive_weights.get('text_visual_weight', 200)
    visual_flava_weight = contrastive_weights.get('visual_flava_weight', 120)
    temperature = contrastive_weights.get('temperature', 0.4)
    
    # Best model tracking
    best_test_loss = float('inf')
    best_model_path = os.path.join(results_dir, 'best_model.pth')

    # Start training
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_mse_loss = 0.0
        running_text_visual_loss = 0.0
        running_visual_flava_loss = 0.0
        
        # Training loop
        for rgb_imgs, depth_imgs, targets, ingredient_texts, _ in train_loader:
            rgb_imgs = rgb_imgs.to(device)
            depth_imgs = depth_imgs.to(device)
            targets = targets.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass with text
            outputs = model(rgb_imgs, depth_imgs, ingredient_texts)
            predictions, visual_text_embeddings, text_embeddings, visual_image_embeddings, flava_image_embeddings = outputs
            
            # Main loss (MSE)
            mse_loss = criterion(predictions, targets)
            
            # Contrastive losses
            contrastive_losses_dict = calculate_contrastive_losses(
                visual_text_embeddings, text_embeddings,
                visual_image_embeddings, flava_image_embeddings,
                temperature=temperature
            )
            
            text_visual_loss = contrastive_losses_dict['text_visual_loss']
            visual_flava_loss = contrastive_losses_dict['visual_flava_loss']
            
            # Combine all losses
            loss = mse_loss + text_visual_weight * text_visual_loss + visual_flava_weight * visual_flava_loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Track losses
            batch_size = rgb_imgs.size(0)
            running_loss += loss.item() * batch_size
            running_mse_loss += mse_loss.item() * batch_size
            running_text_visual_loss += text_visual_loss.item() * batch_size
            running_visual_flava_loss += visual_flava_loss.item() * batch_size
            
        # Calculate average losses
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_mse_loss = running_mse_loss / len(train_loader.dataset)
        epoch_text_visual_loss = running_text_visual_loss / len(train_loader.dataset)
        epoch_visual_flava_loss = running_visual_flava_loss / len(train_loader.dataset)
        
        # Save the losses
        train_losses.append(epoch_train_loss)
        mse_losses.append(epoch_mse_loss)
        contrastive_losses['text_visual'].append(epoch_text_visual_loss)
        contrastive_losses['visual_flava'].append(epoch_visual_flava_loss)
        
        # Step scheduler
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        lr_history.append(current_lr)

        # Evaluate on test set
        model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for rgb_imgs, depth_imgs, targets, _, _ in test_loader:
                rgb_imgs = rgb_imgs.to(device)
                depth_imgs = depth_imgs.to(device)
                targets = targets.to(device)

                # For testing, we only need the predictions (no alignment losses)
                outputs = model.visual_processor(rgb_imgs, depth_imgs)[0]
                loss = criterion(outputs, targets)

                test_loss += loss.item() * rgb_imgs.size(0)

        # Calculate average test loss
        epoch_test_loss = test_loss / len(test_loader.dataset)
        test_losses.append(epoch_test_loss)

        # Print epoch results with all losses
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"MSE Loss: {epoch_mse_loss:.4f} | "
              f"Text-Visual Loss: {epoch_text_visual_loss:.4f} | "
              f"Visual-FLAVA Loss: {epoch_visual_flava_loss:.4f} | "
              f"Test Loss: {epoch_test_loss:.4f} | "
              f"LR: {current_lr:.6f}")
        
        if epoch==0:
            # Get GPU memory usage
            memory_stats = get_gpu_memory_usage()
            print(f"GPU Memory: Allocated: {memory_stats['allocated_mb']:.2f} MB | "
                f"Reserved: {memory_stats['reserved_mb']:.2f} MB | "
                f"Max Allocated: {memory_stats['max_allocated_mb']:.2f} MB")   

        # Save best model
        if epoch_test_loss < best_test_loss:
            best_test_loss = epoch_test_loss
            
            # For deployment, we only need to save the visual processor
            torch.save(model.visual_processor.state_dict(), best_model_path)
            
            print(f"Model saved with test loss: {best_test_loss:.4f}")

        # Save checkpoint after each epoch
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses,
            'test_losses': test_losses,
            'mse_losses': mse_losses,
            'contrastive_losses': contrastive_losses,
            'lr_history': lr_history,
        }, os.path.join(results_dir, 'latest_checkpoint.pth'))

    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'mse_losses': mse_losses,
        'contrastive_losses': contrastive_losses,
        'lr_history': lr_history
    }

def evaluate_with_visual_processor(visual_processor, test_loader, device):
    visual_processor.eval()

    # Lists to store results
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for rgb_imgs, depth_imgs, targets, _, _ in tqdm(test_loader, desc="Evaluating"):
            rgb_imgs = rgb_imgs.to(device)
            depth_imgs = depth_imgs.to(device)

            # Get predictions from visual processor
            outputs, _ = visual_processor(rgb_imgs, depth_imgs)

            all_targets.append(targets.numpy())
            all_predictions.append(outputs.cpu().numpy())

    # Combine all batches
    all_targets = np.vstack(all_targets)
    all_predictions = np.vstack(all_predictions)

    # Calculate metrics
    mean_targets = np.mean(all_targets, axis=0)
    mae = np.mean(np.abs(all_predictions - all_targets), axis=0)
    pmae = (mae / mean_targets) * 100

    # Prepare results
    nutrient_names = ['Calories', 'Mass', 'Fat', 'Carb', 'Protein']
    results = {}

    print("\nEvaluation Results:")
    print(f"{'Nutrient':<10} {'MAE':<10} {'PMAE (%)':<10} {'Mean Value':<10}")
    print("-" * 50)

    for i, name in enumerate(nutrient_names):
        print(f"{name:<10} {mae[i]:.4f} {pmae[i]:.2f}% {mean_targets[i]:.4f}")
        results[name] = {'MAE': mae[i], 'PMAE': pmae[i], 'Mean': mean_targets[i]}

    avg_mae = np.mean(mae)
    avg_pmae = np.mean(pmae)
    print(f"{'Average':<10} {avg_mae:.4f} {avg_pmae:.2f}%")

    return results, all_targets, all_predictions

# Function to plot and save loss curves
def plot_enhanced_loss_curves(history):
    """
    Plot and save loss curves for the enhanced model training
    
    Args:
        history: Dictionary containing training history
            - train_losses: List of total training losses
            - test_losses: List of test losses
            - mse_losses: List of MSE losses
            - contrastive_losses: Dict with lists of different contrastive losses
            - lr_history: List of learning rates
    """
    plt.figure(figsize=(20, 15))

    # Plot total training and test loss
    plt.subplot(3, 2, 1)
    plt.plot(history['train_losses'], label='Train Loss (Total)')
    plt.plot(history['test_losses'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)

    # Plot MSE loss
    plt.subplot(3, 2, 2)
    plt.plot(history['mse_losses'], label='MSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Nutrition Prediction Loss')
    plt.legend()
    plt.grid(True)

    # Plot text-visual contrastive loss
    plt.subplot(3, 2, 3)
    plt.plot(history['contrastive_losses']['text_visual'], label='Text-Visual Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Text-Visual Alignment Loss')
    plt.legend()
    plt.grid(True)

    # Plot visual-flava contrastive loss
    plt.subplot(3, 2, 4)
    plt.plot(history['contrastive_losses']['visual_flava'], label='Visual-FLAVA Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Visual-FLAVA Alignment Loss')
    plt.legend()
    plt.grid(True)

    # Plot learning rate
    plt.subplot(3, 2, 5)
    plt.plot(history['lr_history'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'enhanced_loss_curves.png'))
    plt.close()

def save_training_history(history, results_dir):
    """
    Save the training history to a CSV file
    
    Args:
        history: Dictionary containing training history
        results_dir: Directory to save the CSV file
    """
    # Create a DataFrame with all history data
    history_df = pd.DataFrame({
        'epoch': range(1, len(history['train_losses']) + 1),
        'train_loss': history['train_losses'],
        'test_loss': history['test_losses'],
        'mse_loss': history['mse_losses'],
        'text_visual_loss': history['contrastive_losses']['text_visual'],
        'visual_flava_loss': history['contrastive_losses']['visual_flava'],
        'learning_rate': history['lr_history']
    })
    
    # Save to CSV
    history_df.to_csv(os.path.join(results_dir, 'training_history.csv'), index=False)
    print(f"Training history saved to {os.path.join(results_dir, 'training_history.csv')}")

def get_gpu_memory_usage():
    """Get the current GPU memory usage in MB"""
    if not torch.cuda.is_available():
        return "GPU not available"
    
    # Get the current device
    device = torch.cuda.current_device()
    
    # Get memory stats
    allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)  # Convert to MB
    reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)    # Convert to MB
    max_allocated = torch.cuda.max_memory_allocated(device) / (1024 * 1024)  # Convert to MB
    
    return {
        "allocated_mb": allocated,
        "reserved_mb": reserved,
        "max_allocated_mb": max_allocated
    }

# Run the preloading
preload_images()

# Initialize the enhanced model
model = EnhancedNutritionPredictionModel().to(device)

# Loss function
criterion = WeightedMSELoss()
# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
# Learning rate scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eta_min)

# Train the model
history = train_enhanced_model(
    model, train_loader, test_loader, criterion, optimizer, scheduler, 
    device, num_epochs, contrastive_weights
)

# Plot loss curves
plot_enhanced_loss_curves(history)
# Save training history to CSV
save_training_history(history, results_dir)

# Load best model for evaluation
deployment_model = VisualProcessor().to(device)
deployment_model.load_state_dict(torch.load(os.path.join(results_dir, 'best_model.pth')))
results, targets, predictions = evaluate_with_visual_processor(deployment_model, test_loader, device)

# Save results to text file
with open(os.path.join(results_dir, 'results.txt'), 'w') as f:
    f.write("SwinV2 Tiny Hierarchical Feature Mixing for RGBD Nutrition Prediction\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Training samples: {len(train_dataset)}\n")
    f.write(f"Test samples: {len(test_dataset)}\n\n")

    f.write("Model Architecture: Swin Transformer V2 Tiny with Hierarchical Feature Mixing\n")
    f.write("Training parameters:\n")
    f.write(f"  - Batch size: {batch_size}\n")
    f.write(f"  - Initial learning rate: {initial_lr}\n")
    f.write(f"  - Optimizer: AdamW with weight decay 0.01\n")
    f.write(f"  - Scheduler: CosineAnnealingLR (T_max={num_epochs}, eta_min=1e-7)\n\n")

    f.write("Evaluation Results:\n")
    f.write(f"{'Nutrient':<10} {'MAE':<10} {'PMAE (%)':<10} {'Mean Value':<10}\n")
    f.write("-" * 50 + "\n")

    nutrient_names = ['Calories', 'Mass', 'Fat', 'Carb', 'Protein']
    maes = []
    pmaes = []

    for name in nutrient_names:
        mae = results[name]['MAE']
        pmae = results[name]['PMAE']
        mean = results[name]['Mean']

        maes.append(mae)
        pmaes.append(pmae)

        f.write(f"{name:<10} {mae:.4f} {pmae:.2f}% {mean:.4f}\n")

    avg_mae = np.mean(maes)
    avg_pmae = np.mean(pmaes)

    f.write("-" * 50 + "\n")
    f.write(f"{'Average':<10} {avg_mae:.4f} {avg_pmae:.2f}%\n\n")