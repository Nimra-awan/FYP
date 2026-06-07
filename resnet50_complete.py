import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import glob
from sklearn.metrics import confusion_matrix, classification_report

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}

# Custom Dataset class
class MedicalImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Load dataset with pre-split option
def load_dataset(data_dir, batch_size=32, use_presplit=False):
    train_images = []
    train_labels = []
    val_images = []
    val_labels = []
    test_images = []
    test_labels = []
    class_names = []
    
    if use_presplit:
        # Use pre-split data (train/val/test folders)
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(data_dir, split)
            if os.path.exists(split_dir):
                for class_idx, class_name in enumerate(sorted(os.listdir(split_dir))):
                    if class_name not in class_names:
                        class_names.append(class_name)
                    
                    class_dir = os.path.join(split_dir, class_name)
                    if os.path.isdir(class_dir):
                        for img_name in os.listdir(class_dir):
                            img_path = os.path.join(class_dir, img_name)
                            if os.path.isfile(img_path) and (img_path.lower().endswith('.png') or 
                                                           img_path.lower().endswith('.jpg') or 
                                                           img_path.lower().endswith('.jpeg')):
                                if split == 'train':
                                    train_images.append(img_path)
                                    train_labels.append(class_idx)
                                elif split == 'val':
                                    val_images.append(img_path)
                                    val_labels.append(class_idx)
                                elif split == 'test':
                                    test_images.append(img_path)
                                    test_labels.append(class_idx)
    else:
        # Split data manually (80/10/10)
        for class_idx, class_name in enumerate(sorted(os.listdir(data_dir))):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                class_names.append(class_name)
                image_paths = []
                
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if os.path.isfile(img_path) and (img_path.lower().endswith('.png') or 
                                                   img_path.lower().endswith('.jpg') or 
                                                   img_path.lower().endswith('.jpeg')):
                        image_paths.append(img_path)
                
                # Shuffle and split
                np.random.shuffle(image_paths)
                n_train = int(0.8 * len(image_paths))
                n_val = int(0.1 * len(image_paths))
                
                train_images.extend(image_paths[:n_train])
                train_labels.extend([class_idx] * n_train)
                
                val_images.extend(image_paths[n_train:n_train+n_val])
                val_labels.extend([class_idx] * n_val)
                
                test_images.extend(image_paths[n_train+n_val:])
                test_labels.extend([class_idx] * (len(image_paths) - n_train - n_val))
    
    # Create datasets
    train_dataset = MedicalImageDataset(train_images, train_labels, transform=data_transforms['train'])
    val_dataset = MedicalImageDataset(val_images, val_labels, transform=data_transforms['test']) if val_images else None
    test_dataset = MedicalImageDataset(test_images, test_labels, transform=data_transforms['test']) if test_images else None
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0) if test_dataset else None
    
    return train_loader, val_loader, test_loader, class_names

# Initialize ResNet50 model
def initialize_model(num_classes):
    model = models.resnet50(pretrained=True)
    
    # Freeze early layers
    for param in list(model.parameters())[:-20]:
        param.requires_grad = False
    
    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

# Train the model
def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation phase
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_loss = val_loss / len(val_loader)
            val_acc = 100 * val_correct / val_total
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save the best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, 'best_model.pth')
        else:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    
    return model

# Evaluate the model
def evaluate_model(model, test_loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print('Confusion Matrix:')
    print(cm)
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print('Classification Report:')
    print(report)
    
    return accuracy

# Save the model
def save_model(model, model_path, class_names=None):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
    
    # Save the model
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'class_names': class_names
    }
    
    torch.save(checkpoint, model_path)
    print(f'Model saved to {model_path}')

# Load the model
def load_model(model_path, num_classes=None):
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get class names
    class_names = checkpoint.get('class_names', None)
    
    # Determine number of classes
    if num_classes is None:
        if class_names is not None:
            num_classes = len(class_names)
        elif 'model_state_dict' in checkpoint:
            # Try to infer from the model's fc layer
            if 'fc.weight' in checkpoint['model_state_dict']:
                num_classes = checkpoint['model_state_dict']['fc.weight'].size(0)
            else:
                num_classes = 2  # Default to binary classification
        else:
            num_classes = 2  # Default to binary classification
    
    # Initialize model
    model = initialize_model(num_classes)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Try to adapt the checkpoint to the model
        try:
            model.load_state_dict(checkpoint)
        except:
            # The model might be from a different architecture
            # Let's try to load a MobileNetV3 model instead
            model = models.mobilenet_v3_large(pretrained=False)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
            
            try:
                # Try to load directly
                model.load_state_dict(checkpoint)
            except:
                # Try to extract features if it's a different format
                if isinstance(checkpoint, dict) and 'features' in checkpoint:
                    model.load_state_dict(checkpoint)
                else:
                    print("Warning: Could not load model weights exactly. Using model with random weights.")
    
    model = model.to(device)
    model.eval()
    
    if class_names is None:
        class_names = ['CN', 'AD']  # Default class names for Alzheimer's dataset
    
    print(f"Loaded model with {num_classes} classes: {class_names}")
    return model, class_names

# Simplified GradCAM implementation
def generate_gradcam(model, image_tensor, target_class=None):
    """
    A simplified GradCAM implementation that doesn't rely on hooks
    and works with various model architectures.
    """
    # Set model to evaluation mode
    model.eval()
    
    # Create a copy of the tensor that requires gradients
    input_tensor = image_tensor.clone().detach().to(device)
    input_tensor.requires_grad = True
    
    # Get the model's prediction if target class not provided
    if target_class is None:
        with torch.no_grad():
            output = model(image_tensor)
            target_class = output.argmax().item()
    
    # Create a basic heatmap using the model's attention
    try:
        # Forward pass with gradient tracking
        output = model(input_tensor)
        
        # Get the score for the target class
        score = output[0, target_class]
        
        # Backward pass to get gradients
        model.zero_grad()
        score.backward()
        
        # Get the gradients of the input image
        gradients = input_tensor.grad.data
        
        # Take the maximum absolute value across color channels
        pooled_gradients = torch.mean(torch.abs(gradients), dim=[0, 2, 3], keepdim=True)
        
        # Weight the channels by corresponding gradients
        weighted_image = pooled_gradients * input_tensor
        
        # Average the weighted channels
        cam = torch.mean(weighted_image, dim=1).squeeze().detach().cpu().numpy()
        
        # Apply ReLU to focus on features that have a positive influence
        cam = np.maximum(cam, 0)
        
        # Normalize between 0-1
        cam = cam - np.min(cam)
        cam_max = np.max(cam)
        if cam_max != 0:
            cam = cam / cam_max
        
        # Resize to input size if needed
        if cam.shape != (image_tensor.shape[2], image_tensor.shape[3]):
            cam = cv2.resize(cam, (image_tensor.shape[3], image_tensor.shape[2]))
        
        return cam
    
    except Exception as e:
        print(f"Error generating simplified GradCAM: {str(e)}")
        # Return a simple attention map based on the image itself
        img_np = image_tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
        gray = np.mean(img_np, axis=2)
        # Normalize
        gray = gray - np.min(gray)
        gray_max = np.max(gray)
        if gray_max != 0:
            gray = gray / gray_max
        return gray

# Visualize GradCAM
def visualize_gradcam(model, image_tensor, class_names, target_class=None, alpha=0.5, save_path=None):
    # Get the class index if not provided
    if target_class is None:
        with torch.no_grad():
            output = model(image_tensor)
            target_class = output.argmax().item()
    
    # Generate GradCAM
    cam = generate_gradcam(model, image_tensor, target_class)
    
    # Convert tensor to image
    image = image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    image = (image * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    image = np.clip(image, 0, 1)
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    
    # Overlay heatmap on image
    result = (1 - alpha) * image + alpha * heatmap
    result = np.clip(result, 0, 1)
    
    # Create figure
    plt.figure(figsize=(12, 4))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title(f'Original - Class: {class_names[target_class]}')
    plt.axis('off')
    
    # Heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title('GradCAM Heatmap')
    plt.axis('off')
    
    # Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(result)
    plt.title(f'Overlay (alpha={alpha:.2f})')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved GradCAM visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()

# Batch GradCAM visualization
def batch_gradcam(model_path, test_dir, output_dir, num_per_class=3, alpha=0.5):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model, class_names = load_model(model_path)
    
    # Process each class directory
    for class_name in os.listdir(test_dir):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        # Get image files
        image_files = glob.glob(os.path.join(class_dir, "*.png"))
        if not image_files:
            image_files = glob.glob(os.path.join(class_dir, "*.jpg"))
        
        if not image_files:
            print(f"No images found in {class_dir}")
            continue
        
        # Process limited number of images
        for i, img_path in enumerate(image_files[:num_per_class]):
            try:
                # Load and preprocess image
                img = Image.open(img_path).convert('RGB')
                input_tensor = data_transforms['test'](img).unsqueeze(0).to(device)
                
                # Get class index
                target_class = class_names.index(class_name) if class_name in class_names else 0
                
                # Generate output filename
                base_name = os.path.basename(img_path)
                output_path = os.path.join(output_dir, f"{class_name}_{i+1}_{base_name}")
                
                # Generate and save GradCAM visualization
                print(f"Processing: {img_path}")
                print(f"Saving to: {output_path}")
                
                visualize_gradcam(model, input_tensor, class_names, 
                                 target_class, alpha, output_path)
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
        
        print(f"Processed {min(num_per_class, len(image_files))} images for class {class_name}")

# Main function
def main():
    parser = argparse.ArgumentParser(description='ResNet50 for Medical Image Classification with GradCAM')
    parser.add_argument('--mode', required=True, choices=['train', 'test', 'gradcam', 'batch_gradcam'],
                        help='Operation mode: train, test, gradcam, or batch_gradcam')
    parser.add_argument('--data_dir', type=str, help='Directory containing the dataset')
    parser.add_argument('--model_path', type=str, help='Path to save/load the model')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--image_path', type=str, help='Path to the image for GradCAM visualization')
    parser.add_argument('--target_class', type=int, help='Target class for GradCAM visualization')
    parser.add_argument('--output_dir', type=str, default='gradcam_visualizations', 
                        help='Directory to save GradCAM visualizations')
    parser.add_argument('--num_per_class', type=int, default=3, 
                        help='Number of images per class for batch GradCAM')
    parser.add_argument('--alpha', type=float, default=0.5, 
                        help='Alpha value for GradCAM overlay (0-1)')
    parser.add_argument('--use_presplit', action='store_true', 
                        help='Use pre-split data (train/val/test folders)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Load dataset
        train_loader, val_loader, test_loader, class_names = load_dataset(
            args.data_dir, args.batch_size, args.use_presplit
        )
        
        # Initialize model
        model = initialize_model(len(class_names))
        model = model.to(device)
        
        # Train model
        train_model(model, train_loader, val_loader, args.num_epochs, args.learning_rate)
        
        # Save model
        save_model(model, args.model_path, class_names)
        
        # Evaluate model
        if test_loader:
            evaluate_model(model, test_loader, device, class_names)
    
    elif args.mode == 'test':
        # Load model
        model, class_names = load_model(args.model_path)
        
        # Load dataset
        _, _, test_loader, _ = load_dataset(args.data_dir, args.batch_size, args.use_presplit)
        
        # Evaluate model
        if test_loader:
            evaluate_model(model, test_loader, device, class_names)
        else:
            print("No test data available.")
    
    elif args.mode == 'gradcam':
        # Load model
        model, class_names = load_model(args.model_path)
        
        # Load and preprocess image
        img = Image.open(args.image_path).convert('RGB')
        input_tensor = data_transforms['test'](img).unsqueeze(0).to(device)
        
        # Determine target class
        target_class = args.target_class
        if target_class is None:
            # Try to infer from the image path
            img_dir = os.path.basename(os.path.dirname(args.image_path))
            if img_dir in class_names:
                target_class = class_names.index(img_dir)
            else:
                # Use the predicted class
                with torch.no_grad():
                    output = model(input_tensor)
                    target_class = output.argmax().item()
        
        # Generate output filename
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            base_name = os.path.basename(args.image_path)
            save_path = os.path.join(args.output_dir, f"gradcam_{base_name}")
        else:
            save_path = "gradcam_result.png"
        
        # Visualize GradCAM
        visualize_gradcam(model, input_tensor, class_names, 
                         target_class, args.alpha, save_path)
    
    elif args.mode == 'batch_gradcam':
        # Run batch GradCAM
        batch_gradcam(args.model_path, args.data_dir, args.output_dir, 
                     args.num_per_class, args.alpha)

if __name__ == '__main__':
    main()