import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import time
import copy

# ============ CONFIGURATION ============
# Change these to match your setup
DATA_DIR = './data'  # Your data folder
TRAIN_DIR = f'{DATA_DIR}/data-files/train'
VAL_DIR = f'{DATA_DIR}/data-files/validation'

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
NUM_WORKERS = 2  # For data loading (0 if on Windows)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============ DATA PREPARATION ============
# Data augmentation for training (helps model generalize)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),          # Resize to 224x224
    transforms.RandomHorizontalFlip(),       # Randomly flip images
    transforms.RandomRotation(10),           # Randomly rotate ±10 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust colors
    transforms.ToTensor(),                   # Convert to tensor
    transforms.Normalize(                    # Normalize (ImageNet stats)
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Validation transforms (no augmentation, just resize and normalize)
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load datasets
print("\nLoading datasets...")
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transforms)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

# Print dataset info
print(f"\nDataset Statistics:")
print(f"Classes: {train_dataset.classes}")
print(f"Number of classes: {len(train_dataset.classes)}")
print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Training batches: {len(train_loader)}")

# ============ MODEL DEFINITION ============
# Using transfer learning with pre-trained ResNet18
print("\nLoading pre-trained model...")

# Load ResNet18 pre-trained on ImageNet
model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

num_classes = len(train_dataset.classes)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, num_classes)

print(model)

# Move model to device
model = model.to(device)

print(f"Model loaded: EfficientNet_B3")
print(f"Final layer modified for {num_classes} classes")

# ============ TRAINING SETUP ============
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler (optional but helpful)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# ============ TRAINING FUNCTION ============
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)
    
    return epoch_loss, epoch_acc

# ============ VALIDATION FUNCTION ============
def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = running_corrects.double() / len(val_loader.dataset)
    
    return epoch_loss, epoch_acc

# ============ TRAINING LOOP ============
print("\n" + "="*50)
print("Starting Training")
print("="*50)

best_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())
train_losses, val_losses = [], []
train_accs, val_accs = [], []

start_time = time.time()

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    print("-" * 30)
    
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    # Update learning rate
    scheduler.step()
    
    # Save statistics
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc.item())
    val_accs.append(val_acc.item())
    
    # Print results
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
    
    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        print(f"✓ New best model! (Val Acc: {val_acc:.4f})")

total_time = time.time() - start_time
print("\n" + "="*50)
print(f"Training Complete in {total_time//60:.0f}m {total_time%60:.0f}s")
print(f"Best Validation Accuracy: {best_acc:.4f}")
print("="*50)

# Load best model weights
model.load_state_dict(best_model_wts)

# ============ SAVE MODEL ============
print("\nSaving model...")
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_acc': best_acc,
    'classes': train_dataset.classes,
    'epoch': NUM_EPOCHS
}, 'best_model.pth')
print("Model saved as 'best_model.pth'")

# ============ PLOT TRAINING HISTORY ============
print("\nGenerating training plots...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Loss plot
ax1.plot(train_losses, label='Train Loss', marker='o')
ax1.plot(val_losses, label='Val Loss', marker='o')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True)

# Accuracy plot
ax2.plot(train_accs, label='Train Acc', marker='o')
ax2.plot(val_accs, label='Val Acc', marker='o')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_history.png')
print("Training plots saved as 'training_history.png'")

# ============ VISUALIZE PREDICTIONS ============
def visualize_predictions(model, data_loader, classes, num_images=6):
    """Show some predictions"""
    model.eval()
    images, labels = next(iter(data_loader))
    images = images[:num_images]
    labels = labels[:num_images]
    
    with torch.no_grad():
        outputs = model(images.to(device))
        _, preds = torch.max(outputs, 1)
    
    # Denormalize images for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for idx, ax in enumerate(axes.flat):
        if idx >= len(images):
            break
        
        # Denormalize
        img = images[idx] * std + mean
        img = torch.clamp(img, 0, 1)
        
        # Display
        ax.imshow(img.permute(1, 2, 0))
        
        true_label = classes[labels[idx]]
        pred_label = classes[preds[idx]]
        color = 'green' if labels[idx] == preds[idx] else 'red'
        
        ax.set_title(f'True: {true_label}\nPred: {pred_label}', color=color)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    print("Sample predictions saved as 'predictions.png'")

print("\nGenerating sample predictions...")
visualize_predictions(model, val_loader, train_dataset.classes)

print("\n✅ All done! Your model is trained and saved.")
print(f"\nNext steps:")
print(f"1. Check 'training_history.png' to see how training went")
print(f"2. Check 'predictions.png' to see sample predictions")
print(f"3. Use 'best_model.pth' to make predictions on new images")