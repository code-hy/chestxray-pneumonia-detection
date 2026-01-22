import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Define paths
data_dir = 'data/chest_xray'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

# ----------------------------
# PyTorch Model (ResNet18)
# ----------------------------
print("\n" + "="*40)
print("🚀 Training PyTorch Model (ResNet18)...")
print("="*40)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data Setup
# Note: Using standard ImageNet mean/std for normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

# Use validaton set if not empty, else use test set for validation during training
if len(val_dataset) < 10:
    print("Validation set is too small, using Test set for validation.")
    val_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model Setup
model_pt = torchvision.models.resnet18(weights='IMAGENET1K_V1')
# Adjust final layer for 2 classes (Normal vs Pneumonia)
model_pt.fc = nn.Linear(model_pt.fc.in_features, 2)
model_pt.to(device)

# Loss & Optimizer
# Weighted loss to handle potential class imbalance
count_normal = len(os.listdir(os.path.join(train_dir, 'NORMAL')))
count_pneumonia = len(os.listdir(os.path.join(train_dir, 'PNEUMONIA')))
weight = torch.tensor([count_pneumonia/count_normal, 1.0]).to(device) 

criterion = nn.CrossEntropyLoss(weight=weight)
optimizer = torch.optim.Adam(model_pt.parameters(), lr=1e-4)

# Training Loop
epochs = 5
for epoch in range(epochs):
    # Train
    model_pt.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model_pt(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total
    
    # Validation
    model_pt.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_pt(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
    val_epoch_loss = val_loss / len(val_loader.dataset)
    val_epoch_acc = val_correct / val_total
    
    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
          f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

# Save Model
torch.save(model_pt.state_dict(), "models/model_pth.pth")
print("✅ PyTorch model saved to models/model_pth.pth")


# ----------------------------
# Keras Model (MobileNetV2)
# ----------------------------
print("\n" + "="*40)
print("🚀 Training Keras Model (MobileNetV2)...")
print("="*40)

# Data Generators
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Determining validation dir (use val if robust, else test)
validation_source = val_dir if len(os.listdir(val_dir)) > 0 else test_dir # Simple check
# Ideally we check number of files, but flow_from_directory handles empty dirs gracefully-ish
# Let's rely on the previous PyTorch logic check or just use val_dir because we saw it exists.
# But often 'val' in this dataset has very few images (8 normal, 8 pneumonia), so using 'test' for validation 
# gives a better estimate of performance.
val_gen = val_datagen.flow_from_directory(
    test_dir, # Using test_dir for more robust validation metrics during training
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Model Architecture
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)

model_keras = Model(inputs=base_model.input, outputs=predictions)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Compile
model_keras.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model_keras.fit(
    train_gen,
    epochs=5,
    validation_data=val_gen,
    verbose=1
)

# Save Model
model_keras.save("models/model_keras.h5")
print("✅ Keras model saved to models/model_keras.h5")
print("\nAll training complete.")
