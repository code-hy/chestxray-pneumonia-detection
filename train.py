import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# ----------------------------
# PyTorch Model (ResNet18)
# ----------------------------
print("🚀 Training PyTorch model...")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder("data/chest_xray/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model_pt = models.resnet18(weights='IMAGENET1K_V1')
model_pt.fc = nn.Linear(model_pt.fc.in_features, 2)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_pt.parameters(), lr=1e-4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_pt.to(device)

for epoch in range(5):  # Reduced for demo; increase to 10–15 for better results
    model_pt.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model_pt(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

torch.save(model_pt.state_dict(), "models/model_pth.pth")
print("✅ PyTorch model saved to models/model_pth.pth")

# ----------------------------
# Keras Model (MobileNetV2)
# ----------------------------
print("🚀 Training Keras model...")

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    "data/chest_xray/train",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)

model_keras = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model_keras.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_keras.fit(train_gen, epochs=5, verbose=1)

model_keras.save("models/model_keras.h5")
print("✅ Keras model saved to models/model_keras.h5")