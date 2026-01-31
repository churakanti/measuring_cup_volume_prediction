# Model Architecture Documentation

This document provides detailed technical information about the model architectures used in the Measuring Cup Volume Prediction project.

## Table of Contents
- [Overview](#overview)
- [Approach 1: EfficientNet Transfer Learning](#approach-1-efficientnet-transfer-learning)
- [Approach 2: ES(1+1) Evolutionary Strategy](#approach-2-es11-evolutionary-strategy)
- [Training Strategies](#training-strategies)
- [Performance Comparison](#performance-comparison)
- [Model Files](#model-files)

## Overview

This project implements two distinct approaches to volume prediction:

1. **EfficientNet Transfer Learning** - State-of-the-art CNN for direct volume regression
2. **ES(1+1) Evolutionary Strategy** - Automated hyperparameter optimization for custom architectures

Both approaches are designed for the same task but offer different trade-offs in terms of performance, training time, and customizability.

## Approach 1: EfficientNet Transfer Learning

### Architecture Overview

EfficientNet is a family of convolutional neural networks that achieve state-of-the-art accuracy with fewer parameters through compound scaling.

**Primary Model**: EfficientNetB0 (baseline) or EfficientNetB3 (enhanced)

### Model Configuration

#### Base Architecture
```
Input (224×224×3)
    ↓
EfficientNetB0 (Pre-trained on ImageNet)
    - Total layers: 237
    - Parameters: ~5.3M
    - First 50% frozen during training
    ↓
GlobalAveragePooling2D
    ↓
Dense(256, activation='relu')
    ↓
Dropout(0.3)
    ↓
Dense(1, activation='linear')  # Regression output
    ↓
Output (single continuous value)
```

#### Layer Details

**EfficientNetB0 Base**:
- **Input**: 224×224×3 RGB images
- **Compound scaling**: Depth (d), width (w), resolution (r)
- **MBConv blocks**: Mobile Inverted Bottleneck Convolution
- **Squeeze-and-Excitation**: Channel attention mechanism
- **Swish activation**: f(x) = x · sigmoid(βx)

**Custom Regression Head**:
```python
from keras.applications import EfficientNetB0
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout

# Load pre-trained base
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze first 50% of layers
for layer in base_model.layers[:int(len(base_model.layers) * 0.5)]:
    layer.trainable = False

# Add custom regression head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu', name='dense_1')(x)
x = Dropout(0.3, name='dropout')(x)
predictions = Dense(1, activation='linear', name='output')(x)

model = Model(inputs=base_model.input, outputs=predictions)
```

### Training Configuration

**Optimizer**: Adam
```python
from keras.optimizers import Adam
optimizer = Adam(learning_rate=0.0001)
```

**Loss Function**: Mean Squared Error (MSE)
```python
model.compile(
    optimizer=optimizer,
    loss='mse',
    metrics=['mae', 'mse']
)
```

**Callbacks**:
1. **Early Stopping**: Stop training when validation loss plateaus
   ```python
   EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
   ```

2. **Learning Rate Reduction**: Reduce LR when validation loss plateaus
   ```python
   ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
   ```

3. **Model Checkpoint**: Save best model weights
   ```python
   ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')
   ```

### Data Preprocessing

**Image Augmentation**:
```python
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,              # Normalize to [0,1]
    rotation_range=20,           # Random rotation ±20°
    width_shift_range=0.2,       # Horizontal shift ±20%
    height_shift_range=0.2,      # Vertical shift ±20%
    zoom_range=0.2,              # Zoom 80-120%
    horizontal_flip=False,       # No flip (maintains orientation)
    fill_mode='nearest'          # Fill strategy for empty pixels
)

test_datagen = ImageDataGenerator(rescale=1./255)  # Only normalization for test
```

**Label Normalization**:
```python
# Normalize labels to [0, 1] for stable training
y_train_normalized = (y_train - y_min) / (y_max - y_min)

# Denormalize predictions
y_pred_original = predictions * (y_max - y_min) + y_min
```

### Model Variants

#### EfficientNetB0 (Baseline)
- **Parameters**: ~5.3M
- **Training time**: ~20-30 min (GPU)
- **Memory**: ~2GB VRAM
- **Accuracy**: ~94%

#### EfficientNetB3 (Enhanced)
- **Parameters**: ~12M
- **Training time**: ~40-60 min (GPU)
- **Memory**: ~4GB VRAM
- **Accuracy**: ~95-96% (marginal improvement)

### Performance Metrics

**Achieved Results** (EfficientNetB0):
- **Test Accuracy**: ~94%
- **Test MAE**: ~15.43 units
- **Test MSE**: ~400-500
- **R² Score**: ~0.92-0.95

**Interpretation**:
- MAE of 15.43 means average prediction error is ±15.43 mL
- For a 500 mL measurement, this represents ~3% error
- Highly accurate for practical volume estimation tasks

## Approach 2: ES(1+1) Evolutionary Strategy

### Algorithm Overview

The (1+1) Evolution Strategy is a simple yet effective evolutionary algorithm for hyperparameter optimization.

**Key Concept**:
- Population size: 1 parent
- Offspring: 1 child per generation
- Selection: Keep the better individual (parent or child)
- Adaptation: 1/5 success rule for mutation step size

### Hyperparameter Search Space

The ES(1+1) algorithm optimizes the following hyperparameters:

| Hyperparameter | Range | Type |
|----------------|-------|------|
| Hidden Layers | 1-4 | Discrete |
| Neurons (Layer 1) | 8-512 | Continuous |
| Neurons (Layer 2) | 8-512 | Continuous |
| Neurons (Layer 3) | 8-512 | Continuous |
| Neurons (Layer 4) | 8-512 | Continuous |
| Learning Rate | 0.01-0.1 | Continuous |
| Batch Size | 16-64 | Discrete |
| Activation Function | ReLU, ELU, Sigmoid, Tanh | Categorical |
| Optimizer | SGD, RMSprop, Adam | Categorical |

### Base CNN Architecture

The ES(1+1) optimizes architectures built on this base:

```
Input (224×224×3)
    ↓
Conv2D(32, 3×3, activation='relu')
    ↓
MaxPooling2D(2×2)
    ↓
Conv2D(64, 3×3, activation='relu')
    ↓
MaxPooling2D(2×2)
    ↓
Conv2D(128, 3×3, activation='relu')
    ↓
MaxPooling2D(2×2)
    ↓
Flatten
    ↓
[Dynamic Dense Layers - ES optimized]
    ↓
Dropout(0.3)
    ↓
Dense(22, activation='softmax')  # Classification
    ↓
Output (22 volume classes)
```

### 1/5 Success Rule

The 1/5 success rule adapts the mutation step size (σ) based on success rate:

```python
if success_rate > 1/5:
    σ = σ * 1.5  # Increase step size (explore more)
elif success_rate < 1/5:
    σ = σ * 0.85  # Decrease step size (exploit more)
else:
    σ = σ  # Keep current step size
```

**Success rate** is calculated every 10 generations:
```python
success_rate = successful_mutations / total_mutations
```

### Mutation Strategy

**Continuous Parameters** (learning rate, neurons):
```python
new_value = current_value + N(0, σ)  # Gaussian noise
new_value = clip(new_value, min_val, max_val)
```

**Discrete Parameters** (layers, batch size):
```python
if random() < mutation_probability:
    new_value = random_choice(valid_options)
```

**Categorical Parameters** (activation, optimizer):
```python
if random() < mutation_probability:
    new_value = random_choice(['relu', 'elu', 'sigmoid', 'tanh'])
```

### ES(1+1) Training Process

1. **Initialization**: Generate random hyperparameter configuration
2. **Training**: Train model with current configuration for N epochs
3. **Evaluation**: Measure validation accuracy
4. **Mutation**: Create child configuration by mutating parent
5. **Selection**: Keep better configuration (parent or child)
6. **Adaptation**: Apply 1/5 rule every 10 generations
7. **Repeat**: Until convergence or max generations

### Performance Metrics

**Typical Results**:
- **Convergence**: 50-100 generations
- **Final Accuracy**: 80-95%
- **Training Time**: 4-8 hours (depends on generations)
- **Best Config Found**: Varies by run

**Example Best Configuration**:
```json
{
  "hidden_layers": 3,
  "layer1_neurons": 256,
  "layer2_neurons": 128,
  "layer3_neurons": 64,
  "learning_rate": 0.045,
  "batch_size": 32,
  "activation": "relu",
  "optimizer": "adam"
}
```

## Training Strategies

### Two-Phase Training (EfficientNet)

**Phase 1: Feature Extraction**
- Freeze all base model layers
- Train only custom head
- Epochs: 10-20
- Learning rate: 0.001

**Phase 2: Fine-Tuning**
- Unfreeze top 50% of base model
- Train entire upper portion
- Epochs: 20-40
- Learning rate: 0.0001 (10x lower)

### Transfer Learning Benefits

1. **Faster Convergence**: Pre-trained weights from ImageNet
2. **Better Generalization**: Learned features from 1M+ images
3. **Lower Data Requirements**: Works well with <1000 images
4. **Reduced Overfitting**: Regularization from pre-training

## Performance Comparison

### Quantitative Comparison

| Metric | EfficientNetB0 | ES(1+1) Custom CNN |
|--------|----------------|-------------------|
| **Test Accuracy** | ~94% | 80-95% (varies) |
| **Test MAE** | ~15.43 | Varies |
| **Training Time** | 20-30 min | 4-8 hours |
| **Inference Time** | <10ms/image | <5ms/image |
| **Model Size** | ~21MB | ~5-15MB |
| **GPU Memory** | ~2GB | ~1GB |
| **Convergence** | Stable | Variable |

### Qualitative Comparison

**EfficientNet Strengths**:
- Consistently high accuracy
- Fast training
- State-of-the-art architecture
- Well-documented and tested

**ES(1+1) Strengths**:
- Automatic hyperparameter tuning
- Customizable architecture
- Educational value (demonstrates optimization)
- Potentially lighter models

**Use Case Recommendations**:
- **Production/Portfolio**: Use EfficientNet (reliable, impressive)
- **Research/Experimentation**: Use ES(1+1) (flexible, exploratory)
- **Resource-Constrained**: Use ES(1+1) (smaller models possible)
- **Time-Constrained**: Use EfficientNet (faster convergence)

## Model Files

### Saved Model Format

Models are saved in Keras 3.x format (`.keras`):

```python
# Save model
model.save('best_efficientnet_model.keras')

# Load model
from keras.models import load_model
model = load_model('best_efficientnet_model.keras')
```

### Model Downloads

Pre-trained models are hosted externally:

**EfficientNetB0 (Recommended)**:
- Download: [Insert Google Drive link]
- Size: ~21 MB
- Accuracy: 94%

**EfficientNetB3**:
- Download: [Insert Google Drive link]
- Size: ~48 MB
- Accuracy: 95%

**ES(1+1) Best Configuration**:
- Download: [Insert Google Drive link]
- Size: Varies
- Accuracy: Varies (typically 85-92%)

### Using Pre-trained Models

```python
from keras.models import load_model
import numpy as np
from PIL import Image

# Load model
model = load_model('best_efficientnet_model.keras')

# Load and preprocess image
img = Image.open('test_image.jpg').resize((224, 224))
img_array = np.array(img) / 255.0  # Normalize
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Predict
prediction = model.predict(img_array)
volume = prediction[0][0]  # Extract volume value
print(f"Predicted volume: {volume:.2f} mL")
```

## Advanced Topics

### Model Ensembling

Combine predictions from multiple models for improved accuracy:

```python
# Load multiple models
model1 = load_model('efficientnet_b0.keras')
model2 = load_model('efficientnet_b3.keras')
model3 = load_model('es_best.keras')

# Get predictions
pred1 = model1.predict(img_array)
pred2 = model2.predict(img_array)
pred3 = model3.predict(img_array)

# Average predictions
ensemble_pred = (pred1 + pred2 + pred3) / 3
```

### Model Quantization

Reduce model size for deployment:

```python
import tensorflow as tf

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save quantized model
with open('model_quantized.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Grad-CAM Visualization

Visualize which regions the model focuses on:

```python
import tensorflow as tf
from keras.models import Model

# Get last conv layer
last_conv_layer = model.get_layer('top_conv')  # EfficientNet top layer

# Create grad model
grad_model = Model(
    inputs=model.input,
    outputs=[last_conv_layer.output, model.output]
)

# Generate heatmap
with tf.GradientTape() as tape:
    conv_output, predictions = grad_model(img_array)
    loss = predictions[0]

grads = tape.gradient(loss, conv_output)
# ... (see visualization tutorials for complete implementation)
```

## References

1. **EfficientNet**: Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML 2019.

2. **Evolution Strategies**: Rechenberg, I. (1973). Evolutionsstrategie: Optimierung technischer Systeme nach Prinzipien der biologischen Evolution.

3. **1/5 Success Rule**: Schwefel, H. P. (1977). Numerische Optimierung von Computer-Modellen mittels der Evolutionsstrategie.

4. **Transfer Learning**: Pan, S. J., & Yang, Q. (2010). A Survey on Transfer Learning. IEEE Transactions on Knowledge and Data Engineering.

---

For implementation details, see:
- **EfficientNet**: [code/BMC_VOLUME_README.md](../code/BMC_VOLUME_README.md)
- **ES(1+1)**: [ES_1plus1_README.md](../ES_1plus1_README.md)
- **Technical Reference**: [code/CLAUDE.md](../code/CLAUDE.md)
