# Dataset Documentation

This document provides comprehensive information about the Measuring Cup Volume Prediction dataset.
Link:[https://drive.google.com/drive/folders/10eL1VkEnKwgj-iKgzTuBZNbLM8RwGz5Y?usp=drive_link]
Due to GitHub size restrictions on large datasets, they are available upon request.

## Table of Contents
- [Overview](#overview)
- [Download Instructions](#download-instructions)
- [Dataset Structure](#dataset-structure)
- [Dataset Statistics](#dataset-statistics)
- [Image Specifications](#image-specifications)
- [CSV Format](#csv-format)
- [Data Collection](#data-collection)
- [Usage Guidelines](#usage-guidelines)

## Overview

The **Measuring Cup Volume Prediction Dataset** consists of **900+ high-quality images** of measuring cups at various liquid volume levels, captured from multiple viewing angles.

### Key Features
- **22 volume classes** ranging from 100 mL to 610 mL
- **Two viewing perspectives**: Front View (FV) and Bottom-Up View (BV)
- **High-quality JPG images** at 224×224 resolution
- **Annotated CSV files** with volume labels and metadata
- **Train/Test split** for model evaluation (80/20 split)

### Dataset Variants

This project includes datasets with **TWO DIFFERENT LIQUID TYPES** for cross-domain transfer learning research:

#### Liquid Types

Understanding the dataset naming:
- **OR** = **Orange Juice** (orange-colored, semi-transparent liquid)
- **RED** = **Red Juice** (red-colored, more opaque liquid)

The dual-liquid approach creates a **controlled domain shift** (visual difference in color/transparency) while keeping the underlying task (volume measurement) identical. This tests whether models can transfer knowledge across different visual appearances.

#### Dataset Configurations

1. **BMC_NewResized** (Primary - Orange Juice)
   - **Liquid type**: Orange juice (OR)
   - **Organization**: Organized by volume class with FV/BV subfolders
   - **Size**: 794 training images, 107 test images (901 total)
   - **Used for**: EfficientNet transfer learning baseline

2. **BMC_OR** (Orange Juice - Alternative)
   - **Liquid type**: Orange juice (OR)
   - **Organization**: Alternative folder structure (FV/BV mixed)
   - **Size**: 794 training images, 107 test images
   - **Used for**: ES(1+1) optimization experiments

3. **BMC_RED** (Red Juice - For Cross-Domain)
   - **Liquid type**: Red juice (RED)
   - **Organization**: Similar to BMC_NewResized
   - **Size**: 821 training images, 107 test images (928 total)
   - **Used for**: Cross-domain transfer experiments (OR→RED, RED→OR)

#### Cross-Domain Learning Goals

**Research Question**: Can a model trained on orange juice accurately predict volumes for red juice?

**Key Findings**:
- Same-domain performance (OR→OR): ~94% accuracy, ~15 mL MAE
- Cross-domain without fine-tuning: Higher error due to visual domain shift
- Cross-domain with fine-tuning: **82% error reduction** (214.59 → 38.73 mL)
- Demonstrates that volume prediction concepts successfully transfer across visual domains

## Download Instructions

Due to GitHub's file size limitations, the dataset is hosted externally.

### Option 1: Google Drive (Recommended)

**Download Link**: [Insert your Google Drive link here]

**Steps**:
1. Click the download link above
2. Download the archive file (`BMC_Dataset.zip` or similar)
3. Extract to your project root directory
4. Verify the structure matches the format below

### Option 2: Alternative Hosting

If the Google Drive link is unavailable, you can also download from:
- **Zenodo**: [Insert Zenodo DOI link]
- **Kaggle Datasets**: [Insert Kaggle link]
- **Institutional Repository**: [Insert link if applicable]

### Quick Download (Command Line)

```bash
# Using wget (Linux/macOS)
wget https://your-hosted-url.com/BMC_Dataset.zip

# Using curl (cross-platform)
curl -O https://your-hosted-url.com/BMC_Dataset.zip

# Extract
unzip BMC_Dataset.zip
```

## Dataset Structure

### BMC_NewResized Directory Structure

```
BMC_NewResized/
├── training/                    # Training set (~794 images)
│   ├── 100/                     # Volume class: 100 mL
│   │   ├── B100001.jpg
│   │   ├── B100002.jpg
│   │   └── ...
│   ├── 105/                     # Volume class: 105 mL
│   ├── 150/
│   ├── 155/
│   ├── 200/
│   ├── 205/
│   ├── 250/
│   ├── 255/
│   ├── 300/
│   ├── 305/
│   ├── 350/
│   ├── 355/
│   ├── 400/
│   ├── 405/
│   ├── 450/
│   ├── 455/
│   ├── 500/
│   ├── 505/
│   ├── 550/
│   ├── 555/
│   ├── 600/
│   └── 610/                     # Volume class: 610 mL
│
├── Testing/                     # Test set (~107 images)
│   ├── 100/
│   ├── 105/
│   └── ... (same 22 classes)
│
├── training_labels.csv          # Training annotations
└── testing_labels.csv           # Test annotations
```

### Image Naming Convention

Images follow this naming pattern:
- **Format**: `[B/S][Volume][Sequence].jpg`
- **Examples**:
  - `B100001.jpg` - Big measuring cup, 100 mL, image #1
  - `B450023.jpg` - Big measuring cup, 450 mL, image #23
  - `S505012.jpg` - Small measuring cup (if applicable)

**Prefixes**:
- `B` = Big Measuring Cup (BMC)
- `S` = Small Measuring Cup (SMC)

## Dataset Statistics

### Volume Classes (22 total)

| Class | Volume (mL) | Training Images | Test Images | Total |
|-------|-------------|-----------------|-------------|-------|
| 1     | 100         | ~36             | ~5          | ~41   |
| 2     | 105         | ~36             | ~5          | ~41   |
| 3     | 150         | ~36             | ~5          | ~41   |
| 4     | 155         | ~36             | ~5          | ~41   |
| 5     | 200         | ~36             | ~5          | ~41   |
| 6     | 205         | ~36             | ~5          | ~41   |
| 7     | 250         | ~36             | ~5          | ~41   |
| 8     | 255         | ~36             | ~5          | ~41   |
| 9     | 300         | ~36             | ~5          | ~41   |
| 10    | 305         | ~36             | ~5          | ~41   |
| 11    | 350         | ~36             | ~5          | ~41   |
| 12    | 355         | ~36             | ~5          | ~41   |
| 13    | 400         | ~36             | ~5          | ~41   |
| 14    | 405         | ~36             | ~5          | ~41   |
| 15    | 450         | ~36             | ~5          | ~41   |
| 16    | 455         | ~36             | ~5          | ~41   |
| 17    | 500         | ~36             | ~5          | ~41   |
| 18    | 505         | ~36             | ~5          | ~41   |
| 19    | 550         | ~36             | ~5          | ~41   |
| 20    | 555         | ~36             | ~5          | ~41   |
| 21    | 600         | ~36             | ~5          | ~41   |
| 22    | 610         | ~36             | ~5          | ~41   |
| **Total** |         | **~794**        | **~107**    | **~901** |

### Dataset Split
- **Training**: ~88% (794 images)
- **Testing**: ~12% (107 images)

### Class Balance
The dataset is approximately balanced across all 22 volume classes, with each class containing roughly the same number of images.

## Image Specifications

### Technical Details
- **Format**: JPG (converted from HEIC)
- **Resolution**: 224 × 224 pixels
- **Color Space**: RGB (3 channels)
- **File Size**: ~20-50 KB per image (compressed)
- **Aspect Ratio**: 1:1 (square)

### Image Views
Each volume measurement includes images from two perspectives:

1. **Front View (FV)**
   - Direct frontal view of the measuring cup
   - Shows volume markings and liquid level clearly
   - Used for standard volume reading

2. **Bottom-Up View (BV/BUV)**
   - View from below the cup
   - Shows liquid meniscus from different angle
   - Provides additional perspective for robust prediction

### Preprocessing
Images have been preprocessed as follows:
- Converted from HEIC (iPhone format) to JPG
- Resized to 224×224 pixels (required for EfficientNet input)
- No additional augmentation in raw dataset (augmentation applied during training)

### Sample Images
(Include sample images in `docs/images/` if available)

## CSV Format

### training_labels.csv / testing_labels.csv

CSV files contain annotations for all images with the following structure:

**Columns**:
- `filepath`: Relative path to image file
- `label`: Volume measurement (continuous value)
- `class_id`: Class identifier (0-21)
- `image_name`: Image filename
- `view_type`: FV (Front View) or BV (Bottom View) - optional

**Example**:
```csv
filepath,label,class_id,image_name
training/100/B100001.jpg,100,0,B100001.jpg
training/100/B100002.jpg,100,0,B100002.jpg
training/105/B105001.jpg,105,1,B105001.jpg
training/105/B105002.jpg,105,1,B105002.jpg
```

### Loading CSV Files

```python
import pandas as pd

# Load training labels
train_df = pd.read_csv('BMC_NewResized/training_labels.csv')
print(train_df.head())

# Load testing labels
test_df = pd.read_csv('BMC_NewResized/testing_labels.csv')
print(test_df.head())

# Check class distribution
print(train_df['label'].value_counts().sort_index())
```

## Data Collection

### Liquid Types and Domain Shift Experiment

To enable cross-domain transfer learning research, two distinct liquid types were recorded:

#### 1. Orange Juice (OR Dataset)
- **Color spectrum**: Orange wavelengths (590-620nm)
- **Visual properties**: Semi-transparent, lighter appearance
- **Dataset size**: 901 total images (794 train, 107 test)
- **Purpose**: Baseline training and source domain for transfer learning experiments

#### 2. Red Juice (RED Dataset)
- **Color spectrum**: Red wavelengths (620-750nm)
- **Visual properties**: More opaque, darker appearance
- **Dataset size**: 928 total images (821 train, 107 test)
- **Purpose**: Target domain for cross-domain transfer experiments

#### Domain Shift Design

The color difference between orange and red liquids creates a **controlled visual domain shift** while the measurement task remains identical across both datasets. This experimental design enables research on:

- **Cross-domain transfer learning**: Training on OR, testing on RED (and vice versa)
- **Model robustness**: Testing if models learn volume concepts vs. memorizing liquid appearance
- **Domain adaptation techniques**: Fine-tuning strategies for new visual domains
- **Few-shot learning**: Adapting to new liquid types with minimal retraining data

**Real-World Impact**: Models that can handle domain shifts are more practical for deployment, as they can generalize to new liquid types, cup designs, or lighting conditions without requiring complete retraining.

### Collection Process
1. **Measurement Setup**: Precise volume measurements using calibrated measuring cups
2. **Image Capture**: High-quality smartphone camera (iPhone with HEIC format)
3. **View Angles**: Systematic capture from front view and bottom-up view
4. **Lighting Conditions**: Controlled indoor lighting for consistency
5. **Background**: Neutral background to minimize distractions

### Quality Control
- Manual inspection of all images
- Removal of blurry or mislabeled images
- Verification of volume measurements
- Consistent image capture protocol

### Original Format Conversion
Original images were captured in HEIC (High Efficiency Image Container) format and converted to JPG for compatibility:

```python
# Example conversion script (if needed)
from PIL import Image
import os

def convert_heic_to_jpg(heic_path, jpg_path):
    img = Image.open(heic_path)
    img = img.convert('RGB')
    img.save(jpg_path, 'JPEG', quality=95)
```

See `code/SUBSET_README.md` for more details on the conversion process.

## Usage Guidelines

### Loading Images with Keras

```python
from keras.preprocessing.image import ImageDataGenerator

# Create data generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2
)

# Load training data
train_generator = train_datagen.flow_from_directory(
    'BMC_NewResized/training',
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse'  # For classification
)
```

### Loading Images with PyTorch

```python
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

class MeasuringCupDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.labels_df.iloc[idx]['filepath'])
        image = Image.open(img_path).convert('RGB')
        label = self.labels_df.iloc[idx]['label']

        if self.transform:
            image = self.transform(image)

        return image, label

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset
train_dataset = MeasuringCupDataset(
    csv_file='BMC_NewResized/training_labels.csv',
    root_dir='BMC_NewResized',
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

### Dataset Verification Script

```python
import os
import pandas as pd

def verify_dataset(base_path='BMC_NewResized'):
    """Verify dataset integrity"""

    # Check directories exist
    assert os.path.exists(f'{base_path}/training'), "Training directory not found"
    assert os.path.exists(f'{base_path}/Testing'), "Testing directory not found"

    # Check CSV files exist
    assert os.path.exists(f'{base_path}/training_labels.csv'), "Training CSV not found"
    assert os.path.exists(f'{base_path}/testing_labels.csv'), "Testing CSV not found"

    # Load and verify CSV
    train_df = pd.read_csv(f'{base_path}/training_labels.csv')
    test_df = pd.read_csv(f'{base_path}/testing_labels.csv')

    print(f"✓ Training images: {len(train_df)}")
    print(f"✓ Testing images: {len(test_df)}")
    print(f"✓ Total images: {len(train_df) + len(test_df)}")
    print(f"✓ Number of classes: {train_df['label'].nunique()}")
    print(f"✓ Volume range: {train_df['label'].min()} - {train_df['label'].max()} mL")

    return True

# Run verification
verify_dataset()
```

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{measuring_cup_volume_2025,
  title={Measuring Cup Volume Prediction Dataset},
  author={[Your Name]},
  year={2025},
  publisher={[Publisher/Institution]},
  note={Dataset for deep learning-based volume prediction from measuring cup images}
}
```

## License

The dataset is released under the same license as the code (MIT License). See [LICENSE](../LICENSE) for details.

## Contact

For questions about the dataset, data collection methodology, or to report issues:
- Open an issue on GitHub
- Email: [your-email@example.com]

---

**Note**: If you create derivative datasets or modifications, please maintain attribution and share your improvements with the community.
