# Dataset Documentation

This document provides comprehensive information about the Measuring Cup Volume Prediction dataset.

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

This project includes two main dataset configurations:

1. **BMC_NewResized** (Primary)
   - Organized by volume class
   - ~794 training images, ~107 test images
   - Used for EfficientNet transfer learning

2. **BMC_OR** (Object Recognition variant)
   - Alternative organization (FV/BV mixed in same folder)
   - Used for ES(1+1) optimization experiments

## Download Instructions

Due to GitHub's file size limitations, the dataset is hosted externally.

### Option 1: Google Drive (Recommended)

**Download Link**: [(https://drive.google.com/drive/folders/10eL1VkEnKwgj-iKgzTuBZNbLM8RwGz5Y?usp=sharing)]

**Steps**:
1. Click the download link above after requested
2. Download the archive file (`BMC_Dataset.zip` or similar)
3. Extract to your project root directory
4. Verify the structure matches the format below


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
  

**Prefixes**:
- `B` = Big Measuring Cup (BMC)



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
