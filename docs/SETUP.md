# Setup Guide - Measuring Cup Volume Prediction

This guide provides detailed instructions for setting up the Measuring Cup Volume Prediction project on your local machine.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [GPU Configuration](#gpu-configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements
- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **RAM**: Minimum 4GB, recommended 8GB+
- **Storage**: At least 5GB free space (for code, dataset, and models)
- **GPU** (optional but recommended): NVIDIA GPU with CUDA support for faster training

### Software Requirements
- **Python**: Version 3.8, 3.9, 3.10, or 3.11
- **pip**: Latest version (comes with Python)
- **Git**: For cloning the repository

### Checking Python Version

```bash
python --version
# or
python3 --version
```

If Python is not installed, download it from [python.org](https://www.python.org/downloads/).

## Installation

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/yourusername/measuring-cup-volume-prediction.git

# Navigate to the project directory
cd measuring-cup-volume-prediction
```

### Step 2: Create Virtual Environment

Creating a virtual environment isolates the project dependencies from your system Python.

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the beginning of your command prompt, indicating the virtual environment is active.

### Step 3: Upgrade pip

```bash
pip install --upgrade pip
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including:
- Keras 3.11+
- TensorFlow 2.15+
- PyTorch
- OpenCV
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn
- Jupyter Lab

**Installation time**: 5-15 minutes depending on your internet speed.

### Step 5: Verify Installation

```bash
python -c "import keras; import tensorflow as tf; import torch; print(f'Keras: {keras.__version__}')"
```

Expected output:
```
Keras: 3.11.x
```

## Dataset Setup

### Option 1: Download from External Hosting

The dataset is hosted externally to keep the repository lightweight.

1. **Download the dataset** from the provided link (see [docs/DATASET.md](DATASET.md) for links)
2. **Extract the archive** to the project root directory
3. **Verify the structure**:

```bash
# The directory structure should look like this:
measuring-cup-volume-prediction/
├── BMC_NewResized/
│   ├── training/
│   │   ├── 100/
│   │   ├── 105/
│   │   └── ... (22 classes total)
│   ├── Testing/
│   ├── training_labels.csv
│   └── testing_labels.csv
```

### Option 2: Use Sample Dataset (for Quick Testing)

If you want to test the code without downloading the full dataset:

```bash
# Create a minimal dataset structure for testing
python scripts/create_sample_dataset.py  # (if provided)
```

### Verify Dataset

```bash
# Check training images count
ls BMC_NewResized/training/*/  # Should show ~794 images across 22 folders

# Check CSV files exist
ls BMC_NewResized/*.csv  # Should show training_labels.csv and testing_labels.csv
```

## GPU Configuration

### NVIDIA GPU (CUDA)

If you have an NVIDIA GPU, follow these steps for GPU acceleration:

1. **Check CUDA compatibility**:
```bash
nvidia-smi
```

2. **Install CUDA Toolkit** (if not already installed):
   - Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
   - Recommended: CUDA 11.8 or 12.x

3. **Verify TensorFlow GPU support**:
```bash
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
```

4. **Verify PyTorch GPU support**:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```

### CPU-Only Configuration

If you don't have a GPU, the project will automatically use CPU. Training will be slower but still functional.

**Expected training times**:
- **GPU**: 20-30 minutes for EfficientNet
- **CPU**: 1-2 hours for EfficientNet

## Verification

### Run Quick Verification Script

Create and run this verification script:

```python
# verify_setup.py
import sys
import keras
import tensorflow as tf
import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns

print("✓ Python version:", sys.version)
print("✓ Keras version:", keras.__version__)
print("✓ TensorFlow version:", tf.__version__)
print("✓ PyTorch version:", torch.__version__)
print("✓ OpenCV version:", cv2.__version__)
print("✓ NumPy version:", np.__version__)
print("✓ Pandas version:", pd.__version__)
print("✓ Matplotlib version:", matplotlib.__version__)
print("✓ Seaborn version:", sns.__version__)
print("\n✓ GPU Available (TensorFlow):", len(tf.config.list_physical_devices('GPU')) > 0)
print("✓ CUDA Available (PyTorch):", torch.cuda.is_available())
print("\nSetup verification complete!")
```

Run it:
```bash
python verify_setup.py
```

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'keras'`

**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: GPU not detected

**Solution**:
1. Verify NVIDIA drivers are installed: `nvidia-smi`
2. Check CUDA version compatibility with TensorFlow/PyTorch
3. Reinstall TensorFlow with GPU support:
   ```bash
   pip install tensorflow[and-cuda]
   ```

### Issue: Out of memory errors

**Solution**:
- Reduce batch size in training scripts (e.g., from 32 to 16 or 8)
- Use CPU instead of GPU for smaller models
- Close other applications to free up RAM

### Issue: `ImportError: DLL load failed` (Windows)

**Solution**:
- Install Visual C++ Redistributable from [Microsoft](https://aka.ms/vs/17/release/vc_redist.x64.exe)
- Restart your computer after installation

### Issue: Jupyter kernel not found

**Solution**:
```bash
python -m ipykernel install --user --name=venv
```

### Issue: Dataset not found

**Solution**:
- Verify dataset is in the correct location: `BMC_NewResized/`
- Check that CSV files exist: `training_labels.csv` and `testing_labels.csv`
- See [docs/DATASET.md](DATASET.md) for dataset structure details

## Next Steps

After successful setup:

1. **Explore the dataset**: Check out [docs/DATASET.md](DATASET.md)
2. **Understand the models**: Read [docs/MODELS.md](MODELS.md)
3. **Run your first training**: Open `BMC_NewResized/edBMC_1.ipynb` in Jupyter Lab
4. **Try ES(1+1) optimization**: Follow [QUICKSTART_ES.md](../QUICKSTART_ES.md)

## Additional Resources

- **Keras Documentation**: https://keras.io/
- **TensorFlow Tutorials**: https://www.tensorflow.org/tutorials
- **PyTorch Documentation**: https://pytorch.org/docs/
- **OpenCV Tutorials**: https://docs.opencv.org/

## Getting Help

If you encounter issues not covered here:

1. Check existing documentation in the `docs/` directory
2. Review the troubleshooting section above
3. Open an issue on GitHub with:
   - Your operating system and Python version
   - Full error message
   - Steps to reproduce the issue

---
