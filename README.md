# Measuring Cup Volume Prediction using Deep Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Keras](https://img.shields.io/badge/Keras-3.11+-red.svg)](https://keras.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://www.tensorflow.org/)

A comprehensive computer vision project for automated volume measurement prediction from measuring cup images using state-of-the-art deep learning techniques.

## 🎯 Project Overview

This project implements two complementary approaches topredict liquid volume measurements from images of measuring cups filled with **two different colored liquids** (red juice and orange juice), demonstrating cross-domain transfer learning:

1. **EfficientNet Transfer Learning** - Direct volume regression using pre-trained EfficientNetB0/B3
2. **ES(1+1) Evolutionary Strategy** - Automated hyperparameter optimization for custom CNN architectures

### Key Features

- **High Accuracy**: Achieves ~94% accuracy with MAE ~15.43 units on test set
- **Transfer Learning**: Leverages ImageNet pre-trained EfficientNet models
- **Automated Optimization**: ES(1+1) evolutionary algorithm for hyperparameter tuning
- **Robust Dataset**: 900+ images across 22 volume classes (100-610 mL)
- **Multiple Views**: Front View (FV) and Bottom-Up View (BV) image perspectives
- **Production Ready**: Clean, documented code with comprehensive guides

### 🎨 Dataset Variants - Cross-Domain Learning

The project tests model robustness across **two distinct liquid types**:

| Dataset | Liquid Type | Training | Testing | Total | Visual Properties |
|---------|------------|----------|---------|-------|-------------------|
| **OR** | Orange Juice | 794 | 107 | 901 | Semi-transparent, orange color |
| **RED** | Red Juice | 821 | 107 | 928 | More opaque, red color |

**Why Two Liquids?** This creates a controlled **visual domain shift** while keeping the measurement task identical - demonstrating the model's ability to transfer knowledge across different visual appearances.

**Cross-Domain Results**: Achieves 82% error reduction when transferring from orange juice to red juice predictions.

## 📊 Results

### Performance Metrics

| Approach | Test Accuracy | Test MAE | Training Time |
|----------|--------------|----------|---------------|
| **EfficientNetB0** | ~94% | ~15.43 units | ~20-30 min (GPU) |
| **ES(1+1) Custom CNN** | 80-95% | Varies | 50-100 generations |

### Volume Classes
22 classes ranging from 100 mL to 610 mL:
`100, 105, 150, 155, 200, 205, 250, 255, 300, 305, 350, 355, 400, 405, 450, 455, 500, 505, 550, 555, 600, 610`

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- GPU recommended (CUDA-compatible for faster training)
- 4GB+ RAM

### Installation

```bash
# Clone the repository
git clone https://github.com/churakanti/measuring-cup-volume-prediction.git
cd measuring-cup-volume-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset (see docs/DATASET.md for links)
# Extract to BMC_NewResized/ directory
```

### Usage

#### Option 1: EfficientNet Transfer Learning

```python
# Using Jupyter Notebook (recommended)
jupyter notebook BMC_NewResized/edBMC_1.ipynb

# Or using Python script
python code/BMC_Volume_Prediction.py
```

#### Option 2: ES(1+1) Evolutionary Optimization

```python
# Run ES(1+1) hyperparameter optimization
python BMC_ES_1plus1.py

# See QUICKSTART_ES.md for detailed configuration
```

## 📁 Project Structure

```
measuring-cup-volume-prediction/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git exclusion rules
│
├── docs/                              # Documentation
│   ├── SETUP.md                       # Detailed installation guide
│   ├── DATASET.md                     # Dataset documentation
│   └── MODELS.md                      # Model architecture details
│
├── BMC_ES_1plus1.py                   # ES(1+1) implementation
├── ES_1plus1_README.md                # ES algorithm documentation
├── QUICKSTART_ES.md                   # ES quick start guide
├── ES_IMPLEMENTATION_SUMMARY.md       # Implementation details
│
├── BMC_NewResized/                    # Main dataset directory
│   ├── training/                      # Training images (~794)
│   ├── Testing/                       # Test images (~107)
│   ├── training_labels.csv            # Training annotations
│   ├── testing_labels.csv             # Test annotations
│   └── edBMC_1.ipynb                  # Primary training notebook
│
├── code/                              # Alternative implementations
│   ├── BMC_Volume_Prediction.py       # EfficientNet Python script
│   ├── ES_1plus1_optimization.py      # ES optimization script
│   ├── BMC_EfficientNet_Merged.ipynb  # Complete pipeline notebook
│   ├── BMC_VOLUME_README.md           # EfficientNet documentation
│   └── CLAUDE.md                      # Technical reference
│
└── Final/                             # Papers and documentation
    ├── IEEE_Paper_ES_FewShot_Transfer.tex
    └── collab_2_siriv2.pdf
```

## 🧠 Model Architectures

### 1. EfficientNet Transfer Learning

**Architecture**:
- Base: EfficientNetB0 (pre-trained on ImageNet)
- Strategy: Freeze first 50% of layers, fine-tune top layers
- Custom Head: GlobalAveragePooling → Dense(256, ReLU) → Dropout(0.3) → Dense(1, linear)
- Input: 224×224×3 RGB images, normalized [0,1]
- Output: Single continuous value (volume prediction)

**Training Strategy**:
- Optimizer: Adam (lr=0.0001)
- Loss: Mean Squared Error (MSE)
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Data Augmentation: Rotation (±20°), zoom (0.8-1.2), shifts

### 2. ES(1+1) Evolutionary Strategy

**Algorithm**: (1+1) Evolution Strategy with 1/5 success rule

**Optimized Hyperparameters**:
- Hidden layers: 1-4
- Neurons per layer: 8-512
- Learning rate: 0.01-0.1
- Batch size: 16-64
- Activation: ReLU, ELU, Sigmoid, Tanh
- Optimizer: SGD, RMSprop, Adam

**CNN Base Architecture**:
- Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → MaxPool → Flatten
- Dynamic dense layers (ES-optimized)
- Dropout(0.3) regularization
- Output: Dense(22, softmax) for classification

## 📖 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Setup Guide](docs/SETUP.md)** - Detailed installation and configuration
- **[Dataset Documentation](docs/DATASET.md)** - Dataset structure and download links
- **[Model Documentation](docs/MODELS.md)** - Architecture details and benchmarks

Additional specialized guides:
- **[ES(1+1) README](ES_1plus1_README.md)** - Evolutionary strategy algorithm
- **[Quick Start ES](QUICKSTART_ES.md)** - ES(1+1) quick start
- **[EfficientNet Guide](code/BMC_VOLUME_README.md)** - Transfer learning details
- **[Technical Reference](code/CLAUDE.md)** - Comprehensive technical documentation

## 🔬 Research & Papers

This project implements techniques from:
- EfficientNet: Rethinking Model Scaling for CNNs (Tan & Le, 2019)
- Evolution Strategies for Deep Learning (Rechenberg, 1973; Schwefel, 1977)
- Few-Shot Transfer Learning for Computer Vision

Related publications:
- IEEE Paper: "ES Few-Shot Transfer Learning" (see `Final/IEEE_Paper_ES_FewShot_Transfer.tex`)

## 🛠️ Technologies Used

- **Deep Learning**: Keras 3.11+, TensorFlow 2.15+, PyTorch
- **Computer Vision**: OpenCV 4.11+, Pillow 11.3+
- **Data Science**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Development**: Jupyter Lab, Python 3.8+

## 📊 Dataset

The dataset consists of **900+ images** of measuring cups across **22 volume classes**.


<img width="366" height="245" alt="image" src="https://github.com/user-attachments/assets/8246e856-3282-4035-888d-261ae9a5c055" />


**Image Specifications**:
- Format: JPG (converted from HEIC)
- Resolution: 224×224 pixels
- Views: Front View (FV) and Bottom-Up View (BV)
- Volumes: 100-610 mL in increments of 5 or 50 mL

**Dataset Organization**:
```
BMC_NewResized/
├── training/           # ~794 training images
│   ├── 100/
│   ├── 105/
│   └── ... (22 classes)
├── Testing/            # ~107 test images
├── training_labels.csv
└── testing_labels.csv
```

For dataset download instructions, see **[docs/DATASET.md](docs/DATASET.md)**.

## 🎓 Getting Started

1. **Installation**: Follow the [Setup Guide](docs/SETUP.md)
2. **Download Dataset**: See [Dataset Documentation](docs/DATASET.md)
3. **Choose Approach**:
   - For quick results: Use EfficientNet transfer learning (`edBMC_1.ipynb`)
   - For experimentation: Try ES(1+1) optimization (`BMC_ES_1plus1.py`)
4. **Train Model**: Follow notebook instructions or run Python scripts
5. **Evaluate**: Check generated results, plots, and metrics

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- EfficientNet implementation based on Keras Applications
- ES(1+1) algorithm inspired by classical evolution strategies literature
- Dataset collected and annotated for measuring cup volume prediction research

## 📧 Contact

For questions, issues, or collaboration opportunities, please open an issue on GitHub.

---

**Note**: This is a research/portfolio project demonstrating deep learning for computer vision regression tasks. For production deployment, additional validation and testing are recommended.
