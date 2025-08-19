# AABC AI/ML Workshop 2025

A machine learning project for image segmentation using TensorFlow/Keras with ResNet architectures and custom utility functions.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Before you begin, ensure you have the following installed on your system:
- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Git
- CUDA-compatible GPU (recommended for training but not necessary)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Michael-S-Phillips/AI_workshop_2025
cd AI_workshop_2025
```

## Environment Setup

### Option 1: Using environment.yml (Recommended)

An `environment.yml` file is provided in the repository:

```bash
conda env create -f environment.yml
conda activate aiml
```

### Option 2: Manual Setup

Create the environment manually with all dependencies:

```bash
# Create environment
conda create -n aiml python=3.9

# Activate environment
conda activate aiml

# Install packages
conda install -c conda-forge \
    pandas \
    numpy \
    matplotlib \
    scikit-learn \
    tqdm \
    jupyter \
    ipykernel

# Install TensorFlow and related packages
pip install tensorflow keras-cv openai

```

## Project Structure

```
AI_workshop_2025/
├── README.md
├── environment.yml
├── util_files/
│   └── util_functions.py
├── data/
│   ├── images/
│   └── masks/
├── models/
├── notebooks/
└── scripts/
```

## Usage

### 1. Activate Environment

Always activate the conda environment before running the code:

```bash
conda activate aiml
```

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**
   ```bash
   # Check CUDA version
   nvidia-smi
   
   # Install specific TensorFlow version for your CUDA version
   pip install tensorflow==2.13.0  # Adjust version as needed
   ```

2. **Import Errors**
   ```bash
   # Ensure you're in the correct environment
   conda activate aiml
   
   # Check if packages are installed
   pip list | grep tensorflow
   ```

3. **Memory Issues**
   - Reduce batch size in your training scripts
   - Use mixed precision training
   - Monitor GPU memory usage

4. **Module Not Found Errors**
   ```bash
   # Add project root to Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

### Environment Recreation

If you need to recreate the environment:

```bash
# Remove existing environment
conda env remove -n aiml

# Recreate following the installation steps above
```

## Additional Notes

- **GPU Memory**: Large ResNet models (ResNet152) require significant GPU memory. Consider using ResNet50 for initial testing.
- **OpenAI API**: Set up your OpenAI API key as an environment variable:
  ```bash
  export OPENAI_API_KEY="your-api-key-here"
  ```
- **Data Organization**: Organize your image and mask data in the `data/` directory following the expected structure for your utility functions.
