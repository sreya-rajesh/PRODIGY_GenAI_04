# PRODIGY_GenAI_04
# Pix2Pix - Image-to-Image Translation using GANs

A PyTorch implementation of Pix2Pix for image-to-image translation using conditional Generative Adversarial Networks (cGANs). This project transforms architectural facade sketches into realistic building images using the U-Net generator architecture and PatchGAN discriminator.

## 🎯 Overview

This project implements the Pix2Pix model that learns to translate input sketches/semantic maps to photorealistic images. The model uses a conditional GAN architecture where both the generator and discriminator are conditioned on input images, enabling controlled image generation.

**Key Architecture Components:**
- **U-Net Generator**: Encoder-decoder with skip connections for preserving fine details
- **PatchGAN Discriminator**: Classifies overlapping image patches as real/fake
- **Combined Loss**: Adversarial loss + L1 loss for both realism and pixel-level accuracy

*This project was developed as **Task 4** during my internship at **Prodigy InfoTech**.*

## ✨ Features

- **Complete Pix2Pix Implementation**: Full generator and discriminator architecture
- **Automated Dataset Handling**: Automatic download and preprocessing of facades dataset
- **U-Net Architecture**: Skip connections preserve spatial information
- **PatchGAN Discriminator**: 70×70 receptive field for detailed texture assessment
- **Training Visualization**: Real-time progress monitoring with sample generations
- **GPU Acceleration**: CUDA support for faster training
- **Model Persistence**: Save and load trained models

## 🛠️ Technologies Used

- **Python 3.7+**
- **PyTorch 1.7+** - Deep learning framework
- **torchvision** - Computer vision utilities
- **NumPy** - Numerical computations
- **Matplotlib** - Visualization and plotting
- **PIL (Pillow)** - Image processing
- **tqdm** - Progress bars for training
- **requests** - HTTP library for dataset download

## 📁 Project Structure

```
pix2pix-project/
│
├── pix2pix.py          # Main implementation file
├── generator.pth       # Saved generator model (after training)
├── facades/            # Downloaded dataset (auto-created)
│   └── train/          # Training images
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- CUDA-compatible GPU (recommended for faster training)
- At least 4GB of free disk space for the dataset

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/pix2pix-gan.git
   cd pix2pix-gan
   ```

2. **Install dependencies**
   ```bash
   pip install torch torchvision numpy matplotlib pillow tqdm requests
   ```

   Or create a `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the project**
   ```bash
   python pix2pix.py
   ```

The script will automatically:
- Download the facades dataset (~30MB)
- Extract and prepare the data
- Start training the model
- Display results every 10 epochs

## 🏗️ Model Architecture

### Generator (U-Net)
```
Input (3,256,256) → Encoder → Bottleneck → Decoder → Output (3,256,256)
├── Skip Connections preserve spatial details
├── 8 encoder blocks with increasing channels: 64→128→256→512
└── 8 decoder blocks with decreasing channels: 512→256→128→64→3
```

### Discriminator (PatchGAN)
```
Input: Concatenated(input_image, target_image) = (6,256,256)
├── 5 convolutional layers
├── Output: (1,30,30) patch predictions
└── Receptive field: 70×70 pixels
```

## 🎯 Training Process

The model trains for **50 epochs** with the following configuration:

- **Batch Size**: 16
- **Learning Rate**: 0.0002 (both G and D)
- **Optimizer**: Adam (β₁=0.5, β₂=0.999)
- **L1 Weight**: 100 (lambda_L1)
- **Image Size**: 256×256

### Loss Functions
- **Generator Loss**: `L_G = L_GAN + λ * L_L1`
- **Discriminator Loss**: `L_D = (L_real + L_fake) / 2`

## 📊 Results

The model generates realistic building facades from architectural sketches:

### Training Progress
- **Epoch 10**: Basic structure formation
- **Epoch 20**: Improved texture details
- **Epoch 30**: Refined architectural features
- **Epoch 50**: High-quality realistic outputs

### Sample Results
```
Input (Sketch) → Generated Image → Ground Truth
     🏗️      →        🏢        →      🏢
```

*Visual results are displayed during training every 10 epochs*

## ⚙️ Customization

### Modify Training Parameters

Edit these variables in `pix2pix.py`:

```python
num_epochs = 50        # Training duration
lambda_L1 = 100        # L1 loss weight
batch_size = 16        # Batch size
lr = 0.0002           # Learning rate
```

### Use Your Own Dataset

1. **Prepare your data**: Images should be concatenated horizontally (input|target)
2. **Update dataset path**: Modify the `FacadesDataset` class
3. **Adjust image dimensions** if needed in the transform

### Change Model Architecture

- **Generator**: Modify encoder/decoder blocks in the `Generator` class
- **Discriminator**: Adjust convolutional layers in the `Discriminator` class

## 🔧 Technical Details

### Memory Requirements
- **GPU Memory**: ~4GB VRAM recommended
- **RAM**: ~8GB for dataset loading
- **Storage**: ~100MB for saved models

### Performance Optimization
- Uses `torch.cuda.is_available()` for automatic GPU detection
- Implements efficient data loading with PyTorch DataLoader
- Includes gradient accumulation for stable training

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size
   dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
   ```

2. **Dataset Download Failed**
   ```bash
   # Manual download
   wget http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz
   tar -xzf facades.tar.gz
   ```

3. **Truncated Images Error**
   ```python
   # Already handled in code
   from PIL import ImageFile
   ImageFile.LOAD_TRUNCATED_IMAGES = True
   ```

## 📈 Monitoring Training

The training script displays:
- Progress bars for each epoch
- Loss values for both Generator and Discriminator
- Visual comparisons every 10 epochs
- Automatic model saving after training completion

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional datasets support
- Hyperparameter optimization
- Model architecture variants
- Evaluation metrics implementation




