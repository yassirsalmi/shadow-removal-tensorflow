# Shadow Removal using Conditional GAN

This project implements a shadow removal system using a Conditional Generative Adversarial Network (CGAN) in TensorFlow. The model is designed to remove shadows from images while preserving the original image quality.

## Model Architecture

The model consists of two main components:

1. **Shadow Detection (G1 + D1)**
   - G1: UNet-based generator for shadow mask prediction
   - D1: PatchGAN discriminator for validating shadow masks

2. **Shadow Removal (G2 + D2)**
   - G2: UNet-based generator for shadow-free image generation
   - D2: PatchGAN discriminator for validating shadow-free images

## Dataset

The model is trained on the ISTD dataset which contains:
- `train_A/`: Shadow images
- `train_B/`: Shadow masks
- `train_C/`: Shadow-free images

## Requirements and Installation

### Prerequisites
- Python 3.8+
- TensorFlow 2.10+

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yassirsalmi/shadow-removal-tensorflow
cd tensorflow
```

2. Create and activate virtual environment:
```bash
python -m venv firstenv
source firstenv/bin/activate  
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

The requirements include:
- **Core ML Framework**:
  - tensorflow>=2.10.0
  - tensorflow-addons>=0.19.0
  - numpy>=1.21.0
  
- **Image Processing**:
  - opencv-python>=4.7.0
  - Pillow>=9.0.0
  - scikit-image>=0.19.0
  
- **Progress and Logging**:
  - tqdm>=4.65.0
  - logging>=0.5.1.2
  
- **Data Processing & Utilities**:
  - pandas>=1.5.0
  - matplotlib>=3.5.0
  - ipython>=8.0.0

## Training

To train the model:

```bash
python train.py --data_dir "/path/to/ISTD_Dataset/train" --batch_size 4 --epochs 200
```

### Training Parameters
- `--batch_size`: Number of images per batch (default: 4)
- `--epochs`: Number of training epochs (default: 200)
- `--lr_G`: Generator learning rate (default: 0.0002)
- `--lr_D`: Discriminator learning rate (default: 0.0002)
- `--lambda_L1`: Weight for L1 loss (default: 100.0)

### Training Progress
The training progress is displayed with a progress bar showing:
- Current epoch and batch progress
- Generator losses (G1, G2)
- Discriminator losses (D1, D2)
- Time per epoch

Model weights are automatically saved every 5 epochs in the `weights` directory.

## Project Structure

```
tensorflow/
├── src/
│   ├── models/
│   │   ├── stcgan_g.py  # Generator architecture
│   │   └── stcgan_d.py  # Discriminator architecture
│   ├── cgan.py          # Main CGAN implementation
│   ├── dataset.py       # Data loading and preprocessing
│   ├── loss.py          # Loss functions
│   ├── networks.py      # Network utilities
│   └── transform.py     # Image transformations
├── train.py             # Training script
├── requirements.txt     # Project dependencies
└── README.md
```
