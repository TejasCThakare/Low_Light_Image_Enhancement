# Retinex++: Hybrid Low-Light Image Enhancement Pipeline

A hybrid low-light image enhancement model combining KinD++, Zero-DCE++, and MIRNet architectures with Retinex-style decomposition. Features automatic LOL dataset detection, comprehensive multi-loss training, and complete evaluation metrics.



## Installation

```
pip install torch torchvision pillow tqdm scikit-image lpips pytorch-msssim kornia numpy
```

## Dataset Structure

```
dataset/
├── train/
│   ├── low/     # Low-light images
│   └── high/    # Ground truth enhanced images
└── test/
    ├── low/
    └── high/
```

## Usage

### Training
```
# Basic training with auto-detection
python main.py --mode train --data_dir ./dataset

# Training with specific parameters
python main.py --mode train --data_dir ./dataset --epochs 100 --batch_size 4 --lr 1e-4 --patch_size 192
```

### Evaluation  
```
# Evaluate model with metrics
python main.py --mode eval --data_dir ./dataset/test --checkpoint checkpoints/best_model.pth --eval_save_dir results_eval

# Evaluation with intermediates and LPIPS
python main.py --mode eval --data_dir ./dataset/test --checkpoint checkpoints/best_model.pth --save_intermediates --calc_lpips
```

### Inference
```
# Single image enhancement
python main.py --mode inference --input_image input.jpg --checkpoint checkpoints/best_model.pth --output_path enhanced.jpg

# With intermediate outputs and gamma correction
python main.py --mode inference --input_image input.jpg --checkpoint checkpoints/best_model.pth --save_intermediates --apply_gamma
```

## Model Architecture

- **DecompositionNet**: Separates input into reflectance (R) and illumination (I) components using encoder-decoder with ResBlocks
- **RelightNet**: Enhances illumination using Zero-DCE++ curve estimation with iterative enhancement steps  
- **FusionNet**: Combines R and I_enh using gated fusion mechanism with learned attention weights
- **RefinerNet**: Final refinement using MIRNet-style multi-scale attention (RRB + MFFA + SEBlock)


## Command Arguments

### Training Mode
- `--mode train` (required)
- `--data_dir`: Dataset root path (required)
- `--epochs`: Training epochs (default: 100)
- `--batch_size`: Batch size (default: 4)
- `--lr`: Learning rate (default: 1e-4)
- `--patch_size`: Auto-adjusts based on LOL version
- `--save_dir`: Checkpoint directory (default: "checkpoints")
- `--version`: Force LOL version ["auto", "lolv1", "lolv2"]

### Evaluation Mode
- `--mode eval` (required)
- `--data_dir`: Test data directory (required)
- `--checkpoint`: Model checkpoint path (required)
- `--eval_save_dir`: Output directory (default: "results_eval")
- `--eval_patch_size`: Processing patch size (default: 512)
- `--save_intermediates`: Save R, I, I_enh components
- `--calc_lpips`: Compute LPIPS perceptual metrics

### Inference Mode
- `--mode inference` (required)
- `--input_image`: Input image path (required)
- `--checkpoint`: Model checkpoint path (required)
- `--output_path`: Output path (default: "results/enhanced.jpg")
- `--save_intermediates`: Save intermediate components
- `--apply_gamma`: Apply gamma correction (1/2.2)

```

## Author

**[Tejas Thakare]**  
```
