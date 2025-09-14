# Retinex++: Hybrid Low-Light Image Enhancement Pipeline

Retinex++ is a complete low-light image enhancement model that combines techniques from KinD++, Zero-DCE++, and MIRNet in a four-stage neural architecture. It uses Retinex-style decomposition to separate reflectance and illumination, enhances the illumination using curve estimation and attention mechanisms, applies gated fusion, and refines the result with multi-scale attention blocks. This model supports both LOL-v1 and LOL-v2 datasets with automatic detection and adaptation.

The design focuses on interpretable enhancement with complete intermediate visualization, producing natural-looking results with improved brightness, color consistency, and structural clarity through a comprehensive 12-component loss function.

---

## Project Structure
```
retinex-pipeline/  
├── main.py              → CLI entrypoint for training, evaluation, and inference  
├── train.py             → Training loop with 12-component multi-loss optimization  
├── inference.py         → Single image enhancement with optional intermediate saves  
├── eval.py              → Comprehensive evaluation with PSNR, SSIM, and LPIPS metrics  
├── retinex_model.py     → Hybrid 4-stage architecture (Decomp→Relight→Fusion→Refine)  
├── losses.py            → 12 loss functions: L1, LAB, LPIPS, VGG, MS-SSIM, Edge, etc.  
├── dataloader.py        → LOL-v1/v2 dataset handler with automatic version detection  
├── utils.py             → Intermediate visualization utilities (R, I, I_enh saves)  
├── requirements.txt     → Python dependencies  
├── checkpoints/         → Model weights storage (epoch + best model)  
├── results/             → Enhanced outputs from inference  
├── results_eval/        → Enhanced outputs from evaluation  
├── visuals/             → Training intermediate outputs (R, I, I_enh components)  
└── README.md            → This file  
```

---

## Installation
Install the required dependencies with:

```bash
pip install -r requirements.txt
```

**requirements.txt**:
```
torch>=1.10.0
torchvision>=0.11.0
pillow>=8.0.0
tqdm>=4.62.0
scikit-image>=0.18.0
lpips>=0.1.4
pytorch-msssim>=0.2.1
kornia>=0.6.0
numpy>=1.21.0
```

---

## Dataset Format
Organize the LOL dataset like this:

```
LOL-dataset/  
├── train/
│   ├── low/     # Low-light training images
│   └── high/    # Normal-light ground truth
├── test/
│   ├── low/     # Low-light test images  
│   └── high/    # Normal-light test ground truth
└── val/         # Optional validation split
    ├── low/
    └── high/
```

Auto-detection: LOL-v1 (≤1000px → 192 patches) vs LOL-v2 (>1000px → 512 patches).  
Use `--version lolv1` or `--version lolv2` to force specific version.

---

## Training and Evaluation
Train the model with auto-detection:

```bash
python main.py --mode train --data_dir ./LOL-dataset
```

Evaluate the model:

```bash
python main.py --mode eval --data_dir ./LOL-dataset/test \
  --checkpoint checkpoints/best_model.pth \
  --calc_lpips --save_intermediates
```

Run inference on a single image:

```bash
python main.py --mode inference \
  --input_image ./samples/dark_image.jpg \
  --checkpoint checkpoints/best_model.pth \
  --output_path results/enhanced.jpg \
  --save_intermediates --apply_gamma
```

The pipeline automatically detects LOL-v1 vs LOL-v2 and adjusts patch sizes accordingly.  
You can override with `--patch_size` or force version with `--version lolv1/lolv2`.

---

## Model Overview
- **DecompositionNet**: Separates input into reflectance (R) and illumination (I) using encoder-decoder with ResBlocks  
- **RelightNet**: Enhances illumination (I → I_enh) using Zero-DCE++ curve estimation with 8 iterative enhancement steps  
- **FusionNet**: Combines R and I_enh using gated fusion mechanism with learned attention weights  
- **RefinerNet**: Final refinement using MIRNet-style multi-scale attention (RRB + MFFA + SEBlock)  

Training uses **12-component multi-objective loss** with total weight of 4.5x for comprehensive supervision.

---

## Loss Objectives

| Loss Component          | Weight | Description                                  |
|--------------------------|--------|----------------------------------------------|
| L1 Loss                 | 1.0    | Pixel-level reconstruction accuracy          |
| LAB Color Loss          | 0.5    | Perceptual color space preservation          |
| LPIPS Loss              | 0.4    | Learned perceptual similarity matching       |
| VGG Loss                | 0.4    | High-level feature content preservation      |
| MS-SSIM Loss            | 0.4    | Multi-scale structural similarity            |
| Edge Loss               | 0.3    | Reflectance edge structure preservation      |
| Histogram Loss          | 0.3    | Global tone distribution alignment           |
| Tone Contrast Loss      | 0.3    | Local contrast level maintenance             |
| Illumination Smoothness | 0.2    | Edge-aware illumination smoothness           |
| Total Variation         | 0.2    | Enhanced illumination smoothness regulation  |
| Reflectance SSIM        | 0.3    | Structural similarity on reflectance maps    |
| Curve Regularization    | 0.1    | Enhancement curve parameter regularization   |

---

## Citations
This project incorporates concepts from the following works:

```
@inproceedings{zhang2019kindling,
  title={Kindling the Darkness: A Practical Low-light Image Enhancer},
  author={Zhang, Yue and Zhang, Jiawan and Guo, Xian},
  booktitle={ACM Multimedia}, year={2019}
}

@inproceedings{zhang2021beyond,
  title={Beyond Brightening Low-Light Images},
  author={Zhang, Yue and Zhang, Jiawan and Guo, Xian},
  booktitle={CVPR}, year={2021}
}

@inproceedings{zamir2020learning,
  title={Learning Enriched Features for Real Image Restoration and Enhancement},
  author={Zamir, Syed Waqas and Arora, Akash and Khan, Salman and others},
  booktitle={ECCV}, year={2020}
}

@inproceedings{li2021zero,
  title={Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement},
  author={Li, Chongyi and Guo, Chunle and Loy, Chen Change},
  booktitle={CVPR}, year={2021}
}
```

---

## Author
[Your Name]  
GitHub: https://github.com/yourusername  
Email: your.email@domain.com  

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.
