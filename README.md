# L2.5: Retinex-Based Low-Light Image Enhancement

L2.5 is a full-fidelity low-light enhancement model combining architectural insights from KinD++, MIRNet, and Zero-DCE++. It performs Retinex-style decomposition, illumination enhancement, and realistic fusion with perceptual and statistical losses. Built for real-world use on LOL-v1 and LOL-v2 datasets.

The goal of this project is to provide high-quality enhancement with realistic tone, structure preservation, and minimal visual artifacts.

---

## Project Structure

Retinex-L2.5/  
├── main.py              → Entrypoint for training, evaluation, and inference  
├── train.py             → Full training loop with multi-loss support  
├── inference.py         → Run enhancement on single images  
├── eval.py              → Evaluate PSNR, SSIM, and LPIPS metrics  
├── retinex_model.py     → Core Retinex-based generator architecture  
├── losses.py            → LAB, LPIPS, VGG, SSIM, TV, histogram losses  
├── dataloader.py        → Handles LOL-v1 / LOL-v2 datasets  
├── utils.py             → Image saving and preprocessing helpers  
├── requirements.txt     → Dependencies for L2.5 model  
├── .gitignore           → Ignore weights, results, datasets, etc.  
├── checkpoints/         → Folder to store trained models (*.pth)  
├── results/             → Output from inference and evaluation  
├── visuals/             → Optional intermediate outputs (R, I, I_enh)  
└── README.md            → This file  

---

## Installation

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Dataset Format

Organize LOL-v1 or LOL-v2 datasets like this:

LOL-v1/  
├── train/low/  
├── train/high/  
└── test/  
  ├── low/  
  └── high/  

Use `--patch_size 192` for LOL-v1 and `--patch_size 512` for LOL-v2.

---

## Training and Evaluation

Train on LOL-v1:

```bash
python main.py --mode train --data_dir ./LOL-v1 --patch_size 192
```

Fine-tune on LOL-v2 Real:

```bash
python main.py --mode train --data_dir ./LOL-v2/real \
  --patch_size 512 --fine_tune \
  --resume_ckpt checkpoints/best_model.pth --lr 5e-5 --epochs 20
```

Evaluate on LOL-v1:

```bash
python main.py --mode eval --data_dir ./LOL-v1/test \
  --checkpoint checkpoints/best_model.pth --patch_size 192 \
  --calc_lpips --save_intermediates
```

Inference on a single image:

```bash
python main.py --mode inference \
  --input_image ./samples/night.jpg \
  --checkpoint checkpoints/best_model.pth \
  --output_path results/night_enhanced.jpg \
  --patch_size 192 --apply_gamma --save_intermediates
```

---

## Model Overview

- Retinex decomposition separates reflectance (R) and illumination (I)  
- A deep network enhances illumination (I → I_enh) using attention modules  
- The final image is reconstructed by combining R and I_enh  
- Losses used for training include perceptual, color, structural, and tone-aware constraints  

---

## Loss Objectives

| Loss Name         | Purpose                                              |
|-------------------|------------------------------------------------------|
| L1 Loss           | Pixel-level accuracy                                 |
| SSIM Loss         | Preserves structural details                         |
| LAB Color Loss    | Ensures color constancy in perceptual space          |
| LPIPS Loss        | Measures perceptual similarity using deep features   |
| VGG Loss          | Content reconstruction from high-level features      |
| TV Loss           | Promotes smooth illumination maps                    |
| Tone Loss         | Improves brightness and tone balance                 |
| Histogram Loss    | Aligns tone distribution                             |
| Chroma Reg Loss   | Stabilizes color saturation                          |

---

## Citations

This model incorporates ideas from:

```bibtex
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
  author={Zamir, Syed Waqas and Arora, Akash and Khan, Salman and ...},
  booktitle={ECCV}, year={2020}
}

@inproceedings{li2021learning,
  title={Zero-Reference Deep Curve Estimation for Low-Light Enhancement},
  author={Li, Chongyi and Gu, Shuochen and Loy, Chen Change},
  booktitle={CVPR}, year={2021}
}
```

---

## Author

**Tejas Chandrakant Thakare**  
GitHub: https://github.com/TejasCThakare  
Email: tejas2iitmadras.gmail.com  


---

## License

This project is licensed under the MIT License.  
See the [LICENSE](./LICENSE) file for details.
