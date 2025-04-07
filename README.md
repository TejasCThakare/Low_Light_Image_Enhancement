# L2.5: Retinex-Based Low-Light Image Enhancement

L2.5 is a complete low-light image enhancement model inspired by techniques from KinD++, MIRNet, and Zero-DCE++. It uses a Retinex-style decomposition to separate reflectance and illumination, enhances the illumination using a deep attention-based network, and fuses the components back into a visually improved image. This model is trained and evaluated on the LOL-v1 dataset.

The design focuses on natural-looking results with improved brightness, color consistency, and structural clarity.

---

## Project Structure

Retinex-L2.5/  
├── main.py              → Entrypoint for training, evaluation, and inference  
├── train.py             → Full training loop with multi-loss support  
├── inference.py         → Run enhancement on single images  
├── eval.py              → Evaluate PSNR, SSIM, and LPIPS metrics  
├── retinex_model.py     → Core Retinex-based generator architecture  
├── losses.py            → LAB, LPIPS, VGG, SSIM, TV, histogram losses  
├── dataloader.py        → Handles LOL-v1 dataset  
├── utils.py             → Image saving and preprocessing helpers  
├── requirements.txt     → Python dependencies  
├── .gitignore           → Ignore weights, results, datasets, etc.  
├── checkpoints/         → Folder to store trained models  
├── results/             → Output from inference and evaluation  
├── visuals/             → Optional intermediate outputs (R, I, I_enh)  
└── README.md            → This file  

---

## Installation

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

---

## Dataset Format

Organize the LOL-v1 dataset like this:

LOL-v1/  
├── train/low/  
├── train/high/  
└── test/  
  ├── low/  
  └── high/  

Use `--patch_size 192` for training and inference on LOL-v1.

---

## Training and Evaluation

Train the model on LOL-v1:

```bash
python main.py --mode train --data_dir ./LOL-v1 --patch_size 192
```

Evaluate the model:

```bash
python main.py --mode eval --data_dir ./LOL-v1/test \
  --checkpoint checkpoints/best_model.pth --patch_size 192 \
  --calc_lpips --save_intermediates
```

Run inference on a single image:

```bash
python main.py --mode inference \
  --input_image ./samples/night.jpg \
  --checkpoint checkpoints/best_model.pth \
  --output_path results/night_enhanced.jpg \
  --patch_size 192 --apply_gamma --save_intermediates
```

> You can also fine-tune or apply this model to other datasets like LOL-v2 using a similar approach.

---

## Model Overview

- Retinex decomposition separates the image into reflectance (R) and illumination (I)  
- An attention-based network enhances the illumination (I → I_enh)  
- The final output is formed by combining R with the enhanced illumination  
- The training uses a combination of perceptual, structural, color, and tone-based loss functions  

---

## Loss Objectives

| Loss Name         | Description                                        |
|-------------------|----------------------------------------------------|
| L1 Loss           | Preserves pixel-level accuracy                     |
| SSIM Loss         | Maintains structural consistency                   |
| LAB Color Loss    | Enforces perceptual color similarity               |
| LPIPS Loss        | Measures perceptual difference using deep features|
| VGG Loss          | Matches high-level content structure               |
| TV Loss           | Promotes smooth illumination                       |
| Tone Loss         | Guides brightness and exposure levels              |
| Histogram Loss    | Aligns tone distribution between input and output  |
| Chroma Reg Loss   | Prevents color oversaturation                      |

---

## Citations

This project incorporates concepts from the following works:

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

**Tejas Thakare**  
GitHub: https://github.com/TejasCThakare  
Email: tejas2iitmadras@gmail.com  

---

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
