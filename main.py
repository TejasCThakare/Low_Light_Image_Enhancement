import argparse
from train import train
from inference import enhance_image
from eval import evaluate_model
from retinex_model import RetinexEnhancer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "inference", "eval"], required=True)
    
    # For training
    parser.add_argument("--data_dir", type=str, help="Path to training or validation data directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patch_size", type=int, default=192)
    
    # For inference
    parser.add_argument("--checkpoint", type=str, help="Path to .pth model file")
    parser.add_argument("--input_image", type=str, help="Input low-light image for enhancement")
    parser.add_argument("--output_path", type=str, default="results/enhanced.jpg")
    parser.add_argument("--save_intermediates", action="store_true")
    
    # For evaluation
    parser.add_argument("--save_dir", type=str, default="results_eval")

    args = parser.parse_args()

    if args.mode == "train":
        train(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patch_size=args.patch_size,
            lr=args.lr
        )

    elif args.mode == "inference":
        assert args.input_image and args.checkpoint, " --input_image and --checkpoint are required for inference!"
        enhance_image(
            input_path=args.input_image,
            model_path=args.checkpoint,
            output_path=args.output_path,
            patch_size=args.patch_size,
            save_intermediates=args.save_intermediates
        )

    elif args.mode == "eval":
        assert args.data_dir and args.checkpoint, " --data_dir and --checkpoint are required for evaluation!"
        model = RetinexEnhancer().to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        psnr_val, ssim_val = evaluate_model(
            model,
            val_data_dir=args.data_dir,
            patch_size=args.patch_size,
            save_dir=args.save_dir,
            save_intermediates=args.save_intermediates
        )
        print(f"\n Evaluation Complete")
        print(f" PSNR: {psnr_val:.2f}")
        print(f" SSIM: {ssim_val:.4f}")
