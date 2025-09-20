import argparse
import torch
from train import train
from inference import enhance_image
from eval import evaluate_model
from retinex_model import RetinexEnhancer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "eval", "inference"], required=True)

    # Training args
    parser.add_argument("--data_dir", type=str, help="Path to dataset root (should contain train/test/val)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patch_size", type=int, default=None)  # Will auto-adjust based on LOL version
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--version", type=str, default="auto", choices=["auto", "lolv1", "lolv2"], help="LOL dataset version")

    # Inference args
    parser.add_argument("--checkpoint", type=str, help="Path to model .pth file")
    parser.add_argument("--input_image", type=str, help="Input low-light image path")
    parser.add_argument("--output_path", type=str, default="results/enhanced.jpg")
    parser.add_argument("--save_intermediates", action="store_true")
    parser.add_argument("--apply_gamma", action="store_true")
    
    # Evaluation args
    parser.add_argument("--calc_lpips", action="store_true", help="Also compute LPIPS")
    parser.add_argument("--eval_patch_size", type=int, default=512)
    parser.add_argument("--eval_save_dir", type=str, default="results_eval")

    args = parser.parse_args()

    if args.mode == "train":
        assert args.data_dir, "⚠️ --data_dir is required for training"
        train(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patch_size=args.patch_size,
            lr=args.lr,
            save_dir=args.save_dir
        )

    elif args.mode == "inference":
        assert args.input_image and args.checkpoint, "⚠️ --input_image and --checkpoint are required for inference"
        enhance_image(
            input_path=args.input_image,
            model_path=args.checkpoint,
            output_path=args.output_path,
            patch_size=args.patch_size,
            save_intermediates=args.save_intermediates,
            apply_gamma=args.apply_gamma
        )

    elif args.mode == "eval":
        assert args.data_dir and args.checkpoint, "⚠️ --data_dir and --checkpoint are required for evaluation"
        model = RetinexEnhancer().to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

        psnr_val, ssim_val, lpips_val = evaluate_model(
            model,
            val_data_dir=args.data_dir,
            patch_size=args.eval_patch_size,
            save_dir=args.eval_save_dir,
            save_intermediates=args.save_intermediates,
            calc_lpips=args.calc_lpips
        )

        print("\n✅ Evaluation Complete")
        print(f"📈 PSNR: {psnr_val:.2f}")
        print(f"📈 SSIM: {ssim_val:.4f}")
        if args.calc_lpips:
            print(f"📉 LPIPS: {lpips_val:.4f}")
