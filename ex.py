from dataloader import get_dataloader

dataset_path = "/mnt/data/Tejas/gs/HDR-GS/data_hdr/synthetic/"
dataloader = get_dataloader(dataset_path)

print(f"Total dataset size: {len(dataloader.dataset)}")

for i, (input_img, target_exposures) in enumerate(dataloader):
    print(f"Batch {i+1}: Input Image Shape: {input_img.shape}, Target Shape: {target_exposures.shape}")
    if i == 5:  # Print first 5 batches only
        break
