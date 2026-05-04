"""
Plot training results: SSIM, PSNR, and Loss curves.
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_history(log_file):
    """
    Load training history from text file.
    """
    data = np.loadtxt(log_file, skiprows=1, delimiter='\t')
    
    history = {
        'epoch': data[:, 0],
        'train_g_loss': data[:, 1],
        'train_rec_loss': data[:, 2],
        'train_adv_local': data[:, 3],
        'train_adv_global': data[:, 4],
        'train_fm': data[:, 5],
        'val_ssim_global': data[:, 6],
        'val_psnr_global': data[:, 7],
        'val_ssim_tiles': data[:, 8],
        'val_psnr_tiles': data[:, 9],
        'val_l1_global': data[:, 10],
        'val_l1_tiles': data[:, 11]
    }
    
    return history


def plot_results(history, save_dir):
    """
    Plot SSIM, PSNR, and Loss curves.
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = history['epoch']
    
    # Plot 1: SSIM curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_ssim_global'], label='SSIM Global', linewidth=2)
    plt.plot(epochs, history['val_ssim_tiles'], label='SSIM Tiles', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('SSIM', fontsize=12)
    plt.title('Validation SSIM', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ssim_curves.png'), dpi=300)
    plt.close()
    
    # Plot 2: PSNR curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_psnr_global'], label='PSNR Global', linewidth=2)
    plt.plot(epochs, history['val_psnr_tiles'], label='PSNR Tiles', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title('Validation PSNR', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'psnr_curves.png'), dpi=300)
    plt.close()
    
    # Plot 3: Generator Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_g_loss'], label='Total G Loss', linewidth=2)
    plt.plot(epochs, history['train_rec_loss'], label='Reconstruction Loss', linewidth=2)
    plt.plot(epochs, history['train_adv_local'], label='Adversarial Local', linewidth=2)
    plt.plot(epochs, history['train_adv_global'], label='Adversarial Global', linewidth=2)
    plt.plot(epochs, history['train_fm'], label='Feature Matching', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Generator Training Losses', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale for better visualization
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'generator_losses.png'), dpi=300)
    plt.close()
    
    # Plot 4: Combined metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # SSIM
    axes[0, 0].plot(epochs, history['val_ssim_global'], label='Global', linewidth=2)
    axes[0, 0].plot(epochs, history['val_ssim_tiles'], label='Tiles', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('SSIM')
    axes[0, 0].set_title('SSIM')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # PSNR
    axes[0, 1].plot(epochs, history['val_psnr_global'], label='Global', linewidth=2)
    axes[0, 1].plot(epochs, history['val_psnr_tiles'], label='Tiles', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('PSNR (dB)')
    axes[0, 1].set_title('PSNR')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # L1 Loss
    axes[1, 0].plot(epochs, history['val_l1_global'], label='Global', linewidth=2)
    axes[1, 0].plot(epochs, history['val_l1_tiles'], label='Tiles', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('L1 Loss')
    axes[1, 0].set_title('L1 Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Generator Loss
    axes[1, 1].plot(epochs, history['train_g_loss'], label='Total', linewidth=2)
    axes[1, 1].plot(epochs, history['train_rec_loss'], label='Reconstruction', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Generator Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'combined_metrics.png'), dpi=300)
    plt.close()
    
    print(f"Plots saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Plot training results')
    parser.add_argument('--log_file', type=str, required=True,
                       help='Path to training_history.txt')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save plots (default: same as log_file directory)')
    
    args = parser.parse_args()
    
    if args.save_dir is None:
        args.save_dir = os.path.dirname(args.log_file)
    
    history = load_history(args.log_file)
    plot_results(history, args.save_dir)


if __name__ == '__main__':
    main()
