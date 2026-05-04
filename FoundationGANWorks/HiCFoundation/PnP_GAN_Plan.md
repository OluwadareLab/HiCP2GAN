P&P GAN (Option C: multi-scale) — concise step-by-step outline

Data setup

Build paired LR/HR from the same loci (both KR-normalized, same transforms).

For each sample, keep a global crop (ideally 224×224) and its local tiles (your generator tiles, e.g., 40×40).

Models

G (local generator): 40×40 LR → 40×40 SR.

D_local: small PatchGAN/CNN on 40×40 (real HR tiles vs fake SR tiles).

D_global: HiCFoundation resolution checkpoint as backbone + lightweight GAN head; expose intermediate features.

Freezing

Freeze HiCFoundation encoder (or unfreeze only last 1–2 blocks later).

Train only the GAN head (and any small adapters) in D_global.

Forward pass per batch

Sample one region: get LR_global, HR_global + the aligned LR_tiles, HR_tiles.

Run G on each LR tile → SR_tiles.

Stitch SR_tiles → SR_global (and HR_tiles → HR_global if needed) so D_global sees a full-context patch.

Update D_local

Compute GAN loss on (HR_tiles = real) vs (SR_tiles = fake).

Backprop/update D_local.

Update D_global (HiCFoundation critic)

Compute GAN loss on (HR_global = real) vs (SR_global = fake).

Backprop/update only D_global head/adapters (encoder frozen).

Update G

Reconstruction: L1 or Huber on tiles: L_rec = ||SR_tiles − HR_tiles||.

Local adversarial: encourage D_local(SR_tiles) → real.

Global adversarial: encourage D_global(SR_global) → real.

Feature matching (recommended): match HiCFoundation intermediate features: ||f(HR_global) − f(SR_global)||.

Total: L_G = λrec*L_rec + λl*L_adv_local + λg*L_adv_global + λfm*L_fm.

Warm start: first few epochs set λl=λg=0, keep λrec (and small λfm).

Validation

Run inference on held-out chromosomes/regions.

Compute metrics at:

tile scale (40×40): L1/Huber, SSIM, PSNR

global scale (stitched 224×224): SSIM, PSNR (+ correlations if you want)

Metrics choice

Primary: SSIM

Secondary: PSNR

Usually skip SNR (less informative with KR-normalized Hi-C).

Checkpointing

Save best models by val SSIM (and keep PSNR as a sanity check).

Log example LR/SR/HR tiles + stitched global patches each epoch, save the results per epoch in a text file, and generate a plot using the data for both SSIM and PSNR and Loss