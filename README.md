## Usage

# Compute sorted indexes from a pre-trained U-Net model across convolutional layers


```
bash

python layerwise_sorted_index_generation.py \
    --ckpt checkpoints/Unet_model-m.ckpt \
    --output pruned_indexes/B3_B4/sorted_indexes_dict.pkl  
```  


# Obtain a pruned network with pre-defined channel scaling parameters (b3(dp),b4(p)), and save the pruned U-net parameters 

```
bash
python pruned_unet_dict_creation.py \
    --ckpt checkpoints/Unet_model-m.ckpt \
    --idx-dict pruned_indexes/B3_B4/sorted_indexes_dict.pkl \
    --output checkpoints/pruned/l1_unet_pruned_p2_dp2.pt \
    --p 2 \
    --dp 2# 

```

### Arguments

- `--ckpt` : pre-trained UNet checkpoint path
- `--idx-dict` : pruning index dictionary
- `--output` : U-Net pruned checkpoint  
- `--p` : block-4 channel multiplier
- `--dp` : block-3 channel multiplier



# AudioLDM Checkpoint Merger

Merge pruned UNet weights into a full AudioLDM checkpoint.

## Usage
```
python merge_pruned_checkpoint.py \
    --pruned-ckpt checkpoints/pruned/l1_unet_pruned_p2_dp2.pt \
    --full-ckpt checkpoints/original/audioldm-m-full.ckpt \
    --output checkpoints/l1_audioldm-m-full_p2_dp2.ckpt
```
### Arguments

- `--pruned-ckpt` : Pruned UNet weights (.pt)
- `--full-ckpt` : Original AudioLDM checkpoint (.ckpt)
- `--output` : Output checkpoint path
- `--prefix` : Key prefix (default: `model.diffusion_model.`)







