

# Efficient AudioLDM via Structured  U-Net Pruning

This repository presents a **structured pruning framework for compressing the AudioLDM-M-Full text-to-audio generative model**. The proposed approach reduces the computational complexity and memory footprint of the diffusion U-Net by removing redundant convolutional channels while preserving audio generation quality.

The pruning pipeline consists of three main stages:

1. **Channel Importance Estimation**  
   Compute layer-wise channel importance rankings from a pretrained AudioLDM U-Net.

2. **Structured U-Net Pruning**  
   Generate compact U-Net architectures using predefined block-wise channel scaling factors.

3. **AudioLDM Checkpoint Reconstruction**  
   Merge the pruned U-Net weights into the full AudioLDM checkpoint for inference or further fine-tuning.

This repository provides scripts for generating pruning indexes, creating pruned U-Net checkpoints, and reconstructing efficient AudioLDM models.

The implementation is built on the official **AudioLDM training, fine-tuning, inference, and evaluation framework**:
[AudioLDM Training & Fine-tuning Repository](https://github.com/haoheliu/AudioLDM-training-finetuning)





### Installation and AudioLDM training/finetuning framework

Please use  this [Link](https://github.com/haoheliu/AudioLDM-training-finetuning)  which does "AudioLDM training, finetuning, inference and evaluation" repository to setup environment, finetuning code.
Thanks to Haohe Liu for great efforts on AudioLDM github repository. 


### Compute sorted indexes from a pre-trained U-Net model across convolutional layers


```

python layerwise_sorted_index_generation.py \
    --ckpt checkpoints/Unet_model-m.ckpt \
    --output pruned_indexes/B3_B4/sorted_indexes_dict.pkl  
```  


### Obtain a pruned network with pre-defined channel scaling parameters, and save the pruned U-net parameters 

```
python pruned_unet_dict_creation.py \
    --ckpt checkpoints/Unet_model-m.ckpt \
    --idx-dict pruned_indexes/B3_B4/sorted_indexes_dict.pkl \
    --output checkpoints/pruned/l1_unet_pruned_p2_dp2.pt \
    --p 2 \
    --dp 2# 

```

#### Arguments

- `--ckpt` : pre-trained U-Net checkpoint path
- `--idx-dict` : pruning index dictionary
- `--output` : U-Net pruned checkpoint  
- `--p` : block-4 channel multiplier
- `--dp` : block-3 channel multiplier



### AudioLDM Checkpoint Merger


Merge pruned UNet weights into a full AudioLDM checkpoint.

```
python merge_pruned_checkpoint.py \
    --pruned-ckpt checkpoints/pruned/l1_unet_pruned_p2_dp2.pt \
    --full-ckpt checkpoints/original/audioldm-m-full.ckpt \
    --output checkpoints/l1_audioldm-m-full_p2_dp2.ckpt
```
#### Arguments

- `--pruned-ckpt` : Pruned UNet weights (.pt)
- `--full-ckpt` : Original AudioLDM checkpoint (.ckpt)
- `--output` : Output checkpoint path
- `--prefix` : Key prefix (default: `model.diffusion_model.`)







