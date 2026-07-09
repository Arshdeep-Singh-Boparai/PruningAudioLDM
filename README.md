

# Efficient AudioLDM via Structured  U-Net Pruning

This repository presents a **structured pruning framework for compressing the AudioLDM-M-Full text-to-audio generative model**. The proposed approach reduces the computational complexity and memory footprint of the diffusion U-Net by removing redundant convolutional filters while preserving audio generation quality.

The pruning pipeline consists of three main stages:

1. **Filter Importance Estimation** 
   Compute layer-wise channel importance rankings from a pretrained AudioLDM U-Net.

2. **Structured U-Net Pruning** 
   Generate compact U-Net architectures using predefined block-wise channel scaling factors.

3. **AudioLDM Checkpoint Reconstruction** 
   Merge the pruned U-Net weights into the full AudioLDM checkpoint for inference or further finetuning.
   
4. **Finetuning the pruned AudioLDM** 
   Finetune the pruned AudioLDM-M-Full using similar training setup as used by [original AudioLDM](https://github.com/haoheliu/AudioLDM-training-finetuning).
   
5. **Semantic Quality Analysis** 
    Using PANNs, we obtain top-10 sound events predicted given the generated sounds. Then, we analyse capture rate or recall to analyse what sound events are getting affected by Pruning and how does finetuning helps to recover missed sound events.

This repository provides scripts for generating pruning indexes, creating pruned U-Net checkpoints, and reconstructing efficient AudioLDM models.







### Installation and AudioLDM training/finetuning framework

The implementation is built on the official **AudioLDM training, fine-tuning, inference, and evaluation framework**:
[AudioLDM Training & Fine-tuning Repository](https://github.com/haoheliu/AudioLDM-training-finetuning)

Please use  this to setup environment, finetuning code. Thanks to Haohe Liu for great efforts on AudioLDM github repository. 


### Download checkpoints 
 Please dowloand the pre-trained AudioLDM-M-Full and U-Net model checkpoints from [zenodo link](https://github.com/haoheliu/AudioLDM-training-finetuning).

### 1. Generate Layer-wise Sorted Channel Indexes
This step computes channel importance rankings for convolutional layers in the pretrained U-Net.

The generated index dictionary is used during pruning.

```

python layerwise_sorted_index_generation.py \
    --ckpt checkpoints/Unet_model-m.ckpt \
    --output pruned_indexes/B3_B4/sorted_indexes_dict.pkl  
```  

| Argument   | Description                                     |
| ---------- | ----------------------------------------------- |
| `--ckpt`   | Path to pretrained AudioLDM U-Net checkpoint    |
| `--output` | Output path for sorted channel index dictionary |

### 2. Generate Pruned U-Net Checkpoint

This step creates a compact U-Net by applying structured channel pruning.

The pruning ratio is controlled using block-wise channel scaling factors:

p: Block-4 channel scaling factor
dp: Block-3 channel scaling factor
```
python pruned_unet_dict_creation.py \
    --ckpt checkpoints/Unet_model-m.ckpt \
    --idx-dict pruned_indexes/B3_B4/sorted_indexes_dict.pkl \
    --output checkpoints/pruned/l1_unet_pruned_p2_dp2.pt \
    --p 2 \
    --dp 2# 

```

| Argument     | Description                             |
| ------------ | --------------------------------------- |
| `--ckpt`     | Pre-trained U-Net checkpoint path       |
| `--idx-dict` | Layer-wise pruning index dictionary     |
| `--output`   | Output path for pruned U-Net checkpoint |
| `--p`        | Block-4 channel multiplier              |
| `--dp`       | Block-3 channel multiplier              |


### 3. AudioLDM Checkpoint Merger

The generated pruned U-Net checkpoint contains only the modified diffusion model parameters.

This step merges the pruned U-Net weights into the complete AudioLDM checkpoint while keeping other model components unchanged.


```
python merge_pruned_checkpoint.py \
    --pruned-ckpt checkpoints/pruned/l1_unet_pruned_p2_dp2.pt \
    --full-ckpt checkpoints/original/audioldm-m-full.ckpt \
    --output checkpoints/l1_audioldm-m-full_p2_dp2.ckpt
```
| Argument        | Description                                                    |
| --------------- | -------------------------------------------------------------- |
| `--pruned-ckpt` | Path to pruned U-Net weights (`.pt`)                           |
| `--full-ckpt`   | Original AudioLDM checkpoint (`.ckpt`)                         |
| `--output`      | Output merged AudioLDM checkpoint                              |
| `--prefix`      | U-Net parameter key prefix (default: `model.diffusion_model.`) |



### 4. Finetuning pruned AudioLDM-M-FUll


