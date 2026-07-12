

# Efficient Text-to-Audio generative model via Pruning

This repository presents a **structured pruning framework for compressing the AudioLDM-M-Full text-to-audio generative (TTA) model**. The proposed approach reduces the computational complexity and memory footprint of the diffusion U-Net by removing redundant convolutional filters while preserving audio generation quality.


The TTA model is evaluated on AudioCaps test dataset using FAD and KL divergence metrics.

The pruning pipeline consists of three main stages:

1. **Filter Importance Estimation:** 
   Compute layer-wise channel importance rankings from a pretrained AudioLDM U-Net.

2. **Structured U-Net Pruning:** 
   Generate compact U-Net architectures using predefined block-wise channel scaling factors.

3. **AudioLDM Checkpoint Reconstruction:** 
   Merge the pruned U-Net weights into the full AudioLDM checkpoint for inference or further finetuning.
   
4. **Finetuning the pruned AudioLDM:** 
   Finetune the pruned AudioLDM-M-Full using similar training setup as used by [original AudioLDM](https://github.com/haoheliu/AudioLDM-training-finetuning), except the dataset which is AudioCaps training dataset.
   
5. **Semantic Quality Analysis:** 
    Using PANNs, we obtain top-10 sound events predicted given the generated sounds. Then, we analyse capture rate or recall to analyse what sound events are getting affected by Pruning and how does finetuning helps to recover missed sound events.

This repository provides scripts for generating pruning indexes, creating pruned U-Net checkpoints, and reconstructing efficient AudioLDM models.







### Installation and AudioLDM training/finetuning framework

The implementation is built on the official **AudioLDM training, fine-tuning, inference, and evaluation framework**:
[AudioLDM Training & Fine-tuning Repository](https://github.com/haoheliu/AudioLDM-training-finetuning)

Please use  this to setup environment, finetuning code. Thanks to Haohe Liu for great efforts on AudioLDM github repository. 


### Download checkpoints 
 Please dowloand the pre-trained AudioLDM-M-Full and U-Net model checkpoints from [zenodo link to be updated soon]().

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

p: Block-4 channel scaling factor (here, e.g. p=2)
dp: Block-3 channel scaling factor (here, e.g. dp=2)
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



### 4. Finetuning pruned AudioLDM-M-Full

Finetuning of the pruned model follows exactly same setup as shared in [AudioLDM Training & Fine-tuning Repository](https://github.com/haoheliu/AudioLDM-training-finetuning). 
It requires an update in the confoguration file: "audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml":
     
        channel_mult [1,2,3,5]--> [1,2,dp,p] and useage of the pruned model checkpoint with [1,2,dp,p] configuration.
        
 
  
```  
python3 audioldm_train/train/latent_diffusion.py -c audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml --reload_from_ckpt data/checkpoints/{pruned model checkpoint}.ckpt

```

Please follow [AudioLDM Training & Fine-tuning Repository](https://github.com/haoheliu/AudioLDM-training-finetuning) for evaluation of the model output.


### 5. Semantic Quality Analysis

This section will be updated soon. To listen generated audios, please visit [project page](https://arshdeep-singh-boparai.github.io/EfficientAudioLDM/).


## Cite this work

if you find this work interesting and useful, please cite it as:

```
To be updated soon.
```

### Acknowledgment
We greatly appreciate the open-soucre code base of the Haohe Liu's [AudioLDM Training & Fine-tuning Repository](https://github.com/haoheliu/AudioLDM-training-finetuning). 

This work was supported by the Engineering and Physical Sciences Research Council (EPSRC)  [grant number EP/Y028805/1].  For the purpose of open access, the authors have applied a Creative Commons Attribution (CC BY) licence to any Author Accepted Manuscript version arising


