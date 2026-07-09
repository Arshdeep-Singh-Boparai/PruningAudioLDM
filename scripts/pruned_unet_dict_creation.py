# l1norm pruning

from collections import OrderedDict
import shutil
import sys, os
sys.path.append(os.getcwd())

import argparse
import yaml
import torch

from audioldm_train.modules.diffusionmodules.openaimodel import UNetModel
  
 
# --------------------------------------------------
# Import UNet (AudioLDM / Stable Diffusion style)
# --------------------------------------------------


#%%
# ==================================================
# COMMAND LINE ARGUMENTS
# ==================================================

parser = argparse.ArgumentParser(
    description="AudioLDM L1-Norm Structured Pruning"
)

parser.add_argument(
    "--ckpt",
    required=True,
    help="Path to original checkpoint"
)

parser.add_argument(
    "--idx-dict",
    required=True,
    help="Path to pruning index dictionary (.pkl)"
)

parser.add_argument(
    "--output",
    required=True,
    help="Output checkpoint path"
)

parser.add_argument(
    "--p",
    type=int,
    required=True,
    choices=[1, 2, 3, 4, 5],
    help="channel multiplier p"
)

parser.add_argument(
    "--dp",
    type=int,
    required=True,
    choices=[1, 2, 3],
    help="channel multiplier dp"
)

args = parser.parse_args()
#%%
unet = UNetModel(
    image_size=64,
    extra_film_condition_dim=512,
    in_channels=8,
    out_channels=8,
    model_channels=192,
    attention_resolutions=(8, 4, 2),
    num_res_blocks=2,
    channel_mult=(1, 2, 3, 5),
    num_head_channels=32,
    use_spatial_transformer=True,
    transformer_depth=1,
    extra_sa_layer=False,
)


# ==================================================
# 4. PRUNING FUNCTION (Conv / Norm only)
# ==================================================


def prune_with_indices(old_sd, new_sd, idx_dict, layer_map):

    pruned_sd = OrderedDict()

    for k, v_new in new_sd.items():

        # Skip if not in old_sd (same as your original function)
        if k not in old_sd:
            continue

        v_old = old_sd[k]

        # =========================================================
        # -------- CASE 0: SHAPE MATCH → KEEP ----------------------
        # =========================================================
        if v_old.shape == v_new.shape:
            pruned_sd[k] = v_old
            continue

        # =========================================================
        # -------- CASE 1: STRUCTURED (layer_map) ------------------
        # =========================================================
        if k in layer_map:

            idx1_name, idx2_name = layer_map[k]

            # -------- output indices --------
            out_k = v_new.shape[0]
            out_idx_full = idx_dict[idx1_name]
            out_idx = out_idx_full[:out_k]

            # -------- Conv / Linear --------
            if v_old.ndim == 4 or v_old.ndim == 2:

                if idx2_name is not None:
                    in_k = v_new.shape[1]
                    in_idx_full = idx_dict[idx2_name]
                    in_idx = in_idx_full[:in_k]
                else:
                    in_idx = slice(None)

                pruned_sd[k] = v_old[out_idx][:, in_idx]

            # -------- Bias / Norm --------
            elif v_old.ndim == 1:
                pruned_sd[k] = v_old[out_idx]

            else:
                pruned_sd[k] = v_old

        # =========================================================
        # -------- CASE 2: FALLBACK (your original logic) ----------
        # =========================================================
        else:

            # Conv2d weights
            if v_old.ndim == 4:
                pruned_sd[k] = v_old[:v_new.shape[0], :v_new.shape[1]]

            # Linear weights
            elif v_old.ndim == 2:
                pruned_sd[k] = v_old[:v_new.shape[0], :v_new.shape[1]]

            # Bias / Norm
            elif v_old.ndim == 1:
                pruned_sd[k] = v_old[:v_new.shape[0]]

            else:
                pruned_sd[k] = v_old

    return pruned_sd
# --------------------------------------------------
# MAIN SCRIPT
# --------------------------------------------------

# ==================================================
# 1. CONFIG
# ==================================================
ORIG_CKPT = args.ckpt
OUT_CKPT = args.output

p = args.p
dp = args.dp

print("\n========== CONFIG ==========")
print(f"Checkpoint : {ORIG_CKPT}")
print(f"Index file : {args.idx_dict}")
print(f"Output     : {OUT_CKPT}")
print(f"p          : {p}")
print(f"dp         : {dp}")
print("============================\n")

# Original model configuration
orig_config = dict(
    image_size=64,
    extra_film_condition_dim=512,
    in_channels=8,
    out_channels=8,
    model_channels=192,
    attention_resolutions=(8, 4, 2),
    num_res_blocks=2,
    channel_mult=(1, 2, 3, 5),
    num_head_channels=32,
    use_spatial_transformer=True,
    transformer_depth=1,
    extra_sa_layer=False, 
)

# Pruned model config
pruned_config = dict(
    image_size=64,
    extra_film_condition_dim=512,
    in_channels=8,
    out_channels=8,
    model_channels=192,
    attention_resolutions=(8, 4, 2),
    num_res_blocks=2,
    channel_mult=(1, 2, dp, p),
    num_head_channels=32,
    use_spatial_transformer=True,
    transformer_depth=1,
    extra_sa_layer=False, 
)

# ==================================================
# 2. LOAD ORIGINAL MODEL
# ==================================================
print("Loading original UNet...")

orig_unet = UNetModel(**orig_config)

# ==================================================    
# Load checkpoint (all)
# ==================================================
ckpt = torch.load(ORIG_CKPT, map_location="cpu")
# Handle different checkpoint formats
if isinstance(ckpt, dict):
    # If it has a 'state_dict' key, extract it
    if 'state_dict' in ckpt:
        orig_sd = ckpt['state_dict']
    else:
        orig_sd = ckpt
else:
    # If it's a model object, get its state_dict
    orig_sd = ckpt.state_dict()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

orig_unet.load_state_dict(orig_sd, strict=True)

print("Original UNet loaded ✔")
#
# ==================================================
# 3. BUILD PRUNED MODEL
# ==================================================
print("Building pruned UNet...")

pruned_unet = UNetModel(**pruned_config)

target_sd = pruned_unet.state_dict()

#  prepare layer_map for pruning function
layer_map = {
    # -------- INPUT BLOCKS --------
    "input_blocks.7.0.in_layers.2.weight": ("input_blocks.7.0.in_layers.2.weight", None),
    "input_blocks.7.0.out_layers.3.weight": ("input_blocks.7.0.out_layers.3.weight", "input_blocks.7.0.in_layers.2.weight"),


    "input_blocks.8.0.in_layers.2.weight": ("input_blocks.8.0.in_layers.2.weight", "input_blocks.7.0.in_layers.2.weight"),
    "input_blocks.8.0.out_layers.3.weight": ("input_blocks.8.0.out_layers.3.weight", "input_blocks.8.0.in_layers.2.weight"),

    "input_blocks.9.0.op.weight": ("input_blocks.9.0.op.weight", "input_blocks.8.0.in_layers.2.weight"),



    "input_blocks.10.0.in_layers.2.weight": ("input_blocks.10.0.in_layers.2.weight", "input_blocks.9.0.op.weight"),
    "input_blocks.10.0.out_layers.3.weight": ("input_blocks.10.0.out_layers.3.weight", "input_blocks.10.0.in_layers.2.weight"),

    "input_blocks.11.0.in_layers.2.weight": ("input_blocks.11.0.in_layers.2.weight", "input_blocks.10.0.in_layers.2.weight"),
    "input_blocks.11.0.out_layers.3.weight": ("input_blocks.11.0.out_layers.3.weight", "input_blocks.11.0.in_layers.2.weight"),

    # -------- MIDDLE BLOCK --------
    "middle_block.0.in_layers.2.weight": ("middle_block.0.in_layers.2.weight", "input_blocks.11.0.in_layers.2.weight"),
    "middle_block.0.out_layers.3.weight": ("middle_block.0.out_layers.3.weight", "middle_block.0.in_layers.2.weight"),

    "middle_block.2.in_layers.2.weight": ("middle_block.2.in_layers.2.weight", "middle_block.0.out_layers.3.weight"),
    "middle_block.2.out_layers.3.weight": ("middle_block.2.out_layers.3.weight", "middle_block.2.in_layers.2.weight"),

    # -------- OUTPUT BLOCKS --------
    "output_blocks.0.0.in_layers.2.weight": ("output_blocks.0.0.in_layers.2.weight", "middle_block.2.out_layers.3.weight"),
    "output_blocks.0.0.out_layers.3.weight": ("output_blocks.0.0.out_layers.3.weight", "output_blocks.0.0.in_layers.2.weight"),

    "output_blocks.1.0.in_layers.2.weight": ("output_blocks.1.0.in_layers.2.weight", "output_blocks.0.0.out_layers.3.weight"),
    "output_blocks.1.0.out_layers.3.weight": ("output_blocks.1.0.out_layers.3.weight", "output_blocks.1.0.in_layers.2.weight"),

   # "output_blocks.2.0.in_layers.2.weight": ("output_blocks.2.0.in_layers.2.weight", None),
    "output_blocks.2.0.out_layers.3.weight": ("output_blocks.2.0.out_layers.3.weight", "output_blocks.2.0.in_layers.2.weight"),

    "output_blocks.2.2.conv.weight": ("output_blocks.2.2.conv.weight", "output_blocks.2.0.out_layers.3.weight"),

    #"output_blocks.3.0.in_layers.2.weight": ("output_blocks.3.0.in_layers.2.weight", None),
    "output_blocks.3.0.out_layers.3.weight": ("output_blocks.3.0.out_layers.3.weight", "output_blocks.3.0.in_layers.2.weight"),

    #"output_blocks.4.0.in_layers.2.weight": ("output_blocks.4.0.in_layers.2.weight", None),
    "output_blocks.4.0.out_layers.3.weight": ("output_blocks.4.0.out_layers.3.weight", "output_blocks.4.0.in_layers.2.weight"),

    #"output_blocks.5.0.in_layers.2.weight": ("output_blocks.5.0.in_layers.2.weight", None),
    "output_blocks.5.0.out_layers.3.weight": ("output_blocks.5.0.out_layers.3.weight", "output_blocks.5.0.in_layers.2.weight"),

    "output_blocks.5.2.conv.weight": ("output_blocks.5.2.conv.weight", "output_blocks.5.0.out_layers.3.weight"),

    #"output_blocks.6.0.in_layers.2.weight": ("output_blocks.6.0.in_layers.2.weight", None),





    # -------- BIASES --------
    "input_blocks.7.0.in_layers.2.bias": ("input_blocks.7.0.in_layers.2.weight", None),
    "input_blocks.7.0.out_layers.3.bias": ("input_blocks.7.0.out_layers.3.weight", None),


    "input_blocks.8.0.in_layers.2.bias": ("input_blocks.8.0.in_layers.2.weight", None),
    "input_blocks.8.0.out_layers.3.bias": ("input_blocks.8.0.out_layers.3.weight", None),

    "input_blocks.9.0.op.bias": ("input_blocks.9.0.op.weight", None),

    "input_blocks.10.0.in_layers.2.bias": ("input_blocks.10.0.in_layers.2.weight", None),
    "input_blocks.10.0.out_layers.3.bias": ("input_blocks.10.0.out_layers.3.weight", None),

    "input_blocks.11.0.in_layers.2.bias": ("input_blocks.11.0.in_layers.2.weight", None),
    "input_blocks.11.0.out_layers.3.bias": ("input_blocks.11.0.out_layers.3.weight", None),

    "middle_block.0.in_layers.2.bias": ("middle_block.0.in_layers.2.weight", None),
    "middle_block.0.out_layers.3.bias": ("middle_block.0.out_layers.3.weight", None),

    "middle_block.2.in_layers.2.bias": ("middle_block.2.in_layers.2.weight", None),
    "middle_block.2.out_layers.3.bias": ("middle_block.2.out_layers.3.weight", None),

    "output_blocks.0.0.in_layers.2.bias": ("output_blocks.0.0.in_layers.2.weight", None),
    "output_blocks.0.0.out_layers.3.bias": ("output_blocks.0.0.out_layers.3.weight", None),  # fixed typo

    "output_blocks.1.0.in_layers.2.bias": ("output_blocks.1.0.in_layers.2.weight", None),
    "output_blocks.1.0.out_layers.3.bias": ("output_blocks.1.0.out_layers.3.weight", None),

    #"output_blocks.2.0.in_layers.2.bias": ("output_blocks.2.0.in_layers.2.weight", None),
    "output_blocks.2.0.out_layers.3.bias": ("output_blocks.2.0.out_layers.3.weight", None),

    "output_blocks.2.2.conv.bias": ("output_blocks.2.2.conv.weight", None),

    #"output_blocks.3.0.in_layers.2.bias": ("output_blocks.3.0.in_layers.2.weight", None),
    "output_blocks.3.0.out_layers.3.bias": ("output_blocks.3.0.out_layers.3.weight", None),

    #"output_blocks.4.0.in_layers.2.bias": ("output_blocks.4.0.in_layers.2.weight", None),
    "output_blocks.4.0.out_layers.3.bias": ("output_blocks.4.0.out_layers.3.weight", None),
    
    #"output_blocks.5.0.in_layers.2.bias": ("output_blocks.5.0.in_layers.2.weight", None),
    "output_blocks.5.0.out_layers.3.bias": ("output_blocks.5.0.out_layers.3.weight", None),
    
    "output_blocks.5.2.conv.bias": ("output_blocks.5.2.conv.weight", None),
    
    #"output_blocks.6.0.in_layers.2.bias": ("output_blocks.6.0.in_layers.2.weight", None),

}
# ==================================================
# load idx_dict
# ==================================================
try:
    idx_dict = torch.load(
        args.idx_dict,
        map_location="cpu",
        weights_only=False,
    )

except TypeError:

    idx_dict = torch.load(
        args.idx_dict,
        map_location="cpu"
    )

except Exception:

    import pickle

    with open(args.idx_dict, "rb") as f:
        idx_dict = pickle.load(f)

print(f"Loaded idx_dict with {len(idx_dict)} keys")


# ==================================================
# 5. APPLY PRUNING
# ==================================================
print("Pruning weights...")

pruned_sd = prune_with_indices(orig_sd, target_sd, idx_dict, layer_map)

missing, unexpected = pruned_unet.load_state_dict(pruned_sd, strict=True)

print(f"Missing keys     : {len(missing)} (expected)")
print(f"Unexpected keys  : {len(unexpected)} (expected)")


# ==================================================
# 6. SAVE PRUNED MODEL
# ==================================================
os.makedirs(
    os.path.dirname(OUT_CKPT),
    exist_ok=True
)
torch.save(pruned_unet.state_dict(), OUT_CKPT)  # save model after pruning 

print(f"Pruned UNet saved to: {OUT_CKPT}")

#%%

state_dict_pruned = torch.load(OUT_CKPT, map_location="cpu")

missing, unexpected = pruned_unet.load_state_dict(state_dict_pruned, strict=True)

print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

def count_params(m):
    return sum(p.numel() for p in m.parameters())

print("Original:", count_params(orig_unet))
print("Pruned  :", count_params(pruned_unet))






