#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 16:09:49 2026

@author: arshdeep
"""
import shutil
import sys, os
sys.path.append(os.getcwd())

import argparse
import yaml
import torch
from collections import OrderedDict
from audioldm_train.modules.diffusionmodules.openaimodel import UNetModel




# ==================================================
# FUNCTION 1: LOAD sd_pruned
# ==================================================
def load_sd_pruned(path):
    """Load pruned state dict from file"""
    print("="*80)
    print("STEP 1: LOADING sd_pruned")
    print("="*80)
    
    sd_pruned = torch.load(path, map_location="cpu")
    print(f"\n✓ Loaded sd_pruned from: {path}")
    print(f"  Total keys: {len(sd_pruned.keys())}")
    
    # Show first few keys
    print("\nFirst 5 keys from sd_pruned:")
    for i, key in enumerate(list(sd_pruned.keys())[:5], 1):
        print(f"  {i}. {key}")
    
    return sd_pruned


# ==================================================
# FUNCTION 2: RENAME KEYS IN sd_pruned
# ==================================================
def rename_sd_pruned_keys(sd_pruned, prefix="model.diffusion_model."):
    """Rename all keys in sd_pruned by adding prefix"""
    print("\n" + "="*80)
    print("STEP 2: RENAMING sd_pruned KEYS")
    print("="*80)
    
    sd_pruned_renamed = OrderedDict()
    for key, value in sd_pruned.items():
        new_key = f"{prefix}{key}"
        sd_pruned_renamed[new_key] = value
    
    print(f"\n✓ Renamed {len(sd_pruned_renamed)} keys in sd_pruned")
    print(f"  Prefix used: '{prefix}'")
    
    # Show first few renamed keys
    print("\nFirst 5 renamed keys:")
    for i, key in enumerate(list(sd_pruned_renamed.keys())[:5], 1):
        print(f"  {i}. {key}")
    
    return sd_pruned_renamed


# ==================================================
# FUNCTION 3: LOAD CHECKPOINT
# ==================================================
def load_checkpoint(path):
    """Load full checkpoint and extract state dict"""
    print("\n" + "="*80)
    print("STEP 3: LOADING CHECKPOINT")
    print("="*80)
    
    cpt = torch.load(path, map_location="cpu")
    
    # Handle different checkpoint formats
    if isinstance(cpt, dict):
        if 'state_dict' in cpt:
            sd_all = cpt['state_dict']
            print(f"✓ Extracted 'state_dict' key from checkpoint")
        else:
            sd_all = cpt
            print(f"✓ Using checkpoint dict directly as state_dict")
    else:
        sd_all = cpt.state_dict()
        print(f"✓ Extracted state_dict from model object")
    
    print(f"\n✓ Loaded checkpoint from: {path}")
    print(f"  Type: {type(cpt)}")
    print(f"  Total keys in sd_all: {len(sd_all.keys())}")
    
    # Show first few keys
    print("\nFirst 5 keys from sd_all:")
    for i, key in enumerate(list(sd_all.keys())[:5], 1):
        print(f"  {i}. {key}")
    
    return cpt, sd_all


# ==================================================
# FUNCTION 4: REPLACE VALUES IN sd_all
# ==================================================
def replace_sd_all_with_pruned(sd_all, sd_pruned_renamed):
    """Replace values in sd_all with values from sd_pruned_renamed"""
    print("\n" + "="*80)
    print("STEP 4: REPLACING VALUES IN sd_all")
    print("="*80)
    
    replaced_count = 0
    missing_count = 0
    
    for key, value in sd_pruned_renamed.items():
        if key in sd_all:
            sd_all[key] = value
            replaced_count += 1
        else:
            missing_count += 1
    
    print(f"\n✓ Replaced {replaced_count} keys in sd_all")
    print(f"✗ {missing_count} keys from sd_pruned_renamed were not found in sd_all")
    
    if missing_count > 0:
        print("\nFirst 5 missing keys:")
        missing = []
        for key in sd_pruned_renamed.keys():
            if key not in sd_all:
                missing.append(key)
        for key in missing[:5]:
            print(f"  - {key}")
    
    return sd_all


# ==================================================
# FUNCTION 5: SAVE UPDATED sd_all
# ==================================================
def save_updated_sd_all(sd_all, output_path):
    """Save the updated state dict to file"""
    print("\n" + "="*80)
    print("STEP 5: SAVING UPDATED sd_all")
    print("="*80)
    
    torch.save(sd_all, output_path)
    print(f"\n✓ Saved updated sd_all to: {output_path}")
    print(f"  Total keys in updated sd_all: {len(sd_all)}")
    
    return output_path


# ==================================================
# FUNCTION 5B: SAVE sd_all AS CHECKPOINT
# ==================================================
def save_sd_all_as_checkpoint(sd_all, cpt_original, output_ckpt_path="checkpoints/sd_all_updated.ckpt"):
    """Save sd_all as a checkpoint, preserving the original checkpoint structure"""
    print("\n" + "="*80)
    print("STEP 5B: SAVING sd_all AS CHECKPOINT")
    print("="*80)
    
    # Create checkpoint with the same structure as the original
    if isinstance(cpt_original, dict):
        if 'state_dict' in cpt_original:
            ckpt_updated = cpt_original.copy()
            ckpt_updated['state_dict'] = sd_all
            print("✓ Updated 'state_dict' key in checkpoint structure")
        else:
            ckpt_updated = sd_all
            print("✓ Saving sd_all directly as checkpoint dict")
    else:
        ckpt_updated = sd_all
        print("✓ Saving sd_all as checkpoint")
    
    # Save the checkpoint
    torch.save(ckpt_updated, output_ckpt_path)
    print(f"\n✓ Saved checkpoint to: {output_ckpt_path}")
    print(f"  Total keys in checkpoint: {len(ckpt_updated) if isinstance(ckpt_updated, dict) else 'N/A'}")
    
    return output_ckpt_path



# ==================================================
# MAIN EXECUTION
# ==================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Replace AudioLDM checkpoint weights with pruned UNet weights"
    )

    parser.add_argument(
        "--pruned-ckpt",
        required=True,
        help="Path to pruned UNet state_dict (.pt)"
    )

    parser.add_argument(
        "--full-ckpt",
        required=True,
        help="Path to original AudioLDM checkpoint (.ckpt)"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output checkpoint path"
    )

    parser.add_argument(
        "--prefix",
        default="model.diffusion_model.",
        help="Prefix to prepend to pruned keys"
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    print("\n" + "=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(f"Pruned U-Net checkpoint : {args.pruned_ckpt}")
    print(f"Full basline (AudioLDM) checkpoint   : {args.full_ckpt}")
    print(f"Pruned AudioLDM checkpoint : {args.output}")
    print(f"Prefix            : {args.prefix}")
    print("=" * 80)

    # Create output directory if necessary
    output_dir = os.path.dirname(args.output)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------
    # Step 1: Load sd_pruned
    # --------------------------------------------------
    sd_pruned = load_sd_pruned(
        args.pruned_ckpt
    )

    # --------------------------------------------------
    # Step 2: Rename keys
    # --------------------------------------------------
    sd_pruned_renamed = rename_sd_pruned_keys(
        sd_pruned,
        prefix=args.prefix
    )

    # --------------------------------------------------
    # Step 3: Load checkpoint
    # --------------------------------------------------
    cpt, sd_all = load_checkpoint(
        args.full_ckpt
    )

    # --------------------------------------------------
    # Step 4: Replace values
    # --------------------------------------------------
    sd_all = replace_sd_all_with_pruned(
        sd_all,
        sd_pruned_renamed
    )

    # --------------------------------------------------
    # Step 5: Save checkpoint
    # --------------------------------------------------
    output_ckpt_path = save_sd_all_as_checkpoint(
        sd_all,
        cpt,
        output_ckpt_path=args.output
    )



