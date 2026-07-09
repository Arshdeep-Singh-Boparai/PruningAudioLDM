#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 13:53:03 2026

@author: arshdeep
"""
#!/usr/bin/env python3

"""
Generate sorted filter importance indexes using L1-norm scores.

Example:
python layerwise_sorted_index_generation.py \
    --ckpt checkpoints/Unet_model-m.ckpt \
    --output pruned_indexes/B3_B4/sorted_indexes_dict.pkl
"""

import argparse
import os
import pickle

import numpy as np
import torch


def l1_imp_index(weights):
    """
    Compute filter-wise L1 importance scores.

    Args:
        weights: numpy array of shape
                 (out_channels, in_channels, H, W)

    Returns:
        normalized scores
    """
    scores = []

    for i in range(weights.shape[0]):
        scores.append(np.sum(np.abs(weights[i])))

    scores = np.array(scores)

    if np.max(scores) > 0:
        scores = scores / np.max(scores)

    return scores


def load_checkpoint(ckpt_path):
    """
    Load checkpoint and return state_dict.
    """

    print(f"Loading checkpoint: {ckpt_path}")

    ckpt = torch.load(
        ckpt_path,
        map_location="cpu"
    )

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        return ckpt

    return ckpt.state_dict()


def get_default_layers():
    """
    Edit this list as required.
    """

    return [
        # --- 3x3 layers ---
        "input_blocks.7.0.in_layers.2.weight",
        "input_blocks.7.0.out_layers.3.weight",

        "input_blocks.8.0.in_layers.2.weight",
        "input_blocks.8.0.out_layers.3.weight",

        "input_blocks.9.0.op.weight",




        "input_blocks.10.0.in_layers.2.weight",
        "input_blocks.10.0.out_layers.3.weight",

        "input_blocks.11.0.in_layers.2.weight",
        "input_blocks.11.0.out_layers.3.weight",

        "middle_block.0.in_layers.2.weight",
        "middle_block.0.out_layers.3.weight",

        "middle_block.2.in_layers.2.weight",
        "middle_block.2.out_layers.3.weight",

        "output_blocks.0.0.in_layers.2.weight",
        "output_blocks.0.0.out_layers.3.weight",

        "output_blocks.1.0.in_layers.2.weight",
        "output_blocks.1.0.out_layers.3.weight",

        "output_blocks.2.0.in_layers.2.weight",
        "output_blocks.2.0.out_layers.3.weight",
        "output_blocks.2.2.conv.weight",

        "output_blocks.3.0.in_layers.2.weight",
        "output_blocks.3.0.out_layers.3.weight",

        "output_blocks.4.0.in_layers.2.weight",
        "output_blocks.4.0.out_layers.3.weight",


        "output_blocks.5.0.in_layers.2.weight",
        "output_blocks.5.0.out_layers.3.weight",

        "output_blocks.5.2.conv.weight",

        "output_blocks.6.0.in_layers.2.weight",

    ]


def main():

    parser = argparse.ArgumentParser(
        description="Generate sorted filter importance indexes."
    )

    parser.add_argument(
        "--ckpt",
        required=True,
        help="Path to checkpoint file"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output pickle file"
    )

    args = parser.parse_args()

    state_dict = load_checkpoint(args.ckpt)

    conv_layers = get_default_layers()

    sorted_indexes_dict = {}

    print("\nComputing importance scores...\n")

    for layer_name in conv_layers:

        if layer_name not in state_dict:
            print(f"[WARNING] Layer not found: {layer_name}")
            continue

        weights = state_dict[layer_name].cpu().numpy()

        scores = l1_imp_index(weights)

        sorted_idx = np.argsort(scores)

        sorted_indexes_dict[layer_name] = sorted_idx.tolist()
        #{
      #      "scores": scores,
         #   "sorted_indexes": sorted_idx
        #}

        print(
            f"{layer_name} "
            f"({weights.shape[0]} filters)"
        )

    output_dir = os.path.dirname(args.output)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "wb") as f:
        pickle.dump(sorted_indexes_dict, f)

    print("\nSaved results to:")
    print(args.output)

    print(
        f"\nProcessed {len(sorted_indexes_dict)} layers."
    )


if __name__ == "__main__":
    main()