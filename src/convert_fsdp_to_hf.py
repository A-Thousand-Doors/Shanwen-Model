from collections import defaultdict
from pathlib import Path
import re

import torch


def load_sharded_model(fsdp_checkpoint_path):
    state_dict = defaultdict(list)
    checkpoint_dir = Path(fsdp_checkpoint_path)

    shard_files = list(checkpoint_dir.glob("model_world_size_*_rank_*.pt"))
    print("fsdp_checkpoint_path: ", fsdp_checkpoint_path)
    print("shard_files: ", shard_files)
    if not shard_files:
        raise ValueError(f"No checkpoint files found in {fsdp_checkpoint_path}")

    pattern = re.compile(r"model_world_size_(\d+)_rank_(\d+)\.pt")
    world_sizes = set()
    for file in shard_files:
        match = pattern.match(file.name)
        if match:
            world_sizes.add(int(match.group(1)))

    if len(world_sizes) != 1:
        raise ValueError(
            f"Inconsistent world_size found in checkpoint files: {world_sizes}"
        )

    world_size = world_sizes.pop()
    print(f"Found checkpoints with world_size = {world_size}")

    for rank in range(world_size):
        filepath = checkpoint_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
        if not filepath.exists():
            raise ValueError(f"Missing shard file: {filepath}")

        print(f"Loading shard: {filepath}")
        shard_dict = torch.load(filepath, weights_only=False)

        for key, value in shard_dict.items():
            if hasattr(value, "to_local"):
                value = value.to_local()
            state_dict[key].append(value)

    consolidated_state_dict = {}
    for key in state_dict:
        try:
            consolidated_state_dict[key] = torch.cat(state_dict[key], dim=0)
        except (RuntimeError, TypeError):
            consolidated_state_dict[key] = state_dict[key][0]
            print(
                f"Parameter '{key}' does not need concatenation, using first shard value"
            )

    return consolidated_state_dict


def convert_fsdp_to_hf(fsdp_checkpoint_path, hf_model_path, output_path):
    """
    Convert a FSDP checkpoint to a Hugging Face model checkpoint.
    """
    # Load the FSDP checkpoint
    state_dict = load_sharded_model(fsdp_checkpoint_path)

    # Load the Hugging Face model
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(hf_model_path)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    model = AutoModelForCausalLM.from_config(config)

    # Update the model's state_dict with the FSDP checkpoint
    model.load_state_dict(state_dict)

    # Save the model and tokenizer
    model.save_pretrained(output_path, max_shard_size="10GB")
    tokenizer.save_pretrained(output_path)

    print(f"Converted FSDP checkpoint saved to {output_path}")
    print(f"Tokenizer saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert a FSDP checkpoint to a Hugging Face model checkpoint."
    )
    parser.add_argument("--fsdp_checkpoint_path", type=str, help="Path to the FSDP checkpoint.")
    parser.add_argument("--hf_model_path", type=str, help="Path to the Hugging Face model.")
    parser.add_argument("--output_path", type=str, help="Path to save the converted model.")

    args = parser.parse_args()

    convert_fsdp_to_hf(
        fsdp_checkpoint_path=args.fsdp_checkpoint_path,
        hf_model_path=args.hf_model_path,
        output_path=args.output_path,
    )
    print("Conversion completed.")