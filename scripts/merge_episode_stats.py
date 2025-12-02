"""
Merge pre-computed per-episode normalization statistics into a global one.

This script is an alternative to `compute_norm_stats.py`. It's useful when you
have already computed the statistics for each episode individually and want to
combine them without re-iterating through the entire dataset.

It reads a JSON Lines (.jsonl) file, where each line contains the mean, std, and
count for a single episode, and merges them into a final `norm_stats.json`.

Usage:
    python scripts/merge_episode_stats.py <config_name> --stats_file /path/to/your/stats.jsonl
"""
import json
import pathlib

import numpy as np
import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as _config


class StatsMerger:
    """A class to merge running statistics from multiple sources."""

    def __init__(self):
        self.total_count = 0
        self.mean = None
        self.mean_of_squares = None  # E[X^2], crucial for merging variances

    def add(self, mean: np.ndarray, std: np.ndarray, count: int):
        """
        Add a new set of statistics (mean, std, count) to the current aggregate.
        """
        if count == 0:
            return

        # Convert to numpy arrays for vectorized operations
        mean = np.asarray(mean, dtype=np.float64)
        std = np.asarray(std, dtype=np.float64)

        # The formula for variance is Var(X) = E[X^2] - (E[X])^2
        # So, the mean of squares is E[X^2] = Var(X) + (E[X])^2 = std^2 + mean^2
        variance = std**2
        mean_of_squares = variance + mean**2

        if self.total_count == 0:
            self.mean = mean
            self.mean_of_squares = mean_of_squares
            self.total_count = count
        else:
            new_total_count = self.total_count + count

            # Calculate the new combined mean as a weighted average
            new_mean = (self.mean * self.total_count + mean * count) / new_total_count

            # Calculate the new combined mean of squares as a weighted average
            new_mean_of_squares = (
                self.mean_of_squares * self.total_count + mean_of_squares * count
            ) / new_total_count

            # Update the stored values
            self.mean = new_mean
            self.mean_of_squares = new_mean_of_squares
            self.total_count = new_total_count

    def get_statistics(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the final global mean and standard deviation.

        Returns:
            A tuple of (global_mean, global_std).
        """
        if self.total_count < 2:
            raise ValueError("Cannot compute statistics for less than 2 total samples.")

        # Calculate final variance from the combined mean and mean of squares
        # Var(X) = E[X^2] - (E[X])^2
        variance = self.mean_of_squares - self.mean**2
        # Clamp variance at 0 to avoid numerical issues (sqrt of negative number)
        std = np.sqrt(np.maximum(0, variance))

        return self.mean, std


def main(config_name: str, stats_file: pathlib.Path):
    """
    Main function to run the statistics merging process.

    Args:
        config_name: The name of the training config (e.g., 'pi0_franka_text_custom').
                     This is used to determine the output path for norm_stats.json.
        stats_file: The path to the .jsonl file containing per-episode statistics.
    """
    if not stats_file.is_file():
        raise FileNotFoundError(f"Statistics file not found: {stats_file}")

    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    # Keys to process from the JSON files and their final names in norm_stats
    # In your JSON it's "observation.state", but the final file needs "state".
    key_mapping = {"observation.state": "state", "action": "actions"}
    mergers = {key: StatsMerger() for key in key_mapping.values()}

    print(f"Reading episodes from {stats_file}. Merging...")

    with open(stats_file) as f:
        # Read all lines to get a total for the progress bar
        lines = f.readlines()
        for line in tqdm.tqdm(lines, desc="Merging episode stats"):
            # Skip empty lines
            if not line.strip():
                continue

            data = json.loads(line)
            episode_stats = data.get("stats", {})
            for source_key, target_key in key_mapping.items():
                if source_key in episode_stats:
                    stats = episode_stats[source_key]
                    # The 'count' in your JSON is a list with one element, e.g., [121]
                    count = stats.get("count")[0] if isinstance(stats.get("count"), list) else stats.get("count")
                    if count and count > 0:
                        mergers[target_key].add(stats["mean"], stats["std"], count)

    # Finalize statistics
    # Note: This method does not compute quantiles (q01, q99) as it requires
    # the full data distribution, which is not available from pre-computed stats.
    # We will save them as None, which is compatible with the NormStats class.
    final_norm_stats = {}
    for key, merger in mergers.items():
        mean, std = merger.get_statistics()
        final_norm_stats[key] = normalize.NormStats(mean=mean, std=std, q01=None, q99=None)

    # Determine output path, same as in compute_norm_stats.py
    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing final merged stats to: {output_path}")
    normalize.save(output_path, final_norm_stats)

    print("Done.")


if __name__ == "__main__":
    tyro.cli(main)