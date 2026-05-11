# MIT License
# Copyright (c) 2026 Gong.io
# See LICENSE file for full license text

import logging
import os
from typing import Any, Dict, List, Optional

from matplotlib import pyplot as plt

from .utils import save_file_to_wandb

logger = logging.getLogger("ExperimentLogger")


def plot_performance(
    config: Dict[str, Any],
    wandb_run: Optional[Any],
    methods: List[str],
    average_metrics_dict: Dict[str, Dict[int, Dict[str, Dict[str, Any]]]],
    target_names: List[str] = ["Negative", "Positive"],
) -> None:
    """
    Plot performance metrics across different methods and few-shot configurations.

    Args:
        config: Configuration dictionary
        wandb_run: WandB run object or None
        methods: List of method names
        average_metrics_dict: Nested dictionary containing average metrics
        target_names: List of label names
    """
    categories = target_names + ["macro avg", "micro avg", "weighted avg"]
    metrics = ["precision", "recall", "f1-score"]
    for category in categories:
        for metric in metrics:
            plt.figure(figsize=(8, 6))
            markers = ["^", "*", "o", "s", "D", "v", "p"]
            for i, method in enumerate(methods):
                means = []
                stds = []
                for num_few_shot_examples in config["num_few_shot_examples"]:
                    mean_value = average_metrics_dict[method][num_few_shot_examples][
                        "mean"
                    ][category][metric]
                    std_value = average_metrics_dict[method][num_few_shot_examples][
                        "std"
                    ][category][metric]
                    means.append(mean_value)
                    stds.append(std_value)
                plt.errorbar(
                    config["num_few_shot_examples"],
                    means,
                    yerr=stds,
                    label=method,
                    marker=markers[i % len(markers)],
                    capsize=5,
                )
            plt.title(f"{category.title()} - {metric.title()}", fontsize=14)
            plt.xlabel("Number of Few-Shots", fontsize=12)
            plt.ylabel(f"{metric.title()}", fontsize=12)
            plt.legend(loc="best")
            plt.grid(True)
            os.makedirs(config["model_output_dir"], exist_ok=True)
            for ext in ["pdf", "png"]:
                plot_filename = f"{category.replace(' ', '_').lower()}_{metric}.{ext}"
                plot_path = os.path.join(config["model_output_dir"], plot_filename)
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            if wandb_run is not None:
                save_file_to_wandb(
                    wandb_run,
                    plot_path,
                    f"{category.replace(' ', '_').lower()}_{metric}_plot",
                )
            plt.close()
