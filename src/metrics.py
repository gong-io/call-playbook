# MIT License
# Copyright (c) 2026 Gong.io
# See LICENSE file for full license text

import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
)
import wandb

from .utils import flatten_dict, save_file_to_wandb

logger = logging.getLogger("ExperimentLogger")


def get_classification_report(
    true_labels: Iterable[int],
    predicted_labels: Iterable[int],
    labels: List[int] = [0, 1],
    target_names: List[str] = ["Negative", "Positive"],
) -> Dict[str, Any]:
    """
    Generate a classification report with metrics.

    Args:
        true_labels: Ground truth labels
        predicted_labels: Predicted labels
        labels: List of label integers
        target_names: List of label names

    Returns:
        Dictionary containing classification metrics
    """
    classification_report_res = classification_report(
        list(true_labels),
        list(predicted_labels),
        labels=labels,
        target_names=target_names,
        output_dict=True,
    )
    if "accuracy" not in classification_report_res:
        classification_report_res["accuracy"] = accuracy_score(
            true_labels, predicted_labels
        )
    # sklearn omits "micro avg" for binary classification (it equals accuracy),
    # but we need it for consistent aggregation across all averaging strategies.
    if "micro avg" not in classification_report_res:
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, labels=labels, average="micro"
        )
        classification_report_res["micro avg"] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": len(true_labels),
        }
    return classification_report_res


def log_average_metrics(
    config: Dict[str, Any],
    wandb_run: Optional[Any],
    reports: List[Dict[str, Any]],
    method: str,
    target_names: List[str] = ["Negative", "Positive"],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Calculate and log average metrics across multiple reports.

    Args:
        config: Configuration dictionary
        wandb_run: WandB run object or None
        reports: List of classification reports
        method: Method name
        target_names: List of label names

    Returns:
        Tuple containing dictionaries of mean and standard deviation metrics
    """
    # Build a collector dict with one list per metric per label, ready for aggregation.
    # "accuracy" is a scalar so it gets a flat list; all other labels are dicts of lists.
    aggregated_metrics = {
        target_name: {"precision": [], "recall": [], "f1-score": [], "support": []}
        for target_name in target_names
    }
    aggregated_metrics.update(
        {
            "accuracy": [],
            "macro avg": {"precision": [], "recall": [], "f1-score": [], "support": []},
            "micro avg": {"precision": [], "recall": [], "f1-score": [], "support": []},
            "weighted avg": {
                "precision": [],
                "recall": [],
                "f1-score": [],
                "support": [],
            },
        }
    )
    for report in reports:
        aggregated_metrics["accuracy"].append(report["accuracy"])
        all_labels = target_names + ["macro avg", "micro avg", "weighted avg"]
        for label in all_labels:
            aggregated_metrics[label]["precision"].append(report[label]["precision"])
            aggregated_metrics[label]["recall"].append(report[label]["recall"])
            aggregated_metrics[label]["f1-score"].append(report[label]["f1-score"])
            aggregated_metrics[label]["support"].append(report[label]["support"])

    mean_report = {}
    std_report = {}
    for label, metrics in aggregated_metrics.items():
        if label == "accuracy":
            mean_report[label] = sum(metrics) / len(metrics) if len(metrics) > 0 else 0
            std_report[label] = pd.Series(metrics).std(ddof=0)
        else:
            mean_report[label] = {
                "precision": (
                    sum(metrics["precision"]) / len(metrics["precision"])
                    if len(metrics["precision"]) > 0
                    else 0
                ),
                "recall": (
                    sum(metrics["recall"]) / len(metrics["recall"])
                    if len(metrics["recall"]) > 0
                    else 0
                ),
                "f1-score": (
                    sum(metrics["f1-score"]) / len(metrics["f1-score"])
                    if len(metrics["f1-score"]) > 0
                    else 0
                ),
                "support": (
                    sum(metrics["support"]) / len(metrics["support"])
                    if len(metrics["support"]) > 0
                    else 0
                ),
            }
            std_report[label] = {
                "precision": pd.Series(metrics["precision"]).std(ddof=0),
                "recall": pd.Series(metrics["recall"]).std(ddof=0),
                "f1-score": pd.Series(metrics["f1-score"]).std(ddof=0),
                "support": pd.Series(metrics["support"]).std(ddof=0),
            }

    mean_file_path = os.path.join(
        config["model_output_dir"], f"{method}_average_metrics.json"
    )
    os.makedirs(config["model_output_dir"], exist_ok=True)
    with open(mean_file_path, "w") as f:
        json.dump(mean_report, f, indent=2)
    std_file_path = os.path.join(
        config["model_output_dir"], f"{method}_std_metrics.json"
    )
    with open(std_file_path, "w") as f:
        json.dump(std_report, f, indent=2)
    logger.info("Average %s report: %s", method, json.dumps(mean_report, indent=2))

    if wandb_run is not None:
        # Log full dicts as a single step, then also write flattened scalars to the
        # run summary so they appear in W&B's overview table for easy comparison.
        wandb_run.log({f"{method}_mean": mean_report, f"{method}_std": std_report})
        flatten_mean_dict = flatten_dict(mean_report)
        for key, value in flatten_mean_dict.items():
            wandb_run.summary[f"mean_{key}"] = value
        flatten_std_dict = flatten_dict(std_report)
        for key, value in flatten_std_dict.items():
            wandb_run.summary[f"std_{key}"] = value
        save_file_to_wandb(wandb_run, mean_file_path, f"{method}_mean_json")
        save_file_to_wandb(wandb_run, std_file_path, f"{method}_std_json")
    return mean_report, std_report


def compute_average_metrics_dict(
    config: Dict[str, Any],
    wandb_run: Optional[Any],
    reports: Dict[str, Dict[int, List[Dict[str, Any]]]],
    target_names: List[str] = ["Negative", "Positive"],
) -> Dict[str, Dict[int, Dict[str, Dict[str, Any]]]]:
    """
    Compute average metrics for all methods and few-shot configurations.

    Args:
        config: Configuration dictionary
        wandb_run: WandB run object or None
        reports: Nested dictionary containing classification reports
        target_names: List of label names

    Returns:
        Nested dictionary containing average metrics
    """
    average_metrics_dict = {}
    for method, method_reports in reports.items():
        average_metrics_dict[method] = {}
        for num_few_shot_examples, report in method_reports.items():
            mean_report, std_report = log_average_metrics(
                config,
                wandb_run,
                report,
                f"{method}_{num_few_shot_examples}",
                target_names=target_names,
            )
            average_metrics_dict[method][num_few_shot_examples] = {
                "mean": mean_report,
                "std": std_report,
            }
    return average_metrics_dict
