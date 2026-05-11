# MIT License
# Copyright (c) 2026 Gong.io
# See LICENSE file for full license text

import json
import logging
import os
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import wandb

from .metrics import get_classification_report
from .utils import fillna_to_all_types, save_file_to_wandb

logger = logging.getLogger("ExperimentLogger")


def setup_logger(
    log_dir: Union[str, Path], log_filename: str = "experiment.log"
) -> None:
    """
    Set up a custom logger for the experiment.

    Args:
        log_dir: Directory to save log files
        log_filename: Name of the log file
    """
    _logger = logging.getLogger("ExperimentLogger")
    _logger.setLevel(logging.INFO)
    # Clear existing handlers to prevent duplicate log lines if setup_logger is called again.
    if _logger.handlers:
        _logger.handlers.clear()
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / log_filename)
    console_handler = logging.StreamHandler()
    file_handler.setLevel(logging.INFO)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    _logger.addHandler(file_handler)
    _logger.addHandler(console_handler)


def init_reports_dict(
    methods: List[str], few_shot_examples: List[int]
) -> Dict[str, Dict[int, List]]:
    """
    Initialize a dictionary to store reports for each method and few-shot configuration.

    Args:
        methods: List of methods to include in the reports dictionary
        few_shot_examples: List of few-shot example counts

    Returns:
        Nested dictionary for storing experiment reports
    """
    reports = {}
    for method in methods:
        reports[method] = {}
        for num_few_shot_examples in few_shot_examples:
            reports[method][num_few_shot_examples] = []
    return reports


async def run_experiment(
    config: Dict[str, Any],
    wandb_run: Optional[Any],
    classifier: Any,
    samples_df: pd.DataFrame,
    method: str,
    few_shot_examples: Optional[Dict[str, str]] = None,
    criteria: Optional[Dict[str, str]] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a classification experiment and collect results.

    Args:
        config: Configuration dictionary
        wandb_run: WandB run object or None
        classifier: Classifier instance
        samples_df: DataFrame containing test samples
        method: Method name
        few_shot_examples: Dictionary of few-shot examples or None
        criteria: Dictionary of criteria or None
        description: Description string or None

    Returns:
        Classification report dictionary
    """
    # Classify the entire test set using whichever knowledge artifact was produced
    # for this method (raw examples, criteria, or description). The classifier sends
    # prompts in async batches and returns one prediction row per test sample.
    tic = time()
    responses = await classifier.get_llm_responses(
        samples_df,
        few_shot_examples=few_shot_examples,
        criteria=criteria,
        description=description,
    )
    # responses_df is indexed by the original dataset row index so that the merge
    # correctly realigns predictions with their source rows even after batching.
    responses_df = pd.DataFrame(responses).set_index("index")
    toc = time()
    logger.info("Time for method %s: %s seconds.", method, toc - tic)

    # Attach predictions to the original test rows and persist the full output —
    # this CSV contains ground-truth labels, LLM rationales, and predicted labels,
    # making it the primary artifact for error analysis.
    samples_w_responses_df = pd.merge(
        samples_df, responses_df, left_index=True, right_index=True
    )
    samples_w_responses_df = fillna_to_all_types(samples_w_responses_df)
    experiment_file_name = f"{method}_samples_w_responses_df.csv"
    samples_w_responses_path = os.path.join(
        config["model_output_dir"], experiment_file_name
    )
    os.makedirs(config["model_output_dir"], exist_ok=True)
    samples_w_responses_df.to_csv(samples_w_responses_path, index=False)
    logger.info(
        "Samples with responses dataframe saved to %s", samples_w_responses_path
    )

    # Compute the classification report for this single iteration; the caller collects
    # these across iterations to compute mean/std in compute_average_metrics_dict.
    classification_report_res = get_classification_report(
        samples_w_responses_df["label"],
        samples_w_responses_df["predicted_label"],
        labels=classifier.labels,
        target_names=classifier.target_names,
    )
    classification_report_path = os.path.join(
        config["model_output_dir"], f"{method}_classification_report.json"
    )
    with open(classification_report_path, "w") as f:
        json.dump(classification_report_res, f, indent=2)
        logger.info("Classification report saved to %s", classification_report_path)

    if wandb_run is not None:
        experiment_classification_report_name = f"{method}_classification_report"
        wandb_run.log(
            {
                experiment_classification_report_name: classification_report_res,
                "time": toc - tic,
            }
        )
        dataset_with_responses_table = wandb.Table(dataframe=samples_w_responses_df)
        experiment_table_name = f"{method}_samples_w_responses_table"
        wandb_run.log({experiment_table_name: dataset_with_responses_table})
        save_file_to_wandb(wandb_run, samples_w_responses_path, experiment_table_name)
    return classification_report_res
