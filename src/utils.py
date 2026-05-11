# MIT License
# Copyright (c) 2026 Gong.io
# See LICENSE file for full license text

import json
import logging
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import wandb

logger = logging.getLogger("ExperimentLogger")


def convert_keys_to_int(label_map: Dict[Any, str]) -> Dict[Union[int, Any], str]:
    """
    Convert dictionary keys to integers if all keys are numeric strings.

    Args:
        label_map: Dictionary with keys to potentially convert

    Returns:
        Dictionary with numeric keys converted to integers if applicable
    """
    if not label_map:
        return {}
    if all(str(key).isdigit() for key in label_map):
        return {int(key): value for key, value in label_map.items()}
    else:
        return label_map


def process_label_map(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process the label map in the configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Updated configuration dictionary
    """
    if isinstance(config.get("label_map"), str):
        config["label_map"] = json.loads(config["label_map"])
    config["label_map"] = convert_keys_to_int(config["label_map"])
    return config


def format_examples(
    few_shot_examples_df: Optional[pd.DataFrame], mix_examples: bool = False
) -> Optional[Dict[str, str]]:
    """
    Format examples for use in prompts.

    Args:
        few_shot_examples_df: DataFrame containing examples
        mix_examples: Whether to mix examples or group them by label

    Returns:
        Dictionary of formatted examples or None if input is None
    """
    if few_shot_examples_df is None:
        return None

    if mix_examples:
        few_shot_examples = {}
        few_shot_examples["mixed"] = "\n".join(
            [
                f"Example {idx + 1}: {row['text']} | Label: {row['label'].title()}"
                for idx, row in few_shot_examples_df.iterrows()
            ]
        )
    else:
        labels = few_shot_examples_df["label"].unique()
        few_shot_examples = {}
        for label in labels:
            relevant_examples = few_shot_examples_df[
                few_shot_examples_df["label"] == label
            ]
            few_shot_examples[label.title()] = "\n".join(
                [
                    f"Example {idx + 1}: {text}"
                    for idx, text in enumerate(relevant_examples["text"])
                ]
            )
    return few_shot_examples


def maybe_load_few_shot_examples(
    config: Dict[str, Any],
    method: str,
    num_few_shot_examples: int,
    iter_num: int,
    mix_examples: bool = False,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, str]]]:
    """
    Load few-shot examples from disk if they already exist.

    Args:
        config: Configuration dictionary
        method: Method name
        num_few_shot_examples: Number of few-shot examples
        iter_num: Iteration number
        mix_examples: Whether to mix the examples or group them by label

    Returns:
        Tuple containing loaded few-shot examples DataFrame and formatted examples dict (or None if not found)
    """
    few_shot_prefix = f"{method}_{num_few_shot_examples}_iter_{iter_num}"
    few_shot_examples_file_path = os.path.join(
        config["output_dir"], f"{few_shot_prefix}.csv"
    )
    if os.path.exists(few_shot_examples_file_path):
        logger.info("Loading few-shot examples from %s", few_shot_examples_file_path)
        few_shot_examples_df = pd.read_csv(few_shot_examples_file_path)
        few_shot_examples = format_examples(few_shot_examples_df, mix_examples)
        return few_shot_examples_df, few_shot_examples
    else:
        if num_few_shot_examples > 0:
            logger.info(
                "Few-shot examples file not found: %s. Generating new examples.",
                few_shot_examples_file_path,
            )
        return None, None


def maybe_load_artifact(
    config: Dict[str, Any],
    method: str,
    num_few_shot_examples: int,
    iter_num: int,
    artifact_type: str,
) -> Optional[Any]:
    """
    Load artifact (criteria or description) from disk if it already exists.

    Args:
        config: Configuration dictionary
        method: Method name
        num_few_shot_examples: Number of few-shot examples
        iter_num: Iteration number
        artifact_type: Type of artifact ('criteria' or 'description')

    Returns:
        Loaded artifact or None if not found
    """
    prefix = f"{method}_{num_few_shot_examples}_iter_{iter_num}"
    file_path = os.path.join(config["model_output_dir"], f"{prefix}.json")
    if os.path.exists(file_path):
        logger.info("Loading %s from %s", artifact_type, file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            artifact = json.load(f)
        return artifact
    else:
        logger.info("%s file not found: %s", artifact_type.capitalize(), file_path)
        return None


def maybe_load_criteria(
    config: Dict[str, Any], method: str, num_few_shot_examples: int, iter_num: int
) -> Optional[Dict[str, str]]:
    """
    Load criteria from disk if they already exist.

    Args:
        config: Configuration dictionary
        method: Method name
        num_few_shot_examples: Number of few-shot examples
        iter_num: Iteration number

    Returns:
        Loaded criteria dictionary or None if not found
    """
    return maybe_load_artifact(
        config, method, num_few_shot_examples, iter_num, artifact_type="criteria"
    )


def maybe_load_description(
    config: Dict[str, Any], method: str, num_few_shot_examples: int, iter_num: int
) -> Optional[str]:
    """
    Load description from disk if it already exists.

    Args:
        config: Configuration dictionary
        method: Method name
        num_few_shot_examples: Number of few-shot examples
        iter_num: Iteration number

    Returns:
        Loaded description or None if not found
    """
    return maybe_load_artifact(
        config, method, num_few_shot_examples, iter_num, artifact_type="description"
    )


def sample_labels_by_distribution(
    dataset_df: pd.DataFrame, num_few_shot_examples: int, label_column: str
) -> pd.DataFrame:
    """
    Ensures sampling from all labels while maintaining the total number of few-shot examples.

    Args:
        dataset_df: The dataset containing examples
        num_few_shot_examples: The number of examples to sample
        label_column: The column name representing the label

    Returns:
        The sampled few-shot examples DataFrame
    """
    dataset_df_copy = dataset_df.sample(frac=1).copy()
    label_counts = dataset_df_copy[label_column].value_counts()
    non_zero_labels = label_counts[label_counts > 0]
    label_distribution = (
        non_zero_labels / non_zero_labels.sum()
        if non_zero_labels.sum() > 0
        else non_zero_labels
    )
    min_examples_per_label = 1
    # Reserve one slot per label, then distribute the remainder proportionally.
    remaining_examples = num_few_shot_examples - len(non_zero_labels)
    if remaining_examples < 0:
        # Fewer requested than number of labels — just take one per label up to the limit.
        sample_sizes = pd.Series(
            {label: (1) for label in non_zero_labels.index[:num_few_shot_examples]}
        )
    else:
        additional_samples = (
            (label_distribution * remaining_examples).round().astype(int)
        )
        sample_sizes = pd.Series(
            {
                label: (min_examples_per_label + additional)
                for label, additional in additional_samples.items()
            }
        )
        # Rounding can make the total differ by 1 — nudge the largest bucket to fix it.
        while sample_sizes.sum() != num_few_shot_examples:
            if sample_sizes.sum() < num_few_shot_examples:
                label_to_adjust = label_distribution.idxmax()
                sample_sizes[label_to_adjust] += 1
            else:
                label_to_adjust = sample_sizes.idxmax()
                sample_sizes[label_to_adjust] -= 1

    sampled_dfs = []
    for label, size in sample_sizes.items():
        group = dataset_df_copy[dataset_df_copy[label_column] == label]
        actual_size = min(len(group), size)
        sampled_dfs.append(group.sample(actual_size, replace=False))
    few_shot_example_df = pd.concat(sampled_dfs)
    return few_shot_example_df


def random_sampling(
    dataset_df: pd.DataFrame, num_few_shot_examples: int
) -> pd.DataFrame:
    """
    Randomly samples few-shot examples from the dataset.

    Args:
        dataset_df: The dataset containing examples
        num_few_shot_examples: The number of examples to sample

    Returns:
        The sampled few-shot examples DataFrame
    """
    if dataset_df.empty:
        return pd.DataFrame(columns=dataset_df.columns)
    if num_few_shot_examples <= 0:
        return pd.DataFrame(columns=dataset_df.columns)
    dataset_df_copy = dataset_df.sample(frac=1).copy()
    num_few_shot_examples = min(num_few_shot_examples, len(dataset_df_copy))
    sample_indices = random.sample(range(len(dataset_df_copy)), num_few_shot_examples)
    few_shot_example_df = dataset_df_copy.iloc[sample_indices]
    return few_shot_example_df


def create_few_shot_examples(
    dataset_df: pd.DataFrame,
    num_few_shot_examples: int = 5,
    mix_examples: bool = False,
    example_column: str = "text",
    label_column: str = "label",
    sampling_method: str = "random",
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, str]]]:
    """
    Create few-shot examples from the dataset.

    Args:
        dataset_df: Dataset DataFrame
        num_few_shot_examples: Number of few-shot examples to sample
        mix_examples: Whether to mix the examples or group them by label
        example_column: Name of the column containing the examples
        label_column: Name of the column containing the labels
        sampling_method: Sampling method to use (random or distribution)

    Returns:
        Tuple containing the few-shot examples DataFrame and a dictionary of formatted examples,
        or (None, None) if num_few_shot_examples is 0
    """
    if num_few_shot_examples == 0:
        return None, None
    if sampling_method == "label_distribution":
        few_shot_example_df = sample_labels_by_distribution(
            dataset_df, num_few_shot_examples, label_column
        )
    else:
        few_shot_example_df = random_sampling(dataset_df, num_few_shot_examples)
    # Normalize to canonical column names ("text", "label") expected by format_examples.
    few_shot_example_df = few_shot_example_df[[example_column, "textual_label"]].rename(
        columns={"textual_label": label_column}
    )
    few_shot_example_df = few_shot_example_df.reset_index().rename(
        columns={example_column: "text", label_column: "label"}
    )
    few_shot_examples = format_examples(few_shot_example_df, mix_examples)
    return few_shot_example_df, few_shot_examples


def save_file_to_wandb(
    wandb_run: Any,
    file_path: Union[str, Path],
    artifact_name: str,
    artifact_type: str = "dataset",
) -> None:
    """
    Save a file as a WandB artifact.

    Args:
        wandb_run: WandB run object
        file_path: Path to the file to save
        artifact_name: Name of the artifact
        artifact_type: Type of the artifact
    """
    if not os.path.exists(file_path):
        logger.warning("File not found for W&B logging: %s", file_path)
        return
    artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
    artifact.add_file(str(file_path))
    wandb_run.log_artifact(artifact)


def save_few_shot_examples(
    few_shot_examples_df: Optional[pd.DataFrame],
    config: Dict[str, Any],
    method: str,
    num_few_shot_examples: int,
    iter_num: int,
    wandb_run: Optional[Any] = None,
) -> None:
    """
    Save few-shot examples to disk and optionally to WandB.

    Args:
        few_shot_examples_df: DataFrame containing few-shot examples
        config: Configuration dictionary
        method: Method name
        num_few_shot_examples: Number of few-shot examples
        iter_num: Iteration number
        wandb_run: WandB run object or None
    """
    if few_shot_examples_df is None or len(few_shot_examples_df) <= 0:
        return
    few_shot_prefix = f"{method}_{num_few_shot_examples}_iter_{iter_num}"
    few_shot_examples_file_path = os.path.join(
        config["output_dir"], f"{few_shot_prefix}.csv"
    )
    few_shot_examples_df.to_csv(few_shot_examples_file_path, index=False)
    logger.info("Few-shot examples saved to %s", few_shot_examples_file_path)
    few_shot_examples_model_file_path = os.path.join(
        config["model_output_dir"], f"{few_shot_prefix}.csv"
    )
    few_shot_examples_df.to_csv(few_shot_examples_model_file_path, index=False)
    logger.info("Few-shot examples saved to %s", few_shot_examples_model_file_path)
    if wandb_run is not None:
        few_shot_examples_table = wandb.Table(dataframe=few_shot_examples_df)
        few_shot_examples_table_name = f"{few_shot_prefix}_table"
        wandb_run.log({few_shot_examples_table_name: few_shot_examples_table})
        save_file_to_wandb(wandb_run, few_shot_examples_file_path, few_shot_prefix)


def fillna_to_all_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaN values in a DataFrame based on column data types.

    Args:
        df: DataFrame to process

    Returns:
        DataFrame with NaN values filled
    """
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=["float", "int"]).columns
    if not numeric_cols.empty:
        df_copy.loc[:, numeric_cols] = df_copy.loc[:, numeric_cols].fillna(0)
    string_cols = df_copy.select_dtypes(include=["object", "string"]).columns
    if not string_cols.empty:
        df_copy.loc[:, string_cols] = df_copy.loc[:, string_cols].fillna("")
    bool_cols = df_copy.select_dtypes(include=["bool"]).columns
    if not bool_cols.empty:
        df_copy.loc[:, bool_cols] = df_copy.loc[:, bool_cols].fillna(False)
    return df_copy


def save_artifact(
    artifact: Any,
    config: Dict[str, Any],
    method: str,
    num_few_shot_examples: int,
    iter_num: int,
    artifact_type: str,
    wandb_run: Optional[Any] = None,
) -> None:
    """
    Save an artifact (criteria or description) to disk and optionally to WandB.

    Args:
        artifact: The artifact to save
        config: Configuration dictionary
        method: Method name
        num_few_shot_examples: Number of few-shot examples
        iter_num: Iteration number
        artifact_type: Type of artifact ('criteria' or 'description')
        wandb_run: WandB run object or None
    """
    if artifact is None:
        return
    os.makedirs(config["model_output_dir"], exist_ok=True)
    artifact_prefix = f"{method}_{num_few_shot_examples}_iter_{iter_num}"
    artifact_file_path = os.path.join(
        config["model_output_dir"], f"{artifact_prefix}.json"
    )
    with open(artifact_file_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)
    logger.info("%s saved to %s", artifact_type.capitalize(), artifact_file_path)
    if wandb_run is not None:
        artifact_name = f"{artifact_prefix}_json"
        save_file_to_wandb(wandb_run, artifact_file_path, artifact_name)


def save_criteria(
    criteria: Optional[Dict[str, str]],
    config: Dict[str, Any],
    method: str,
    num_few_shot_examples: int,
    iter_num: int,
    wandb_run: Optional[Any] = None,
) -> None:
    """
    Save criteria to disk and optionally to WandB.

    Args:
        criteria: Dictionary containing criteria
        config: Configuration dictionary
        method: Method name
        num_few_shot_examples: Number of few-shot examples
        iter_num: Iteration number
        wandb_run: WandB run object or None
    """
    save_artifact(
        criteria,
        config,
        method,
        num_few_shot_examples,
        iter_num,
        artifact_type="criteria",
        wandb_run=wandb_run,
    )


def save_description(
    description: Optional[str],
    config: Dict[str, Any],
    method: str,
    num_few_shot_examples: int,
    iter_num: int,
    wandb_run: Optional[Any] = None,
) -> None:
    """
    Save description to disk and optionally to WandB.

    Args:
        description: Description string
        config: Configuration dictionary
        method: Method name
        num_few_shot_examples: Number of few-shot examples
        iter_num: Iteration number
        wandb_run: WandB run object or None
    """
    save_artifact(
        description,
        config,
        method,
        num_few_shot_examples,
        iter_num,
        artifact_type="description",
        wandb_run=wandb_run,
    )


def maybe_load_classification_report(
    config: Dict[str, Any], method: str
) -> Optional[Dict[str, Any]]:
    """
    Load a classification report if it exists and override is not specified.

    Args:
        config: Configuration dictionary
        method: Method name

    Returns:
        Classification report dictionary or None if it doesn't exist or override is True
    """
    if not config.get("override", False):
        classification_report_path = os.path.join(
            config["model_output_dir"], f"{method}_classification_report.json"
        )
        if os.path.exists(classification_report_path):
            logger.info(
                "Classification report already exists for %s. Skipping...", method
            )
            with open(classification_report_path, "r", encoding="utf-8") as f:
                classification_report_res = json.load(f)
            return classification_report_res
    return None


def flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    """
    Flatten a nested dictionary.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key for nested items
        sep: Separator for joining keys

    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
