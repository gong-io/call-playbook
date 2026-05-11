# MIT License
# Copyright (c) 2026 Gong.io
# See LICENSE file for full license text

import argparse
import asyncio
from dotenv import load_dotenv
import json
import logging
import os
from typing import Dict

import wandb

from src.classifiers import BinaryClassifier
from src.criteria_creator import BinaryCriteriaCreator
from src.dataset_loader import DatasetLoader
from src.description_creator import BinaryDescriptionCreator
from src.experiment import init_reports_dict, run_experiment, setup_logger
from src.metrics import compute_average_metrics_dict
from src.models import get_llm
from src.utils import (
    convert_keys_to_int,
    create_few_shot_examples,
    maybe_load_classification_report,
    maybe_load_criteria,
    maybe_load_description,
    maybe_load_few_shot_examples,
    process_label_map,
    save_criteria,
    save_description,
    save_few_shot_examples,
)
from src.visualization import plot_performance

logger = logging.getLogger("ExperimentLogger")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the classification experiment.

    Returns:
        Parsed arguments as a Namespace object
    """
    parser = argparse.ArgumentParser(
        description="Classification experiments using LLMs with various prompting strategies."
    )
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model_id", type=str, default="gpt-4o", help="ID of the model to use."
    )
    model_group.add_argument(
        "--model_source",
        type=str,
        default="azure",
        choices=["azure", "bedrock"],
        help="Source of the model (e.g., azure, bedrock).",
    )
    model_group.add_argument(
        "--objective", type=str, help="Objective of the experiment."
    )
    model_group.add_argument(
        "--data_dir", type=str, help="Directory containing the dataset."
    )
    model_group.add_argument(
        "--batch_size", type=int, default=10, help="Batch size for model inference."
    )
    fewshot_group = parser.add_argument_group("Few-shot Configuration")
    fewshot_group.add_argument(
        "--num_few_shot_examples",
        type=int,
        nargs="+",
        default=[0, 10, 25, 50, 75, 100],
        help="List of few-shot example counts to use.",
    )
    fewshot_group.add_argument(
        "--num_experiments",
        type=int,
        default=5,
        help="Number of experiments to run for statistical significance.",
    )
    fewshot_group.add_argument(
        "--mix_examples",
        action="store_true",
        default=False,
        help="Allow mixing of examples across classes instead of grouping by label.",
    )
    fewshot_group.add_argument(
        "--sampling_method",
        type=str,
        default="label_distribution",
        choices=["random", "label_distribution"],
        help="Sampling method to use for sampling the few-shot examples. \
            'random' for random sampling, \
            'label_distribution' ensures sampling from all labels while maintaining the label distribution.",
    )
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--output_dir", type=str, help="Directory to save the outputs."
    )
    output_group.add_argument(
        "--use_wandb",
        action="store_true",
        default=False,
        help="Enable logging with Weights & Biases.",
    )
    output_group.add_argument(
        "--override",
        action="store_true",
        default=False,
        help="Override existing outputs.",
    )
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument(
        "--label_column",
        type=str,
        default="label",
        help="Name of the label column in the dataset.",
    )
    data_group.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Name of the text column in the dataset.",
    )
    data_group.add_argument(
        "--example_column",
        type=str,
        default="text",
        help="Name of the example column in the dataset.",
    )
    data_group.add_argument(
        "--label_map",
        type=convert_keys_to_int,
        default='{"0": "Negative", "1": "Positive"}',
        help='Label map as a JSON string, e.g., \'{"0": "Negative", "1": "Positive"}\'',
    )
    wandb_group = parser.add_argument_group("Weights & Biases Configuration")
    wandb_group.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Entity name for Weights & Biases logging.",
    )
    wandb_group.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="W&B project name.",
    )
    wandb_group.add_argument(
        "--wandb_run", type=str, default=None, help="W&B run name."
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to the configuration JSON file."
    )
    parser.add_argument(
        "--env_file", type=str, default=".env", help="Path to the .env file"
    )
    return parser.parse_args()


async def main(config: Dict):
    """
    Main function to run experiments.

    Args:
        config: Configuration dictionary
    """
    # --- Setup: logging, output directories, W&B ---
    wandb_run = None
    if config.get("use_wandb", False):
        wandb_run = wandb.init(
            entity=config["wandb_entity"],
            project=config["wandb_project"],
            name=config["wandb_run"],
            config=config,
        )
    if not os.path.exists(config["output_dir"]):
        os.makedirs(config["output_dir"], exist_ok=True)
    # Bedrock custom models use an ARN as model_id (e.g. "arn:aws:bedrock:.../<name>").
    # Extract just the last path segment so the output directory has a readable name.
    if config["model_id"].startswith("arn:"):
        config["model_output_dir"] = os.path.join(
            config["output_dir"], config["model_id"].split("/")[-1]
        )
    else:
        config["model_output_dir"] = os.path.join(
            config["output_dir"], config["model_id"]
        )
    if not os.path.exists(config["model_output_dir"]):
        os.makedirs(config["model_output_dir"], exist_ok=True)
    setup_logger(config["model_output_dir"])
    logger.info("Starting experiment with config: %s", json.dumps(config, indent=2))

    # --- Data loading ---
    dataset_loader = DatasetLoader(config)
    logger.info("Loaded dataset loader: %s", dataset_loader)
    train_df, test_df = dataset_loader.load_datasets()
    logger.info(
        "Loaded train and test datasets: train_df: %s, test_df: %s",
        train_df.shape,
        test_df.shape,
    )
    # Derive human-readable class names from the label map (e.g. {0: "negative"} → ["Negative"]).
    target_names = list(map(lambda label: label.title(), config["label_map"].values()))
    logger.info("Target names: %s", target_names)

    # --- Model and component initialization ---
    llm = get_llm(config)
    if config["model_source"] == "azure":
        logger.info("Loaded Azure LLM: %s", llm.deployment_name)
    else:
        logger.info("Loaded Bedrock LLM: %s", llm.model_id)
    if len(config["label_map"]) <= 2:
        classifier = BinaryClassifier(
            llm=llm,
            objective=config["objective"],
            label_map=config["label_map"],
            target_names=target_names,
            text_column=config["text_column"],
            batch_size=config["batch_size"],
        )
        criteria_creator = BinaryCriteriaCreator(llm=llm, objective=config["objective"])
        description_creator = BinaryDescriptionCreator(
            llm=llm, objective=config["objective"]
        )
    else:
        raise ValueError(
            "Only binary classification is supported in this version. "
            "Please update the config to use a binary label map."
        )

    # --- Experiment configuration ---
    # The five methods compared in the paper: raw examples baseline plus four
    # knowledge-distillation variants (criteria / description, from examples / from each other).
    methods = [
        "Examples",
        "Criteria-Ex",
        "Criteria-De",
        "Description-Ex",
        "Description-Cr",
    ]
    config["num_few_shot_examples"] = sorted(
        [int(num_few_shots) for num_few_shots in config["num_few_shot_examples"]]
    )
    # Always include the zero-shot baseline, even if the user didn't request it explicitly.
    if 0 not in config["num_few_shot_examples"]:
        config["num_few_shot_examples"].insert(0, 0)
    # Accumulates per-iteration classification reports; keyed by method → num_shots → [reports].
    reports = init_reports_dict(methods, config["num_few_shot_examples"])

    # --- Main experiment loop ---
    # Outer loop: independent repetitions for variance estimation.
    # Inner loop: sweep over few-shot counts; at each count, run all five methods.
    for iter_num in range(1, config["num_experiments"] + 1):
        logger.info(
            "Running experiment iteration %s/%s...", iter_num, config["num_experiments"]
        )
        for num_few_shot_examples in config["num_few_shot_examples"]:
            logger.info(
                "Iteration %s - Few-shot examples: %s", iter_num, num_few_shot_examples
            )

            # ---- Method: Examples (raw few-shot baseline) ----
            # First, sample the few-shot examples for this (iter, num_shots) pair.
            # The same sample is then reused across all five methods so every method
            # sees identical training signal — ensuring a controlled comparison.
            # Each artifact produced here (examples, criteria, descriptions, reports)
            # is persisted to disk so a crashed run can resume without re-issuing LLM calls.
            method = "Examples"
            # Load the few-shot sample from a previous run, or draw a fresh one from the train set.
            few_shot_examples_df, few_shot_examples = maybe_load_few_shot_examples(
                config,
                method="Examples",
                num_few_shot_examples=num_few_shot_examples,
                iter_num=iter_num,
                mix_examples=config["mix_examples"],
            )
            if few_shot_examples_df is None:
                if num_few_shot_examples > 0:
                    logger.info(
                        "Few-shot examples not found for %s examples. Generating new examples...",
                        num_few_shot_examples,
                    )
                # Sample k labeled examples from the training set.
                few_shot_examples_df, few_shot_examples = create_few_shot_examples(
                    train_df,
                    num_few_shot_examples=num_few_shot_examples,
                    mix_examples=config["mix_examples"],
                    example_column=config["example_column"],
                    label_column=config["label_column"],
                    sampling_method=config["sampling_method"],
                )
                if num_few_shot_examples > 0:
                    save_few_shot_examples(
                        few_shot_examples_df,
                        config,
                        method,
                        num_few_shot_examples,
                        iter_num,
                        wandb_run,
                    )
                    logger.info(
                        "Few-shot examples saved for %s examples.",
                        num_few_shot_examples,
                    )
            samples_report = maybe_load_classification_report(
                config,
                method=f"{method}_num_samples_{num_few_shot_examples}_iter_{iter_num}",
            )
            if samples_report is None:
                logger.info(
                    "Classification report not found for %s. Running experiment...",
                    method,
                )
                # Classify using raw examples directly in the prompt (baseline).
                samples_report = await run_experiment(
                    config,
                    wandb_run,
                    classifier,
                    test_df,
                    few_shot_examples=few_shot_examples,
                    method=f"{method}_num_samples_{num_few_shot_examples}_iter_{iter_num}",
                )
            logger.info(
                "Finished running experiment for %s; with %s examples. (iter %s)",
                method,
                num_few_shot_examples,
                iter_num,
            )
            reports[method][int(num_few_shot_examples)].append(samples_report)

            # ---- Method: Criteria-Ex (criteria derived from examples) ----
            # Offline extraction step: the LLM generalizes the labeled examples into
            # explicit positive/negative criteria. At inference time, these compact
            # criteria replace the raw examples in the classification prompt.
            method = "Criteria-Ex"
            criteria_report = maybe_load_classification_report(
                config,
                method=f"{method}_num_samples_{num_few_shot_examples}_iter_{iter_num}",
            )
            # Load criteria from a previous run, or generate them now from the examples.
            criteria_from_examples = maybe_load_criteria(
                config, method, num_few_shot_examples, iter_num
            )
            if criteria_report is None:
                if criteria_from_examples is None:
                    logger.info(
                        "Criteria not found for %s. Generating new criteria...", method
                    )
                    # Extract positive/negative criteria from the labeled examples.
                    criteria_from_examples = await criteria_creator.get_llm_responses(
                        few_shot_examples=few_shot_examples
                    )
                    save_criteria(
                        criteria_from_examples,
                        config,
                        method,
                        num_few_shot_examples,
                        iter_num,
                        wandb_run,
                    )
                    logger.info("Generated and saved criteria for %s", method)
                logger.info(
                    "Classification report not found for %s. Running experiment...",
                    method,
                )
                # Classify the test set using the extracted criteria in the prompt.
                criteria_report = await run_experiment(
                    config,
                    wandb_run,
                    classifier,
                    test_df,
                    criteria=criteria_from_examples,
                    method=f"{method}_num_samples_{num_few_shot_examples}_iter_{iter_num}",
                )
            logger.info(
                "Finished running experiment for %s; with %s examples. (iter %s)",
                method,
                num_few_shot_examples,
                iter_num,
            )
            reports[method][int(num_few_shot_examples)].append(criteria_report)

            # ---- Method: Description-Ex (free-text description derived from examples) ----
            # Same offline extraction step as Criteria-Ex, but the LLM produces a
            # free-text task description instead of a structured criteria list.
            # Captures nuanced, context-dependent patterns that rigid criteria may miss.
            method = "Description-Ex"
            description_report = maybe_load_classification_report(
                config,
                method=f"{method}_num_samples_{num_few_shot_examples}_iter_{iter_num}",
            )
            # Load description from a previous run, or generate it now from the examples.
            description_from_examples = maybe_load_description(
                config, method, num_few_shot_examples, iter_num
            )
            if description_report is None:
                if description_from_examples is None:
                    logger.info(
                        "Description not found for %s. Generating new description...",
                        method,
                    )
                    # Extract a free-text task description from the labeled examples.
                    description_from_examples = (
                        await description_creator.get_llm_responses(
                            few_shot_examples=few_shot_examples
                        )
                    )
                    save_description(
                        description_from_examples,
                        config,
                        method,
                        num_few_shot_examples,
                        iter_num,
                        wandb_run,
                    )
                    logger.info("Generated and saved description for %s", method)
                logger.info(
                    "Classification report not found for %s. Running experiment...",
                    method,
                )
                # Classify the test set using the extracted description in the prompt.
                description_report = await run_experiment(
                    config,
                    wandb_run,
                    classifier,
                    test_df,
                    description=description_from_examples,
                    method=f"{method}_num_samples_{num_few_shot_examples}_iter_{iter_num}",
                )
            logger.info(
                "Finished running experiment for %s; with %s examples. (iter %s)",
                method,
                num_few_shot_examples,
                iter_num,
            )
            reports[method][int(num_few_shot_examples)].append(description_report)

            # ---- Method: Criteria-De (criteria derived from Description-Ex output) ----
            # Two-stage pipeline: examples → description (Description-Ex) → criteria.
            # Adding a second distillation step allows the LLM to refine the task
            # knowledge before committing to the structured criteria format.
            method = "Criteria-De"
            criteria_report = maybe_load_classification_report(
                config,
                method=f"{method}_num_samples_{num_few_shot_examples}_iter_{iter_num}",
            )
            if num_few_shot_examples == 0:
                # With zero examples there is no description to derive criteria from,
                # so Criteria-De is identical to Criteria-Ex — reuse its result directly.
                criteria_from_description = criteria_from_examples
                criteria_report = reports["Criteria-Ex"][int(num_few_shot_examples)][
                    iter_num - 1
                ]
                save_criteria(
                    criteria_from_description,
                    config,
                    method,
                    num_few_shot_examples,
                    iter_num,
                    wandb_run,
                )
                logger.info(
                    "Using and saving zero-shot criteria for %s from Criteria-Ex",
                    method,
                )
            else:
                # Load criteria from a previous run, or derive them from the Description-Ex output.
                criteria_from_description = maybe_load_criteria(
                    config, method, num_few_shot_examples, iter_num
                )
                if criteria_report is None:
                    if criteria_from_description is None:
                        logger.info(
                            "Criteria not found for %s. Generating from description...",
                            method,
                        )
                        # Second distillation step: description → criteria.
                        criteria_from_description = (
                            await criteria_creator.get_llm_responses(
                                description=description_from_examples
                            )
                        )
                        save_criteria(
                            criteria_from_description,
                            config,
                            method,
                            num_few_shot_examples,
                            iter_num,
                            wandb_run,
                        )
                        logger.info("Generated and saved criteria for %s", method)
                    logger.info(
                        "Classification report not found for %s. Running experiment...",
                        method,
                    )
                    # Classify the test set using the description-derived criteria.
                    criteria_report = await run_experiment(
                        config,
                        wandb_run,
                        classifier,
                        test_df,
                        criteria=criteria_from_description,
                        method=f"{method}_num_samples_{num_few_shot_examples}_iter_{iter_num}",
                    )
                logger.info(
                    "Finished running experiment for %s; with %s examples. (iter %s)",
                    method,
                    num_few_shot_examples,
                    iter_num,
                )
            reports[method][int(num_few_shot_examples)].append(criteria_report)

            # ---- Method: Description-Cr (description derived from Criteria-Ex output) ----
            # Two-stage pipeline: examples → criteria (Criteria-Ex) → description.
            # The reverse of Criteria-De: starts from structured criteria and synthesizes
            # them into a cohesive narrative description for the final classifier.
            method = "Description-Cr"
            description_report = maybe_load_classification_report(
                config,
                method=f"{method}_num_samples_{num_few_shot_examples}_iter_{iter_num}",
            )
            if num_few_shot_examples == 0:
                # With zero examples there are no criteria to derive a description from,
                # so Description-Cr is identical to Description-Ex — reuse its result directly.
                description_from_criteria = description_from_examples
                description_report = reports["Description-Ex"][
                    int(num_few_shot_examples)
                ][iter_num - 1]
                save_description(
                    description_from_criteria,
                    config,
                    method,
                    num_few_shot_examples,
                    iter_num,
                    wandb_run,
                )
                logger.info(
                    "Using and saving zero-shot description for %s from Description-Ex",
                    method,
                )
            else:
                # Load description from a previous run, or derive it from the Criteria-Ex output.
                description_from_criteria = maybe_load_description(
                    config, method, num_few_shot_examples, iter_num
                )
                if description_report is None:
                    if description_from_criteria is None:
                        logger.info(
                            "Description not found for %s. Generating from criteria...",
                            method,
                        )
                        # Second distillation step: criteria → description.
                        description_from_criteria = (
                            await description_creator.get_llm_responses(
                                criteria=criteria_from_examples
                            )
                        )
                        save_description(
                            description_from_criteria,
                            config,
                            method,
                            num_few_shot_examples,
                            iter_num,
                            wandb_run,
                        )
                        logger.info("Generated and saved description for %s", method)
                    logger.info(
                        "Classification report not found for %s. Running experiment...",
                        method,
                    )
                    # Classify the test set using the criteria-derived description.
                    description_report = await run_experiment(
                        config,
                        wandb_run,
                        classifier,
                        test_df,
                        description=description_from_criteria,
                        method=f"{method}_num_samples_{num_few_shot_examples}_iter_{iter_num}",
                    )
                logger.info(
                    "Finished running experiment for %s; with %s examples. (iter %s)",
                    method,
                    num_few_shot_examples,
                    iter_num,
                )
            reports[method][int(num_few_shot_examples)].append(description_report)

    # --- Post-experiment: aggregation and visualization ---
    logger.info("Computing average metrics across all iterations...")
    average_metrics_dict = compute_average_metrics_dict(
        config, wandb_run, reports, target_names=target_names
    )
    logger.info("Creating performance visualizations...")
    plot_performance(
        config, wandb_run, methods, average_metrics_dict, target_names=target_names
    )
    logger.info("Experiment completed successfully")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    args = parse_args()
    if args.env_file:
        load_dotenv(args.env_file)
    # Config file takes full precedence over CLI flags when provided.
    if args.config is not None:
        with open(args.config, "r") as f:
            config = json.load(f)
    else:
        config = vars(args)
        config.pop("config", None)
        config.pop("env_file", None)
    config = process_label_map(config)
    asyncio.run(main(config))
