# MIT License
# Copyright (c) 2026 Gong.io
# See LICENSE file for full license text

import os
from typing import Any, Dict, Optional, Union

from langchain_aws import ChatBedrock
from langchain_openai import AzureChatOpenAI


def get_bedrock_llm(
    model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> ChatBedrock:
    """
    Initialize a Bedrock LLM with the specified configuration.

    Args:
        model_id: ID of the Bedrock model to use
        model_kwargs: Additional model parameters

    Returns:
        Configured ChatBedrock instance
    """
    if model_kwargs is None:
        model_kwargs = {
            "temperature": 0.0,
            "top_p": 1,
            "timeout": 120,
            "max_retries": 3,
        }

    if "anthropic" in model_id:
        provider = "anthropic"
    elif "meta" in model_id:
        provider = "meta"
    elif "mistral" in model_id:
        provider = "mistral"
    elif "amazon" in model_id:
        provider = "amazon"
    else:
        raise ValueError(f"Unable to determine provider for model_id: {model_id}")

    credentials_profile_name = os.environ.get("AWS_PROFILE")
    region = os.environ.get("AWS_REGION", "us-east-1")
    endpoint_url = os.environ.get("AWS_BEDROCK_ENDPOINT")

    llm = ChatBedrock(
        model_id=model_id,
        credentials_profile_name=credentials_profile_name,
        region=region,
        endpoint_url=endpoint_url,
        provider=provider,
        model_kwargs=model_kwargs,
    )
    return llm


def get_azure_llm(
    model_id: str = "gpt-4o",
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> AzureChatOpenAI:
    """
    Initialize an Azure OpenAI LLM with the specified configuration.

    Args:
        model_id: ID of the Azure OpenAI model to use
        model_kwargs: Additional model parameters

    Returns:
        Configured AzureChatOpenAI instance
    """
    if model_kwargs is None:
        model_kwargs = {"temperature": 0.0, "top_p": 1}

    model_version = os.environ.get("AZURE_OPENAI_MODEL_VERSION", None)

    llm = AzureChatOpenAI(
        deployment_name=model_id,
        model_version=model_version,
        temperature=model_kwargs.get("temperature", 0.0),
        top_p=model_kwargs.get("top_p", 1),
        timeout=600,
        max_retries=3,
    )
    return llm


def get_llm(
    config: Dict[str, Any],
    model_kwargs: Optional[Dict[str, Any]] = None,
    credentials_profile_name: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    region: Optional[str] = None,
) -> Union[ChatBedrock, AzureChatOpenAI]:
    """
    Get an LLM based on the configuration.

    Args:
        config: Configuration dictionary
        model_kwargs: Additional model parameters
        credentials_profile_name: AWS credentials profile (for Bedrock); falls back to env/default chain if None
        endpoint_url: Bedrock endpoint URL (for Bedrock); uses default if None
        region: AWS region (for Bedrock); uses default if None

    Returns:
        Configured LLM instance (either Azure or Bedrock)
    """
    if config["model_source"].lower() == "azure":
        llm = get_azure_llm(model_id=config["model_id"], model_kwargs=model_kwargs)
    elif config["model_source"].lower() == "bedrock":
        llm = get_bedrock_llm(model_id=config["model_id"], model_kwargs=model_kwargs)
    else:
        raise ValueError(f"Unsupported model source: {config['model_source']}")
    return llm
