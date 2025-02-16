"""
title: Azure OpenAI SDK
author: davidsewell
date: 2025-02-16
version: 0.1
license: MIT
description: A pipeline for integrating with Azure OpenAI using the Azure OpenAI SDK.
requirements: azure-ai-inference, azure-identity, azure-core, pydantic
environment_variables: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_MODEL
"""

from typing import List, Union, Generator, Iterator, Optional
from pydantic import BaseModel
import os

from openai import AzureOpenAI, ChatCompletion
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential


class Pipeline:
    class Valves(BaseModel):
        # You can add your custom valves here.
        AZURE_OPENAI_API_KEY: Optional[str] = None
        AZURE_OPENAI_ENDPOINT: str
        AZURE_OPENAI_API_VERSION: str
        AZURE_OPENAI_MODELS: str
        AZURE_OPENAI_MODEL_NAMES: str

    def __init__(self):
        self.type = "manifold"
        self.name = "Azure OpenAI SDK: "
        self.valves = self.Valves(
            **{
                "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY", None),
                "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT", "your-azure-openai-endpoint-here"),
                "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
                "AZURE_OPENAI_MODELS": os.getenv("AZURE_OPENAI_MODELS", "gpt-4o-mini"),
                "AZURE_OPENAI_MODEL_NAMES": os.getenv("AZURE_OPENAI_MODEL_NAMES", "GPT-4o-MINI"),
            }
        )

        self.client = self._openai_client()

        self.set_pipelines()
        pass

    def _openai_client(self) -> AzureOpenAI:
        """
        Create an OpenAI client. Requires Azure Credentials. See the DefaultAzureCredential documentation for details
        of the authentication process (it will cascade through multiple authentication methods until it finds one that
        works, including a Workload Identity, an SPN via Env Vars, or az login credentials when running locally).

        :return: AzureOpenAI client
        """
        default_credential = DefaultAzureCredential(exclude_environment_credential=True)
        token = default_credential.get_token(
            "https://cognitiveservices.azure.com/.default"
        )

        try:
            client = AzureOpenAI(
                api_version=self.valves.AZURE_OPENAI_API_VERSION,
                azure_endpoint=self.valves.AZURE_OPENAI_ENDPOINT,
                api_key=self.valves.AZURE_OPENAI_API_KEY or token.token
            )
            print("AzureOpenAI client created")
        except Exception as e:
            return f"Error: {e}"

        return client

    def set_pipelines(self):
        models = self.valves.AZURE_OPENAI_MODELS.split(";")
        model_names = self.valves.AZURE_OPENAI_MODEL_NAMES.split(";")
        self.pipelines = [
            {"id": model, "name": name} for model, name in zip(models, model_names)
        ]
        print(f"azure_openai_sdk_pipeline - models: {self.pipelines}")
        pass

    async def on_valves_updated(self):
        self.set_pipelines()

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    def pipe(
            self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        print(messages)
        print(user_message)

        allowed_params = {'messages', 'temperature', 'role', 'content', 'contentPart', 'contentPartImage',
                          'enhancements', 'dataSources', 'n', 'stream', 'stop', 'max_tokens', 'presence_penalty',
                          'frequency_penalty', 'logit_bias', 'user', 'function_call', 'funcions', 'tools',
                          'tool_choice', 'top_p', 'log_probs', 'top_logprobs', 'response_format', 'seed'}

        # remap user field
        if "user" in body and not isinstance(body["user"], str):
            body["user"] = body["user"]["id"] if "id" in body["user"] else str(body["user"])

        stream = body.get("stream", False)

        # Base parameters for the API call
        parameters = {
            "model": model_id,
            "messages": messages,
            "stream": stream,
            "temperature": body.get("temperature", 0.5),
            "max_tokens": body.get("max_tokens", 1000),
            "top_p": body.get("openai_top_p", None),
            "frequency_penalty": body.get("openai_frequency_penalty", None),
            "presence_penalty": body.get("openai_presence_penalty", None),
        }

        response: ChatCompletion = None
        try:
            response = self.client.chat.completions.create(**parameters)

            if stream:
                return self.stream_response(response)
            else:
                return response.choices[0].message.content

        except Exception as e:
            if response:
                text = response.choices[0].message.content
                return f"Error: {e} ({text})"
            else:
                return f"Error: {e}"

    def stream_response(self, response: ChatCompletion):
        for chunk in response:
            choices = chunk.get("choices")
            if choices and len(choices) > 0:
                content = choices[0]["delta"].get("content", "")
                if content:
                    print(f"Chunk: {content}")
                    yield content