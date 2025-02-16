from typing import List, Union, Generator, Iterator, Optional
from pydantic import BaseModel
import os

from azure.ai.inference import ChatCompletionsClient
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential


class Pipeline:
    class Valves(BaseModel):
        # You can add your custom valves here.
        AZURE_OPENAI_API_KEY: Optional[str] = None
        AZURE_OPENAI_ENDPOINT: str
        AZURE_OPENAI_API_VERSION: str
        AZURE_OPENAI_MODEL: str

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

        if not self.valves.AZURE_OPENAI_API_KEY:
            self.client = ChatCompletionsClient(
                endpoint=self.valves.AZURE_OPENAI_ENDPOINT,
                credential=DefaultAzureCredential(exclude_environment_credential=True),
                credential_scopes=["https://cognitiveservices.azure.com/.default"]
            )
        else:
            self.client = ChatCompletionsClient(
                endpoint=self.valves.AZURE_OPENAI_ENDPOINT,
                credential=AzureKeyCredential(self.valves.AZURE_OPENAI_API_KEY)
            )

        self.set_pipelines()
        pass

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

        try:
            response = self.client.complete(
                messages = messages,
                model = model_id,
                stream = body.get("stream", False),
                max_tokens = body.get("max_tokens", 1000),
                max_tokens = body.get("temperature", 0.5)
            )

            if body.get("stream", False):
                return self.stream_response(response)
            else:
                return response.choices[0].message.content

        except Exception as e:
            if response:
                text = response.choices[0].message.content
                return f"Error: {e} ({text})"
            else:
                return f"Error: {e}"

    def stream_response(self, response):
        for chunk in response:
            choices = chunk.get("choices")
            if choices and len(choices) > 0:
                content = choices[0]["delta"].get("content", "")
                if content:
                    print(f"Chunk: {content}")
                    yield content