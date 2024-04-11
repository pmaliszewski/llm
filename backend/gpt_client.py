import openai
import os
from enum import Enum

SYSTEM_MESSAGE = """
        You are a helpful assistant highly specialized in assisting with programming tasks.
        You can help with debugging, writing code, and explaining concepts.
        You can also provide general advice on programming and computer science.
        Assume the user is knowledgable and more often than not you will be expected to provide working and complete code blocks.""".strip()


class GPTModel(Enum):
    GPT4_TURBO = "gpt-4-turbo-preview"
    GPT4 = "gpt-4"
    GPT3_TURBO = "gpt-3.5-turbo"


class GPTClient:
    def __init__(
        self,
    ) -> None:
        self.api_key = os.environ.get("OPENAI_API_KEY", None)
        if not self.api_key:
            self.client = None
        else:
            self.client = openai.OpenAI(api_key=self.api_key)

        self._system_message = SYSTEM_MESSAGE

        self._system_message = " ".join(
            line.strip() for line in self._system_message.split("\n") if line.strip()
        )

        self._history = [{"role": "system", "content": self.system_message}]
        self.total_tokens = 0
        self.temperature = 1
        self.frequency_penalty = 0
        self.model = GPTModel.GPT4_TURBO.value

    @property
    def history(self):
        return self._history

    @history.setter
    def history(self, value):
        self._history.append(value)

    @property
    def system_message(self):
        return self._system_message

    @system_message.setter
    def system_message(self, value):
        self._system_message = " ".join(
            line.strip() for line in value.split("\n") if line.strip()
        )

    def clear_history(self):
        self._history = [{"role": "system", "content": self.system_message}]

    def handle_parameters(
        self,
        api_key: str,
        system_message: str,
        frequency_penalty: float,
        temperature: int,
        model: str,
    ) -> None:
        if all((self.api_key, api_key, api_key != self.api_key)) or not self.api_key:
            self.api_key = api_key
            self.client = openai.OpenAI(api_key=self.api_key)

        if system_message != self.system_message:
            self.system_message = system_message
            self.clear_history()

        self.frequency_penalty = frequency_penalty
        self.temperature = temperature
        self.model = model

    def create_completion(
        self,
        prompt: str,
    ) -> str:
        self.history = {"role": "user", "content": prompt}
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
            temperature=self.temperature,
            frequency_penalty=self.frequency_penalty,
        )
        self.total_tokens = response.usage.total_tokens
        answer = response.choices[0].message.content
        self.history = {"role": "assistant", "content": answer}
        return answer
