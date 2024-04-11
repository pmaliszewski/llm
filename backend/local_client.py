import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)
from threading import Thread
from enum import Enum

MODEL_NAME = "togethercomputer/RedPajama-INCITE-Chat-3B-v1"

class Devices(Enum):
    CPU = "cpu"
    GPU = "cuda"


class DTypes(Enum):
    FLOAT16 = torch.float16
    FLOAT32 = torch.float32


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, _, **kwargs) -> bool:
        stop_ids = [29, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class LocalClient:
    def __init__(self):
        self.device = Devices.CPU.value
        self.dtype = DTypes.FLOAT32.value
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=self.dtype)
        self.model = self.model.to(self.device)
        self.max_new_tokens = 1024
        self.do_sample = True
        self.top_p = 0.95
        self.top_k = 1000
        self.temperature = 1.0
        self.num_beams = 1
        self.history = []

    def handle_parameters(
        self,
        device: str,
        dtype,
        max_new_tokens: int,
        do_sample: bool,
        top_p: float,
        top_k: int,
        temperature: float,
        num_beams: int,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.model = self.model.to(self.device)
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.num_beams = num_beams

    def create_completion(self, message):
        self.history.append([message, ""])
        stop = StopOnTokens()

        messages = "".join(
            [
                "".join(["\n<human>:" + item[0], "\n<bot>:" + item[1]])
                for item in self.history
            ]
        )

        model_inputs = self.tokenizer([messages], return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(
            self.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
        )
        generate_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            top_p=self.top_p,
            top_k=self.top_k,
            temperature=self.temperature,
            num_beams=self.num_beams,
            stopping_criteria=StoppingCriteriaList([stop]),
        )
        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()

        partial_message = ""
        for new_token in streamer:
            if new_token != "<":
                partial_message += new_token
                self.history[-1][1] = partial_message
                yield partial_message
