import gradio as gr
from backend.gpt_client import GPTClient, GPTModel
from backend.local_client import LocalClient, Devices, DTypes

gpt_client = GPTClient()
local_client = LocalClient()

css = """
.gradio-container {
    height: 100%;
    width: 100%;
}
"""

def gpt_interface(
    message: str,
    _,
    api_key: str,
    system_message: str,
    frequency_penalty: float,
    temperature: int,
    model: GPTModel,
):
    gpt_client.handle_parameters(
        api_key=api_key,
        system_message=system_message,
        frequency_penalty=frequency_penalty,
        temperature=temperature,
        model=model,
    )

    response = gpt_client.create_completion(message)
    return response


def local_interface(
    message: str,
    _,
    device: Devices,
    dtype: DTypes,
    max_new_tokens: int,
    do_sample: bool,
    top_p: float,
    top_k: int,
    temperature: float,
    num_beams: int,
):
    local_client.handle_parameters(
        device=device,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_beams=num_beams,
    )

    yield from local_client.create_completion(message)


gpt = gr.ChatInterface(
    gpt_interface,
    additional_inputs=[
        gr.Textbox(
            label="API key",
            value="",
            info="If not provided here, it will be read from the environment variable.",
        ),
        gr.Textbox(
            gpt_client.system_message,
            label="System message",
            info="Changing system message will clear the internal context.",
        ),
        gr.Slider(minimum=-2.0, maximum=2.0, value=0, label="Frequency penalty"),
        gr.Slider(minimum=0, maximum=2, value=1, label="Temperature"),
        gr.Dropdown(
            choices=[(model.name, model) for model in GPTModel],
            label="Model",
            value=GPTModel.GPT4_TURBO,
            multiselect=False,
        ),
    ],
    css=css,
)


local = gr.ChatInterface(
    local_interface,
    additional_inputs=[
        gr.Dropdown(
            choices=[(device.name, device) for device in Devices],
            label="Device",
            value=Devices.CPU,
            multiselect=False,
            info="The device to run the model on. GPU for CUDA enabled GPUs. CPU for CPU."
        ),
        gr.Dropdown(
            choices=[(dtype.name, dtype) for dtype in DTypes],
            label="Data type",
            value=DTypes.FLOAT32,
            multiselect=False,
            info="The data type to use for the model.",
        ),
        gr.Number(
            label="Max new tokens",
            value=1024,
            minimum=0,
            maximum=2048,
            step=1,
            info="The maximum number of tokens to generate.",
        ),
        gr.Checkbox(
            label="Do sample",
            value=True,
            info="Whether to use sampling or greedy decoding.",
        ),
        gr.Slider(
            label="Top p",
            value=0.95,
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            info="The cumulative probability for sampling.",
        ),
        gr.Number(
            label="Top k",
            value=1000,
            minimum=0,
            maximum=2048,
            step=1,
            info="The maximum number of tokens to consider for sampling.",
        ),
        gr.Slider(
            label="Temperature",
            value=1.0,
            minimum=0.0,
            maximum=2.0,
            step=0.01,
            info="The temperature for sampling.",
        ),
        gr.Number(
            label="Number of beams",
            value=1,
            minimum=1,
            maximum=8,
            step=1,
            info="The number of beams for beam search.",
        ),
    ],
    css=css,
)
demo = gr.TabbedInterface([gpt, local], ["GPT", "local"])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
