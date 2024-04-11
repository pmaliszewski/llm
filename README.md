# LLM

## Description

A wrapper for OpenAI's GPT and an implementation for RedPajama's INCITE.

## Installation

Please note that `local_client.py` will attempt to download the model shards hosted on HuggingFace.

### docker-compose

Use the following command to start the service:

```bash
docker-compose up
```

You can set the `OPENAI_API_KEY` as an environment variable in `docker-compose.yaml`. Alternatively, the UI provides an option for manually supplying the API key.

### manual

Follow these steps for manual installation:


```bash
pip install poetry
poetry install
gradio ui.py
```

For both methods, the service is available on port `7860`. You can access the UI through your localhost. This is a very naive approach, though - if the ports are used, you might want to adjust
the bindings in `ui.py` and perhaps `Dockerfile` and `docker-compose.yaml`. The current implementation doesn't check for usage.

## UI

Please be aware of the `Additional Inputs` dropdown menu. In the GPT tab, it includes options to provide an API key (if not set or to override the existing one), a system message, and model choices.

For the local model, options are available for adjusting torch-related parameters (such as the device or data type) and model-specific parameters.

Descriptions of the parameters themselves are provided in the info tabs.