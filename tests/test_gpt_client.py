import pytest
from backend.gpt_client import GPTClient, SYSTEM_MESSAGE


@pytest.fixture
def gpt_client():
    return GPTClient()


def test_clear_history(gpt_client):
    gpt_client.history = ["a", "b", "c"]
    gpt_client.clear_history()
    processed_system_message = " ".join(
        line.strip() for line in SYSTEM_MESSAGE.split("\n") if line.strip()
    )
    assert gpt_client.history == [{"role": "system", "content": processed_system_message}]


def test_new_system_message(gpt_client):
    gpt_client.system_message = "New system message"
    assert gpt_client.system_message == "New system message"
    
