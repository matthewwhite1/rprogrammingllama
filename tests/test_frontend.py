import pytest
from unittest.mock import MagicMock

# Assume your module is named r_llama.py
import frontend

@pytest.fixture(autouse=True)
def mock_model_and_tokenizer(monkeypatch):
    # Mock the tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_tensors = "pt"
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 0
    mock_tokenizer.__call__.return_value = {
        "input_ids": [[1, 2, 3]],
        "attention_mask": [[1, 1, 1]]
    }
    mock_tokenizer.decode.return_value = "print('Hello, world!')"

    # Mock the model
    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.generate.return_value = [[[4, 5, 6, 7]]]

    # Patch the global MODEL and TOKENIZER
    monkeypatch.setattr(frontend, "MODEL", mock_model)
    monkeypatch.setattr(frontend, "TOKENIZER", mock_tokenizer)

def test_r_chat_fn_basic():
    message = "Create a vector of numbers from 1 to 10"
    history = []  # history isn't used inside r_chat_fn
    output = frontend.r_chat_fn(message, history)
    
    assert isinstance(output, str)
    assert "print" in output or "c(" in output or "1:10" in output
