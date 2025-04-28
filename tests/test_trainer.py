import pytest
from unittest import mock

from trainer import train

@mock.patch("trainer.SFTTrainer")
@mock.patch("trainer.AutoModelForCausalLM.from_pretrained")
@mock.patch("trainer.AutoTokenizer.from_pretrained")
@mock.patch("trainer.load_dataset")
def test_train_mocked(mock_load_dataset, mock_tokenizer_from_pretrained, mock_model_from_pretrained, mock_sft_trainer):
    # Setup mocks

    # Mock dataset
    mock_dataset = mock.MagicMock()
    mock_dataset.map.return_value = mock_dataset  # .map() should return another mock dataset
    mock_load_dataset.return_value = mock_dataset

    # Mock tokenizer
    mock_tokenizer = mock.MagicMock()
    mock_tokenizer.pad_token = mock_tokenizer.eos_token = 0
    mock_tokenizer_from_pretrained.return_value = mock_tokenizer

    # Mock model
    mock_model = mock.MagicMock()
    mock_model.gradient_checkpointing_enable.return_value = None
    mock_model.enable_input_require_grads.return_value = None
    mock_model.to.return_value = None
    mock_model.config = mock.MagicMock()
    mock_model.config.pad_token_id = None
    mock_model_from_pretrained.return_value = mock_model

    # Mock trainer
    mock_trainer_instance = mock.MagicMock()
    mock_sft_trainer.return_value = mock_trainer_instance

    # Now run train()
    train()

    # Assertions (optional but recommended)
    mock_load_dataset.assert_called_once()
    mock_tokenizer_from_pretrained.assert_called_once()
    mock_model_from_pretrained.assert_called_once()
    mock_sft_trainer.assert_called_once()
    mock_trainer_instance.train.assert_called_once()
    mock_model.save_pretrained.assert_called_once_with("./r_model")
    mock_tokenizer.save_pretrained.assert_called_once_with("./r_model")
