import torch
import transformers as tr
from src.fine_tuning import to_tokens


def test_to_tokens():
    # Define a sample tokenizer and label map
    tokenizer = tr.models.t5.tokenization_t5_fast.T5TokenizerFast.from_pretrained("t5-small")
    label_map = {"positive": 1, "negative": 0}

    # Define a sample dataset
    dataset = [{"text": "This is a positive sentence.", "label": "positive"},
               {"text": "This is a negative sentence.", "label": "negative"}]

    # Get the closure function
    apply_fn = to_tokens(tokenizer, label_map)

    # Apply the closure function to the dataset
    batch_encoding = apply_fn(dataset)

    # Check that the batch encoding has the expected keys
    assert set(batch_encoding.keys()) == {"input_ids", "attention_mask", "text_target"}

    # Check that the input_ids and attention_mask tensors have the expected shape
    assert batch_encoding["input_ids"].shape == torch.Size([2, 6])
    assert batch_encoding["attention_mask"].shape == torch.Size([2, 6])

    # Check that the text_target tensor has the expected values
    assert torch.equal(batch_encoding["text_target"], torch.tensor([[1, 0], [0, 1]]))