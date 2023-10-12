# %%
import os

import transformers as tr
from datasets import load_dataset

# %%
imdb_ds = load_dataset("imdb")
# %%
model_checkpoint = "t5-small"
cache_dir = "data/cached/"
# load the tokenizer that was used for the t5-small model
tokenizer = tr.AutoTokenizer.from_pretrained(model_checkpoint, cache_dir=cache_dir)  # Use a pre-cached model
# %%


def to_tokens(tokenizer: tr.models.t5.tokenization_t5_fast.T5TokenizerFast, label_map: dict) -> callable:
    """
    Returns a closure that takes a formatted dataset `x` and returns a batch encoding `token_res`.

    Args:
        tokenizer (tr.models.t5.tokenization_t5_fast.T5TokenizerFast): A tokenizer object.
        label_map (dict): A dictionary mapping label strings to integers.

    Returns:
        callable: A closure that takes a formatted dataset `x` and returns a batch encoding `token_res`.
    """

    def apply(x) -> tr.tokenization_utils_base.BatchEncoding:
        """
        From a formatted dataset `x` a batch encoding `token_res` is created.

        Args:
            x (dict): A dictionary containing the keys "text" and "label".

        Returns:
            tr.tokenization_utils_base.BatchEncoding: A batch encoding object.
        """
        target_labels = [label_map[y] for y in x["label"]]
        token_res = tokenizer(
            x["text"],
            text_target=target_labels,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        return token_res

    return apply


imdb_label_lookup = {0: "negative", 1: "positive", -1: "unknown"}

imdb_to_tokens = to_tokens(tokenizer, imdb_label_lookup)
tokenized_dataset = imdb_ds.map(imdb_to_tokens, batched=True, remove_columns=["text", "label"])
# %%
local_training_root = "models/train/"
checkpoint_name = "trainer"
local_checkpoint_path = os.path.join(local_training_root, checkpoint_name)
training_args = tr.TrainingArguments(
    local_checkpoint_path,
    num_train_epochs=1,  # default number of epochs to train is 3
    per_device_train_batch_size=16,
    optim="adamw_torch",
    report_to=["tensorboard"],
)
# %%
# load the pre-trained model
cache_dir = "models/pretrained/"
model = tr.AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, cache_dir=cache_dir)  # Use a pre-cached model
# %%
# used to assist the trainer in batching the data
data_collator = tr.DataCollatorWithPadding(tokenizer=tokenizer)
trainer = tr.Trainer(
    model,
    training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)
# %%
tensorboard_display_dir = f"{local_checkpoint_path}/runs"


# %%
trainer.train()

# save model to the local checkpoint
trainer.save_model()
trainer.save_state()

# COMMAND ----------

# persist the fine-tuned model
final_model_path = f"models/fine_tuning/{checkpoint_name}"
trainer.save_model(output_dir=final_model_path)
# %%
