from collections import Counter 
import random 

import torch

# TODO : add dataset
from models.base import BaseProbe 
from models.logistic import LogisticProbe
from dataset import CircuitBreakerDataset

def parse_dtype(dtype):
    if dtype == "float16":
        return torch.float16
    elif dtype == "float32":
        return torch.float32
    elif dtype == "float64":
        return torch.float64
    else:
        raise ValueError(f"Invalid dtype: {dtype}")

def get_data(
    dataset: str,
    num_samples: int,
    max_num_generate: int = 0,
    do_few_shot: bool = False,
    label_key: str | None = None,
    cache_dir: str | None = None,
):
    messages = get_toxic_completions_messages(max_messages=num_samples + 2*max_num_generate, do_few_shot=do_few_shot, cache_dir=cache_dir)

    prompt_counts = Counter([message["prompt"] for message in messages])
    all_prompts = list(prompt_counts.keys())
    random.Random(0).shuffle(all_prompts)

    num_train = int(0.8*num_samples)
    num_val = num_samples - num_train
    train_count = 0
    train_prompts, val_prompts = [], []
    for prompt in all_prompts:
        if train_count < num_train:
            train_prompts.append(prompt)
            train_count += prompt_counts[prompt]
        else:
            val_prompts.append(prompt)

    train_messages = [message for message in messages if message["prompt"] in train_prompts]
    val_messages = [message for message in messages if message["prompt"] in val_prompts]

    completion_messages = []
    seen_prompts = set()
    for message in val_messages:
        if message["prompt"] not in seen_prompts:
            completion_messages.append(message)
            seen_prompts.add(message["prompt"])
        if len(completion_messages) >= max_num_generate:
            break
    val_messages = val_messages[:num_val]
    
    return train_messages, val_messages, completion_messages, num_train



# def get_cb_data(circuit_breaker_dataset, num_samples, max_num_generate):
#     harmless_messages = circuit_breaker_dataset.harmless_set
#     harmful_messages = circuit_breaker_dataset.circuit_breaker.orig
#     val_messages = circuit_breaker_dataset.val_orig

#     train_messages = harmless_messages + harmful_messages
    
#     # Function to split the user prompt and assistant response
#     def split_prompt_response(message):
#         user_tag = "user\n\n"
#         assistant_tag = "assistant\n\n"
#         separator = "<SEPARATOR>"

#         # Case 1: <SEPARATOR> present, but no user prompt
#         if separator in message:
#             prompt_part, response_part = message.split(separator)
#             prompt = prompt_part.replace(user_tag, "").replace(assistant_tag, "").strip()
#             response = response_part.strip()
#         # Case 2: with user and assistant tags but no <SEPARATOR>
#         elif user_tag in message and assistant_tag in message:
#             parts = message.split(assistant_tag)
#             prompt = parts[0].replace(user_tag, "").strip()
#             response = parts[1].strip()
#         else:
#             return None  # If the message doesn't have the expected format, skip it

#     # Format the train and validation messages
#     train_messages = [split_prompt_response(message) for message in train_messages]
#     val_messages = [split_prompt_response(message) for message in val_messages]
    
#     # Filter out any None values that might result from improperly formatted messages
#     train_messages = [msg for msg in train_messages if msg is not None]
#     val_messages = [msg for msg in val_messages if msg is not None]

#     # Select completion messages for validation
#     completion_messages = []
#     seen_prompts = set()
#     for message in val_messages:
#         if message["prompt"] not in seen_prompts:
#             completion_messages.append(message)
#             seen_prompts.add(message["prompt"])
#         if len(completion_messages) >= max_num_generate:
#             break
    
#     # Return the processed data
#     return train_messages, val_messages, completion_messages


def train_probe(probe, train_xs, train_labels, device="cpu") -> BaseProbe:
    if probe == "logistic":
        model = LogisticProbe(normmalize=True, l2_penalty=1.0)
        model.fit(train_xs, train_labels)
    else:
        raise NotImplementedError(f"Invalid probe: {probe}")
    return model 