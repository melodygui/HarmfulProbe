import copy
from typing import Literal, Tuple 

import torch
import tqdm
import numpy as np
import torch.nn.functional as F 
from numpy.typing import NDArray
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from baukit import TraceDict

def get_tokens(
        tokenizer: PreTrainedTokenizerBase,
        message: dict[str, str],
        system_prompt: str = "",
        max_length: int = 1024,
        max_assistant_tokens: int = 32,
        truncation: bool = True,
):
    history = [] if "conversation_history" not in message or message["conversation_history"] is None else copy.deepcopy(message["conversation_history"])
    history.extend([
        {"role": "user", "content": message["prompt"]},
        {"role": "assistant", "content": message["completion"]},
    ])
    if system_prompt:
        history = [{"role": "system", "content": system_prompt}] + history

    prompt = tokenizer.apply_chat_template(history, tokenize=False)
    if message["completion"]:
        # TODO this does not work for short answers, e.g. for MMLU ... 
        # prefix = prompt.split(message["answer"])[0].strip()
        history_ = copy.deepcopy(history)
        history_.pop()
        history_.append({"role": "assistant", "content": ""})
        prompt_ = tokenizer.apply_chat_template(history_, tokenize=False)
        num_prefix_tokens = len(tokenizer(prompt_, add_special_tokens=False)["input_ids"]) - 2 # remove the last two tokens for the empty answer TODO does this hold for all tokenizers?? (I think so)
    else:
        prefix = prompt.strip()
        num_prefix_tokens = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])

    inputs = tokenizer(prompt, return_tensors="pt", truncation=False, add_special_tokens=False)

    # truncate right (assistant response)
    if truncation and max_assistant_tokens > 0:
        for key in inputs:
            inputs[key] = inputs[key][:, :num_prefix_tokens + max_assistant_tokens]
    num_completion_tokens = len(inputs["input_ids"][0]) - num_prefix_tokens

    # truncate left (history)
    if truncation and max_length > 0:
        for key in inputs:
            inputs[key] = inputs[key][:, -max_length:]
        num_completion_tokens = min(max_length, num_completion_tokens)

    return inputs, num_completion_tokens


RepresentationType = Literal[
    "hiddens", "pre-attn", "queries", "keys", "values", "heads", "mlp", "post-attn"
]

@torch.no_grad()
def compute_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    system_prompt: str = "",
    representation: RepresentationType = "hiddens",
    ctx_len: int = 16,
    max_assistant_tokens: int = 32,
    max_input_len = 1024,
    truncation = True,
    max_messages = -1, 
    output_dtype = torch.float32,
    return_metrics: bool = False,
) -> Tuple[list[NDArray], NDArray, dict]:
    
    device = next(model.parameters()).device
    activations = []
    labels = []

    metrics = {
        "ppl": [],
        "ce_loss": [],
    } 

    trace_modules = []
    target_key = None
    error_msg = None 

    if max_messages < 0:
        max_messages = len(messages)
    
    with TraceDict(model, trace_modules) as trace:
        with tqdm.tqdm(total=min(len(messages), max_messages), desc="Activations", smoothing=0.01) as pbar:
            for msg in messages:
                if len(activations) >= max_messages:
                    break 

                inputs, num_completion_tokens = get_tokens(
                    tokenizer, 
                    msg,
                    system_prompt=system_prompt,
                    max_length=max_input_len,
                    max_assistant_tokens=max_assistant_tokens,
                    truncation=truncation,
                )
                inputs = inputs.to(device)
                outputs = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=True)

                if return_metrics:
                    if len(outputs.logits[0, -num_completion_tokens:]) > 1:
                        ce_loss = F.cross_entropy(
                            outputs.logits[0, -num_completion_tokens:-1],
                            inputs["input_ids"][0, -num_completion_tokens+1:]
                        ).to(torch.float32)
                        ppl = torch.exp(ce_loss) # calculates perplexity score
                        metrics["ppl"].append(ppl.item())
                        metrics["ce_loss"].append(ce_loss.item())
                    else:
                        metrics["ppl"].append(np.nan) # nan = not a number
                        metrics["ce_loss"].append(np.nan)

                if representation == "hiddens":
                    reps = torch.cat([outputs.hidden_states[1:]]).squeeze(1)
                else:
                    raise ValueError(f"Unknown representation: {representation}")
                
                reps = reps[:, -num_completion_tokens:-1]

                if ctx_len > 0:
                    reps = reps[:, :ctx_len]

                activations.append(reps.cpu().to(output_dtype).transpose(0, 1))
                labels.append(msg["label"])
                pbar.update(1)
        
    labels = torch.tensor(labels, dtype=torch.long)
    if return_metrics:
        return activations, labels, metrics
    return activations, labels 