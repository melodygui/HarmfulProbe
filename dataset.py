from torch.utils.data import Dataset
import datasets
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer
from typing import Dict
import torch
import numpy as np
from tqdm import tqdm
import json
import random
import csv
random.seed(0)

class CircuitBreakerDataset(Dataset):
    
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        num_examples,
        model_name_or_path,
    ):
        super(CircuitBreakerDataset, self).__init__()

        self.model_name_or_path = model_name_or_path.lower()
        self.max_length = 1024

        one_shot_template = "{user_tag}{instruction}{assistant_tag}<SEPARATOR>{response}"

        # ============= Model and Chat Template Config  ============= #
        user_tag="<|start_header_id|>user<|end_header_id|>\n\n"
        assistant_tag="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        switch_select = [0, 1]
        sep_token = ""

        self.user_tag = user_tag
        self.assistant_tag = assistant_tag
        self.sep_token = sep_token

        # ============= Benign Set ============= #
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
        benign = []
        for convo in ds:
            messages = convo["messages"]
            if len(messages) < 2: continue 

            # randombly switch between one shot and few shot 
            switch = np.random.choice(switch_select)
            if switch == 0:
                formatted_input = tokenizer.apply_chat_template(messages, tokenize=False).replace(tokenizer.bos_token, "")
            elif switch == 1:
                formatted_input = one_shot_template.format(
                    user_tag=user_tag, assistant_tag=assistant_tag,
                    instruction="", response=messages[1]["content"])  

            benign.append(formatted_input)

            if len(benign) > num_examples:
                break
        
        self.harmless_set = benign 
        random.shuffle(self.harmless_set)
        print("benign[0]: ", benign[0])
        print("Harmless set's length: ", len(self.harmless_set))

        # ======================= Borderline Retain ======================= #
        # from https://github.com/paul-rottger/exaggerated-safety
        with open(f'data/xstest_v2_completions_gpt4_gpteval.csv', newline='') as f:
            data = [dict(row) for row in csv.DictReader(f)]
            data = [row for row in data if row['final_label'] == "1_full_compliance"]
        
        borderline_benign = []
        for i, d in enumerate(data * 50):
            switch = np.random.choice(switch_select)
            if switch == 0:
                formatted_input = one_shot_template.format(
                    user_tag=user_tag, assistant_tag=assistant_tag,
                    instruction=d['prompt'], response=d['completion'])
            elif switch == 1:
                formatted_input = one_shot_template.format(
                    user_tag=user_tag, assistant_tag=assistant_tag,
                    instruction="", response=d['completion'])
            
            borderline_benign.append(formatted_input)

        self.harmless_set += borderline_benign
        random.shuffle(self.harmless_set)
        print("borderline_benign[0]", borderline_benign[0])
        print("Harmless set length:", len(self.
        ))

        # ======================= Circuit Breaker ======================= #
        with open("data/circuit_breakers_train.json") as file:
            dataset = json.load(file)
        circuit_breaker_orig = []

        for i, d in tqdm(enumerate(dataset)):
            cb_output = d['output']
            switch = np.random.choice(switch_select)
            if switch == 0:
                formatted_input = one_shot_template.format(
                    user_tag=user_tag, assistant_tag=assistant_tag,
                    instruction=d['prompt'], response=cb_output)
            elif switch == 1:
                formatted_input = one_shot_template.format(
                    user_tag=user_tag, assistant_tag=assistant_tag,
                    instruction="", response=cb_output)
            
            circuit_breaker_orig.append(formatted_input)
        
        self.circuit_breaker_orig = circuit_breaker_orig
        random.shuffle(self.circuit_breaker_orig)
        print("circuit_breaker_orig[0]", circuit_breaker_orig[0])
        print("Harmful (citcuit breaker) set length:", len(self.circuit_breaker_orig))

        # ======================= Val ======================= #
        with open("data/circuit_breakers_val.json") as file:
            dataset = json.load(file)
        val_orig = []
        for i, d in tqdm(enumerate(dataset)):
            val_orig.append(one_shot_template.format(
                user_tag=user_tag, assistant_tag=assistant_tag,
                instruction=d['prompt'], response=d['output']))

        self.val_orig = val_orig
        self.tokenizer = tokenizer

        def __len__(self):
            return min(len(self.harmless_set), len(self.circuit_breaker_orig))
        
        def __getitem__(self, i) -> Dict[str, torch.Tensor]:
            """
               Returns a dictionary with tokenized inputs for the circuit breaker, 
               retain, and validation data, along with their corresponding 
               attention masks.
            """
            harmless_set = self.harmless_set[i]
            citcuit_breaker_orig = self.circuit_breaker_orig[i]
            val_orig = self.val_orig[i % len(self.val_orig)]

            cb_tokenized_kwargs = dict(max_length=512, padding='max_length', truncation=True, return_tensors="pt")
            tokenize_kwargs = dict(max_length=1024, padding="max_length", truncation=True, return_tensors="pt")

            # =========== Circuit Breaker Inputs ===========
            # === split to [request, response] shape [512,512] to support different mask configs ===
            cb_request, cb_response = circuit_breaker_orig.split('<SEPARATOR>')
            self.tokenizer.padding_side = "left"
            tokenized_request_circuit_breaker = self.tokenizer(cb_request, **cb_tokenized_kwargs)
            self.tokenizer.padding_side = "right"
            response_tokenized_circuit_breaker = self.tokenizer(cb_response, add_special_tokens=False, **cb_tokenized_kwargs)
            self.tokenizer.padding_side = "left"

            combined_input_ids_circuit_breaker = torch.cat([tokenized_request_circuit_breaker["input_ids"], response_tokenized_circuit_breaker["input_ids"]], dim=1)
            combined_attention_mask_circuit_breaker = torch.cat([tokenized_request_circuit_breaker["attention_mask"], response_tokenized_circuit_breaker["attention_mask"]], dim=1)

            # =========== Val Inputs ===========
            tokenized_inputs_val = self.tokenizer(val_orig.replace('<SEPARATOR>', self.sep_token), **tokenize_kwargs)

            return dict(
            input_ids_circuit_breaker=combined_input_ids_circuit_breaker,
            attention_mask_circuit_breaker=combined_attention_mask_circuit_breaker,
            input_ids=tokenized_inputs_retain["input_ids"],
            attention_mask=tokenized_inputs_retain["attention_mask"],
            input_ids_val=tokenized_inputs_val["input_ids"],
            attention_mask_val=tokenized_inputs_val["attention_mask"],
        )

# ================= construct the circuit breaker dataset ================= #
model_name_or_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        #cache_dir=training_args.cache_dir,
        #model_max_length=training_args.model_max_length,
        padding_side="left",
        #use_fast="LlamaForCausalLM" not in config.architectures,
    )
train_dataset = CircuitBreakerDataset(tokenizer, num_examples=10000, model_name_or_path=model_name_or_path)
print("TRAIN LEN: ", len(train_dataset))