#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import math
import os
import pathlib
import random
import datasets
import torch
import torch.distributed as dist
import copy
from functools import partial
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from typing import Optional, Dict, Sequence
from util.templator import Qwen2Templator, Llama3Templator, GemmaTemplator
import json

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
)
from trl import DPOConfig
from dpo_trainer import DPOTrainer

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--model_type",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--mask_knowledge",
        action="store_true",
        help="If passed, will mask the knowledge part of the input.",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="The maximum total sequence length (prompt+completion) of each training example.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="Initial dpo beta to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0, help="Ratio of total training steps used for warmup."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="Log the training loss and learning rate every logging_steps steps.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--use_special_tokens",
        action="store_true",
        help=(
            "Use special tokens."
        ),
    )

    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None:
        raise ValueError("Need either a dataset name or a training file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["json", "jsonl"], "`train_file` should be a json/jsonl file."
    return args


def encode_with_messages_format(example, tokenizer, max_seq_length, model_type, context_markups=None, mask_knowledge=False):
    '''
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    '''
    
    messages = example
    if len(messages) == 0:
        raise ValueError('messages field is empty.')
    
    for message in messages:
        if "from" in message:
            message["role"] = "user" if message["from"] == "human" else "assistant"
            message.pop("from")
        if "value" in message:
            message["content"] = message["value"]
            message.pop("value")
            
    def _concat_messages(messages, add_end_splitter=False):
        if model_type.lower()=="qwen":
            template = Qwen2Templator()               
            return template.wrap(messages, force_system_prompt=True, add_end_splitter=add_end_splitter)
        elif model_type.lower()=="llama3":
            template = Llama3Templator()
            return template.wrap(messages, add_end_splitter=add_end_splitter)
        elif model_type.lower() == "gemma":
            template = GemmaTemplator()
            return template.wrap(messages, add_end_splitter=add_end_splitter)
    
    example_text = _concat_messages(messages, add_end_splitter=False).strip()
    
    assert(type(example_text) == str, f"You must pass a string to the tokenizer. example_text: {example_text}")
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
            
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]), return_tensors='pt', max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx+1]["role"] == "assistant":
                # here we also ignore the role of the assistant
                messages_so_far = _concat_messages(messages[:message_idx+1], add_end_splitter=True)
            else:
                messages_so_far = _concat_messages(messages[:message_idx+1], add_end_splitter=True)
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors='pt', 
                max_length=max_seq_length, 
                truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100
            
            if message_end_idx >= max_seq_length:
                break
    
    if mask_knowledge:
        labels = labels.flatten()
        if context_markups is not None:
            context_start = False
            for j, orig_token in enumerate(labels[:]):
                if context_start is False and orig_token == context_markups[0]:
                    context_start = True
                    assert labels[j] == context_markups[0]
                    start_idx = j
                    end_idx = None
                    for k, orig_token_2 in enumerate(labels[start_idx:]):
                        if orig_token_2 == context_markups[1]:
                            end_idx = start_idx + k
                            break
                    if end_idx is None:
                        end_idx =  start_idx + k
                    else:
                        assert labels[end_idx] == context_markups[1]
                    labels[start_idx+1:end_idx] = -100
                    context_start = False

    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }
        
def encode_dpo(
    source,
    tokenizer: transformers.PreTrainedTokenizer,
    max_seq_length=512,
    model_type="llama3",
    context_markups=None, 
    mask_knowledge=False
) -> Dict:
    prompt = source['conversations']
    chosen = source['conversations'] + source['chosen']
    rejected = source['conversations'] + source['rejected']
    prompt_tokens = encode_with_messages_format(prompt, tokenizer, max_seq_length, model_type, context_markups=context_markups, mask_knowledge=mask_knowledge)
    
    chosen_tokens = encode_with_messages_format(chosen, tokenizer, max_seq_length, model_type, context_markups=context_markups, mask_knowledge=mask_knowledge)
    chosen_labels = chosen_tokens['labels']
    chosen_labels[:len(prompt_tokens['input_ids'])] = -100
    
    rejected_tokens = encode_with_messages_format(rejected, tokenizer, max_seq_length, model_type, context_markups=context_markups, mask_knowledge=mask_knowledge)
    rejected_labels = rejected_tokens['labels']
    rejected_labels[:len(prompt_tokens['input_ids'])] = -100
    
    return dict(
        chosen_input_ids=chosen_tokens['input_ids'].tolist(),
        chosen_attention_mask=chosen_tokens['attention_mask'].tolist(),
        chosen_labels=chosen_tokens['labels'].tolist(),
        rejected_input_ids=rejected_tokens['input_ids'].tolist(),
        rejected_attention_mask=rejected_tokens['attention_mask'].tolist(),
        rejected_labels=rejected_tokens['labels'].tolist(),
        prompt_input_ids=prompt_tokens['input_ids'].tolist(),
        prompt_attention_mask=prompt_tokens['attention_mask'].tolist(),
    )

def main():
    args = parse_args()
    # A hacky way to make llama work with flash attention
    # if args.use_flash_attn:
    #     from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
    #     replace_llama_attn_with_flash_attn()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    with logging_redirect_tqdm():
        logger.info(accelerator.state, main_process_only=False)
        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()

        # If passed along, set the training seed now.
        if args.seed is not None:
            set_seed(args.seed)

        if accelerator.is_main_process:
            if args.output_dir is not None:
                os.makedirs(args.output_dir, exist_ok=True)
        
        accelerator.wait_for_everyone()

        # Load pretrained model and tokenizer
        if args.config_name:
            config = AutoConfig.from_pretrained(args.config_name)
        elif args.model_name_or_path:
            config = AutoConfig.from_pretrained(args.model_name_or_path)
        else:
            raise ValueError(
                "You are instantiating a new config instance from scratch. This is not supported by this script."
            )

        if args.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
        elif args.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script."
                "You can do it from another script, save it, and load it from here, using --tokenizer_name."
            )
        
        if args.model_name_or_path:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                low_cpu_mem_usage=args.low_cpu_mem_usage,
            )
            reference = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                low_cpu_mem_usage=args.low_cpu_mem_usage,
            )
        else:
            logger.info("Training new model from scratch")
            model = AutoModelForCausalLM.from_config(config)
            reference = AutoModelForCausalLM.from_config(config)
        
        embedding_size = model.get_input_embeddings().weight.shape[0]
        model.resize_token_embeddings(len(tokenizer))
        reference.resize_token_embeddings(len(tokenizer))

        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True})
        reference.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True})
        
        logger.info(f"Special tokens: {tokenizer.special_tokens_map}")
        context_markups = []
        for token in ["<knowledge>", "</knowledge>"]:
            context_markups.append(tokenizer.convert_tokens_to_ids(token))
        logger.info(f"context_markups: {context_markups}")
        
        if args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
            )
        else:
            data_files = {}
            dataset_args = {}
            if args.train_file is not None:
                data_files["train"] = args.train_file
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                **dataset_args,
            )
        preprocess = partial(
            encode_dpo, 
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length, 
            model_type=args.model_type, 
            context_markups=context_markups if args.use_special_tokens is True else None, 
            mask_knowledge=True if args.mask_knowledge else False
        )
        
        with accelerator.main_process_first():
            train_dataset = raw_datasets["train"].map(preprocess)

        with open("processed.json", "w") as outfile:
            new_data = []
            for idx, item in enumerate(train_dataset):
                new_data.append({"idx": idx, "train_dataset": item, "raw_datasets": raw_datasets["train"][idx]})
            json.dump(new_data[:10], outfile)
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        
        os.environ["WANDB_DISABLED"] = "true"

        output_dir = args.output_dir
        training_arguments = DPOConfig(
            output_dir=output_dir,
            eval_strategy="no",
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            logging_steps=5,
            learning_rate=args.learning_rate,
            fp16 = True,
            beta = args.beta,
            num_train_epochs=args.num_train_epochs,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type=args.lr_scheduler_type,
            max_length=8192,
            max_prompt_length=3072,
            max_target_length=3072,
            gradient_checkpointing=True,
            report_to="tensorboard",
            save_strategy="no",
            save_total_limit=2,
            rpo_alpha=1.0
        )
        
        trainer = DPOTrainer(
            model,
            reference, 
            args=training_arguments,
            train_dataset=train_dataset,
            tokenizer=tokenizer
        )
        
        if list(pathlib.Path(output_dir).glob("checkpoint-*")):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
        
        # Save the final model
        if args.output_dir is not None:
            trainer.save_model(args.output_dir)
            logger.info(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()