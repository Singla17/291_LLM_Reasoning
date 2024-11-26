import os
os.environ["CUDA_VISIBLE_DEVICES"]="4"

import pickle
import time

# # import GPUtil
from datasets import load_dataset
from evaluate import load
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, TrainingArguments, BitsAndBytesConfig, \
     Trainer, AutoConfig, DataCollatorWithPadding,AutoModelForSequenceClassification
from QSTConfig import QSTConfig
from typing import Dict
from modeling_opt_qst import QSTOPTForSequenceClassification

import warnings

# Filter out the specific warning
warnings.filterwarnings("ignore",
                        message="Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.")

torch.backends.cuda.matmul.allow_tf32 = True


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        # output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        # output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        # output_embeddings_data[-num_new_tokens:] = output_embeddings_avg


def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        if "blackbone" in name:
            param.requires_grad = False
        if "model.layer" in name:
            param.requires_grad = False
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.bits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )

task_to_keys = {
    "rte": ("premise", "hypothesis"),
    "boolq": ("question", "passage"),
    "cb": ("premise", "hypothesis")
}

SUPERGLUE_TASKS = ["boolq", "rte"]
DEFAULT_PAD_TOKEN = "[PAD]"


def train(task, parameters):
    batch_size = parameters[task]["batch_size"]
    model_checkpoint = parameters["model_checkpoint"]
    epoch = parameters[task]["epoch"]
    r = parameters[task]["r"]
    alpha_r = parameters[task]["alpha_r"]
    learning_rate = parameters[task]["learning_rate"]
    max_len = parameters[task]["max_seqlen"]
    qst_checkpoint = parameters['qst_checkpoint']

    actual_task = "mnli" if task == "mnli-mm" else task

    print(f"Loading dataset for task: {actual_task}")
    dataset = load_dataset("super_glue", task)
    metric = load('super_glue', task)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, max_length=max_len)

    num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    LLM = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, 
                                                         quantization_config=quant_config, torch_dtype=torch.bfloat16,
                                                         num_labels=num_labels,device_map="auto")

    if tokenizer._pad_token is None:
        # smart_tokenizer_and_embedding_resize(
        #     special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
        #     tokenizer=tokenizer,
        #     model=LLM,
        # )
        tokenizer.pad_token = tokenizer.eos_token

    sentence1_key, sentence2_key = task_to_keys[task]

    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True, padding=True, )
        return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, padding=True, )

    encoded_dataset = dataset.map(preprocess_function, batched=True)

    if task == "boolq":
        encoded_dataset['train'] = encoded_dataset['train'].shuffle(seed=42).select(range(2500))
        encoded_dataset['validation'] = encoded_dataset['validation'].shuffle(seed=42).select(range(500))


    # config = AutoConfig.from_pretrained(model_checkpoint)
    # config.hidden_size = 64



    validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
    num_samples = len(encoded_dataset[validation_key])
    num_batches = num_samples // batch_size
    valid_samples = num_batches * batch_size

    encoded_dataset[validation_key] = encoded_dataset[validation_key].select(range(valid_samples))

    config = AutoConfig.from_pretrained(model_checkpoint)
    config.pad_token_id = config.eos_token_id

    LLM.config.torch_dtype = torch.float32

    qst_config = QSTConfig(
        add_layer_norm_before_adapter=False,
        add_layer_norm_after_adapter=True,
        r=r,
        alpha_r=alpha_r,
        dropout=0.1,
        activation="swish",
        fan_in_fan_out=False,
        peft_hidden_size=16  # here
    )

    model = QSTOPTForSequenceClassification(LLM, config, qst_config)
    model.config.pad_token_id = tokenizer.pad_token_id
    # LLaMA tokenizer may not have correct special tokens set.
    # Check and add them if missing to prevent them from being parsed into different tokens.
    # Note that these are present in the vocabulary.
    # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.

    if qst_checkpoint:
        print("Loading QST from checkpoint.")
        model.load_qst_state(qst_checkpoint)
    else:
        print(f'initing QST modules...')

    # use 16bit as the compute data type, comment it if you want to use 32bit
    for name, module in model.named_modules():
        if 'qst' or 'z' or 'downsample' or 'upsample' in name:
            module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if task != "stsb":
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = predictions[:, 0]
        return metric.compute(predictions=predictions, references=labels)

    train_args = TrainingArguments(
        f"{model_checkpoint}-QST-{task}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epoch,
        weight_decay=0.01,
        warmup_ratio=0.06,
        lr_scheduler_type="linear",
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        logging_dir=f"{model_checkpoint}-QST-{task}-log",
        logging_strategy="epoch",
        bf16=True
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    start_time = time.time()
    # memory_callback = MemoryLoggingCallback()

    trainer = Trainer(
        model,
        train_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset[validation_key],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        # callbacks=[memory_callback]
    )

    trainer.train()
    end_time = time.time()
    results = trainer.evaluate()

    return results, trainer.state.log_history, (end_time - start_time)

parameters = {
    "model_checkpoint": 'facebook/opt-1.3b',
    "qst_checkpoint": None,
    "rte": {"batch_size": 16, "epoch": 7, "r": 16, "alpha_r": 16, "max_seqlen": 512,
            "learning_rate": 5E-04},
    "boolq": {"batch_size": 16, "epoch": 7, "r": 16, "alpha_r": 16, "max_seqlen": 512,
                "learning_rate": 4E-04},
    "axb": {"batch_size": 16, "epoch": 7, "r": 16, "alpha_r": 16, "max_seqlen": 512,
                "learning_rate": 4E-04},
    "cb": {"batch_size": 16, "epoch": 7, "r": 16, "alpha_r": 16, "max_seqlen": 512,
                "learning_rate": 4E-04},
}

result_dict = {}
for task in SUPERGLUE_TASKS:
    if task == "qnli":
        continue

    result_dict[task] = {}
    result, log, train_time = train(task, parameters)

    values = []
    for elem in log:
        if "eval_loss" not in elem.keys():
            continue
        if task == "cola":
            values.append(elem['eval_matthews_correlation'])
        elif task == "stsb":
            values.append(elem['eval_pearson'])
        else:
            values.append(elem['eval_accuracy'])

    best_acc = max(values)
    result_dict[task]["acc"] = best_acc
    result_dict[task]["time"] = train_time
    # result_dict[task]["memory_usage"] = memory_usage

    print(f"Task:{task}: Best acc {best_acc}, Total training time {train_time}")

model_name = os.path.basename(parameters["model_checkpoint"])
with open(f"superglue_qst_{task}_{model_name}_{4}.pickle", 'wb') as f:
    pickle.dump(result_dict, f)