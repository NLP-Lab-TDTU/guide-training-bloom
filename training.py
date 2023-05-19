import os

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)

model_name = 'bigscience/bloom-7b1' # change to your model name here
dataset_name = 'openwebtext' # change to your dataset name here

dataset_dict = load_dataset(dataset_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    examples = tokenizer.eos_token.join([ex for ex in batch['content']]) # packing trick
    output = tokenizer(
        examples,
        truncation=True,
        max_length=1024, # change to your max length here
        stride=512, # change to your stride here
        return_overflowing_tokens=True,
    )
    return output

tokenized_datasets = dataset_dict.map(tokenize, batched=True, num_proc=64, remove_columns=dataset_dict['train'].column_names)
tokenized_datasets = tokenized_datasets.shuffle(seed=42)
dict_dataset = tokenized_datasets['train'].train_test_split(test_size=100, seed=42)
train_dataset = dict_dataset['train']
eval_dataset = dict_dataset['test']

model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=False)

args = TrainingArguments(
    output_dir="checkpoints", # change to your output dir here
    prediction_loss_only=True,
    per_device_train_batch_size=1, # change to your batch size here
    per_device_eval_batch_size=1, # change to your batch size here
    evaluation_strategy="steps",
    logging_steps=1,
    eval_steps=10,
    gradient_accumulation_steps=128, # change to your gradient accumulation steps here
    num_train_epochs=3, # change to your num train epochs here
    weight_decay=0.01,
    warmup_steps=0,
    learning_rate=1e-5,
    save_steps=10,
    fp16=True,
    auto_find_batch_size=True, # if you set this to True, and per_device_train_batch_size > 1, it will try to find the best batch size for you
    dataloader_num_workers=32,
    save_total_limit=20,
    # optim="adafactor",
    ddp_find_unused_parameters=False,
    gradient_checkpointing=True,
    report_to=[],
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train(resume_from_checkpoint=False)