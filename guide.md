# LANGUAGE MODELING

## Requirements
- CUDA 11.0 or newer version
- NCCL 2.10 or newer version
- Python: Only verify on python 3
- PyTorch: 1.8.0 or newer version

## Library
- torch 2.0.0
- transformers 4.28.1
- datasets 2.11.0
- accelerate 0.18.0

## Installation
```bash
pip install -r requirements.txt
```

## Datasets
- English dataset for training Language model: [Click here](https://huggingface.co/datasets?task_categories=task_categories:text-generation&task_ids=task_ids:language-modeling&language=language:en&sort=downloads)
- Japanese dataset for training Language model: [Click here](https://huggingface.co/datasets?task_categories=task_categories:text-generation&task_ids=task_ids:language-modeling&language=language:ja&sort=downloads)

If you want to use your own dataset, refer to [this](https://huggingface.co/docs/datasets/nlp_load) link.

## Training

### Step 1. Model name and dataset name

Set the model name and dataset name in 'training.py' file.\
Dataset name you can find in the link above.

```python
# training.py
...

model_name = 'bigscience/bloom-7b1' # change to your model name here
dataset_name = 'openwebtext' # change to your dataset name here

...
```

### Step 2. Config Trainer

You can set batch size, learning rate, epochs, etc. in 'training.py' file.

```python
# training.py
...

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

...
```

The final batch size will be:

```
batch_size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus

Example:
per_device_train_batch_size = 1
gradient_accumulation_steps = 128
num_gpus = 8

batch_size = 1 * 128 * 8 = 1024
```

You can set optimizer "adafactor", default is "adamw". AdamW is better than Adafactor, but Adafactor requires less GPU memory.

```python
# training.py
...

args = TrainingArguments(
    ...
    optim="adafactor",
    ...
)

...
```

### Step 3. Config Accelerator

Run this command to config accelerator and config like this:

```bash
accelerate config
```

```
---------------------------------------------------------------------------------------------------------------------------------------------------
In which compute environment are you running?
This machine
---------------------------------------------------------------------------------------------------------------------------------------------------
Which type of machine are you using?
multi-GPU
How many different machines will you use (use more than 1 for multi-node training)? [1]: 1
Do you wish to optimize your script with torch dynamo?[yes/NO]:yes
---------------------------------------------------------------------------------------------------------------------------------------------------
Which dynamo backend would you like to use?
inductor
Do you want to customize the defaults sent to torch.compile? [yes/NO]: yes
---------------------------------------------------------------------------------------------------------------------------------------------------
Which mode do you want to use?
default
Do you want the fullgraph mode or it is ok to break model into several subgraphs? [yes/NO]:
Do you want to enable dynamic shape tracing? [yes/NO]:
Do you want to use DeepSpeed? [yes/NO]: yes
Do you want to specify a json file to a DeepSpeed config? [yes/NO]:
---------------------------------------------------------------------------------------------------------------------------------------------------
What should be your DeepSpeed's ZeRO optimization stage?
2
---------------------------------------------------------------------------------------------------------------------------------------------------
Where to offload optimizer states?
none
---------------------------------------------------------------------------------------------------------------------------------------------------
Where to offload parameters?
none
How many gradient accumulation steps you're passing in your script? [1]: 128
Do you want to use gradient clipping? [yes/NO]:
Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]:
How many GPU(s) should be used for distributed training? [1]:8
---------------------------------------------------------------------------------------------------------------------------------------------------
Do you wish to use FP16 or BF16 (mixed precision)?
fp16
```

### Step 4. Run training

Run this command to start training:

```bash
accelerate launch training.py
```

If you want continue training, change this line in 'training.py' file. Then run the command above.

```python
# training.py

...

trainer.train(resume_from_checkpoint=True)
```