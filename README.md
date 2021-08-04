# task1

First I relabeled the data. Since the task was classifying hate speech, I set labels  'Acceptable speech' to 0, all other labels presented some sort of hate speech and were labeled 1.

# Simpletransformers
I started with monolingual dataset `lgbt-en.train.tsv`. My first attempt was with simpletransformers library.

I prepared a classification model as follows:

```python
from simpletransformers.classification import ClassificationModel

model_args = {
    "num_train_epochs": 5,
    "learning_rate": 1e-5,
    "overwrite_output_dir": True,
    "train_batch_size": 40
}

model = ClassificationModel(
    "roberta", "roberta-base", use_cuda=True,
    args=model_args
    
)
```

and got a confusion matrix: 'tp': 181, 'tn': 672, 'fp': 68, 'fn': 96
I also calculated accuracy and F_1 score:
```
Accuracy:  0.8387413962635202
F1 score:  0.688212927756654
```

With distilbert I was able to get the accuracy at most to 0.78 with F_1 score of 0.5.

# HuggingFace

I tried training a pretrained model from different checkpoints:

* bert-base-uncased
* distilbert-base-uncased-finetuned-sst-2-english
* roberta-base

The optimization looked as follows:

```python
import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification

# Same as before
checkpoint = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = list(train.text.values)
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

# This is new
batch["labels"] = torch.tensor(list(train.labels.values))

optimizer = AdamW(model.parameters())
loss = model(**batch).loss
loss.backward()
optimizer.step()
```

Unfortunately, all attempts raised
```python
RuntimeError: [enforce fail at CPUAllocator.cpp:71] . DefaultCPUAllocator: can't allocate memory: you tried to allocate 60637052928 bytes. Error code 12 (Cannot allocate memory)


```