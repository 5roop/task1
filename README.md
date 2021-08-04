# task1

First I relabeled the data. Since the task was classifying hate speech, I set labels  'Acceptable speech' to 0, all other labels presented some sort of hate speech and were labeled 1.

# Simpletransformers
I started with monolingual dataset `lgbt-en.train.tsv`. My first attempt was with simpletransformers library.

I prepared a classification model as follows:

```
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