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

## Results - roberta base
|  language | accuracy  |  f1 |
|---|---|---|
|en   |0.843   |  0.693 |
|sl|0.597|0.6048|
|hr| 0.780| 0.8336|


Other than roberta I also tried to find the most suitable checkpoint. For English it would seem that `distilbert-base-uncased-finetuned-sst-2-english` is the natural best fit, but I also tried it on the other languages.

## Results - distilbert 
|  language | accuracy  |  f1 |
|---|---|---|
|en   | 0.769  |   0.475|
|sl|0.535|0.542|
|hr| 0.6996| 0.7799|

The training took about $1~\mathrm{minute}$ for all languages.


I found a `bert` checkpoint on huggingface, `"IMSyPP/hate_speech_slo"`. Interestingly it did not perform much better on slovenian data:


|  language | accuracy  |  f1 |
|---|---|---|
|sl|0.558|0.53|


# HuggingFace

I tried training a pretrained model from different checkpoints:

* bert-base-uncased
* distilbert-base-uncased-finetuned-sst-2-english
* roberta-base

The optimization was attempted with the following approach, based on the HuggingFace documentation:

```python
import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification


checkpoint = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = list(train.text.values)
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")


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

# Fasttext

When researching Fasttext we noticed that the input seems to be a saved file with a specific format:

```
__label__Offensive Lorem Ipsum dolor sit amet.
```
We prepared a few helper functions that sorted this preprocessing for us. Again, all labels have beed downsampled to either `Acceptable` or `Offensive`.

At first glance the API is easier to work with than HuggingFace. In the first try precision and recall were equal at 0.54. After increasing the number of epochs a bit those metrics rose to 0.62, they were still equal, and did not rose even after increasing the number of epoch to ridiculous numbers.

Training times were way shorter than both previously tried methods. 100 epochs only needed 515 ms.

I tried improving the statistics by fiddling with the learning rate and n-gram settings, but never reached precisions more than 0.62.

Aformentioned tests were performed on Slovenian data, which might be harder to grasp than English. When trying the same 'tricks' for English, better results were obtained. Instead of 0.62 the resulting precision was 0.75. When using 2-grams instead of 3-grams, precision rose marginally, and when only using 1-grams, it marginally fell.

In order to compare the metrics directly I also computed the accuracies and f_1 scores for the true and predicted labels.

## Results:
Hyperparameters used: ` epoch=1000, lr=0.05`.
|  language | accuracy   | f_1 score |
|---|---| --- |
|  en |  0.744 |  0.640|
|sl| 0.62 |0.619|
|hr|0.72|0.69|

Training times was $<10~\mathrm{s}$ for all training sessions.

# `sk-learn` toolbox

Scikit learn also offers a few options for working with text data. Following a few suggestions on their webpage I prepared a `CountVectorizer` that performed text tokenization for the input text. Features were then extracted using `tf-idf` method with sklearn tools,  and a SVM classifier was trained with these inputs.

When evaluating the classifier training data had to be processed with the same count vectorizer and tf-idf transformer and only then it was fed to the SVM classifier. Count vectorizer was set to use 1-, 2- and 3-grams. Other hyperparameters were left at default values.

## Results
|  language | accuracy   | f_1 score |
|---|---| --- |
|  en |  0.757 |  0.220|
|sl| 0.578 |0.505|
|hr|0.704|0.810|

Execution of the whole pipeline took about 10 s.

# Concluding remarks

* After initial problems with the virtual machine were resolved, it worked like a charm. I found the VS Code ssh functionality incredibly useful; I started a jupyter lab server on the remote machine and VS Code took care of ssh tunneling without any complications, the same goes for git integration. This way it was hardly noticeable that I was running the entire process on a remote machine.
* Slovenian data consistently performs worse. This might be due to the low quality of the input data, I encountered weird, Hojsian punctuation style (e.g. misplaced .periods ,and commas). Additional preprocessing should no doubt improve classification accuracy, but since this was not the purpose of this exercise, it was not performed.
* HuggingFace seems versatile, but proved difficult to handle. simpletransformers on the other hand offer a much simpler API with no clutter and quicker results.
* Fasttext requires a specific formatting and so far I was unable to get it to work with input other than in file format, which is a bit cumbersome, but it is incredibly fast and the results obtained are not bad at all.
* sklearn NPL toolbox is a bit low level and it would be nice to have a wrapper around the individual parts of the pipeline, but once all the parts of the puzzle are in place, it performs decently enough, not to mention that it offers the user the whole palette of classifiers with the full power of their customizability.
