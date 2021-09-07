# Summary

The objective was performing a series of practice binary classification tasks on [FRENK](https://www.clarin.si/repository/xmlui/handle/11356/1433) LGBT dataset in three languages (HR, EN and SL). Tools used:
* `fasttext` 
* `simpletransformers`
* `scikit learn` NLP toolbox
  
## Results

#### Fasttext:
|  language | accuracy   | macro F1 |
|---|---| --- |
|  en |  0.744 |  0.640|
|sl| 0.62 |0.619|
|hr|0.72|0.69|

#### `sk-learn` char n-grams:
|  language | accuracy  |  macro F1 |
|---|---|---|
|en| 0.76 | 0.233 |
|sl| 0.616 | 0.539 |
|hr| 0.736 | 0.83 |

#### Stratified dummy classifier:

|  language | accuracy  |  macro F1 |
|---|---|---|
|en| 0.624 | 0.308 |
|sl| 0.498 | 0.52 |
|hr| 0.545 | 0.65 |

#### `simpletransformers`

All the models were finetuned on lgbt domain only and with the same hyperparameters.


|model name| model type | language | accuracy | macro F1|
|---|---|---|---|---|
|IMSyPP/hate_speech_slo|bert|sl|0.579|0.579|
|EMBEDDIA/sloberta|camembert|sl|0.730|0.729|
|EMBEDDIA/sloberta|roberta|sl|Error|Can't load tokenizer for 'EMBEDDIA/sloberta'|
|EMBEDDIA/crosloengual-bert|bert|sl|0.687|0.686|
|EMBEDDIA/crosloengual-bert|bert|hr|0.844|0.829|
|classla/bcms-bertic|electra|hr|0.849|0.832|
|xlm-roberta-base|xlm-roberta|sl|Error|'xlm-roberta'|
|xlm-roberta-base|xlm-roberta|hr|Error|'xlm-roberta'|
|xlm-roberta-base|xlm-roberta|en|Error|'xlm-roberta'|
|xlm-roberta-large|xlm-roberta|en|Error|'xlm-roberta'|
|roberta-base|roberta|en|0.850|0.802|









# task1 chronological report

First I relabeled the data. Since the task was classifying hate speech, I set labels  'Acceptable speech' to 0, all other labels presented some sort of hate speech and were labeled 1. Should I redo this task from scratch I would wait until it was clear this was needed.

In all tests training was performed on `*.train.tsv` datasets and `*.test.tsv` files were used for evaluation.

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

and got a confusion matrix: `'tp': 181, 'tn': 672, 'fp': 68, 'fn': 96`.

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

## Results - `distilbert` 
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

## Results -  `sangrimlee/bert-base-multilingual-cased-nsmc`
Without much deliberation I also windowshopped a bit and found the aforementioned checkpoint that I tested on all three languages.

|  language | accuracy  |  f1 |
|---|---|---|
|en   |  0.794 |   0.584   |
|sl|   0.657  |  0.668  |
|hr|    0.787 |   0.836  |

Training took around 2 minutes, significantly longer than previous examples.

I also wanted to try  `unitary/multilingual-toxic-xlm-roberta`, but couldn't get it to work properly (training phase worked OK, but upon evaluating it crashed due to `TypeError`. I suspect the fact that I could not set the appropriate model type when initialising the classifier, because of a warning `You are using a model of type xlm-roberta to instantiate a model of type roberta.`, but I could not figure out the proper model type that would not raise this warning or even worse, a `KeyError`.)
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

# Multilabel classification

## simpletransformers


The nature of the problem demanded I add `num_labels=6`, but precisely in the right spot, otherwise it would not work. Because of an unknown reason training with option `use_cuda=True` kept raising `RuntimeError` messages, so the training took a bit longer than previously (I estimate an increase of at least two orders of magnitude). I used `roberta-base` as it yielded better results in the binary classification tests.

After leaving it running and checking it in the evening I can finally report the following scores:


|  language | accuracy  |  f1 |
|---|---|---|
|en| 0.814 | 0.33 |
|sl| 0.429 | 0.153 |
|hr| 0.645 | 0.29 |


## fasttext

The library performed admirably and achieved suprisingly high accuracies for such a high number of labels. Data preprocessing was modified to accomodate new labels.

|  language | accuracy  |  f1 |
|---|---|---|
|en| 0.705 | 0.228 |
|sl| 0.47 | 0.233 |
|hr| 0.605 | 0.349|

## `sk-learn` toolbox

No new suprises were waiting for me when generalising to full label set. We see the metrics dip a bit compared to binary classification tests, as we would expect.


|  language | accuracy  |  f1 |
|---|---|---|
|en| 0.739 | 0.171 |
|sl| 0.436 | 0.11 |
|hr| 0.612 | 0.29 |

# Concluding remarks

* After initial problems with the virtual machine were resolved, it worked like a charm. I found the VS Code ssh functionality incredibly useful; I started a `jupyter lab` server on the remote machine and VS Code took care of ssh tunneling without any complications, the same goes for git integration. This way it was hardly noticeable that I was running the entire process on a remote machine. Once I got the IJS VPN working as well, I finally felt ready for working from home. Every lockdown I learn something new...
* Slovenian data consistently performs worse. This might be due to the low quality of the input data, I encountered weird, Hojsian punctuation style (e.g. misplaced .periods ,and commas [sic]), misspellings like "lejzbika", "muslimanska vira", "gdo nebi", "buzarantskega turizma"... as well as many emojis and links to websites. I did not examine the whole dataset exhaustively but I did not see many such instances in the other two datasets.
 Additional preprocessing should no doubt improve classification accuracy, but since this was not the purpose of this exercise, it was not performed.
* `HuggingFace` seems versatile, but proved difficult to handle. Even when I could tackle the API issues and got things to run, weird performance issues surfaced and sabotaged the tests.
* `simpletransformers` on the other hand offer a much simpler API with no clutter and quicker results. HF model hub offers snippets for using the models in HF, but to use them in `simpletransformers` not only the checkpoint but the model type is needed, and this can sometimes be an annoying guessing game at best.
* On non-binary classification `simpletransformers` needed an absurd amount of time, but at least on the English dataset outperformed any other method.
* `Fasttext` requires a specific formatting and so far I was unable to get it to work with input other than in file format, which is a bit cumbersome, but it is incredibly fast and the results obtained are not bad at all.
* `sklearn` NPL toolbox is a bit low level and it would be nice to have a wrapper around the individual parts of the pipeline, but once all the parts of the puzzle are in place, it performs decently enough, not to mention that it offers the user the whole palette of classifiers with the full power of their customizability.
    * After tokenization many classifiers can be used here, all of them could be further optimized with grid search for optimal hyperparameters, which exponentially increases the complexity of the problem, so at this stage this was not performed, but it should be mentioned that it could be done and that the results presented here can be improved a bit. Maybe a follow-up with a systematic `AutoML` optimisation would be interesting.
* Due to the imbalance in the dataset it might be wise to report different metrics than accuracy. Perhaps Cohen's kappa coefficient or Matthews Correlation Coefficient?
* The results presented correspond to the `lgbt` dataset. It could easily be recalculated for the `migrants` dataset as well, but I did not combine the two datasets and train classifiers on the resulting conglomerate. Please advise if this should be done.
* Only in my final tests, when I was certain how to setup the pipeline, was the pipeline run with a bit more automatization. In the future, especially with the expected consistent input formats, I expect to be able to run the experiments with much less human intervention and automatic result reporting.



# Addendum

## Dummy classifiers

As a dummy standard against which we compare our results a `dummy classifier` was introduced. All but one strategy was used and the dummy results are as follows:

- Strategy: `most_frequent`

|  language | accuracy  |  f1 |
|---|---|---|
|en| 0.728 | 0.0 |
|sl| 0.432 | 0.0 |
|hr| 0.651 | 0.789 |

- Strategy: `prior`

|  language | accuracy  |  f1 |
|---|---|---|
|en| 0.728 | 0.0 |
|sl| 0.432 | 0.0 |
|hr| 0.651 | 0.789 |


- Strategy: `uniform`

|  language | accuracy  |  f1 |
|---|---|---|
|en| 0.485 | 0.35 |
|sl| 0.534 | 0.564 |
|hr| 0.515 | 0.576 |

- Strategy: `stratified`
  
|  language | accuracy  |  f1 |
|---|---|---|
|en| 0.624 | 0.308 |
|sl| 0.498 | 0.52 |
|hr| 0.545 | 0.65 |

​The available, but unused strategy was `constant`, which always predicts a constant value. If the value givent to it would be the most common value, the accuracy would be equivalent to the accuracy of the dummy classifier working in the `most_frequent` regime, otherwise it would be its reverse (i.e. 1-value).

Since we assume the training and testing datasets were sampled correctly and we hope that they are both representative samples, the most useful setting for the dummy classifier is probably indeed `stratified`.

So to compare results obtained on training binary classifiers the lowest bar to clear should probably be:
|Strategy:| stratified||
|---|---|---|
|  language | accuracy  |  f1 |
|en| 0.624 | 0.308 |
|sl| 0.498 | 0.52 |
|hr| 0.545 | 0.65 |


## Char-ngrams

Good catch, previous experiments were indeed performed on word n-grams. After changing that to character n-grams in the recommended bracket I was able to produce the following result:
|  language | accuracy  |  f1 |
|---|---|---|
|en| 0.76 | 0.233 |
|word:en |	0.757 |	0.220|
|dummy:en| 0.624 | 0.308 |
|sl| 0.616 | 0.539 |
|word:sl |	0.578 |	0.505|
|dummy:sl| 0.498 | 0.52 |
|hr| 0.736 | 0.83 |
|word:hr| 	0.704 |	0.810|
|dummy:hr| 0.545 | 0.65 |,

In this awkward table (which should really be multiindexed) the prefix `dummy:` denotes the dummy classifier results and the prefix `word:` denotes results from testing on word n-grams.

## Better suited models based on the input language

### `EMBEDDIA/sloberta`
|  language | accuracy  |  f1 |
|---|---|---|
|sl| 0.693 | 0.709 |
|dummy: sl| 0.616 | 0.539 |

Training took 4 minutes. 

### `EMBEDDIA/crosloengual-bert`, Slovenian

|  language | accuracy  |  f1 |
|---|---|---|
|sl|  0.693| 0.701 |
|dummy: sl| 0.616 | 0.539 |

Training took 1 minute and 16 seconds.

### `EMBEDDIA/crosloengual-bert`, Croatian

|  language | accuracy  |  f1 |
|---|---|---|
|hr|  0.842| 0.879 |
|dummy:hr| 0.736 | 0.83 |

Training took 1 minute and 37 seconds.

### `classla/bcms-bertic`

I noticed I can instantiate the classifier with `model_type="bert"` and it will still run, although with warnings about it.

|language|accuracy|f1 score|
|---|---|---|
|hr|0.758|0.819|
|dummy:hr| 0.736 | 0.83 |

Training took 1min 50s.

If instead I instantiate it as a type `electra`, as the warnings suggest, it also works, but better:


|language|accuracy|f1 score|
|---|---|---|
|hr|0.859|0.893|
|dummy:hr| 0.736 | 0.83 |

Training took 1min 35s.

## Remarks