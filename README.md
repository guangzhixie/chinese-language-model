# chinese-language-model

Tri-gram Chinese Language Model.

You can download a trained model by:

```shell
wget --show-progress -P 'model/' 'https://www.dropbox.com/s/xos3ojg7grtr4c9/lm.pkl?dl=1'
```

The model will be placed under [model](model) directory. It is using Kneser-Ney Smoothing, and trained with a 9M corpus. The test perplexity is 39.26 against an 8k corpus.

A sample usage is provided in the file [ngram-usage.ipynb](ngram-usage.ipynb).

You can also train your own model, with either Add-k Smoothing or Kneser-Ney Smoothing. See [ngram.ipynb](ngram.ipynb) for an example.

Note: You may need to install dill by:

```shell
pip install dill
```

