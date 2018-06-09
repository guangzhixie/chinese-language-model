# chinese-language-model

3-gram Chinese Language Model.

You can download a trained model by:

```shell
wget --show-progress -P 'model/' 'https://www.dropbox.com/s/xos3ojg7grtr4c9/lm.pkl?dl=0'
```

The model will be placed at [model/](model/) directory. It is using Kneser-Ney Smoothing, and trained with a 9M corpus. The test perplexity is 39.26 against an 8k corpus.

A sample usage is provided in the file [ngram-udage.ipynb](ngram-udage.ipynb).

You can also train your own model, with either Add-k Smoothing or Kneser-Ney Smoothing. See [ngram.ipynb](ngram.ipynb) for an example.
