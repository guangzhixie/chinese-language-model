# chinese-language-model

A 3-gram Chinese Language Model.

A trained model is located at model/lm.pkl. It is using Kneser-Ney Smoothing, and trained with a 9M corpus. The test perplexity is 39.26 against an 8k corpus.

A sample usage is provided in the file ngram-udage.ipynb.

You can also train your own model, with either Add-k Smoothing or Kneser-Ney Smoothing. See ngram.ipynb for an example.
