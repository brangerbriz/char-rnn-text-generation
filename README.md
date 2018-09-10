# Character Embeddings Recurrent Neural Network Text Generation Model

Originally authored by [YuXuan Tay](https://github.com/yxtay) and inspired by [Andrej Karpathy](https://github.com/karpathy/)'s 
[The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/).

We've made considerable changes to the [original repository](https://github.com/yxtay/char-rnn-text-generation), namely we've:

- Removed all implementations except the keras model
- Heavily restructured/rewritten the keras model to better fit our needs (added `train.py`, `generate.py`, etc.)
- Added `utils.io_batch_generator()` that lazy-loads (and releases) data from disk into RAM, effectively allowing limitless training / validation data to be used. In the original implementation, training data was significantly limited by computer memory resources.
- Added `scripts/` and default parameters for twitter bot creation using the twitter_cikm_2010 dataset.
