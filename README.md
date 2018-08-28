## What is tf_kaldi_io?
A python package: provide a custom tensorflow dataset for kaldi io

Python is its wrapper, C++ is its backend implemention. It depends on two things:

Through [kaldi-io lib](https://github.com/open-speech/kaldi-io.git), it is able to:
- direct read from kaldi rspecifier(scp, ark, in text or binary, just as [kaldi](https://github.com/kaldi-asr/kaldi))
- support multiple feature transforms:
  - delta
  - cmvn
  - splice
  - sampling
- compute fast: kaldi Matrix|Vector with blas math lib is used

Through [tensorflow dataset](https://www.tensorflow.org/guide/datasets), it is able to:
- shuffling
- batching at frame or utt level
- bucketing with input sequence lengths
- and all other tensorflow native dataset manipulations and features (parellel, prefetch, ..)

## usage

```python
import tensorflow as tf
from tf_kaldi_io import KaldiDataset

with tf.Session() as sess:
    kaldi_dataset = KaldiDataset(input_rspecifier="ark:test/data/feats.ark",
                                 target_rspecifier="ark:test/data/labels.ark",
                                 batch_size=2, batch_mode="utt", # batch_mode="frame",
                                 # delta_order=2,
                                 # norm_means=True, norm_vars=True, global_cmvn_file="test/data/global.cmvn"
                                 # left_context=1, right_context=1,
                                 # num_downsample=2, offset=0,
                                 )

    iterator = tf.data.Iterator.from_structure(
      kaldi_dataset.dataset.output_types,
      kaldi_dataset.dataset.output_shapes)

    next_element = iterator.get_next()
    # next_element: 
    #		in utt mode: (utt_keys, inputs, input_lengths, [targets, target_lengths])
    #   in frame mode: (inputs, [targets])

    iterator_init_op = iterator.make_initializer(kaldi_dataset.dataset)

    sess.run(iterator_init_op)

    try:
      while True:
        print(sess.run(next_element))
    except tf.errors.OutOfRangeError:
      pass

```

## Install

### 1. requirements
- tensorflow >= 1.4
- kaldi-io lib [requments](https://github.com/open-speech/kaldi-io.git)
  - kaldi-io lib is compiled automatically in `pip install`, it requires blas math lib installed already.
    - recommendation: use conda python env, and `conda install mkl`
    - ubuntu: `sudo apt-get install libatlas3-base`

### 2. install
```bash
# install from pypi
pip install tf_kaldi_io # it may take a minute, as it compiles kaldi-io lib
# or install from local, then you can run test*.py in test dir
git clone https://github.com/open-speech/tf_kaldi_io.git
cd tf_kaldi_io
pip install .
```

### 3. test

```bash
cd test
python test_tf_kaldi_dataset.py # test KaldiDataset: a python class wrapper of custom dataset
python test_tf_kaldi_io.py # test custom dataset: KaldiReaderDataset
```

## More
- Only `input_rspecifier` is required argument, others are optional or have default values(see in `tf_kaldi_io.py`).
- If use num_downsample in `utt` mode: just the inputs get sampling, the target will not. It's sensible for sequence traing(CTC).
- There are many tf kaldi io implementions, but with one or more defects:
    - just python - io itself is slow.
    - sequential with training - have to wait io done.
    - just kaldi ark in text or binary - text is big, binary is unreadable.
    - no transformations support - you need prepare many feature varieties for one task.
    - no way to become tensorflow native io(dataset) - no parallel, prefetch, shuffle, bucket, ...
    - depend on TFRecodes(protobuf) - unnecessary(need convert to it then to tensor), and protobuf is a nightmare(version incompatible) everytime we meet.
    - all of above disappointments make tf_kaldi_io appear. May kaldi user happy to enter tensorflow world.
    