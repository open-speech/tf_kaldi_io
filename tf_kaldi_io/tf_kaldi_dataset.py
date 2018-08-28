from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_kaldi_io import KaldiReaderDataset


class KaldiDataset:
  """
  kaldi dataset.
  """

  def __init__(self, input_rspecifier, target_rspecifier=None,
               delta_order=None,
               norm_means=False, norm_vars=False, global_cmvn_file=None,
               left_context=None, right_context=None,
               num_downsample=None, offset=None,
               shuffle=False, shuffle_buffer_size=None, num_threads=4,
               batch_size=2, batch_mode="utt",  # or "frame"
               use_bucket=False, bucket_width=100, num_buckets=10,
               name="kaldi_dataset"):

    self._input_rspecifier = input_rspecifier
    self._target_rspecifier = target_rspecifier
    self._delta_order = delta_order
    self._norm_means = norm_means
    self._norm_vars = norm_vars
    self._global_cmvn_file = global_cmvn_file
    self._left_context = left_context
    self._right_context = right_context
    self._num_downsample = num_downsample
    self._offset = offset
    self._shuffle = shuffle
    self._batch_size = batch_size
    self._num_threads = num_threads
    self._use_bucket = use_bucket
    self._bucket_width = bucket_width
    self._num_buckets = num_buckets
    self._num_batches = 2 # int(math.ceil(len(self._tfrecords_lst) / float(self._batch_size)))

    self._batch_model = batch_mode

    if self._batch_model == "frame" and self._use_bucket:
      raise ValueError("Cannot use bucket with `frame` batch model")

    if not target_rspecifier:
      self._only_input = True
    else:
      self._only_input = False

    if not shuffle_buffer_size:
      self._shuffle_buffer_size = 1000 + batch_size * (num_threads + 1)
    else:
      self._shuffle_buffer_size = shuffle_buffer_size

    self._build()

  def __call__(self):
    self._build()
    return self._dataset

  def _build(self):

    # get [utt, source, target] by KaldiReaderDataset
    dataset = KaldiReaderDataset(input_rspecifier=self._input_rspecifier,
                                 target_rspecifier=self._target_rspecifier,
                                 delta_order=self._delta_order,
                                 norm_means=self._norm_means,
                                 norm_vars=self._norm_vars,
                                 global_cmvn_file=self._global_cmvn_file,
                                 left_context=self._left_context,
                                 right_context=self._right_context,
                                 num_downsample=self._num_downsample,
                                 offset=self._offset,
                                 mode=self._batch_model,
                                 )

    if not self._only_input:

      # [random shuffle]
      if self._shuffle:
        dataset = dataset.shuffle(
          self._shuffle_buffer_size,
          seed=tf.random_uniform((), maxval=777, dtype=tf.int64),
          reshuffle_each_iteration=True)

      # [add in sequence lengths]
      if self._batch_model == "utt":
        dataset = dataset.map(
          lambda utt, src, tgt: (utt, src, tf.shape(src)[0], tgt, tf.shape(tgt)[0]),
          num_parallel_calls=self._num_threads)
      elif self._batch_model == "frame":
        dataset = dataset.map(
          lambda utt, src, tgt: (src, tgt),
          num_parallel_calls=self._num_threads)
      else:
        raise ValueError("Unexpected batch model: ", self._batch_model)

      # [batching and dynamic padding]
      def make_batch(ds):
        if self._batch_model == "utt":
          return ds.padded_batch(
            self._batch_size,
            # The 2nd and 4th entries are the source and target line rows; these have unknown-length vectors.
            # The 1st, 3rd and 5th entries are the source and target row sizes; these are scalars.
            padded_shapes=(
              tf.TensorShape([]),  # utt_key
              tf.TensorShape([None, None]),  # input
              tf.TensorShape([]),  # input_len
              tf.TensorShape([None]),  # target
              tf.TensorShape([])))  # target_len
        elif self._batch_model == "frame":
          return ds.apply(tf.contrib.data.unbatch()).batch(self._batch_size)
        else:
          raise ValueError("Unexpected batch model: ", self._batch_model)

      # [grouped by input_sequence_length]
      if self._use_bucket:
        def key_func(utt, src, src_len, tgt, tgt_len):
          # Calculate bucket_width by maximum source sequence length.
          # Pairs with length [0, bucket_width) go to bucket 0, length
          # [bucket_width, 2 * bucket_width) go to bucket 1, etc.
          # Pairs with length over ((num_bucket-1) * bucket_width) source all
          # go into the last bucket.

          # Bucket sentence pairs by the length of their source
          # sentence and target sentence.
          bucket_id = tf.maximum(src_len // self._bucket_width,
                                 tgt_len // self._bucket_width)
          return tf.to_int64(tf.minimum(self._num_buckets, bucket_id))

        def reduce_func(unused_key, windowed_data):
          return make_batch(windowed_data)

        dataset = dataset.apply(
          tf.contrib.data.group_by_window(
            key_func=key_func,
            reduce_func=reduce_func,
            window_size=self._batch_size))
      else:
        dataset = make_batch(dataset)

      self._dataset =  dataset

    else:

      # [random shuffle]
      dataset = dataset.shuffle(
        self._shuffle_buffer_size,
        seed=tf.random_uniform((), maxval=777, dtype=tf.int64),
        reshuffle_each_iteration=True)

      # [add in sequence lengths]
      if self._batch_model == "utt":
        dataset = dataset.map(
          lambda utt, src: (utt, src, tf.shape(src)[0]),
          num_parallel_calls=self._num_threads)
      elif self._batch_model == "frame":
        dataset = dataset.map(
          lambda utt, src: src,
          num_parallel_calls=self._num_threads)
      else:
        raise ValueError("Unexpected batch model: ", self._batch_model)

      # [batching and dynamic padding]
      def make_batch(ds):
        if self._batch_model == "utt":
          return ds.padded_batch(
            self._batch_size,
            # The 2nd and 4th entries are the source and target line rows; these have unknown-length vectors.
            # The 1st, 3rd and 5th entries are the source and target row sizes; these are scalars.
            padded_shapes=(
              tf.TensorShape([]),  # utt_key
              tf.TensorShape([None, None]),  # input
              tf.TensorShape([])),  # input_len
          )
        elif self._batch_model == "frame":
          return ds.apply(tf.contrib.data.unbatch()).batch(self._batch_size)
        else:
          raise ValueError("Unexpected batch model: ", self._batch_model)

      # [grouped by input_sequence_length]
      if self._use_bucket:
        def key_func(utt, src, src_len):
          # Calculate bucket_width by maximum source sequence length.
          # Pairs with length [0, bucket_width) go to bucket 0, length
          # [bucket_width, 2 * bucket_width) go to bucket 1, etc.
          # Pairs with length over ((num_bucket-1) * bucket_width) source all
          # go into the last bucket.

          # Bucket sentence pairs by the length of their source
          # sentence and target sentence.
          bucket_id = tf.maximum(src_len // self._bucket_width)
          return tf.to_int64(tf.minimum(self._num_buckets, bucket_id))

        def reduce_func(unused_key, windowed_data):
          return make_batch(windowed_data)

        dataset = dataset.apply(
          tf.contrib.data.group_by_window(
            key_func=key_func,
            reduce_func=reduce_func,
            window_size=self._batch_size))
      else:
        dataset = make_batch(dataset)

      self._dataset =  dataset

  @property
  def only_input(self):
    return self._only_input

  @property
  def dataset(self):
    return self._dataset


if __name__ == "__main__":
  # Create a KaldiReaderDataset and print its elements.
  with tf.Session() as sess:
    kaldi_dataset = KaldiDataset(input_rspecifier="ark:../test/data/feats.ark",
                                 target_rspecifier="ark:../test/data/labels.ark",
                                 batch_size=2, batch_mode="utt",
                                 )

    iterator = tf.data.Iterator.from_structure(
      kaldi_dataset.dataset.output_types,
      kaldi_dataset.dataset.output_shapes)

    next_element = iterator.get_next()

    iterator_init_op = iterator.make_initializer(kaldi_dataset.dataset)

    sess.run(iterator_init_op)

    try:
      while True:
        print(sess.run(next_element))
    except tf.errors.OutOfRangeError:
      pass
