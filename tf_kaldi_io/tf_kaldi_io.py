from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.util import convert
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops

# TODO(b/64974358): Increase default buffer size to 256 MB.
_DEFAULT_READER_BUFFER_SIZE_BYTES = 256 * 1024  # 256 KB

import tensorflow as tf
from tf_kaldi_io import kaldi_reader_dataset_module


class KaldiReaderDataset(tf.data.Dataset):

  def __init__(self,
               matrix_rspecifier=None,
               vector_rspecifier=None,
               int_vector_rspecifier=None,
               buffer_size=None,  # buffer_size currently not useful
               delta_order=None,
               norm_means=False, norm_vars=False, global_cmvn_file=None,
               left_context=None, right_context=None,
               num_downsample=None, offset=None,
               mode=None):
    if not (matrix_rspecifier or vector_rspecifier or int_vector_rspecifier):
      raise ValueError("all supported reader is None")

    self._readers_idx = [1 if r else 0 for r in
                         [matrix_rspecifier, vector_rspecifier, int_vector_rspecifier]]

    super(KaldiReaderDataset, self).__init__()

    self._matrix_rspecifier = convert.optional_param_to_tensor(
      "matrix_rspecifier", matrix_rspecifier, "", dtypes.string)
    self._vector_rspecifier = convert.optional_param_to_tensor(
      "vector_rspecifier", vector_rspecifier, "", dtypes.string)
    self._int_vector_rspecifier = convert.optional_param_to_tensor(
      "int_vector_rspecifier", int_vector_rspecifier, "", dtypes.string)
    self._buffer_size = convert.optional_param_to_tensor(
      "buffer_size", buffer_size, _DEFAULT_READER_BUFFER_SIZE_BYTES)

    self._delta_order = convert.optional_param_to_tensor(
      "delta_order", delta_order, 0)
    self._norm_means = convert.optional_param_to_tensor(
      "norm_means", norm_means, False, dtypes.bool)
    self._norm_vars = convert.optional_param_to_tensor(
      "norm_vars", norm_vars, False, dtypes.bool)
    self._global_cmvn_file = convert.optional_param_to_tensor(
      "global_cmvn_file", global_cmvn_file, "", dtypes.string)
    self._left_context = convert.optional_param_to_tensor(
      "left_context", left_context, 0)
    self._right_context = convert.optional_param_to_tensor(
      "right_context", right_context, 0)
    self._num_downsample = convert.optional_param_to_tensor(
      "num_downsample", num_downsample, 1)
    self._offset = convert.optional_param_to_tensor(
      "offset", offset, 0)

    self._mode = convert.optional_param_to_tensor(
      "mode", mode, "utt", dtypes.string)

  def _as_variant_tensor(self):
    return kaldi_reader_dataset_module.kaldi_reader_dataset(
      self._matrix_rspecifier, self._vector_rspecifier, self._int_vector_rspecifier,
      self._buffer_size,
      self._delta_order, self._norm_means, self._norm_vars, self._global_cmvn_file,
      self._left_context, self._right_context, self._num_downsample, self._offset,
      self._mode)

  @property
  def output_types(self):
    if self._readers_idx == [0, 0, 1]:
      return tf.string, tf.int32
    elif self._readers_idx in [[0, 1, 0], [1, 0, 0]]:
      return tf.string, tf.float32
    elif self._readers_idx in [[0, 1, 1], [1, 0, 1]]:
      return tf.string, tf.float32, tf.int32
    elif self._readers_idx == [1, 1, 0]:
      return tf.string, tf.float32, tf.float32
    elif self._readers_idx == [1, 1, 1]:
      return tf.string, tf.float32, tf.float32, tf.int32

  @property
  def output_shapes(self):
    if self._readers_idx in [[0, 0, 1], [0, 1, 0]]:
      return tf.TensorShape([]), tf.TensorShape([None])
    elif self._readers_idx == [0, 1, 1]:
      return tf.TensorShape([]), tf.TensorShape([None]), tf.TensorShape([None])
    elif self._readers_idx == [1, 0, 0]:
      return tf.TensorShape([]), tf.TensorShape([None, None])
    elif self._readers_idx in [[1, 0, 1], [1, 1, 0]]:
      return tf.TensorShape([]), tf.TensorShape([None, None]), tf.TensorShape([None])
    elif self._readers_idx == [1, 1, 1]:
      return tf.TensorShape([]), tf.TensorShape([None, None]), tf.TensorShape([None]), tf.TensorShape([None])

  @property
  def output_classes(self):
    num_readers = sum(self._readers_idx)
    if num_readers == 1:
      return tf.Tensor, tf.Tensor
    elif num_readers == 2:
      return tf.Tensor, tf.Tensor, tf.Tensor
    elif num_readers == 3:
      return tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor


if __name__ == "__main__":
  # Create a KaldiReaderDataset and print its elements.
  with tf.Session() as sess:
    kaldi_dataset = KaldiReaderDataset(matrix_rspecifier="ark:../test/data/matrix.ark",
                                       int_vector_rspecifier="ark:../test/data/int_vec.ark",
                                       )
    iterator = kaldi_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    try:
      while True:
        print(sess.run(next_element))
    except tf.errors.OutOfRangeError:
      pass
