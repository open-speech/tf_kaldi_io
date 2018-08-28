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
               input_rspecifier, target_rspecifier=None, buffer_size=None, # buffer_size currently not useful
               delta_order=None,
               norm_means=False, norm_vars=False, global_cmvn_file=None,
               left_context=None, right_context=None,
               num_downsample=None, offset=None,
               mode=None):
    self._only_input = target_rspecifier is None

    super(KaldiReaderDataset, self).__init__()

    self._input_rspecifier = ops.convert_to_tensor(
      input_rspecifier, dtype=dtypes.string, name="input_rspecifier")
    self._target_rspecifier = convert.optional_param_to_tensor(
      "target_rspecifier", target_rspecifier, "", dtypes.string)
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
      self._input_rspecifier, self._target_rspecifier, self._buffer_size,
      self._delta_order, self._norm_means, self._norm_vars, self._global_cmvn_file,
      self._left_context, self._right_context, self._num_downsample, self._offset,
      self._mode)

  @property
  def output_types(self):
    if self._only_input:
      return tf.string, tf.float32
    else:
      return tf.string, tf.float32, tf.int32

  @property
  def output_shapes(self):
    if self._only_input:
      return tf.TensorShape([]), tf.TensorShape([None, None])
    else:
      return tf.TensorShape([]), tf.TensorShape([None, None]), tf.TensorShape([None])

  @property
  def output_classes(self):
    if self._only_input:
      return tf.Tensor, tf.Tensor
    else:
      return tf.Tensor, tf.Tensor, tf.Tensor


if __name__ == "__main__":
  # Create a KaldiReaderDataset and print its elements.
  with tf.Session() as sess:
    kaldi_dataset = KaldiReaderDataset(input_rspecifier="ark:../test/data/feats.ark",
                                       target_rspecifier="ark:../test/data/labels.ark",
                                       )
    iterator = kaldi_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    try:
      while True:
        print(sess.run(next_element))
    except tf.errors.OutOfRangeError:
      pass
