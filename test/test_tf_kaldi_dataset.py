import tensorflow as tf
from tf_kaldi_io import KaldiDataset

if __name__ == "__main__":
  # Create a KaldiDataset and print its elements.
  with tf.Session() as sess:
    kaldi_dataset = KaldiDataset(input_rspecifier="ark:data/feats.ark",
                                 target_rspecifier="ark:data/labels.ark",
                                 batch_size=2, batch_mode="frame", # batch_mode="utt",
                                 # delta_order=2,
                                 # norm_means=True, norm_vars=True, global_cmvn_file="data/global.cmvn"
                                 # left_context=1, right_context=1,
                                 # num_downsample=2, offset=0,
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
