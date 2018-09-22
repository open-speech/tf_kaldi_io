import tensorflow as tf
from tf_kaldi_io import KaldiReaderDataset

if __name__ == "__main__":
  # Create a KaldiReaderDataset and print its elements.
  with tf.Session() as sess:
    kaldi_dataset = KaldiReaderDataset(input_rspecifier="ark:../test/data/matrix.ark",
                                       target_rspecifier="ark:../test/data/int_vec.ark",
                                       )
    iterator = kaldi_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    try:
      while True:
        print(sess.run(next_element))
    except tf.errors.OutOfRangeError:
      pass
