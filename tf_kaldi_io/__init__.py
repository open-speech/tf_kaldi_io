import os
from pkg_resources import resource_filename

MISSING_LIBRARY_ERROR = """
Could not find the kaldi-io shared object (libtf_kaldi_io.so) on the system
search paths, on paths specified in the LD_LIBRARY_PATH environment variable,
or by importing package data from the 'tf_kaldi_io' package.

Please ensure that libtf_kaldi_io.so is built and available, or that the
'tf_kaldi_io' package is installed and available.
"""

# Functions and values included when you run `from tf_kaldi_io import *`
__all__ = ["KaldiReaderDataset", "KaldiDataset"]


def find_shared_library(library_name):
    library_paths = {}

    # Next, traverse LD_LIBRARY_PATH.
    for directory in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if os.path.isdir(directory):
            files = os.listdir(directory)
            for filename in files:
                # Rely on filename to search for shared objects
                if ".so." in filename or filename.endswith(".so"):
                    library_paths[filename] = os.path.join(directory, filename)

    # Filter output to library we're looking for
    object_name = "lib{}.so".format(library_name)
    paths = {name: path for (name, path) in library_paths.items()
             if name.startswith(object_name)}

    # Return None if the list of paths for this library is empty
    if paths:
        return paths
    else:
        return None


def find_kaldi_io_library():
    """Check that libtf_kaldi_io.so can be found. If it can, ensure that
    Tensorflow's tf.load_op_library() can find it by potentially adding it to
    the LD_LIBRARY_PATH as necessary.

    If it is not found, raise a helpful and informative error."""
    try:
        libtf_kaldi_io = resource_filename(__package__, "libtf_kaldi_io.so")
        found = os.path.isfile(libtf_kaldi_io)
    except ImportError:
        # If we can't import tf_kaldi_io, definitely can't get its resources.
        found = False

    if found:
        # If we have a libtf_kaldi_io.so from the tf_kaldi_io Python package,
        # then ensure it gets on the path. We stick it on the front of the
        # path, because it would be confusing if a tf_kaldi_io package used a
        # libtf_kaldi_io.so that didn't correspond to it, just because the user
        # happened to have a custom LD_LIBRARY_PATH set.
        old_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
        lib_dir = os.path.dirname(libtf_kaldi_io)
        os.environ["LD_LIBRARY_PATH"] = lib_dir + ":" + old_ld_library_path

    # Ensure that at this point, no matter what, Tensorflow should be able to
    # load libtf_kaldi_io.so as an op library.
    kaldi_io_lib_paths = find_shared_library("tf_kaldi_io")
    if kaldi_io_lib_paths:
        return kaldi_io_lib_paths["libtf_kaldi_io.so"]
    else:
        raise RuntimeError(MISSING_LIBRARY_ERROR)


# Find the path to the KaldiIO shared library.
libtf_kaldi_io_path = find_kaldi_io_library()

# Load KaldiIO shared library. There is no documentation on how
# tf.load_op_library finds its shared libraries, but looking at the source
# indicates that the string passed to it is simply forwarded to dlopen(), so we
# can pass it an absolute path to libtf_kaldi_io.so.
import tensorflow as tf

kaldi_reader_dataset_module = tf.load_op_library(libtf_kaldi_io_path)

from . import tf_kaldi_io
# aliases for backwards compatibility
KaldiReaderDataset = tf_kaldi_io.KaldiReaderDataset

from . import tf_kaldi_dataset
KaldiDataset = tf_kaldi_dataset.KaldiDataset
