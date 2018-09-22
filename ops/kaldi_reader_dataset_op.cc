//
// Created by songmeixu (songmeixu@outlook.com) on 2018/8/18.
//

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "kaldi-io.h"

namespace KaldiIO {
namespace {

using namespace tensorflow;
using std::string;

Status Matrix2Tensor(const kaldi::MatrixBase<kaldi::BaseFloat> &in, Tensor &out) {
  auto flat = out.flat<float>();
  assert(flat.size() == in.NumRows() * in.NumCols());
  float *to = flat.data();
  for (int r = 0; r < in.NumRows(); ++r, to += in.NumCols()) {
    std::copy_n(in.RowData(r), in.NumCols(), to);
  }

  return Status::OK();
}

Status Vector2Tensor(const kaldi::VectorBase<kaldi::BaseFloat> &in, Tensor &out) {
  auto flat = out.flat<float>();
  assert(flat.size() == in.Dim());
  float *to = flat.data();
  std::copy_n(in.Data(), in.Dim(), to);

  return Status::OK();
}

Status vector2Tensor(std::vector<int> &in, Tensor &out) {
  auto flat = out.flat<int>();
  assert(flat.size() == in.size());
  int *to = flat.data();
  std::copy_n(in.data(), in.size(), to);

  return Status::OK();
}

class KaldiReaderDatasetOp : public DatasetOpKernel {
 public:

  KaldiReaderDatasetOp(OpKernelConstruction *ctx)
      : DatasetOpKernel(ctx) {
    // Parse and validate any attrs that define the dataset using
    // `ctx->GetAttr()`, and store them in member variables.
  }

  void MakeDataset(OpKernelContext *ctx,
                   DatasetBase **output) override {
    // Parse and validate any input tensors 0that define the dataset using
    // `ctx->input()` or the utility function
    // `ParseScalarArgument<T>(ctx, &arg)`.

    // Create the dataset object, passing any (already-validated) arguments from
    // attrs or input tensors.
    const Tensor *matrix_rspecifier_tensor, *vector_rspecifier_tensor, *int_vector_rspecifier_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("matrix_rspecifier", &matrix_rspecifier_tensor));
    OP_REQUIRES(
        ctx, matrix_rspecifier_tensor->dims() == 0,
        errors::InvalidArgument("`matrix filename` must be a scalar of string."));
    OP_REQUIRES_OK(ctx, ctx->input("vector_rspecifier", &vector_rspecifier_tensor));
    OP_REQUIRES(
        ctx, vector_rspecifier_tensor->dims() == 0,
        errors::InvalidArgument("`vector filename` must be a scalar of string."));
    OP_REQUIRES_OK(ctx, ctx->input("int_vector_rspecifier", &int_vector_rspecifier_tensor));
    OP_REQUIRES(
        ctx, int_vector_rspecifier_tensor->dims() == 0,
        errors::InvalidArgument("`int-vector filename` must be a scalar of string."));

    string matrix_rspecifier(matrix_rspecifier_tensor->flat<string>()(0));
    string vector_rspecifier(vector_rspecifier_tensor->flat<string>()(0));
    string int_vector_rspecifier(int_vector_rspecifier_tensor->flat<string>()(0));

    int64 buffer_size = -1;
    OP_REQUIRES_OK(
        ctx, ParseScalarArgument<int64>(ctx, "buffer_size", &buffer_size));
    OP_REQUIRES(ctx, buffer_size >= 0,
                errors::InvalidArgument(
                    "`buffer_size` must be >= 0 (0 == no buffering)"));

    int64 delta_order = 0;
    bool norm_means = false, norm_vars = false;
    string global_cmvn_file, mode;
    int64 left_context, right_context;
    int64 num_downsample = 1, offset = 0;

    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument<int64>(ctx, "delta_order", &delta_order));
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument<bool>(ctx, "norm_means", &norm_means));
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument<bool>(ctx, "norm_vars", &norm_vars));
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument<string>(ctx, "global_cmvn_file", &global_cmvn_file));
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument<int64>(ctx, "left_context", &left_context));
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument<int64>(ctx, "right_context", &right_context));

    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument<int64>(ctx, "num_downsample", &num_downsample));
    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument<int64>(ctx, "offset", &offset));

    OP_REQUIRES_OK(ctx,
                   ParseScalarArgument<string>(ctx, "mode", &mode));

    OP_REQUIRES(ctx, delta_order >= 0,
                errors::InvalidArgument("delta_order must be >= 0 not ", delta_order));
    OP_REQUIRES(ctx, left_context >= 0,
                errors::InvalidArgument("left_context must be >= 0 not ", left_context));
    OP_REQUIRES(ctx, right_context >= 0,
                errors::InvalidArgument("right_context must be >= 0 not ", right_context));

    OP_REQUIRES(ctx, offset >= 0,
                errors::InvalidArgument("offset must be >= 0 not ", offset));

    *output =
        new Dataset(ctx, matrix_rspecifier, vector_rspecifier, int_vector_rspecifier,
                    buffer_size,
                    delta_order,
                    norm_means, norm_vars, global_cmvn_file,
                    left_context, right_context,
                    num_downsample, offset,
                    mode);
  }

 private:
  class Dataset : public GraphDatasetBase {
   public:
    Dataset(OpKernelContext *ctx,
            const string &matrix_rspecifier,
            const string &vector_rspecifier,
            const string &int_vector_rspecifier,
            int64 buffer_size,
            int64 delta_order,
            bool norm_means,
            bool norm_vars,
            const string &global_cmvn_file,
            int64 left_context,
            int64 right_context,
            int64 num_downsample,
            int64 offset,
            const string &mode
    )
        :
        GraphDatasetBase(ctx),
        matrix_rspecifier_(matrix_rspecifier),
        vector_rspecifier_(vector_rspecifier),
        int_vector_rspecifier_(int_vector_rspecifier),
        delta_opts_(delta_order),
        norm_means_(norm_means), norm_vars_(norm_vars),
        left_context_(left_context), right_context_(right_context),
        num_downsample_(num_downsample), offset_(offset),
        mode_(mode) {
      // cmvn
      if (!global_cmvn_file.empty() && (norm_means || norm_vars)) {
        bool binary;
        kaldi::Input ki(global_cmvn_file, &binary);
        cmvn_stats_.Read(ki.Stream(), binary);
      }
    }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string &prefix) const override {
      return std::unique_ptr<IteratorBase>(new Iterator(
          {this, strings::StrCat(prefix, "::KaldiReader")}));
    }

    // Record structure: Each record is represented by a scalar string tensor.
    //
    // Dataset elements can have a fixed number of components of different
    // types and shapes; replace the following two methods to customize this
    // aspect of the dataset.
    const DataTypeVector &output_dtypes() const override {
      bool with_matrix = !matrix_rspecifier_.empty(),
           with_vector = !vector_rspecifier_.empty(),
           with_int_vec = !int_vector_rspecifier_.empty();

      if (!with_matrix && !with_vector && with_int_vec) {
        static auto *const dtypes = new DataTypeVector({DT_STRING, DT_INT32}); // utt, int_vec
        return *dtypes;
      } else if ((!with_matrix && with_vector && !with_int_vec) ||
                 (with_matrix && !with_vector && !with_int_vec)) {
        static auto *const dtypes = new DataTypeVector({DT_STRING, DT_FLOAT});
        return *dtypes;
      } else if ((!with_matrix && with_vector && with_int_vec) ||
                 (with_matrix && !with_vector && with_int_vec)) {
        static auto *const dtypes = new DataTypeVector({DT_STRING, DT_FLOAT, DT_INT32});
        return *dtypes;
      } else if (with_matrix && with_vector && !with_int_vec) {
        static auto *const dtypes = new DataTypeVector({DT_STRING, DT_FLOAT, DT_FLOAT});
        return *dtypes;
      } else if (with_matrix && with_vector && with_int_vec) {
        static auto *const dtypes = new DataTypeVector({DT_STRING, DT_FLOAT, DT_FLOAT, DT_INT32});
        return *dtypes;
      }
    }

    const std::vector<PartialTensorShape> &output_shapes() const override {
      int num_out_shapes = 0;
      if (!matrix_rspecifier_.empty())
        ++num_out_shapes;
      if (!vector_rspecifier_.empty())
        ++num_out_shapes;
      if (!int_vector_rspecifier_.empty())
        ++num_out_shapes;

      static std::vector<PartialTensorShape> *shapes;
      if (num_out_shapes == 1) {
        shapes = new std::vector<PartialTensorShape>({{}});
      } else if (num_out_shapes == 2 ){
        shapes = new std::vector<PartialTensorShape>({{}, {}});
      } else if (num_out_shapes == 3) {
        shapes = new std::vector<PartialTensorShape>({{}, {}, {}});
      }
      return *shapes;
    }

    string DebugString() const override { return "KaldiReaderDatasetOp::Dataset"; }

   protected:
    // Optional: Implementation of `GraphDef` serialization for this dataset.
    //
    // Implement this method if you want to be able to save and restore
    // instances of this dataset (and any iterators over it).
    Status AsGraphDefInternal(DatasetGraphDefBuilder *b,
                              Node **output) const override {
      // Construct nodes to represent any of the input tensors from this
      // object's member variables using `b->AddScalar()` and `b->AddVector()`.
      Node *matrix_rspecifier = nullptr, *vector_rspecifier = nullptr, *int_vector_rspecifier = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(matrix_rspecifier_, &matrix_rspecifier));
      TF_RETURN_IF_ERROR(b->AddScalar(vector_rspecifier_, &vector_rspecifier));
      TF_RETURN_IF_ERROR(b->AddScalar(int_vector_rspecifier_, &int_vector_rspecifier));
      Node *buffer_size = nullptr;
//      TF_RETURN_IF_ERROR(b->AddScalar(options_.buffer_size, &buffer_size));

      Node *delta_order = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(delta_opts_.order, &delta_order));

      Node *norm_means = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(norm_means_, &norm_means));

      Node *norm_vars = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(norm_vars_, &norm_vars));

      Node *global_cmvn_file = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(global_cmvn_file_, &global_cmvn_file));

      Node *left_context = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(left_context_, &left_context));

      Node *right_context = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(right_context_, &right_context));

      Node *num_downsample = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(num_downsample_, &num_downsample));

      Node *offset = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(offset_, &offset));

      Node *mode = nullptr;
      TF_RETURN_IF_ERROR(b->AddScalar(mode_, &mode));

      TF_RETURN_IF_ERROR(
          b->AddDataset(this,
                        {matrix_rspecifier, vector_rspecifier, int_vector_rspecifier,
                         buffer_size,
                         delta_order,
                         norm_means, norm_vars, global_cmvn_file,
                         left_context, right_context,
                         num_downsample, offset,
                         mode},
                        output));
      return Status::OK();
    }

    Status Preprocess(const kaldi::Matrix<kaldi::BaseFloat> &feat,
                      kaldi::Matrix<kaldi::BaseFloat> &feat_ret) const {
      // delta
      kaldi::Matrix<kaldi::BaseFloat> feat_delta;
      kaldi::ComputeDeltas(delta_opts_, feat, &feat_delta);

      // cmvn
      if (norm_means_) {
        kaldi::ApplyCmvn(cmvn_stats_, norm_vars_, &feat_delta);
      }

      // splice
      kaldi::Matrix<kaldi::BaseFloat> feat_splice;
      kaldi::SpliceFrames(feat_delta,
                          left_context_,
                          right_context_,
                          &feat_splice);

      // sampling
      if (num_downsample_ != 1) {
        DownsampleFrames(feat_splice, num_downsample_, offset_, &feat_ret);
      } else {
        feat_ret.Resize(feat_splice.NumRows(), feat_splice.NumCols());
        feat_ret.CopyFromMat(feat_splice);
      }

      return Status::OK();
    }

    void DownsampleFrames(const kaldi::Matrix<kaldi::BaseFloat> &feats,
                          int n,
                          int offset,
                          kaldi::Matrix<kaldi::BaseFloat> *output) const {
      if (n > 0) {
        // This code could, of course, be much more efficient; I'm just
        // keeping it simple.
        kaldi::int32 num_indexes = 0;
        for (kaldi::int32 k = offset; k < feats.NumRows(); k += n)
          num_indexes++; // k is the index.

        if (num_indexes == 0) {
          KALDI_WARN << "This utterance is too short for subsample, "
                     << "keep output as input.";
          output->Resize(feats.NumRows(), feats.NumCols());
          output->CopyFromMat(feats);
          return;
        }
        output->Resize(num_indexes, feats.NumCols());
        kaldi::int32 i = 0;
        for (kaldi::int32 k = offset; k < feats.NumRows(); k += n, i++) {
          kaldi::SubVector<kaldi::BaseFloat> src(feats, k);
          output->Row(i).CopyFromVec(src);
        }
        KALDI_ASSERT(i == num_indexes);
      } else {
        kaldi::int32 repeat = -n;
        output->Resize(feats.NumRows() * repeat, feats.NumCols());
        for (kaldi::int32 i = 0; i < output->NumRows(); i++)
          output->Row(i).CopyFromVec(feats.Row(i / repeat));
      }
    }

    void DownsampleTargets(std::vector<kaldi::int32> &in) const {
      int n = num_downsample_;
      int offset = offset_;
      if (n > 0) {
        // This code could, of course, be much more efficient; I'm just
        // keeping it simple.
        kaldi::int32 num_indexes = 0;
        for (kaldi::int32 k = offset; k < in.size(); k += n)
          num_indexes++; // k is the index.

        if (num_indexes == 0) {
          KALDI_WARN << "This utterance is too short for subsample, "
                     << "keep output as input.";
          return;
        }
        std::vector<kaldi::int32> output(num_indexes);
        kaldi::int32 i = 0;
        for (kaldi::int32 k = offset; k < in.size(); k += n, i++) {
          output[i] = in[k];
        }
        KALDI_ASSERT(i == num_indexes);

        in = output;
      } else {
        kaldi::int32 repeat = -n;
        std::vector<kaldi::int32> output(in.size() * repeat);
        for (kaldi::int32 i = 0; i < output.size(); i++)
          output[i] = in[i / repeat];

        in = output;
      }
    }

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params &params)
          : DatasetIterator<Dataset>(params) {
        if (!dataset()->matrix_rspecifier_.empty()) {
          matrix_reader_.reset(
              new kaldi::SequentialBaseFloatMatrixReader(dataset()->matrix_rspecifier_));
        } else {
          matrix_reader_.reset(nullptr);
        }

        if (!dataset()->vector_rspecifier_.empty()) {
          vector_reader_.reset(
              new kaldi::SequentialBaseFloatVectorReader(dataset()->vector_rspecifier_));
        } else {
          vector_reader_.reset(nullptr);
        }

        if (!dataset()->int_vector_rspecifier_.empty()) {
          int_vec_reader_.reset(
              new kaldi::SequentialInt32VectorReader(dataset()->int_vector_rspecifier_));
        } else {
          int_vec_reader_.reset(nullptr);
        }
      }

      // Implementation of the reading logic.
      //
      // The example implementation in this file yields the string "MyReader!"
      // ten times. In general there are three cases:
      //
      // 1. If an element is successfully read, store it as one or more tensors
      //    in `*out_tensors`, set `*end_of_sequence = false` and return
      //    `Status::OK()`.
      // 2. If the end of input is reached, set `*end_of_sequence = true` and
      //    return `Status::OK()`.
      // 3. If an error occurs, return an error status using one of the helper
      //    functions from "tensorflow/core/lib/core/errors.h".
      Status GetNextInternal(IteratorContext *ctx,
                             std::vector<Tensor> *out_tensors,
                             bool *end_of_sequence) override {
        // NOTE: `GetNextInternal()` may be called concurrently, so it is
        // recommended that you protect the iterator state with a mutex.
        mutex_lock l(mu_);
        do {
          if ((matrix_reader_ && matrix_reader_->Done()) ||
              (vector_reader_ and vector_reader_->Done()) ||
              (int_vec_reader_ and int_vec_reader_->Done())) { // End of any reader, advance to the next.
            *end_of_sequence = true;
            return Status::OK();
          } else {
            string key;
            bool seen_reader = false;

            if(matrix_reader_) { // matrix
              if (key.empty())
                key = matrix_reader_->Key();
              else {
                if (key != matrix_reader_->Key())
                  return errors::InvalidArgument(
                      "pre_reader_key:", key,
                      " is not matched with current_reader_key:", matrix_reader_->Key(),
                      ", something must be wrong(input changed?).");
              }

              if (!seen_reader) {
                Tensor key_tensor(ctx->allocator({}), DT_STRING, {});
                key_tensor.scalar<string>()() = key;
                out_tensors->emplace_back(std::move(key_tensor));
                seen_reader =true;
              }

              kaldi::Matrix <kaldi::BaseFloat> feat_mat(matrix_reader_->Value());

              kaldi::Matrix <kaldi::BaseFloat> feat_preprocessed_mat;
              Status s = dataset()->Preprocess(feat_mat, feat_preprocessed_mat);
              if (!s.ok()) {
                return s;
              }

              Tensor matrix_tensor
                  (ctx->allocator({}),
                   DT_FLOAT,
                   {feat_preprocessed_mat.NumRows(), feat_preprocessed_mat.NumCols()});

              s = Matrix2Tensor(feat_preprocessed_mat, matrix_tensor);
              if (!s.ok()) {
                return s;
              }

              out_tensors->emplace_back(std::move(matrix_tensor));

              matrix_reader_->Next();
            }

            if(vector_reader_) { // vector
              if (key.empty())
                key = vector_reader_->Key();
              else {
                if (key != vector_reader_->Key())
                  return errors::InvalidArgument(
                      "pre_reader_key:", key,
                      " is not matched with current_reader_key:", vector_reader_->Key(),
                      ", something must be wrong(input changed?).");
              }

              if (!seen_reader) {
                Tensor key_tensor(ctx->allocator({}), DT_STRING, {});
                key_tensor.scalar<string>()() = key;
                out_tensors->emplace_back(std::move(key_tensor));
                seen_reader =true;
              }

              kaldi::Vector <kaldi::BaseFloat> feat_vec(vector_reader_->Value());

              Tensor vec_tensor(ctx->allocator({}), DT_FLOAT, {feat_vec.Dim()});

              Status s = Vector2Tensor(feat_vec, vec_tensor);
              if (!s.ok()) {
                return s;
              }

              out_tensors->emplace_back(std::move(vec_tensor));

              vector_reader_->Next();
            }

            if (int_vec_reader_) { // int_vec
              if (key.empty())
                key = int_vec_reader_->Key();
              else {
                if (key != int_vec_reader_->Key())
                  return errors::InvalidArgument(
                      "pre_reader_key:", key,
                      " is not matched with current_reader_key:", int_vec_reader_->Key(),
                      ", something must be wrong(input changed?).");
              }

              if (!seen_reader) {
                Tensor key_tensor(ctx->allocator({}), DT_STRING, {});
                key_tensor.scalar<string>()() = key;
                out_tensors->emplace_back(std::move(key_tensor));
                seen_reader =true;
              }

              std::vector <kaldi::int32> label_vec(int_vec_reader_->Value());

              if (dataset()->num_downsample_ != 1 && dataset()->mode_ == "frame") {
                dataset()->DownsampleTargets(label_vec);
              }

              Tensor output_tensor(ctx->allocator({}), DT_INT32, {int(label_vec.size())});
              Status s = vector2Tensor(label_vec, output_tensor);
              if (!s.ok()) {
                return s;
              }

              out_tensors->emplace_back(std::move(output_tensor));

              int_vec_reader_->Next();
            }

            // We have reached the end of the current key, so maybe
            // move on to next key.
            ++current_key_index_;
            current_key_ = key;

            *end_of_sequence = false;
            return Status::OK();
          }
        } while (true);
      }

     protected:
      // Optional: Implementation of iterator state serialization for this
      // iterator.

      // Implement these two methods if you want to be able to save and restore
      // instances of this iterator.
      Status SaveInternal(IteratorStateWriter* writer) override {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("current_key_index"),
                                               current_key_index_));

        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("current_key"),
                                               current_key_));

        return Status::OK();
      }

      Status RestoreInternal(IteratorContext* ctx,
                             IteratorStateReader* reader) override {
        mutex_lock l(mu_);
        ResetStreamsLocked();
        int64 current_key_index;
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("current_key_index"),
                                              &current_key_index));
        current_key_index_ = size_t(current_key_index);
        if (reader->Contains(full_name("current_key"))) {
          string current_key;
          TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("current_key"), &current_key));
          current_key_ = current_key;
          TF_RETURN_IF_ERROR(SetupStreamsLocked(ctx->env()));
        }
        return Status::OK();
      }

     private:
      // Sets up reader streams to read from the file at `current_key_index_`.
      Status SetupStreamsLocked(Env* env) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        for (size_t i = 0; i <= current_key_index_; ++i) {
          if (matrix_reader_)
            matrix_reader_->Next();
          if (vector_reader_)
            vector_reader_->Next();
          if (int_vec_reader_)
            int_vec_reader_->Next();
        }

        if (current_key_ != matrix_reader_->Key()) {
          return  errors::InvalidArgument(
              "current_key_index_:", current_key_index_,
              " is not matched with current_key:", current_key_,
              ", something must be wrong(input changed?).");
        }

        return Status::OK();
      }

      void ResetStreamsLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_) {
        matrix_reader_.reset();
        vector_reader_.reset();
        int_vec_reader_.reset();
      }

      mutex mu_;
      size_t current_key_index_ GUARDED_BY(mu_) = 0;
      string current_key_ GUARDED_BY(mu_) = "";

      std::unique_ptr<kaldi::SequentialBaseFloatMatrixReader> matrix_reader_ GUARDED_BY(mu_);
      std::unique_ptr<kaldi::SequentialBaseFloatVectorReader> vector_reader_ GUARDED_BY(mu_);
      std::unique_ptr<kaldi::SequentialInt32VectorReader> int_vec_reader_ GUARDED_BY(mu_);
    };

    const string matrix_rspecifier_;
    const string vector_rspecifier_;
    const string int_vector_rspecifier_;
    const string mode_;

    // preprocessing
    /// delta
    kaldi::DeltaFeaturesOptions delta_opts_;
    /// cmvn
    const bool norm_means_, norm_vars_;
    const string global_cmvn_file_;

    /// splice
    const int left_context_, right_context_;
    /// subsampling
    int num_downsample_, offset_;

    kaldi::Matrix<double> cmvn_stats_;
  };
};

// Register the op definition for KaldiReaderDataset.
//
// Dataset ops always have a single output, of type `variant`, which represents
// the constructed `Dataset` object.
//
// Add any attrs and input tensors that define the dataset here.
REGISTER_OP("KaldiReaderDataset")
    .Input("matrix_rspecifier: string")
    .Input("vector_rspecifier: string")
    .Input("int_vector_rspecifier: string")
    .Input("buffer_size: int64")
    .Input("delta_order: int64")
    .Input("norm_means: bool")
    .Input("norm_vars: bool")
    .Input("global_cmvn_file: string")
    .Input("left_context: int64")
    .Input("right_context: int64")
    .Input("num_downsample: int64")
    .Input("offset: int64")
    .Input("mode: string")
    .Output("handle: variant")
    .SetIsStateful()  // TODO(b/65524810): Source dataset ops must be marked
        // stateful to inhibit constant folding.
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // `matrix_rspecifier` must be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      // `vector_rspecifier` must be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      // `int_vector_rspecifier` must be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      // `buffer_size` could only be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      // `delta_order` could only be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      // `norm_means` could only be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));
      // `norm_vars` could only be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));
      // `global_cmvn_file` could only be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &unused));
      // `left_context` could only be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 0, &unused));
      // `right_context` could only be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(9), 0, &unused));
      // `num_downsample` could only be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(10), 0, &unused));
      // `offset` could only be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(11), 0, &unused));
      // `mode` could only be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(12), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

// Register the kernel implementation for KaldiReaderDataset.
REGISTER_KERNEL_BUILDER(Name("KaldiReaderDataset").Device(DEVICE_CPU),
                        KaldiReaderDatasetOp);

}  // namespace
}  // namespace KaldiIO