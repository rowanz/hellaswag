import collections
import csv
import os
import zipfile

import requests
import tensorflow as tf

from adversarial_filtering.bert import modeling
from adversarial_filtering.bert import tokenization
import random
import gcsfs
import numpy as np
from tensorflow.python.lib.io import file_io


def _save_np(absolute_fn, array):
    if absolute_fn.startswith('gs://'):
        with file_io.FileIO(absolute_fn, 'w') as f:
            np.save(f, array)
    else:
        np.save(absolute_fn, array)

def _softmax(x, add_random_noise = True):
    x_copy = x.copy()
    x_copy[np.isnan(x_copy)] = 0
    x_copy[np.isinf(x_copy)] = 1e12

    reduce_dim = x_copy.ndim - 1
    x_minus_max = x_copy - x_copy.max(reduce_dim, keepdims=True)
    x_exp = np.exp(x_minus_max)
    sm = x_exp/x_exp.sum(reduce_dim, keepdims=True)

    if add_random_noise:
        sm += np.random.rand(*sm.shape) * 10e-6
    return sm

def gcs_agnostic_open(fn, mode='r'):
    if fn.startswith('gs://'):
        return gcsfs.GCSFileSystem().open(fn, mode)
    else:
        return open(fn, mode)

def setup_bert(use_tpu, do_lower_case, bert_large, output_dir=None, tpu_name=None, tpu_zone=None,
               gcp_project=None, master=None, iterations_per_loop=1000, num_tpu_cores=8, max_seq_length=512):
    """
    Loads a pretrained BERT model. if not on TPU then we'll download it if necessary
    :param use_tpu:
    :param do_lower_case:
    :param bert_large:
    :param output_dir:
    :param tpu_name: tpu hyperparm
    :param tpu_zone: tpu hyperparm
    :param gcp_project: ok if None
    :param master: ok if None
    :param iterations_per_loop:
    :param num_tpu_cores: 8 should be ok
    :param max_seq_length: just for debugging
    :return:
    """

    #####
    if output_dir is None:
        raise ValueError("Need to set output dir when running on a TPU")
    if tf.gfile.Exists(output_dir):
        tf.gfile.DeleteRecursively(output_dir)
    tf.gfile.MakeDirs(output_dir)

    if use_tpu:
        # Cloud storage has bert-base cased, bert-base uncased, and bert-large uncased, but NOT bert-large cased.
        if bert_large and (not do_lower_case):
            raise ValueError("You need to download bert-large cased and upload it to your own google cloud storage bucket!")

            # Info on how you do this:
            # gsutil cp -R gs://cloud-tpu-checkpoints/bert/cased_L-12_H-768_A-12/ gs://MYBUCKET/bert_models/
            # gsutil cp -R gs://cloud-tpu-checkpoints/bert/uncased_L-12_H-768_A-12/ gs://MYBUCKET/bert_models/
            # gsutil cp -R gs://cloud-tpu-checkpoints/bert/uncased_L-24_H-1024_A-16/ gs://MYBUCKET/bert_models/
            # wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip && unzip cased_L-24_H-1024_A-16.zip && gsutil cp -R cased_L-24_H-1024_A-16/ gs://MYBUCKET/bert_models/


        path = 'gs://cloud-tpu-checkpoints/bert/{}_{}'.format(
            'uncased' if do_lower_case else 'cased',
            'L-24_H-1024_A-16' if bert_large else 'L-12_H-768_A-12',
        )

        if tpu_name is None:
            raise ValueError("Need to set TPU name when running on a TPU")
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            tpu_name, zone=tpu_zone, project=gcp_project)
    else:
        if bert_large:
            if do_lower_case:
                url = 'https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip'
            else:
                url = 'https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip'
        else:
            if do_lower_case:
                url = 'https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip'
            else:
                url = 'https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip'
        download_path = url.split('/')[-1]
        unzip_path = download_path.split('.')[0]

        ####
        if not os.path.exists(unzip_path):
            response = requests.get(url, stream=True)
            with open(download_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=512):
                    if chunk:  # filter out keep-alive new chunks
                        handle.write(chunk)
            with zipfile.ZipFile(download_path) as zf:
                zf.extractall()

        print("BERT HAS BEEN DOWNLOADED")
        path = os.path.join(os.getcwd(), unzip_path)

        # TPU shit
        tpu_cluster_resolver = None

    # Not sure yet if os.path.join works on gs:// files.
    vocab_file = f'{path}/vocab.txt'
    init_checkpoint = f'{path}/bert_model.ckpt'
    bert_config = modeling.BertConfig.from_json_file(f'{path}/bert_config.json')
    tokenization.validate_case_matches_checkpoint(do_lower_case, init_checkpoint)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=master,
        model_dir=output_dir,
        save_checkpoints_steps=iterations_per_loop,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=iterations_per_loop,
            num_shards=num_tpu_cores,
            per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))

    if max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (max_seq_length, bert_config.max_position_embeddings))

    return run_config, bert_config, tokenizer, init_checkpoint


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks. OR if it's a list, you have N of these.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True,
                 masked_lm_positions=None,
                 masked_lm_ids=None,
                 masked_lm_weights=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_ids = masked_lm_ids
        self.masked_lm_weights = masked_lm_weights


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        cand_indexes.append(i)

    random.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)

        masked_token = None
        # 80% of the time, replace with [MASK]
        if random.random() < 0.8:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if random.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = vocab_words[random.randint(0, len(vocab_words) - 1)]

        output_tokens[index] = masked_token

        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)


def bert_pad(tokens_a, tokens_b, max_seq_length, tokenizer, do_mask=True, max_predictions_per_seq=10):
    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    if do_mask:
        # Create the masked tokens
        tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
            tokens,
            # masked_lm_prob=.15,
            masked_lm_prob=.05,
            max_predictions_per_seq=max_predictions_per_seq,
            vocab_words=tokenizer.vocab_words)

        masked_lm_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        # Zero pad
        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)
    else:
        masked_lm_positions, masked_lm_ids, masked_lm_weights = None, None, None

    # NOW do ittt
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights, tokens


def convert_single_example(ex_index, example, max_seq_length, tokenizer, label_length=1, do_mask=True, max_predictions_per_seq=10):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    # This will only be for testing so OK
    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[[0] * max_seq_length] * label_length,
            input_mask=[[0] * max_seq_length] * label_length,
            segment_ids=[[0] * max_seq_length] * label_length,
            label_id=0,
            masked_lm_positions=[[0] * max_predictions_per_seq] * label_length,
            masked_lm_ids=[[0] * max_predictions_per_seq] * label_length,
            masked_lm_weights=[[0.0] * max_predictions_per_seq] * label_length,
            is_real_example=False)

    tokens_bs = [tokenizer.tokenize(tb) for tb in example.text_b]
    if example.text_a:
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_as = [[x for x in tokens_a] for i in range(len(tokens_bs))]

        for i in range(len(tokens_bs)):
            _truncate_seq_pair(tokens_as[i], tokens_bs[i], max_seq_length - 3)

        input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights, tokens = zip(
            *[bert_pad(token_a, token_b, max_seq_length, tokenizer, do_mask=do_mask,
                       max_predictions_per_seq=max_predictions_per_seq)
              for token_a, token_b in zip(tokens_as, tokens_bs)])

    else:
        tokens_bs = [x[-(max_seq_length - 2):] for x in tokens_bs]

        input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights, tokens = zip(
            *[bert_pad(token_b, None, max_seq_length, tokenizer, do_mask=do_mask,
                       max_predictions_per_seq=max_predictions_per_seq)
              for token_b in tokens_bs])
    if not do_mask:
        masked_lm_positions, masked_lm_ids, masked_lm_weights = None, None, None

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("label: %s (id = %d)" % (example.label, example.label))

        for i, these_tokens in enumerate(tokens):
            tf.logging.info("Ending: {} / {}".format(i, len(tokens)))
            tf.logging.info("tokens: {}".format(' '.join([tokenization.printable_text(x) for x in these_tokens])))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids[i]]))
            tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask[i]]))
            tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids[i]]))
            if do_mask:
                tf.logging.info("masked_lm_positions: %s" % " ".join([str(x) for x in masked_lm_positions[i]]))
                tf.logging.info("masked_lm_ids: %s" % " ".join([str(x) for x in masked_lm_ids[i]]))
                tf.logging.info("masked_lm_weights: %s" % " ".join([str(x) for x in masked_lm_weights[i]]))

    assert len(input_ids) == label_length

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=example.label,
        masked_lm_positions=masked_lm_positions,
        masked_lm_ids=masked_lm_ids,
        masked_lm_weights=masked_lm_weights,
        is_real_example=True)
    return feature


def file_based_convert_examples_to_features(
        examples, max_seq_length, tokenizer, output_file, label_length=1, do_mask=True, max_predictions_per_seq=10):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, max_seq_length, tokenizer, label_length=label_length,
                                         do_mask=do_mask,
                                         max_predictions_per_seq=max_predictions_per_seq)

        # Flatten here
        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        # We will have as input
        # [a0 a1 a2 a3]
        # [b0 b1 b2 b3]
        # seq_length is the last dimension
        def create_int_feature_flat(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=[v for x in values for v in x]))
            return f

        def create_float_feature_flat(values):
            f = tf.train.Feature(float_list=tf.train.FloatList(value=[v for x in values for v in x]))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature_flat(feature.input_ids)
        features["input_mask"] = create_int_feature_flat(feature.input_mask)
        features["segment_ids"] = create_int_feature_flat(feature.segment_ids)
        if do_mask:
            features["masked_lm_positions"] = create_int_feature_flat(feature.masked_lm_positions)
            features["masked_lm_ids"] = create_int_feature_flat(feature.masked_lm_ids)
            features['masked_lm_weights'] = create_float_feature_flat(feature.masked_lm_weights)

        features["label_ids"] = create_int_feature([feature.label_id])
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, label_length, is_training,
                                drop_remainder, max_predictions_per_seq=10, do_mask=True,
                                buffer_size=100):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([label_length, seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([label_length, seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([label_length, seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    if do_mask:
        name_to_features["masked_lm_positions"] = tf.FixedLenFeature([label_length, max_predictions_per_seq], tf.int64)
        name_to_features["masked_lm_ids"] = tf.FixedLenFeature([label_length, max_predictions_per_seq], tf.int64)
        name_to_features["masked_lm_weights"] = tf.FixedLenFeature([label_length, max_predictions_per_seq], tf.float32)

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=buffer_size)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)  # <- remove from beginning?
        else:
            tokens_b.pop()
