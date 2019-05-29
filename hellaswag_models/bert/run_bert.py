# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

import tensorflow as tf
from adversarial_filtering.bert.dataloader import setup_bert, InputExample, PaddingInputExample, \
    file_based_convert_examples_to_features, file_based_input_fn_builder, _truncate_seq_pair, gcs_agnostic_open, _save_np, _softmax
from hellaswag_models.bert.modeling import model_fn_builder
import numpy as np
import os
import json
import pandas as pd
import time

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("predict_val", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "predict_test", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 16, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 16, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 2e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_epochs", 3,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "bert_large", False,
    "Use bert large"
)

flags.DEFINE_integer(
    "num_labels", 4,
    "number of labels 2 use.")

flags.DEFINE_integer(
    "num_train", -1,
    "Number of training. Or, -1, for everything")

flags.DEFINE_integer(
    "vals_per_epoch", 1,
    "How often to validate per epoch.")

flags.DEFINE_bool(
    "endingonly", False,
    "Use ONLY THE ENDING"
)


# just use it for something?
hi = FLAGS.use_tpu
# flags.FLAGS._parse_flags()
print("Using \n\n{}\n\n".format({k: v.value for k, v in FLAGS.__flags.items()}), flush=True)

tf.logging.set_verbosity(tf.logging.INFO)

if tf.gfile.Exists(FLAGS.output_dir):
    raise ValueError(f"The output directory {FLAGS.output_dir} exists!")


def _part_a(item):
    if FLAGS.endingonly:
        return ''
    if 'ctx_a' not in item:
        return item['ctx']
    if 'ctx' not in item:
        return item['ctx_a']
    if len(item['ctx']) == len(item['ctx_a']):
        return item['ctx']
    return item['ctx_a']

def _part_bs(item):
    if ('ctx_b' not in item) or len(item['ctx_b']) == 0:
        return item['endings']
    return ['{} {}'.format(item['ctx_b'], x) for x in item['endings']]

# Load dataset examples
examples = {'train': [], 'val': [], 'test': []}
for split in ['train', 'val', 'test']:
    with open(f'../data/hellaswag_{split}.jsonl', 'r') as f:
        for l in f:
            item = json.loads(l)
            examples[split].append(
                InputExample(
                    guid='{}-{}'.format(item['split'], len(examples[item['split']])),
                    text_a=_part_a(item),
                    text_b=_part_bs(item),
                    label=0 if split == 'test' else item['label'],
                )
            )
train_examples = examples['train'][:FLAGS.num_train] if FLAGS.num_train >= 0 else examples['train']
tf.logging.info("@@@@@ {} training examples (num_train={}) @@@@@".format(len(train_examples), FLAGS.num_train))

val_examples = examples['val']
test_examples = examples['test']

run_config, bert_config, tokenizer, init_checkpoint = setup_bert(
    use_tpu=FLAGS.use_tpu,
    do_lower_case=FLAGS.do_lower_case,
    bert_large=FLAGS.bert_large,
    output_dir=FLAGS.output_dir,
    tpu_name=FLAGS.tpu_name,
    tpu_zone=FLAGS.tpu_zone,
    gcp_project=FLAGS.gcp_project,
    master=FLAGS.master,
    iterations_per_loop=FLAGS.iterations_per_loop,
    num_tpu_cores=FLAGS.num_tpu_cores,
    max_seq_length=FLAGS.max_seq_length,
)

# Training
if FLAGS.do_train:
    num_train_steps = (len(train_examples) // FLAGS.train_batch_size) * FLAGS.num_train_epochs
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    assert num_train_steps > 0
else:
    num_train_steps = None
    num_warmup_steps = None

model_fn = model_fn_builder(
    bert_config=bert_config,
    num_labels=FLAGS.num_labels,
    init_checkpoint=FLAGS.init_checkpoint if FLAGS.init_checkpoint else init_checkpoint,
    learning_rate=FLAGS.learning_rate,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    use_tpu=FLAGS.use_tpu,
    use_one_hot_embeddings=FLAGS.use_tpu,
    do_mask=False,
)

estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=FLAGS.use_tpu,
    model_fn=model_fn,
    config=run_config,
    train_batch_size=FLAGS.train_batch_size,
    eval_batch_size=FLAGS.predict_batch_size,
    predict_batch_size=FLAGS.predict_batch_size)

def _predict(examples, name='eval'):
    num_actual_examples = len([x for x in examples if not isinstance(x, PaddingInputExample)])
    if FLAGS.use_tpu:
        # TPU requires a fixed batch size for all batches, therefore the number
        # of examples must be a multiple of the batch size, or else examples
        # will get dropped. So we pad with fake examples which are ignored
        # later on.
        while len(examples) % FLAGS.predict_batch_size != 0:
            examples.append(PaddingInputExample())

    predict_file = os.path.join(FLAGS.output_dir, f"{name}.tf_record")
    if not tf.gfile.Exists(predict_file):
        tf.logging.info(f"***** Recreating {name} file {predict_file} *****")
        file_based_convert_examples_to_features(examples,
                                                FLAGS.max_seq_length, tokenizer,
                                                predict_file, label_length=FLAGS.num_labels,
                                                do_mask=False,
                                                max_predictions_per_seq=0)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(examples), num_actual_examples,
                    len(examples) - num_actual_examples)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    eval_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        label_length=FLAGS.num_labels,
        is_training=False,
        drop_remainder=FLAGS.use_tpu,
        max_predictions_per_seq=0,
        do_mask=False,
    )

    tf.logging.info(f"***** Running {name} *****")
    tf.logging.info("  Num examples = %d", num_actual_examples)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    probs = np.zeros((num_actual_examples, FLAGS.num_labels), dtype=np.float32)
    for i, res in enumerate(estimator.predict(input_fn=eval_input_fn, yield_single_examples=True)):
        if i < num_actual_examples:
            probs[i] = res['scores']
    probs = _softmax(probs)
    return probs


if FLAGS.do_train:
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")

    if not tf.gfile.Exists(train_file):
        tf.logging.info(f"***** Recreating training file at {train_file} *****")
        file_based_convert_examples_to_features(
            train_examples, FLAGS.max_seq_length, tokenizer, train_file,
            label_length=FLAGS.num_labels, do_mask=False, max_predictions_per_seq=0)

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Num epochs = %d", FLAGS.num_train_epochs)
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)

    accuracies = []

    num_steps = 0
    for i in range(FLAGS.num_train_epochs):
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            label_length=FLAGS.num_labels,
            is_training=True,
            drop_remainder=True,
            max_predictions_per_seq=0,
            do_mask=False,
        )
        for val_round in range(FLAGS.vals_per_epoch):
            steps_this_round = num_train_steps//(FLAGS.num_train_epochs * FLAGS.vals_per_epoch)
            estimator.train(input_fn=train_input_fn, steps=steps_this_round)
            num_steps += steps_this_round

            val_probs = _predict(val_examples, name='val')
            _save_np(os.path.join(FLAGS.output_dir, f'val-probs-{i}.npy'), val_probs)

            val_labels = np.array([x.label for x in val_examples])
            acc = np.mean(val_probs.argmax(1) == val_labels)
            tf.logging.info("\n\n&&&& Accuracy on epoch{} ({}iter) is {:.3f} &&&&\n".format(i, num_steps, acc))
            accuracies.append({'num_steps': num_steps,
                               'num_epochs': i,
                               'val_round': val_round,
                               'acc': acc,
                               })

    accuracies = pd.DataFrame(accuracies)
    accuracies.index.name = 'iteration'
    accuracies.to_csv(os.path.join(FLAGS.output_dir, 'valaccs.csv'))


if FLAGS.predict_val:
    probs = _predict(val_examples, name='val')
    val_labels = np.array([x.label for x in val_examples])

    acc = np.mean(probs.argmax(1) == val_labels)
    tf.logging.info("\n\n&&&& VAL Acc IS {:.3f} &&&&\n".format(acc))
    _save_np(os.path.join(FLAGS.output_dir, f'val-probs.npy'), probs)

if FLAGS.predict_test:
    probs = _predict(test_examples, name='test')
    _save_np(os.path.join(FLAGS.output_dir, f'test-probs.npy'), probs)
