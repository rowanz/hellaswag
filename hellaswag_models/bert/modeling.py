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
"""The main BERT model and related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from adversarial_filtering.bert import optimization
from adversarial_filtering.bert.modeling import get_shape_list, BertModel, get_cls_output, \
    get_assignment_map_from_checkpoint, get_masked_lm_output


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, do_mask=False):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # Create model with aux loss
        print(get_shape_list(input_ids, expected_rank=3))
        batch_size, n_way, seq_length = get_shape_list(input_ids, expected_rank=3)

        # THIS IS JUST FOR bert_experiments/, not for AF.
        assert n_way == num_labels

        model = BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=tf.reshape(input_ids, [batch_size * n_way, seq_length]),
            input_mask=tf.reshape(input_mask, [batch_size * n_way, seq_length]),
            token_type_ids=tf.reshape(segment_ids, [batch_size * n_way, seq_length]),
            use_one_hot_embeddings=use_one_hot_embeddings)

        (cls_loss, per_example_cls_loss, logits) = get_cls_output(
            model.get_pooled_output(),
            is_training=is_training,
            num_labels=n_way,
            labels=label_ids,
        )

        if do_mask and is_training:
            masked_lm_positions = features["masked_lm_positions"]
            masked_lm_ids = features["masked_lm_ids"]
            masked_lm_weights = features["masked_lm_weights"]
            masked_shape = get_shape_list(masked_lm_positions, expected_rank=3)
            assert n_way == masked_shape[1]
            assert batch_size == masked_shape[0]

            (masked_lm_loss, masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
                bert_config, model.get_sequence_output(), model.get_embedding_table(),
                tf.reshape(masked_lm_positions, [batch_size * n_way, masked_shape[2]]),
                tf.reshape(masked_lm_ids, [batch_size * n_way, masked_shape[2]]),
                tf.reshape(masked_lm_weights, [batch_size * n_way, masked_shape[2]]))
            tf.logging.info("==== Incorporating Mask LM Loss ====")
            total_loss = cls_loss + masked_lm_loss
        else:
            total_loss = cls_loss

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint and (init_checkpoint != 'False'):
            (assignment_map, initialized_variable_names
             ) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            if use_tpu:
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    scaffold_fn=scaffold_fn)
            else:
                accuracy = tf.metrics.accuracy(label_ids, tf.argmax(logits, axis=-1, output_type=tf.int32))

                if do_mask:
                    logging_info = {
                        'loss': tf.metrics.mean(per_example_cls_loss)[1] + tf.metrics.mean(masked_lm_loss)[1],
                        'lm_loss': tf.metrics.mean(masked_lm_loss)[1],
                    }
                else:
                    logging_info = {
                        'loss': tf.metrics.mean(per_example_cls_loss)[1],
                    }
                logging_info['cls_loss'] = tf.metrics.mean(per_example_cls_loss)[1]
                logging_info['accuracy'] = accuracy[1]

                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    training_hooks=[tf.train.LoggingTensorHook(logging_info, every_n_iter=100)],
                    scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=predictions, weights=is_real_example)
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            eval_metrics = (metric_fn,
                            [per_example_cls_loss, label_ids, logits, is_real_example])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"scores": logits},
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn
