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

from adversarial_filtering.bert import modeling
from adversarial_filtering.bert import optimization
from adversarial_filtering.bert import tokenization
import tensorflow as tf
import requests
import zipfile
import os
import json
from adversarial_filtering.bert.dataloader import setup_bert, InputExample, PaddingInputExample, \
    file_based_convert_examples_to_features, file_based_input_fn_builder, _truncate_seq_pair, gcs_agnostic_open
from adversarial_filtering.bert.modeling import model_fn_builder
import numpy as np
import pandas as pd
from tqdm import tqdm

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")
flags.DEFINE_string(
    "assignments_dir", "../../raw_data/",
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_bool(
    "wikihow", True,
    "True if we should do WIKIHOW, otherwise activitynet")

## Other parameters

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("max_learning_rate", 2e-4, "The initial learning rate for Adam.")
flags.DEFINE_float("min_learning_rate", 5e-5, "The initial learning rate for Adam.")


flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

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

flags.DEFINE_string(
    "log_csv", 'results.csv',
    "We will log the AF results here."
)

flags.DEFINE_string(
    "assignments_fn", '',
    "Initialize the assignments here."
)

flags.DEFINE_integer(
    "num_negatives", 9,
    "number of negatives."
)

flags.DEFINE_integer(
    "subsample", 9,
    "subsample these number of negatives on each iteration from the assignments. must be 1 <=x<= num_negatives."
)

flags.DEFINE_bool(
    "weight_af", False,
    "Weight the AF probabilities"
)

flags.DEFINE_bool(
    "lm_loss", False,
    "have LM loss"
)

flags.DEFINE_integer("random_seed", 123456, "random seed to use")

def _part_a(item):
    if FLAGS.wikihow:
        return item['ctx']
    return item['ctx_a']

def _part_b_gt(item):
    if FLAGS.wikihow:
        return item['gt_detok']
    return '{} {}'.format(item['ctx_b'], item['gt_detok'])

def _part_b_gen(item, i):
    if FLAGS.wikihow:
        return item['gens'][i]
    return '{} {}'.format(item['ctx_b'], item['gens'][i])


def train(train_examples, run_config, bert_config, init_checkpoint, tokenizer, hyperparams):
    num_train_steps = int(
        len(train_examples) / hyperparams['batch_size'] * hyperparams['num_train_epochs'])
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=min(FLAGS.num_negatives, FLAGS.subsample)+1,
        init_checkpoint=init_checkpoint,
        learning_rate=hyperparams['learning_rate'],
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        do_mask=FLAGS.lm_loss,
    )

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=hyperparams['batch_size'],
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    # if FLAGS.do_train:
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    file_based_convert_examples_to_features(
        train_examples, hyperparams['max_seq_length'], tokenizer, train_file, label_length=min(FLAGS.num_negatives, FLAGS.subsample)+1,
        do_mask=FLAGS.lm_loss,
        max_predictions_per_seq=hyperparams['max_predictions_per_seq'])
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", hyperparams['batch_size'])
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=hyperparams['max_seq_length'],
        label_length=min(FLAGS.num_negatives, FLAGS.subsample)+1, # num_labels
        is_training=True,
        drop_remainder=True,
        max_predictions_per_seq=hyperparams['max_predictions_per_seq'],
        do_mask=FLAGS.lm_loss,
    )
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    return estimator


def predict(val_examples, tokenizer, estimator, hyperparams):
    predict_examples = val_examples
    num_actual_predict_examples = len(predict_examples)
    if FLAGS.use_tpu:
        # TPU requires a fixed batch size for all batches, therefore the number
        # of examples must be a multiple of the batch size, or else examples
        # will get dropped. So we pad with fake examples which are ignored
        # later on.
        while len(predict_examples) % FLAGS.predict_batch_size != 0:
            predict_examples.append(PaddingInputExample())

    # TODO: this is pretty slow, we probably want to tokenize only once
    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    file_based_convert_examples_to_features(predict_examples,
                                            hyperparams['max_seq_length'], tokenizer,
                                            predict_file, label_length=1,
                                            do_mask=False)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(predict_examples), num_actual_predict_examples,
                    len(predict_examples) - num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=hyperparams['max_seq_length'],
        label_length=1,
        is_training=False,
        drop_remainder=predict_drop_remainder,
        do_mask=False,
    )

    result = estimator.predict(input_fn=predict_input_fn, yield_single_examples=True)
    human_probs = np.array([x['scores'][0] for i, x in enumerate(result) if i < num_actual_predict_examples])
    # human_probs = np.concatenate([x['scores'][:,0] for i, x in enumerate(result)],0)
    assert human_probs.size == num_actual_predict_examples
    return human_probs


def remap(swag_dataset, remap_inds, tokenizer, estimator, hyperparams):
    # no optimization here...

    # Score everything
    predict_list = []
    # cur_assignments = np.stack([swag_dataset[i]['assignment'][-1] for i in remap_inds])
    for ex_i in remap_inds:
        item = swag_dataset[ex_i]
        predict_list.append(InputExample(guid=f'{ex_i}-human', text_a=_part_a(item),
                                         text_b=[_part_b_gt(item)],
                                         label=0))
        for j, gen in enumerate(item['gens']):
            predict_list.append(InputExample(guid=f'{ex_i}-{j}', text_a=_part_a(item),
                                             text_b=[_part_b_gen(item, j)],
                                             label=0))

    all_probs = predict(val_examples=predict_list, tokenizer=tokenizer, estimator=estimator,
                        hyperparams=hyperparams)

    # each row is [human, machines]
    old_assigned_probs = []
    start_idx = 0
    # First get the old probs
    for ex_i in remap_inds:
        item = swag_dataset[ex_i]
        hp_i = all_probs[start_idx]
        mp_i = all_probs[start_idx + 1:start_idx + 1 + len(item['gens'])]
        start_idx += 1 + len(item['gens'])

        cur_assign = np.array(item['assignment'][-1])
        old_assigned_probs.append(np.append(hp_i, mp_i[cur_assign]))

    old_assigned_probs = np.stack(old_assigned_probs)
    binary_acc = (np.concatenate([old_assigned_probs[:,[0,k]] for k in range(1, old_assigned_probs.shape[1])], 0).argmax(1) == 0).mean()

    # If the model gets x binary classification accuracy, then we should change (x-.5)% of the test data points.
    # NUM2CHANGE_MAX = (old_assigned_probs.shape[1]-1)*(binary_acc-0.5)
    # ehh scratch that, sounds complicated

    if binary_acc <= 0.525:
        NUM2CHANGE_MAX = 0
    elif binary_acc <= 0.575:
        NUM2CHANGE_MAX = 1      # 2% of the dataset
    else:
        NUM2CHANGE_MAX = 2      # 4% of the dataset
    print("CHANGING {} per".format(NUM2CHANGE_MAX))


    new_assigned_probs = []

    start_idx = 0
    num_changed = 0
    for ex_i in remap_inds:
        item = swag_dataset[ex_i]
        hp_i = all_probs[start_idx]
        mp_i = all_probs[start_idx + 1:start_idx + 1 + len(item['gens'])]
        start_idx += 1 + len(item['gens'])

        cur_assign = np.array(item['assignment'][-1])
        new_assign = cur_assign.copy()

        # Needs to be better than some threshold (here we use human perf.) and not already assigned
        adversarial_inds = np.where((mp_i > hp_i) & (~np.in1d(np.arange(mp_i.shape[0]), cur_assign)))[0]
        adversarial_inds = adversarial_inds[np.argsort(-mp_i[adversarial_inds])]  # Order from best <- worst

        # Switch some
        easy_inds = np.where(mp_i[cur_assign] < hp_i)[0]

        assert np.intersect1d(cur_assign[easy_inds], adversarial_inds).size == 0
        num2change = min(NUM2CHANGE_MAX, adversarial_inds.shape[0], easy_inds.shape[0])

        if num2change != 0:
            num_changed += 1

            # Weight the hard indices using a softmax
            adv_p = np.exp(mp_i[adversarial_inds]) # Probability that each one would be picked
            adv_p = adv_p / adv_p.sum()

            # Weight the easy indices using a softmax
            easy_p = np.exp(-mp_i[cur_assign[easy_inds]]) # Inverse probability that each one would be picked
            easy_p = easy_p / easy_p.sum()

            easy_sel = np.random.choice(easy_inds, replace=False, size=num2change,
                                        p=easy_p if FLAGS.weight_af else None)
            adv_sel = np.random.choice(adversarial_inds, replace=False, size=num2change,
                                       p=adv_p if FLAGS.weight_af else None)
            if num_changed <= 3:
                print("Changing {}".format(num2change), flush=True)
                print("Remapping: Easy inds\n{} with scores:\n{}\n---\nprobs {}\n---\nand chose {}\n\n---".format(
                    easy_inds, mp_i[cur_assign[easy_inds]], easy_p if FLAGS.weight_af else '1/n', easy_sel,
                ), flush=True)
                print("Remapping: Hard inds\n{} with scores:\n{}\n---\nprobs {}\n---\nand chose {}\n\n---".format(
                    adversarial_inds, mp_i[adversarial_inds], adv_p if FLAGS.weight_af else '1/n', adv_sel,
                ), flush=True)

            new_assign[easy_sel] = adv_sel
            assert len(set(new_assign.tolist())) == new_assign.size

        item['assignment'] = item['assignment'] + [new_assign.tolist()]
        new_assigned_probs.append(np.append(hp_i, mp_i[new_assign]))

    new_assigned_probs = np.stack(new_assigned_probs)
    if tf.gfile.Exists(os.path.join(FLAGS.assignments_dir, FLAGS.log_csv)):
        pd_log = pd.read_csv(os.path.join(FLAGS.assignments_dir, FLAGS.log_csv))
    else:
        pd_log = pd.DataFrame(columns=['changed'] + [f'before_{i}way' for i in range(2, old_assigned_probs.shape[1]+1)] +
                                      [f'after_{i}way' for i in range(2, old_assigned_probs.shape[1]+1)] + sorted(
            hyperparams.keys()))

    # add new series
    new_row = {k: v for k, v in hyperparams.items()}
    new_row['changed'] = num_changed / len(remap_inds)
    for k in range(2, old_assigned_probs.shape[1]+1):
        new_row[f'before_{k}way'] = (old_assigned_probs[:, :k].argmax(1) == 0).mean()
        new_row[f'after_{k}way'] = (new_assigned_probs[:, :k].argmax(1) == 0).mean()
        print("In a {}-way setup, old accuracy is {:.3f}%. changed {:.3f}% of val. New accuracy is {:.3f}".format(
            k, new_row[f'before_{k}way']*100, new_row['changed']*100, new_row[f'after_{k}way']*100), flush=True)

    # Dump to file
    if not tf.gfile.Exists(FLAGS.assignments_dir):
        tf.gfile.MakeDirs(FLAGS.assignments_dir)

    pd_log = pd_log.append(pd.Series(new_row, name=pd_log.shape[0]))
    pd_log.to_csv(os.path.join(FLAGS.assignments_dir, FLAGS.log_csv), index=False)

    save_fn = os.path.join(FLAGS.assignments_dir, 'swag2gen_af.jsonl')
    for i in range(1000):
        if not tf.gfile.Exists(os.path.join(FLAGS.assignments_dir, f'swag2gen_af_{i}.jsonl')):
            save_fn = os.path.join(FLAGS.assignments_dir, f'swag2gen_af_{i}.jsonl')
            break

    with gcs_agnostic_open(save_fn, mode='w') as f:
        for item in swag_dataset:
            f.write(json.dumps(item) + '\n')

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    np.random.seed(FLAGS.random_seed)

    # First setup the dataset
    swag_gens = []

    if tf.gfile.Exists(FLAGS.assignments_fn):
        print(f"LOADING FROM {FLAGS.assignments_fn}", flush=True)

    with gcs_agnostic_open(FLAGS.assignments_fn if tf.gfile.Exists(FLAGS.assignments_fn) else '../../raw_data/swag2genv2.jsonl',
              'r') as f:
        for lineno, line in enumerate(f):
            item = json.loads(line)
            if len(item['gens']) >= 15:
                if 'assignment' not in item:
                    item['assignment'] = np.random.choice(len(item['gens']), size=(1, FLAGS.num_negatives),
                                                          replace=False).tolist()
                swag_gens.append(item)
            #
            # # # Turn on for debugging
            # if lineno >= 200:
            #     break
    np.random.seed(FLAGS.random_seed + max(len(x['assignment']) for x in swag_gens))

    for i in range(100):
        # Do one round of training and validation
        # We could randomize the hyperparameters here but idk
        hyperparams = {
            'batch_size': FLAGS.train_batch_size,
            'num_train_epochs': FLAGS.num_train_epochs,
            'learning_rate': float(np.exp(np.random.uniform(np.log(FLAGS.min_learning_rate),
                                                            np.log(FLAGS.max_learning_rate)))),
            'do_lower_case': bool(np.random.binomial(1, p=0.5)),
            'max_seq_length': FLAGS.max_seq_length,
            'bert_large': FLAGS.bert_large,
            'max_predictions_per_seq': 16,
        }
        tf.logging.info("\n$$$$$ Iteration{}:$$$$$$\n{}\n\n".format(i, hyperparams))
        run_config, bert_config, tokenizer, init_checkpoint = setup_bert(
            use_tpu=FLAGS.use_tpu,
            do_lower_case=hyperparams['do_lower_case'],
            bert_large=hyperparams['bert_large'],
            output_dir=FLAGS.output_dir,
            tpu_name=FLAGS.tpu_name,
            tpu_zone=FLAGS.tpu_zone,
            gcp_project=FLAGS.gcp_project,
            master=FLAGS.master,
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_tpu_cores=FLAGS.num_tpu_cores,
            max_seq_length=hyperparams['max_seq_length'],
        )

        # # Let's make sure that we have the right sequence length. Last time I ran this it said 135
        # max_seq_length = []
        # print("Double checking seq length")
        # for i, item in enumerate(tqdm(swag_gens)):
        #     ctx_length = len(tokenizer.tokenize(item['ctx']))
        #     gens_length = max([len(tokenizer.tokenize(x)) for x in item['gens'] + [item['gt_detok']]])
        #     max_seq_length.append(3 + gens_length + ctx_length)
        #     if i % 100 == 0:
        #         print(max(max_seq_length))
        #     if i > 10000:
        #         assert False
        # print("Max sequence length is {}".format(max(max_seq_length)), flush=True)
        # assert False

        swag_examples_perm = np.random.permutation(len(swag_gens)).tolist()
        train_examples = []
        n_train = int(len(swag_gens) * .8)

        for i, ex_i in enumerate(swag_examples_perm[:n_train]):
            item = swag_gens[ex_i]

            a2use = np.random.choice(item['assignment'][-1], size=FLAGS.subsample, replace=False).tolist() if \
                FLAGS.subsample <= len(item['assignment'][-1]) else item['assignment'][-1]

            train_examples.append(InputExample(
                guid=f'{ex_i}',
                text_a=_part_a(item),
                text_b=[_part_b_gt(item)] + [
                    _part_b_gen(item, m_idx) for m_idx in a2use
                ],
                label=0,
            ))

        estimator = train(train_examples=train_examples, run_config=run_config,
                          bert_config=bert_config, init_checkpoint=init_checkpoint, tokenizer=tokenizer,
                          hyperparams=hyperparams)
        # human_probs = predict(val_examples=val_examples, tokenizer=tokenizer, estimator=estimator)

        # bert_acc = (human_probs.reshape((human_probs.size // 10, 10)).argmax(1) == 0).mean()
        # print("Bert accuracy on the val set is {:.3f}".format(bert_acc))
        remap(swag_gens,
              remap_inds=swag_examples_perm[n_train:],
              tokenizer=tokenizer,
              estimator=estimator,
              hyperparams=hyperparams
        )


if __name__ == "__main__":
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
