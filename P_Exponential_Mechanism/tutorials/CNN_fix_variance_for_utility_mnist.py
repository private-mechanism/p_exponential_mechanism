# Copyright 2018, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training a CNN on MNIST with differentially private SGD optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags


import numpy as np

import tensorflow as tf



# from tensorflow_privacy.privacy.analysis.privacy_ledger import PrivacyLedger
from tensorflow_privacy.privacy.optimizers import flatten_optimizer
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
config = tf.ConfigProto()
config.allow_soft_placement=True 
config.gpu_options.per_process_gpu_memory_fraction=0.5
config.gpu_options.allow_growth = True


session = tf.Session(config=config)


GradientDescentOptimizer = tf.compat.v1.train.GradientDescentOptimizer

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD. If False, '
    'train with vanilla SGD.')
flags.DEFINE_float('learning_rate', .15, 'Learning rate for training')
flags.DEFINE_float('noise_variance', 0,
                   'variance parameter of our proposed p-exponential mechanism')
flags.DEFINE_float('exponents', 1.2,'exponents of the p-exponential distribution where the noises are sampled')
flags.DEFINE_float('l2_norm_clip', 4, 'Clipping norm')
flags.DEFINE_integer('batch_size', 200, 'Batch size')
flags.DEFINE_integer('epochs', 60, 'Number of epochs')
flags.DEFINE_integer('microbatches', 200, 'Number of microbatches ''(must evenly divide batch_size)')
flags.DEFINE_string('model_dir', None, 'Model directory')


def load_mnist():
  """Loads MNIST and preprocesses to combine training and validation data."""
  train, test = tf.keras.datasets.mnist.load_data()
  train_data, train_labels = train
  test_data, test_labels = test

  train_data = np.array(train_data, dtype=np.float32) / 255
  test_data = np.array(test_data, dtype=np.float32) / 255

  train_labels = np.array(train_labels, dtype=np.int32)
  test_labels = np.array(test_labels, dtype=np.int32)

  assert train_data.min() == 0.
  assert train_data.max() == 1.
  assert test_data.min() == 0.
  assert test_data.max() == 1.
  assert train_labels.ndim == 1
  assert test_labels.ndim == 1

  return train_data, train_labels, test_data, test_labels

def cnn_model_fn(features, labels, mode):
  """Model function for a CNN."""
  # Define CNN architecture using tf.keras.layers.
  input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])
  y = tf.keras.layers.Conv2D(16, 8,
                             strides=2,
                             padding='same',
                             activation='relu').apply(input_layer)
  y = tf.keras.layers.MaxPool2D(2, 1).apply(y)
  y = tf.keras.layers.Conv2D(32, 4,
                             strides=2,
                             padding='valid',
                             activation='relu').apply(y)
  y = tf.keras.layers.MaxPool2D(2, 1).apply(y)
  y = tf.keras.layers.Flatten().apply(y)
  y = tf.keras.layers.Dense(32, activation='relu').apply(y)
  logits = tf.keras.layers.Dense(10).apply(y)

  vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=tf.cast(labels, tf.int32), logits=logits)
  scalar_loss = tf.reduce_mean(input_tensor=vector_loss)
  print("loss:", scalar_loss)
  # Configure the training op (for TRAIN mode).
  global_step = tf.compat.v1.train.get_global_step()
  learing_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.96, staircase=False)
  if mode == tf.estimator.ModeKeys.TRAIN:
      if FLAGS.dpsgd:
          # ledger = PrivacyLedger(
          #     population_size=60000,
          #     selection_probability=(FLAGS.batch_size / 60000))
          optimizer = flatten_optimizer.DP_fixedvarianceoptimizer(
              l2_norm_clip=FLAGS.l2_norm_clip,
              exponents=FLAGS.exponents,
              noise_variance=FLAGS.noise_variance,
              num_microbatches=FLAGS.microbatches,
              learning_rate=learing_rate)
          opt_loss = vector_loss
      else:
          optimizer = GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
          opt_loss = scalar_loss
      train_op = optimizer.minimize(loss=opt_loss, global_step=global_step)
      # In the following, we pass the mean of the loss (scalar_loss) rather than
      # the vector_loss because tf.estimator requires a scalar loss. This is only
      # used for evaluation and debugging by tf.estimator. The actual loss being
      # minimized is opt_loss defined above and passed to optimizer.minimize().
      return tf.estimator.EstimatorSpec(mode=mode,
                                        loss=scalar_loss,
                                        train_op=train_op)

  # Add evaluation metrics (for EVAL mode).
  elif mode == tf.estimator.ModeKeys.EVAL:
      eval_metric_ops = {
          'accuracy':
              tf.compat.v1.metrics.accuracy(
                  labels=labels,
                  predictions=tf.argmax(input=logits, axis=1))
      }

      return tf.estimator.EstimatorSpec(mode=mode,
                                        loss=scalar_loss,
                                        eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  if FLAGS.dpsgd and FLAGS.batch_size % FLAGS.microbatches != 0:
    raise ValueError('Number of microbatches should divide evenly batch_size')


  train_data, train_labels, test_data, test_labels = load_mnist()
  mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,
                                            model_dir=FLAGS.model_dir
                                            )
  train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={'x': train_data},
        y=train_labels,
        batch_size=FLAGS.batch_size,
        num_epochs=FLAGS.epochs,
        shuffle=True)

  eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
      x={'x': test_data},
      y=test_labels,
      num_epochs=1,
      shuffle=False)
  # Training loop.
  steps_per_epoch = 60000 // FLAGS.batch_size
  test_accuracy_list=[]
  for epoch in range(1, FLAGS.epochs + 1):
    mnist_classifier.train(input_fn=train_input_fn, steps=steps_per_epoch)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    test_accuracy = eval_results['accuracy']
    test_accuracy_list.append(test_accuracy)
    print('Test accuracy after %d epochs is: %.3f' % (epoch, test_accuracy))
    print(test_accuracy_list)

if __name__ == '__main__':
  app.run(main)
