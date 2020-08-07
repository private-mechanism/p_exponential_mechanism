# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""PrivacyLedger class for keeping a record of private queries."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import tensorflow as tf
import mpmath as mp

from tensorflow_privacy.privacy.analysis import tensor_buffer
from tensorflow_privacy.privacy.dp_query import dp_query

#开头定义数据容器类型，可迭代，可通过属性名来访问属性值
SampleEntry = collections.namedtuple(  # pylint: disable=invalid-name
    'SampleEntry', ['population_size', 'selection_probability', 'queries'])

P_exponentialSumQueryEntry = collections.namedtuple(  # pylint: disable=invalid-name
    'P_exponentialSumQueryEntry', ['exponents','l2_norm_bound', 'dimension', 'beta'])

sample_array=[]

def format_ledger(sample_array, query_array):
  """Converts array representation into a list of SampleEntries."""
  samples = []
  query_pos = 0
  sample_pos = 0
  for sample in sample_array:
    population_size, selection_probability, num_queries = sample
    queries = []
    for _ in range(int(num_queries)):
      query = query_array[query_pos]
      assert int(query[0]) == sample_pos
      queries.append(P_exponentialSumQueryEntry(*query[1:]))
      query_pos += 1
    samples.append(SampleEntry(population_size, selection_probability, queries))
    sample_pos += 1
  return samples


class P_exponential_PrivacyLedger(object):
  """Class for keeping a record of private queries.

  The PrivacyLedger keeps a record of all queries executed over a given dataset
  for the purpose of computing privacy guarantees.
  """

  def __init__(self,
               population_size,
               selection_probability):
    """Initialize the PrivacyLedger.

    Args:
      population_size: An integer (may be variable) specifying the size of the
        population, i.e. size of the training data used in each epoch.
      selection_probability: A float (may be variable) specifying the
        probability each record is included in a sample.

    Raises:
      ValueError: If selection_probability is 0.
    """
    self._population_size = population_size
    self._selection_probability = selection_probability

    if tf.executing_eagerly():
      if tf.equal(selection_probability, 0):
        raise ValueError('Selection probability cannot be 0.')
      init_capacity = tf.cast(tf.math.ceil(1 / selection_probability), tf.int32)
    else:
      if selection_probability == 0:
        raise ValueError('Selection probability cannot be 0.')
      init_capacity = np.int(np.ceil(1 / selection_probability))

    # The query buffer stores rows corresponding to P_exponentialSumQueryEntries.
    #这里的query buffer需要存储的是p_exponentialsumquery=[exponents,samples_counts, clipping bound, dimension, beta],故每一个query所需的size=4
    self._query_buffer = tensor_buffer.TensorBuffer(
        init_capacity, [5], tf.float32, 'query')
    #弄清楚_sample_var，_sample_buffer，_sample_count的不同，后两者很好理解，主要是_sample_var是什么意思._sample_var=[exponents,population_size,selection_probability,query_count]

    self._sample_var = tf.Variable(
        initial_value=tf.zeros([3]), trainable=False, name='sample')
    # The sample buffer stores rows corresponding to SampleEntries.
    self._sample_buffer = tensor_buffer.TensorBuffer(
        init_capacity, [3], tf.float32, 'sample')
    self._sample_count = tf.Variable(
        initial_value=0.0, trainable=False, name='sample_count')
    self._query_count = tf.Variable(
        initial_value=0.0, trainable=False, name='query_count')
    self._cs = tf.CriticalSection()

  def record_sum_query(self, exponents,l2_norm_bound, dimension, beta):
    """Records that a query was issued.

    Args:
      l2_norm_bound: The maximum l2 norm of the tensor group in the query.
      noise_stddev: The standard deviation of the noise applied to the sum.

    Returns:
      An operation recording the sum query to the ledger.
    """

    def _do_record_query():
      with tf.control_dependencies(
          [tf.compat.v1.assign(self._query_count, self._query_count + 1)]):
        return self._query_buffer.append(
            [self._sample_count, exponents, l2_norm_bound, dimension, beta])

    return self._cs.execute(_do_record_query)


  def record_for_group(self, group,variance,exponents,l2_norm_bound):
    """Records that a query was issued.

    Args:
      group: a list of tensors for adding noise
      l2_norm_bound: The maximum l2 norm of the tensor group in the query.
      beta: The noise magnitude parameter.

    Returns:
      A list of operation recording the sum query to the ledger.
    """
    # tf.nest.map_structure(add_noise, sample_state)

    # def _record_for_group():
    #   with tf.control_dependencies(group):
    #
    #     with tf.control_dependencies(self.record_sum_query(l2_norm_bound, dimension, beta)):
    def _do_record_query1(vector):
      shape_v = vector.get_shape().as_list()
      dimension = np.array(shape_v).prod()
      beta = np.float((variance * dimension * mp.gamma(dimension / exponents) / mp.gamma((dimension + 2) / exponents)) ** (exponents / 2))
      # print ("beta",type(beta))
      return self.record_sum_query(exponents,l2_norm_bound, dimension, beta)

    with tf.control_dependencies([_do_record_query1(vector) for vector in group]):
      c=tf.no_op(name='nooperation')
    return c


  def finalize_sample(self):
    """Finalizes sample and records sample ledger entry."""
    with tf.control_dependencies([
        tf.compat.v1.assign(self._sample_var, [
            self._population_size, self._selection_probability,
            self._query_count
        ])
    ]):
      with tf.control_dependencies([
          tf.compat.v1.assign(self._sample_count, self._sample_count + 1),
          tf.compat.v1.assign(self._query_count, 0)
      ]):
        return self._sample_buffer.append(self._sample_var)

  def get_unformatted_ledger(self):
    return self._sample_buffer.values, self._query_buffer.values

  def get_formatted_ledger(self, sess):
    """Gets the formatted query ledger.

    Args:
      sess: The tensorflow session in which the ledger was created.

    Returns:
      The query ledger as a list of SampleEntries.
    """
    sample_array = sess.run(self._sample_buffer.values)
    query_array = sess.run(self._query_buffer.values)

    return format_ledger(sample_array, query_array)

  def get_formatted_ledger_eager(self):
    """Gets the formatted query ledger.

    Returns:
      The query ledger as a list of SampleEntries.
    """
    sample_array = self._sample_buffer.values.numpy()
    query_array = self._query_buffer.values.numpy()
    print ("foamatted query ledger: ",format_ledger(sample_array, query_array))
    return format_ledger(sample_array, query_array)


class QueryWithLedger(dp_query.DPQuery):
  """A class for DP queries that record events to a PrivacyLedger.

  QueryWithLedger should be the top-level query in a structure of queries that
  may include sum queries, nested queries, etc.
  It should simply wrap another query and contain a reference to the ledger. Any contained queries (including
  those contained in the leaves of a nested query) should also contain a
  reference to the same ledger object.

  For example usage, see privacy_ledger_test.py.
  """

  def __init__(self, query,
               population_size=None, selection_probability=None,
               P_exponential_PrivacyLedger=None):
    """Initializes the QueryWithLedger.

    Args:
      query: The query whose events should be recorded to the ledger. Any
        subqueries (including those in the leaves of a nested query) should also
        contain a reference to the same ledger given here.
      population_size: An integer (may be variable) specifying the size of the
        population, i.e. size of the training data used in each epoch. May be
        None if `ledger` is specified.
      selection_probability: A float (may be variable) specifying the
        probability each record is included in a sample. May be None if `ledger`
        is specified.
      ledger: A PrivacyLedger to use. Must be specified if either of
        `population_size` or `selection_probability` is None.
    """
    self._query = query
    if population_size is not None and selection_probability is not None:
      self.set_ledger(P_exponential_PrivacyLedger(population_size, selection_probability))
    elif P_exponential_PrivacyLedger is not None:
      self.set_ledger(P_exponential_PrivacyLedger)
    else:
      raise ValueError('One of (population_size, selection_probability) or '
                       'p_exponentialledger must be specified.')

  @property
  def ledger(self):
    return self._ledger

  def set_ledger(self, ledger):
    self._ledger = ledger
    self._query.set_ledger(ledger)

  def initial_global_state(self):
    """See base class."""
    return self._query.initial_global_state()

  def derive_sample_params(self, global_state):
    """See base class."""
    return self._query.derive_sample_params(global_state)

  def initial_sample_state(self, template):
    """See base class."""
    return self._query.initial_sample_state(template)

  def preprocess_record(self, params, record):
    """See base class."""
    return self._query.preprocess_record(params, record)

  def accumulate_preprocessed_record(self, sample_state, preprocessed_record):
    """See base class."""
    return self._query.accumulate_preprocessed_record(
        sample_state, preprocessed_record)

  def merge_sample_states(self, sample_state_1, sample_state_2):
    """See base class."""
    return self._query.merge_sample_states(sample_state_1, sample_state_2)

  def get_noised_result(self, sample_state, global_state):
    """Ensures sample is recorded to the ledger and returns noised result."""
    # Ensure sample_state is fully aggregated before calling get_noised_result.
    with tf.control_dependencies(tf.nest.flatten(sample_state)):
      result, new_global_state = self._query.get_noised_result(
          sample_state, global_state)
    # Ensure inner queries have recorded before finalizing.
    with tf.control_dependencies(tf.nest.flatten(result)):
      finalize = self._ledger.finalize_sample()
    # Ensure finalizing happens.
    with tf.control_dependencies([finalize]):
      return tf.nest.map_structure(tf.identity, result), new_global_state

    #搞清楚global state和sample_state的区别？
    # 答：global_state=[l2_norm_clip,exponents, beta];
    #sample_state应该就是梯度，即所要添加噪声对应的对象


