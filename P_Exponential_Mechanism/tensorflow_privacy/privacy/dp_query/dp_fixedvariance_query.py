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

"""Implements DPQuery interface for Gaussian average queries.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from distutils.version import LooseVersion
import tensorflow as tf
import numpy as np
import mpmath as mp

from tensorflow_privacy.privacy.dp_query import dp_query
from tensorflow_privacy.privacy.dp_query import normalized_query
# from tensorflow_privacy.privacy.dp_query import slice_sample


class GeneralSumQuery(dp_query.SumAggregationDPQuery):
    """Implements DPQuery interface for p-exponential sum queries.

    Accumulates clipped vectors, then adds p-exponential noise to the sum.
    """

    # pylint: disable=invalid-name
    _GlobalState = collections.namedtuple(
        '_GlobalState', ['l2_norm_clip', 'exponents', 'noise_variance'])

    # _GlobalState = collections.namedtuple(
    #   '_GlobalState', ['l2_norm_clip', 'exponents'])
    def __init__(self, l2_norm_clip, exponents, noise_variance):
        """Initializes the GaussianSumQuery.

        Args:
          l2_norm_clip: The clipping norm to apply to the global norm of each
            record.
          exponents: The exponents parameter of the p_exponential mechasnism.
          beta: the stddev parameter b
        """
        self._l2_norm_clip = l2_norm_clip
        self.exponents = exponents
        self.noise_variance = noise_variance
        self._ledger = None

    def set_ledger(self, ledger):
        self._ledger = ledger

    # def make_global_state(self, l2_norm_clip, exponents, noise_variance):
    #   """Creates a global state from the given parameters."""
    #   return self._GlobalState(tf.cast(l2_norm_clip, tf.float32), tf.cast(exponents, tf.float32), tf.cast(noise_variance, tf.float32))
    #
    # def initial_global_state(self):
    #   return self.make_global_state(self._l2_norm_clip, self.exponents, self.noise_variance)
    def make_global_state(self, l2_norm_clip, exponents, noise_variance):
        """Creates a global state from the given parameters."""
        return self._GlobalState(tf.cast(l2_norm_clip, tf.float32), tf.cast(exponents, tf.float32),
                                 tf.cast(noise_variance, tf.float32))

    def initial_global_state(self):
        return self.make_global_state(self._l2_norm_clip, self.exponents, self.noise_variance)

    def derive_sample_params(self, global_state):
        return global_state.l2_norm_clip

    def initial_sample_state(self, template):
        return tf.nest.map_structure(
            dp_query.zeros_like, template)

    def preprocess_record_impl(self, params, record):
        """Clips the l2 norm, returning the clipped record and the l2 norm.

        Args:
          params: The parameters for the sample.
          record: The record to be processed.

        Returns:
          A tuple (preprocessed_records, l2_norm) where `preprocessed_records` is
            the structure of preprocessed tensors, and l2_norm is the total l2 norm
            before clipping.
        """
        l2_norm_clip = params
        record_as_list = tf.nest.flatten(record)
        clipped_as_list, norm = tf.clip_by_global_norm(record_as_list, l2_norm_clip)
        return tf.nest.pack_sequence_as(record, clipped_as_list), norm

    def preprocess_record(self, params, record):
        preprocessed_record, _ = self.preprocess_record_impl(params, record)
        return preprocessed_record

    # Add p-exponential noises to the high-dimensional gradients
    def get_noised_result(self, sample_state, global_state):
        """See base class."""
        if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
            def sample_gaussian_noise(v):
                noise = tf.random.normal(tf.shape(input=v), stddev=tf.constant(1.0))
                return noise
        else:
            random_normal = tf.compat.v1.random_normal_initializer(
                stddev=global_state.stddev)

            def sample_gaussian_noise(v):
                return random_normal(tf.shape(input=v))

        #Obtain the parameters to specify the noise distribution Pr(x)=(1/alpha)*exp[(-||x||^exponents)/beta], where x\in R^dimension, and alpha>0 denotes the-
        #-normalization term. The parameters includs the noise dimension, exponents, and beta which reflects the noise variance 
        dimension = 0
        for st in sample_state:
            shape_st = st.get_shape().as_list()
            dimension_st = np.array(shape_st).prod()
            dimension += dimension_st
        sess = tf.Session()
        exponents = global_state.exponents.eval(session=sess)
        noise_variance = global_state.noise_variance.eval(session=sess)
        sess.close()

        def get_beta(dimension, exponents, noise_variance):
            beta = np.float((noise_variance * dimension * mp.gamma(dimension / exponents) / mp.gamma(
                (dimension + 2) / exponents)) ** (exponents / 2))
            return beta

        beta = get_beta(dimension, exponents, noise_variance)

        # Generate the p-exponential noises according to the parameters of dimension, exponents, and beta. These parameters can specify the noise distribution
        def sample_from_p_exponential(sample_state, dimension, exponents, beta):
            '''the args of dimension, exponents, and beta are enough to specify the noise distribution and thus enabling the noise sampling. In particular, we 
            generate the p-exponential noises by first generating the Gaussian noises and then limiting the norm of the Gaussian noises.'''
            norm_totransform = np.power(np.random.gamma(dimension / exponents, beta), 1 / exponents)
            # print(norm_totransform)
            noise_initial = tf.nest.map_structure(sample_gaussian_noise, sample_state)
            # print('33333333333333333333333333333333333333333333333333333333333333333333333333333333,',tf.shape(noise_initial[0]))
            # def transform_gaussian_to_p_exponential(v):
            print('33333333333333333333333333333333333333333333333333333333333333333333333333333333:,',tf.shape(noise_initial[0]))
            # global_norm = tf.linalg.global_norm(v)
            # # print(global_norm)
            # return v * (norm_totransform / global_norm)
            # print(tf.linalg.globalnorm(noise_initial))
            global_norm= tf.sqrt(sum([tf.linalg.norm(t,2)**2 for t in noise_initial]))
            # noise=noise_initial * (norm_totransform / global_norm)
            noise = [n * (norm_totransform / global_norm) for n in noise_initial]
            # noise = transform_gaussian_to_p_exponential(noise)
            # print(noise[0].shape())
            # noise=tf.nest.map_structure(transform_gaussian_to_p_exponential, [noise_initial])[0]
            return noise
        noise = sample_from_p_exponential(sample_state, dimension, exponents, beta)

        # add the generated p-exponential noises to the clipped gradients.
        dependencies = []
        with tf.control_dependencies(dependencies):
            new_sample_state = []
            for i in range(len(sample_state)):
                new_sample_state.append(tf.add(sample_state[i], noise[i]))
            return new_sample_state, global_state


class GeneralAverageQuery(normalized_query.NormalizedQuery):
    """Implements DPQuery interface for Gaussian average queries.

    Accumulates clipped vectors, adds Gaussian noise, and normalizes.

    Note that we use "fixed-denominator" estimation: the denominator should be
    specified as the expected number of records per sample. Accumulating the
    denominator separately would also be possible but would be produce a higher
    variance estimator.
    """

    def __init__(self,
                 l2_norm_clip,
                 exponents,
                 noise_variance,
                 denominator):
        """Initializes the GaussianAverageQuery.
        Args:
          l2_norm_clip: The clipping norm to apply to the global norm of each
            record.
          sum_stddev: The stddev of the noise added to the sum (before
            normalization).
          denominator: The normalization constant (applied after noise is added to
            the sum).
        """
        super(GeneralAverageQuery, self).__init__(
            numerator_query=GeneralSumQuery(l2_norm_clip, exponents, noise_variance),
            denominator=denominator)
