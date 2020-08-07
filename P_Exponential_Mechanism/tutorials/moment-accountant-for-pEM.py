#Copyright [2020] [authors of p-exponential mechanism]
 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compute the total privacy loss using moments accountant technique in p-exponential mechanism."""

import numpy as np
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr, Global
import mpmath as mp
import tensorflow as tf
from absl import app
from absl import flags
import os

mp.dps=350
#the dir where we install the mathamatica software
wlpath = "D:\\Program Files\\Wolfram Research\\Mathematica\\12.0\\WolframKernel.exe"

#"""configure the parameters"""
FLAGS = flags.FLAGS

flags.DEFINE_float('exponents', 2, 'exponents of the p-exoponential distribution where the noises are sampled')
flags.DEFINE_float('noise_variance', 256, 'variance parameter of our proposed p-exponential mechanism')
flags.DEFINE_integer('dimension', 10, 'dimension of the generated noises')
flags.DEFINE_integer('T', 10000, 'number of iterations')
flags.DEFINE_float('clippingbound', 4, 'the clippig bound of the gradient to protect')
flags.DEFINE_float('q', 0.01, 'the sampling ratio of SGD')
flags.DEFINE_float('delta', 10 ** (-5), 'the failure probability of the privacy protection')
flags.DEFINE_integer('num_samples', 10000, 'the number of samples which are used to approximate the moment in the integral')
flags.DEFINE_integer('start_lam', 1, 'the lower bound of lamda in our bisection searching method')
flags.DEFINE_integer('end_lam', 1000, 'the upper bound of lamda in our bisection searching method')


def main(unused_argv):
    beta = np.float(((mp.gamma((FLAGS.dimension) / FLAGS.exponents) * FLAGS.dimension * FLAGS.noise_variance) / (mp.gamma((FLAGS.dimension + 2) / FLAGS.exponents))) ** (FLAGS.exponents / 2))
    # computing the privacy parameter \epsilon
    def _compute_eps(lam):
        session = WolframLanguageSession(wlpath)
        session.evaluate(wlexpr('''
           randomgamma[alpha_, beta_, gamma_, samples_] := RandomVariate[GammaDistribution[alpha, beta, gamma, 0], samples];
         '''))
        random_gamma = session.function(wlexpr('randomgamma'))

        session.evaluate(wlexpr('''
           integrant[exponents_, beta_, dimension_, clippingbound_, lam_, r_, q_] := Mean[NIntegrate[
                           (Sin[x]^(dimension-2)*Gamma[dimension/2]/(Sqrt[Pi]*Gamma[(dimension-1)/2]))*(((1-q)*(1-q+
                           q*Exp[(r^exponents-(r^2+clippingbound^2-2*r*clippingbound*Cos[x])^(exponents/2))/beta])^(lam))
                        +(q*(1-q+q*Exp[((r^2+clippingbound^2+2*r*clippingbound*Cos[x])^(exponents/2)-r^exponents)/beta])^(lam))),{x,0,Pi}
                           ]];
         '''))
        integrant_moment = session.function(wlexpr('integrant'))
        samples = random_gamma(FLAGS.dimension / FLAGS.exponents, beta ** (1 / FLAGS.exponents), FLAGS.exponents,
                               FLAGS.num_samples)
        moment = integrant_moment(FLAGS.exponents,beta, FLAGS.dimension, FLAGS.clippingbound, lam, samples, FLAGS.q)
        eps = (FLAGS.T * mp.log(moment) + mp.log(1 / FLAGS.delta)) / lam
        session.terminate()
        return eps

    # find the minimal epsilon using the bisection searching method
    def Bisection_eps(start, end):
        i = 0
        print(i, (start, end))
        while ((end - start) > 5):
            i += 1
            temp = int((start + end) / 2)
            # print(temp)
            Middle = _compute_eps(temp)
            # print(Middle)
            left = _compute_eps(temp - (end - start) / 4)
            right = _compute_eps(temp + (end - start) / 4)
            if (Middle < left and Middle < right):
                start = temp - int((end - start) / 4)
                end = temp + int((end - start) / 4)
            elif (Middle > left and Middle < right):
                end = temp
            elif (Middle < left and Middle > right):
                start = temp
            else:
                print('the integral approximation based on sampling method has produced inaccurate result in this step. Recompute!')
            print(i, (start, end))
        lam_list = [start + i for i in range(end - start + 1)]
        result_list = [_compute_eps(lam) for lam in lam_list]
        index = result_list.index(min(result_list))
        opt_lam = lam_list[index]
        result = np.min(np.array(result_list))
        return opt_lam, result
   
    opt_lam, eps = Bisection_eps(FLAGS.start_lam, FLAGS.end_lam)
    return opt_lam, eps


if __name__ == '__main__':
    app.run(main)
    
