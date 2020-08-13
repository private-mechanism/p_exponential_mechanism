# p_exponential_mechanism
We propose a mechanism family (p-power exponential mechanism family) of DP to improve the performance of 
the Gaussian mechanism in machine learning.

## TensorFlow Privacy
This folder contains the source code for TensorFlow Privacy, which is a Python
library developed by google (`https://github.com/tensorflow/privacy`) for training machine learning models with differential privacy. 
The detailed procedures for setting up TensorFlow Privacy in our repository are as follows.

### Installing TensorFlow Privacy
First, clone this GitHub repository into a directory of your choice:

`git clone https://github.com/private-mechanism/p_exponential_mechanism`

You can then install the local package in "editable" mode in order to add it to your `PYTHONPATH`:

`cd P_Exponential_Mechanism`

`pip install -e .`

Comparing to the original version of TensorFlow Privacy, we have added one
python file under `privacy/optimizers` and `privacy/dp_query` folders respectively. 

In `dp_fixedvariance_query.py` under `privacy/dp_query` folder, we present how to 
sample noises from the p-exponential distributions.

In `flatten_optimizer.py` under `privacy/optimizers` folder, we present how to generate 
the DP optimizer under the p-exponential mechanism.


## Tutorials directory

The `tutorials/` folder contains the scripts presenting how to apply p-exponential 
mechanism in the machine learning models. In particular, we apply p-exponential 
mechanism into Logistic Regression(LR) on mnist dataset, and Conventional Neural Networks(CNN) on 
both mnist and cifar10 datasets. Additionally, we also present the scripts for 
computing the total privacy loss using moments accountant technique in p-expnential mechanism.

## Contacts

If you have any questions that cannot be addressed by raising an issue, feel
free to contact:

* zfy1454236335 (@stu.xjtu.edu.cn)

