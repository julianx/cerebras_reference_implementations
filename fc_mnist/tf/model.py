# Copyright 2022 Cerebras Systems.
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

"""
Simple FC MNIST model to be used with Estimator
"""
import tensorflow as tf
from common.tf.estimator.cs_estimator_spec import CSEstimatorSpec
from fc_mnist.tf.FCMnistModel import FCMnistModel


def model_fn(features, labels, mode, params):
    model = FCMnistModel(params)
    logits = model(features, mode)
    loss = model.build_total_loss(logits, features, labels, mode)

    train_op = None
    host_call = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = model.build_train_ops(loss)
    elif mode == tf.estimator.ModeKeys.EVAL:
        host_call = (model.build_eval_metric_ops, [logits, labels, features])
    else:
        raise ValueError("Only TRAIN and EVAL modes supported")

    espec = CSEstimatorSpec(
        mode=mode, loss=loss, train_op=train_op, host_call=host_call,
    )

    return espec
