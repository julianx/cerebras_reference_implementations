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

import tensorflow as tf
from common.tf.layers.ActivationLayer import ActivationLayer as Activation
from common.tf.layers.CrossEntropyFromLogitsLayer import (
    CrossEntropyFromLogitsLayer,
)
from common.tf.layers.DenseLayer import DenseLayer as Dense
from common.tf.layers.DropoutLayer import DropoutLayer as Dropout
from common.tf.optimizers.Trainer import Trainer
from common.tf.TFBaseModel import TFBaseModel

NUM_CLASSES = 10


class FCMnistModel(TFBaseModel):
    """
    Fc_Mnist model from TFBaseModel to be used with TF Estimator
    """

    def __init__(self, params):
        if "model" in params:
            mparams = params['model']
        else:
            mparams = params

        super(FCMnistModel, self).__init__(
            mixed_precision=mparams.get("mixed_precision", True)
        )

        self.num_classes = NUM_CLASSES
        self.logging_dict = {}

        # Model Params
        self.depth = mparams.get("depth")
        self.hidden_size = mparams.get("hidden_size")
        self.hidden_sizes = mparams.get("hidden_sizes")
        self.dropout = mparams.get("dropout")
        self.activation_fn = mparams.get("activation_fn")
        self.tf_summary = mparams.get("tf_summary", False)
        self.mixed_precision = mparams.get("mixed_precision", False)
        self.boundary_casting = mparams.get("boundary_casting", False)

        # Optimizer params and default to Adam if none present.
        if "optimizer" in params:
            optimizer_params = params.get("optimizer")
        else:
            optimizer_params = params
        if optimizer_params.get("optimizer_type") is None:
            optimizer_params.update({"optimizer_type": "adam"})

        # Model Trainer
        self.trainer = Trainer(
            params=optimizer_params,
            tf_summary=self.tf_summary,
            mixed_precision=self.mixed_precision,
        )

        # Cs util params for layers
        if "runconfig" in params:
            self.output_dir = params["runconfig"].get("model_dir")
        else:
            self.output_dir = params.get("model_dir")

    def build_model(self, features, mode):

        tf.keras.backend.set_floatx('float16')
        dtype = self.policy

        dropout_layer = Dropout(
            self.dropout, dtype=dtype, tf_summary=self.tf_summary
        )
        # Set depth or hidden_sizes depending on params.
        if self.hidden_sizes:
            # Depth is len(hidden_sizes)
            self.depth = len(self.hidden_sizes)
        else:
            # same hidden size across dense layers
            self.hidden_sizes = [self.hidden_size] * self.depth

        x = features
        for hidden_size in self.hidden_sizes:
            with tf.name_scope("km_disable_scope"):
                dense_layer = Dense(
                    hidden_size, dtype=dtype, tf_summary=self.tf_summary
                )
            act_layer = Activation(
                self.activation_fn, dtype=dtype, tf_summary=self.tf_summary
            )
            with tf.name_scope("km_disable_scope"):
                x = dense_layer(x)
            x = act_layer(x)
            x = dropout_layer(x, training=(mode == tf.estimator.ModeKeys.TRAIN))
        with tf.name_scope("km_disable_scope"):
            # Model has len(hidden_sizes) + 1 Dense layers
            output_dense_layer = Dense(
                NUM_CLASSES, dtype=dtype, tf_summary=self.tf_summary
            )
            logits = output_dense_layer(x)

        return logits

    def build_total_loss(self, model_outputs, features, labels, mode):
        softmax_ce = CrossEntropyFromLogitsLayer(
            boundary_casting=self.boundary_casting,
            tf_summary=self.tf_summary,
            dtype=self.policy,
        )

        loss = tf.reduce_mean(
            tf.cast(softmax_ce(labels, model_outputs), tf.float32)
        )
        loss = tf.cast(loss, model_outputs.dtype)
        tf.compat.v1.summary.scalar('loss', loss)
        return loss

    def build_train_ops(self, total_loss):
        return self.trainer.build_train_ops(total_loss)

    def build_eval_metric_ops(self, model_outputs, labels, features=None):
        return {
            "accuracy": tf.compat.v1.metrics.accuracy(
                labels=labels,
                predictions=tf.argmax(input=model_outputs, axis=1),
            ),
        }
