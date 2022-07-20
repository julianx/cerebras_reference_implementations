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
from common.tf.estimator.cs_estimator_spec import (
    CSEstimatorSpec,
)
from common.tf.hooks.grad_accum_hooks import (
    get_grad_accum_hooks,
)
from gptj.tf.GptJModel import GptJModel


def model_fn(features, labels, mode, params):
    gptj = GptJModel(params)
    outputs = gptj(features, mode)
    loss = gptj.build_total_loss(outputs, features, labels, mode)

    train_op = None
    host_call = None
    eval_metrics = None

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = gptj.build_train_ops(loss)
    elif mode == tf.estimator.ModeKeys.EVAL:
        eval_metrics = gptj.build_eval_metric_ops(outputs, labels, features)
    else:
        raise ValueError(f"Mode {mode} not supported.")

    hooks = []
    if gptj.trainer.is_grad_accum():
        hooks.extend(
            get_grad_accum_hooks(
                gptj.trainer,
                runconfig_params=params["runconfig"],
                summary_dict={"train/lm_cost": loss},
                logging_dict={"loss": loss},
            )
        )

    espec = CSEstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        training_hooks=hooks,
        host_call=host_call,
        eval_metric_ops=eval_metrics,
    )

    return espec
