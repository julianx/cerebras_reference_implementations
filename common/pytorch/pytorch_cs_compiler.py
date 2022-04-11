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

import logging
from typing import Tuple

import torch
from common.pytorch.pytorch_base_runner import PyTorchBaseRunner

import cerebras.framework.torch as cbtorch
import cerebras.framework.torch.core.cb_model as cm

COMPILE_ONLY_MSG = "Compiling the model. This may take a few minutes."


class PyTorchCSCompiler(PyTorchBaseRunner):
    """Class for compiling PyTorch models for Cerebras hardware."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # irrelevant config options for compile only
        self._save_initial_checkpoint = False
        self._save_losses = False

    ##################################################################
    #                         Training Hooks                         #
    ##################################################################

    def on_train_start(self):
        cm.set_run_config(1, 0, 0)

    def on_train_epoch_end(self, early_exit: bool):
        logging.info(COMPILE_ONLY_MSG)

        assert cbtorch.compile(
            fabric_config_file=self._fabric_config_file,
        ), "Compile failed"

        logging.info("Compile for training completed successfully!")

    def on_train_batch_end(self, *args, **kwargs):
        pass  # noop

    def backward(self, loss):
        """Runs the backward pass."""
        self._model.grad_scaler(loss).backward()

    ##################################################################
    #                        Evaluation Hooks                        #
    ##################################################################

    def on_eval_start(self):
        cm.set_run_config(1, 0, 0)

    def eval_forward(self, data):
        outputs = super().eval_forward(data)

        # Need to track eval model outputs to compile
        cbtorch.state().track_object(outputs)

        return outputs

    def on_eval_epoch_end(self, early_exit: bool):
        logging.info(COMPILE_ONLY_MSG)

        assert cbtorch.compile(
            fabric_config_file=self._fabric_config_file,
        ), "Compile Failed"

        logging.info("Compile for evaluation completed successfully!")

    def on_eval_batch_end(self, *args, **kwargs):
        pass  # noop

    def compute_eval_metrics(self):
        pass  # noop

    ##################################################################
    #                   Override Abstract Methods                    #
    ##################################################################

    def train(self, data_loader: torch.utils.data.DataLoader) -> None:
        data_loader = cbtorch.dataloader(data_loader)
        super().train(data_loader)

    def evaluate(self, data_loader: cbtorch.data.DataLoader):
        data_loader = cbtorch.dataloader(data_loader)
        super().evaluate(data_loader)

    def _should_stop(self, epoch_step: int, mode: str) -> Tuple[bool, bool]:
        return True, True

    def _configure_run_steps(self, dataloader, mode: str):
        self._num_epochs = 1
        self._total_steps = 1
        self._checkpoint_steps = 0
        self._fetch_steps = 0

    def _increment_global_step(self):
        self._global_step += 1

    def _write_log(self, loss, step):
        pass  # noop

    def _save_checkpoint(self, step):
        # Should never reach here
        raise RuntimeError("Should not be saving checkpoint in compile only")
