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
import os
import time

import torch
from common.pytorch import modes
from common.pytorch.perf_utils import (
    save_perf,
)
from common.pytorch.pytorch_base_runner import (
    PyTorchBaseRunner,
)

import cerebras.framework.torch as cbtorch
import cerebras.framework.torch.core.cb_model as cm

COMPILE_MSG = (
    "Compiling the model and programming onto fabric. "
    "This may take a few minutes."
)


class PyTorchCSRunner(PyTorchBaseRunner):
    """Class for running PyTorch models on Cerebras hardware."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._device = cm.device()

        # batch size to be inferred on first iteration
        self._batch_size = None

    ##################################################################
    #                         Training Hooks                         #
    ##################################################################

    def on_train_start(self):
        cm.write_to_summary(
            self._writer,
            0,
            dict_to_write={"TensorboardStartTimestamp": time.time()},
        )
        cm.set_run_config(
            self._total_steps, self._checkpoint_steps, self._fetch_steps
        )

    def on_train_end(self, early_exit: bool):
        save_perf(self._perf_dir)

        if self._show_debug_metrics:
            cm.print_metrics_report()

        logging.info("Training Completed Successfully!")

    def on_train_epoch_end(self, early_exit: bool):
        if early_exit:
            cm.mark_step()  # required to complete execution

    def backward(self, loss):
        """Runs the backward pass."""
        self._model.grad_scaler(loss).backward()

    def optimizer_step(self):
        super().optimizer_step()

        if self._global_step == self._initial_step:
            logging.info(COMPILE_MSG)

    ##################################################################
    #                        Evaluation Hooks                        #
    ##################################################################

    def on_eval_start(self):
        cm.write_to_summary(
            self._writer,
            0,
            dict_to_write={"TensorboardStartTimestamp": time.time()},
        )
        cm.set_run_config(self._total_steps, 0, 1)

        logging.info(COMPILE_MSG)

    def on_eval_end(self, early_exit: bool):
        save_perf(self._perf_dir)

        if self._show_debug_metrics:
            cm.print_metrics_report()

        logging.info("Evaluation Completed Successfully!")

    def on_eval_epoch_end(self, early_exit: bool):
        if early_exit:
            cm.mark_step()  # required to complete execution

    ##################################################################
    #                   Override Abstract Methods                    #
    ##################################################################

    def train(self, data_loader: torch.utils.data.DataLoader) -> None:
        data_loader = cbtorch.dataloader(data_loader)

        with cbtorch.Session(data_loader, modes.TRAIN):
            super().train(data_loader)

    def evaluate(self, data_loader: cbtorch.data.DataLoader):
        data_loader = cbtorch.dataloader(data_loader)

        with cbtorch.Session(data_loader, modes.EVAL):
            super().evaluate(data_loader)

    def train_and_eval(
        self,
        train_data_loader: torch.utils.data.DataLoader,
        test_data_loader: torch.utils.data.DataLoader,
    ):
        raise RuntimeError(
            "Training with Eval on CS is not currently supported."
        )

    @property
    def _perf_dir(self) -> str:
        """Return the directory to use for saving perfomance metrics."""
        return os.path.join(self._model_dir, "performance")

    @property
    def world_global_step(self):
        return self._global_step * cm.num_receivers()

    def _increment_global_step(self):
        self._global_step += cm.get_run_step() - self._run_step

    @cm.step_closure
    def _write_log(self, loss, step):
        """Print training update to screen.

        Args:
            loss: The loss tensor.
        """
        cm.print_update(
            self._device,
            step,
            loss.item(),
            self._active_mode,
            summary_writer=self._writer,
        )

    @cm.step_closure
    def _save_checkpoint(self, step):
        """Conditionally add a step closure to save checkpoint."""
        file_name = os.path.join(self._model_dir, f"checkpoint_{step}.mdl")

        state_dict = self._model.get_state()
        state_dict["global_step"] = state_dict.get("global_step", step)

        def post_transfer_callback(state_dict):
            if "optimizer" in state_dict:
                state_dict[
                    "optimizer"
                ] = self._optimizer.convert_state_dict_for_checkpoint(
                    state_dict["optimizer"]
                )
            return state_dict

        cm.save(
            state_dict,
            file_name,
            master_only=True,
            post_transfer_callback=post_transfer_callback,
        )
        logging.info(f"Saved checkpoint at global step: {step}")
