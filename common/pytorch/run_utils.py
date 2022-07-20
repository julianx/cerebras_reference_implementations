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

import inspect
import os
from typing import Callable, Optional

import torch
import yaml
from common.pytorch import cb_model as cm
from common.pytorch import modes
from common.pytorch.pytorch_base_runner import (
    PyTorchBaseRunner,
)
from common.pytorch.PyTorchBaseModel import (
    PyTorchBaseModel,
)
from common.pytorch.utils import (
    get_params_from_args,
    setup_logging,
)

DATA_FN_TYPE = Callable[[dict], torch.utils.data.DataLoader]


def run(
    model_fn: Callable[[dict], PyTorchBaseModel],
    train_data_fn: Optional[DATA_FN_TYPE] = None,
    eval_data_fn: Optional[DATA_FN_TYPE] = None,
    default_params_fn: Optional[Callable[[dict], dict]] = None,
):
    """
    Runs a full end-to-end CS/non-CS workflow for a given model

    Args:
        model_fn: A callable that takes in a 'params' argument
            which it uses to configure and return a PyTorchBaseModel
        train_data_fn: A callable that takes in a 'params' argument
            which it uses to configure and return a PyTorch dataloader
            corresponding to the training dataset
        eval_data_fn: A callable that takes in a 'params' argument
            which it uses to configure and return a PyTorch dataloader
            corresponding to the evaluation dataset
        default_params_fn: An optional callable that takes in the params
            dictionary and updates any missing params
            with default values
    """
    parent = inspect.getouterframes(inspect.currentframe())[1]
    run_dir = os.path.dirname(os.path.abspath(parent.filename))

    # Parse arguments from the command line and get the params
    # from the specified config file
    params = get_params_from_args(run_dir)
    if default_params_fn:
        params = default_params_fn(params) or params

    runconfig_params = params["runconfig"]

    if "seed" in runconfig_params:
        torch.manual_seed(runconfig_params["seed"])

    runner = PyTorchBaseRunner.create(model_fn, params)

    # Set up logging level
    setup_logging(
        runconfig_params.get("logging"),
        runconfig_params.get("streamer_logging"),
    )

    # Save params.yaml only in master task
    mode = runconfig_params["mode"]
    # using this dir structure to keep in sync with runners
    summary_dir = os.path.join(params["runconfig"]["model_dir"], f"{mode}")
    os.makedirs(summary_dir, exist_ok=True)
    if cm.is_master_ordinal():
        with open(
            os.path.join(summary_dir, f"params_{mode}.yaml"), "w+",
        ) as _fout:
            yaml.dump(params, _fout, default_flow_style=False)

    # Initialize the dataloaders depending on the mode
    if mode in (modes.TRAIN, modes.TRAIN_AND_EVAL):
        assert train_data_fn, "Train dataloader function has not been provided"
        train_loader: torch.utils.data.Dataloader = train_data_fn(params)
    if mode in (modes.EVAL, modes.TRAIN_AND_EVAL):
        assert eval_data_fn, "Eval dataloader function has not been provided"
        eval_loader: torch.utils.data.Dataloader = eval_data_fn(params)

    if mode == modes.TRAIN:
        runner.train(train_loader)
    elif mode == modes.EVAL:
        runner.evaluate(eval_loader)
    else:
        raise ValueError(f"Mode {mode} is not supported.")
