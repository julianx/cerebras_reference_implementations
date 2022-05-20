# Cerebras Reference Implementations

This repository contains examples of common deep learning models that can be trained on Cerebras hardware. These models demonstrate the best practices for coding for the Cerebras hardware so that you can take full advantage of this new powerful compute engine.

Refer to the [Documentation Site](https://docs.cerebras.net/en/latest/) for details on latest releases and all the features supported by Cerebras ML Software.

Contact us at [support@cerebras.net] if you need any help or would like to get on board the revolutionary CS-2 and try out these models and more.

# Main Sections

- [PyTorch](#pytorch)
- [TensorFlow](#tensorflow)

# PyTorch

## Directory structure for PyTorch

The PyTorch models are organized as following directories:

- One directory per model.
- The [common/pytorch](common/pytorch) directory containing utilities for running models in PyTorch, including:
  - [PyTorchBaseModel](common/pytorch/PyTorchBaseModel.py#L23), an abstract base class models should inherit from.
  - [PyTorchCSRunner](common/pytorch/pytorch_cs_runner.py#L30) for running PyTorch models on Cerebras hardware.
  - [PyTorchRunner](common/pytorch/pytorch_runner.py#L25) for running PyTorch models on CPU or GPU.
  - [AdamW](common/pytorch/optim/AdamW.py#L21) and [SGD](common/pytorch/optim/SGD.py#L18) optimizers.

A model directory contains the following assets:

* README document with a description of the model.
* `model.py` containing a model class that inherits from `PyTorchBaseModel`.
* `data.py` with input data pipeline implementation.
* `run.py` that contains the training script for Cerebras hardware.
* `configs/` directory of YAML files containing model configurations and training hyperparameters. The easiest way to change a model configuration or hyperparameters is to change parameters in a YAML config file.
* `configs/default_params.yaml` (PyTorch only) YAML file containing default parameter values. A parameter value in this file will be used whenever a config does not specify that parameter. NOTE: this is not a configuration that can be run on its own.

## Compile on CPU

A typical the Cerebras System setup will consist of a support cluster with standard CPU nodes connected to a Cerebras System and managed by an orchestrator like Slurm.

**NOTE**: To either compile or run you must be logged in to a CPU node in the support cluster.

The scripts `csrun_cpu`, for compiling on a CPU node, and the `csrun_wse`, for running on the Cerebras System, are used together in the Cerebras ML workflow.

We recommend that you first use the `csrun_cpu` script to compile your model successfully on a support cluster CPU node, before running it on the CS system using the `csrun_wse` script.

The below ``csrun_cpu`` command will compile the code in the `train` mode for the CS system. Note that this step will only compile the code and will not run training on the CS system.

To use `csrun_cpu`, run:

```
csrun_cpu python-pt run.py --mode train \
    --compile_only
    --params configs/<name-of-the-params-file.yaml> \
```


## Train on the CS system

Execute the `csrun_wse` command to run the training on the CS system. See the command format below:
    **NOTE**: For PyTorch models only, the `cs_ip` flag must include both the IP address and the port number of the CS system. Only the IP address, for example: `--cs_ip 192.168.1.1`, will not be sufficient. You must also include the port number, for example: `--cs_ip 192.168.1.1:9000`.

    ```
    csrun_wse python-pt run.py --mode train \
        --cs_ip <IP:port-number> \
        --params configs/<name-of-the-params-file.yaml> \
    ```

To run evaluation, change the `--mode` from `train` to `eval`.

## Run on CPU or GPU

Execute the following command to run training on a CPU or GPU:

```
python run.py --mode train \
    --params configs/<name-of-the-params-file.yaml>
```

## Where to find model checkpoints and logs?

You can specify a directory for logs and training artifacts in the config `params.yaml` YAML or via a command line argument passed to `run.py`. The default location is `model_dir` directory. This `model_dir` directory will be created in the directory that contains the `run.py` script. See the `params.yaml` file in any model's `configs` directory.  

# TensorFlow

## Directory structure for TensorFlow

The TensorFlow models are organized as following directories:

- One directory per model.
- [common/tf](common/tf) directory that contains:
  - A [library of layers](common/tf/layers).
  - Other common scripts, including a lightweight [Cerebras Estimator wrapper](common/tf/estimator) to make it easy to run the same code on the Cerebras System and traditional hardware such as GPUs or CPUs.

A model directory contains the following assets:

* README document with a detailed description of the model.
* `model.py` that contains the model function definition.
* `data.py` with input data pipeline implementation.
* `params.yaml` YAML file containing a default model configuration and the training hyperparameters. The easiest way to change a model configuration or hyperparameters is to change parameters in this YAML config file.
* `run.py` that contains the training script.
* `utils.py` with the helper scripts.

## Compile on CPU

A typical the Cerebras System setup will consist of a support cluster with standard CPU nodes connected to a Cerebras System and managed by an orchestrator like Slurm.

**NOTE**: To either compile or run you must be logged in to a CPU node in the support cluster.

The scripts `csrun_cpu`, for compiling on a CPU node, and the `csrun_wse`, for running on the Cerebras System, are used together in the Cerebras ML workflow.

We recommend that you first use the `csrun_cpu` script to compile your model successfully on a support cluster CPU node, before running it on the CS system using the `csrun_wse` script.

The below ``csrun_cpu`` command will compile the code in the `train` mode for the CS system. Note that this step will only compile the code and will not run training on the CS system.

To use `csrun_cpu`, run:

```
csrun_cpu python run.py --mode train \
    --compile_only \
    --params configs/<name-of-the-params-file.yaml>
```

## How to train on the CS System

To run TensorFlow models in this repo on the Cerebras System, you must:

1. Specify the IP address of the Cerebras System, either in the  `params.yaml` or as a CLI argument `--cs_ip x.x.x.x`, and
1. Execute the `run.py` script within the Cerebras environment, i.e., within the Singularity container with the Cerebras client software.

Example:

```
csrun_wse python run.py --mode=train \
            --params configs/your-params-file.yaml \
            --model_dir your-model-dir \
            --cs_ip=10.255.253.0
```

To run evaluation, change the `--mode` from `train` to `eval`.

**NOTE**: Many `run.py` scripts support not only `cs_ip` and `mode` parameters but other parameters also. These parameters can be set either in the config `params.yaml` file or passed as a CLI argument.

## Run on CPU or GPU

Execute the following command to run training on a CPU or GPU:

```
python run.py --mode train \
    --params configs/<name-of-the-params-file.yaml>
```

## Where to find model checkpoints and logs?

You can specify a directory for logs and training artifacts in the config `params.yaml` YAML or via a command line argument passed to `run.py`. The default location is `model_dir` directory. This `model_dir` directory will be created in the directory that contains the `run.py` script. See the `params.yaml` file in any model's `configs` directory.  
