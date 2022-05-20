# Checkpoint Utilities

The `convert_hf_ckpt_to_cerebras.py` script can be used to map [publicly available GPT-J-6B weights from HuggingFace](https://huggingface.co/EleutherAI/gpt-j-6B) to a TensorFlow format to be loaded for either continuous pre-training or fine-tuning with this reference implementation on Cerebras hardware.

## Installation

The following pre-requisites are needed to enable a clean run of the script. Below is a setup for a conda environment:

```bash
conda create --name ckpt_env python=3.7.4
conda activate ckpt_env
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c huggingface transformers
pip install tensorflow-gpu==2.2.0
```

## Runnning the Script

To run the script, activate the conda environment and run:

```bash
conda activate ckpt_env
python convert_hf_ckpt_to_cerebras.py
```

__User-Note__: The provided script will download the checkpoint from HuggingFace if it does not exist in the specified path. 
Beware, since the model is very large, the checkpoint size is around 23GB. A single checkpoint conversion will take about 
110GB RAM and about 10-15 mins to run end to end. Please use a system with sufficient compute, memory and storage.
