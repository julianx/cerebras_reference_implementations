# Running BERT: Example Steps

- [Running BERT: Example Steps](#running-bert-example-steps)
	- [Structure of the code](#structure-of-the-code)
	- [Before you start](#before-you-start)
- [Download and prepare the dataset](#download-and-prepare-the-dataset)
	- [Download](#download)
		- [OpenWebText dataset](#openwebtext-dataset)
	- [Extract](#extract)
	- [Allocate subsets for training and validation](#allocate-subsets-for-training-and-validation)
	- [Preprocess Data](#preprocess-data)
		- [Create CSV files for 2-phase pre-training](#create-csvs-for-2-phase-pre-training)
			- [Phase 1: MSL 128](#phase-1-msl-128)
			- [Phase 2: MSL 512](#phase-2-msl-512)
- [Run Pre-training](#run-pre-training)
	- [Run pre-training on the Cerebras System](#run-pre-training-on-the-cerebras-system)
	- [Run pre-training on GPU](#run-pre-training-on-gpu)
	- [MLM Loss Scaling](#mlm-loss-scaling)


This document walks you through an example showing the steps to run a BERT pre-training on the Cerebras Wafer Scale Engine (and on GPUs) using the code in this repo.

> **Note**: You can use any subset of the steps described in this example. For example, if you already downloaded your preferred dataset and created the CSV files, then you can skip the section [Preprocess Data](#preprocess-data).

**Reference**: BERT paper on arXiv.org: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805).


## Structure of the code

* `configs/`: YAML configuration files.
* `huggingface_common`: HuggingFace code that our implementation is based on. The only modifications to the original HuggingFace code are in service of the MLM Loss Scaling method described below.
* `input/`: Input pipeline implementation based on the Open Web Text dataset. This directory also contains the scripts you can use to download and prepare the Open Web Text dataset. Vocab files are located in `bert/vocab/`.
* `layers/`: Implementations of BERT-specific layers.
* `model.py`: Model implementation leveraging HuggingFace's `BertForPreTraining` class.
* `data.py`: The entry point to the data input pipeline code.
* `run.py`: Training script. Performs training and validation.
* `utils.py`: Miscellaneous helper functions.

## Before you start

This example walk-through consists of two main steps:

1. Prepare the dataset
2. Perform the pre-training

This example follows the standard practice of two-phase pre-training for BERT models. In the two-phase pre-training, the model is:

- First pre-trained with the maximum sequence length (MSL) of 128 for 90% of the steps.
- Then the final 10% of the steps are pre-trained with the MSL of 512.

**CSV files for each phase**: You will need to create different CSV files for each of these two phases of pre-training.

# Download and prepare the dataset

## Download

### OpenWebText dataset

The scripts for downloading and preprocessing OpenWebText dataset: [https://skylion007.github.io/OpenWebTextCorpus/](https://skylion007.github.io/OpenWebTextCorpus/) are located [here](../data_processing/scripts/owt/).

> **Note**: The OpenWebText is just one example of the datasets that can be used for BERT training. See [Appendix](#appendix) for other datasets.

The preprocessing comprises of creating CSV files containing sequences and labels.

Start by downloading the OWT dataset by accessing the following link from a browser:

```url
https://drive.google.com/uc?id=1EA5V0oetDCOke7afsktL_JDQ-ETtNOvx
```

and manually download the `tar.xz` file from that location to your preferred local directory.

> **NOTE**: Currently a server side issue with the OWT site prevents using the below `extract.sh` shell script to download this tar file. We will update the script when this issue resolved.

## Extract

To extract the above-downloaded files, run:

```bash
bash extract.sh
```

> **NOTE**: The `extract.sh` may take a while to complete, as it unpacks 40GB of data (8,013,770 documents).

Upon completion, the script will produce `openwebtext` folder in the same folder where the tar file is located. The `openwebtext` folder will have multiple subfolders, each containing a collection of `*.txt` files of raw text data (one document per `.txt` file).

## Allocate subsets for training and validation

In the next step, you will create two subsets of extracted txt files, one for training and the second for validation. These two subsets are then used to create CSV files that will be used for pre-training.

> **IMPORTANT**: The training and validation subsets must contain mutually exclusive .txt files.

Proceed as follows:

Define metadata files that contain paths to subsets of documents in the `openwebtext` folder to be used for training or validation.

For training, in this tutorial we use a subset of 512,000 documents. The associated metadata file can be found in `metadata/train_512k.txt`.

For validation, we choose 5,000 documents that are not in the training set. The metadata file for validation can be found in `metadata/val_files.txt`.

>**NOTE**: You are free to create your own metadata files that define your train and validation data subsets, with your preferred content and sizes. You can also create a data subset for test.

Next, using the metadata file that defines a data subset (for training or for validation), create CSV files containing masked sequences and labels derived from the data subset.

## Preprocess Data

**create_csv.py**

To create CSV files containing sequences and labels derived from the data subset, you will use the Python utility `create_csv.py` located in the [input/scripts](input/scripts) directory.

Refer to [./input/scripts/README.md](./input/scripts/README.md) for more details.

**Prerequisites**

If you do not have [spaCy](https://spacy.io/), the natural language processing (NLP) library, then install it with the following commands:

```bash
pip install spacy
python -m spacy download en
```

**Syntax**

The command-line syntax to run the Python utility `create_csv.py` is as follows:

```bash
python create_csv.py --metadata_files /path/to/metadata_file.txt --input_files_prefix /path/to/raw/data/openwebtext --vocab_file /path/to/vocab.txt --do_lower_case
```

where:

- `metadata_file.txt` is a metadata file containing a list of paths to documents, and
- `/path/to/vocab.txt` contains a vocabulary file to map WordPieces to word IDs.

For example, you can use the supplied `metadata/train_512k.txt` as an input to generate a training set based on 512,000 documents. Sample vocabularies can be found in the `bert/vocab` folder.

For more details, run the command: `python create_csv.py --help` (or `python create_csv_static_masking.py --help` if creating statically masked data).

### Create CSVs for 2-phase pre-training

For the 2-phase BERT pre-training that we are following in this tutorial, you need to generate the following datasets:

- A training dataset for the first phase using sequences with maximum length of 128.
- A second training dataset for the second phase using sequences with maximum sequence length of 512.

If you want to run validation, then:
- Two additional validation datasets, one each for each phase.

In total, to run training and validation for both the pre-training phases, you will need four datasets: a training and a validation dataset for phase 1 with MSL 128, and a training and a validation dataset for phase 2 with MSL 512.

Proceed as follows to run the following commands:

#### Phase 1: MSL 128

**Generate training CSV files**:

```bash
python create_csv.py --metadata_files metadata/train_512k.txt --input_files_prefix /path/to/raw/data/openwebtext --vocab_file ../../../vocab/google_research_uncased_L-12_H-768_A-12.txt --output_dir train_512k_uncased_msl128 --do_lower_case --max_seq_length 128 --max_predictions_per_seq 20
```

**Generate validation CSV files**:

```bash
python create_csv.py --metadata_files metadata/val_files.txt --input_files_prefix /path/to/raw/data/openwebtext --vocab_file ../../../vocab/google_research_uncased_L-12_H-768_A-12.txt --output_dir val_uncased_msl128 --do_lower_case --max_seq_length 128 --max_predictions_per_seq 20
```

#### Phase 2: MSL 512

**Generate training CSV files**:

```bash
python create_csv.py --metadata_files metadata/train_512k.txt --input_files_prefix /path/to/raw/data/openwebtext --vocab_file ../../../vocab/google_research_uncased_L-12_H-768_A-12.txt --output_dir train_512k_uncased_msl512 --do_lower_case --max_seq_length 512 --max_predictions_per_seq 80
```

**Generate validation CSV files**:

```bash
python create_csv.py --metadata_files metadata/val_files.txt --input_files_prefix /path/to/raw/data/openwebtext --vocab_file ../../../vocab/google_research_uncased_L-12_H-768_A-12.txt --output_dir val_uncased_msl512 --do_lower_case --max_seq_length 512 --max_predictions_per_seq 80
```

The above-created CSV files are then used by the `BertCSVDynamicMaskDataProcessor` class to produce inputs to the model.

# Run Pre-training

**IMPORTANT**: See the following notes before proceeding further.

**Parameter settings in YAML config file**: The config YAML files are located in [configs](configs/) directory. Before starting a pre-training run, make sure that in the YAML config file you are using:

   - The `train_input.data_dir` parameter points to the correct dataset, and
   - The `train_input.max_sequence_length` parameter corresponds to the sequence length of the dataset.

**Phase-1 with MSL128 and phase-2 with MSL512**: To continue the pre-training with MSL=512 from a checkpoint that resulted from the first phase of MSL=128, the parameter `model.max_position_embeddings` should be set to 512 for both the phases of pre-training.

**Phase-1 with MSL128**: If you would like to pre-train a model with MSL=128 and do not plan to use that model to continue pre-training with a longer sequence length, then you can change this `model.max_position_embeddings` parameter to 128.

## Run pre-training on the Cerebras System

To run pre-training on the Cerebras System, the training job should be launched inside of the Cerebras environment. In addition, the `cs_ip` should be provided either as a command line argument `--cs_ip` or in the YAML config file.

**Syntax**

Execute the following from within the Cerebras environment:

```
csrun_wse python-pt run.py --mode train --cs_ip x.x.x.x --params configs/params_bert_<model>.yaml --model_dir /path/to/model_dir
```

where:

- `/path/to/yaml` is a path to the YAML config file with model parameters. A few example YAML config files can be found in [configs](configs) directory.
- `/path/to/model_dir` is a path to the directory where you would like to store the logs and other artifacts of the run.

> **Note**: For training on the Cerebras System with an orchestrator like Slurm, also see [Train on the Cerebras System](https://docs.cerebras.net/en/latest/tensorflow-docs/running-a-model/train-eval-predict.html).

## Run pre-training on GPU

**Syntax**

To run pre-training on GPU, use the `run.py` Python utility as follows:
```
python run.py --mode train --params /path/to/yaml --model_dir /path/to/model_dir
```

## MLM Loss Scaling

To increase performance on the Cerebras System, MLM Loss is scaled using a constant factor called `mlm_loss_weight` rather than the exact number of masked tokens in the batch.
