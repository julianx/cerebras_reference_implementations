# OpenWebText (OWT) data preparation scripts

This directory contains scripts that can be used to download the [OpenWebText dataset](https://skylion007.github.io/OpenWebTextCorpus/). This raw data can be preprocessed into TFRecords using scripts found in `bert/tf/input/scripts` or into CSV files using scripts found in `bert/pytorch/input/scripts`.

## Data download and extraction

To download the OWT dataset, access the following link from a browser:

```url
https://drive.google.com/uc?id=1EA5V0oetDCOke7afsktL_JDQ-ETtNOvx
```

and manually download the tar.xz file there to the location you want.

To extract the manually downloaded files, run:

```bash
bash extract.sh
```

Note that `extract.sh` may take a while to complete, as it unpacks 40GB of data (8,013,770 documents). Upon completion, the script will produce `openwebtext` folder in the same location. The folder has multiple subfolders, each containing a collection of `*.txt` files of raw text data (one document per `.txt` file).

## Define train and evaluation datasets

Define metadata files that contain paths to subsets of documents in `openwebtext` folder to be used for training or evaluation. For example, for training, we use a subset of 512,000 documents. The associated metadata file can be found in `metadata/train_512k.txt`.

For evaluation, we choose 5,000 documents that are outside of the training set. The metadata file for evaluation can be found in `metadata/val_files.txt`. Users are free to create their own metadata files to define train and evaluation (as well as test) data subsets of different content and sizes.

These metadata files can be used in conjunction with `create_csv.py` or `create_tfrecords.py` in order to preprocess the data.
