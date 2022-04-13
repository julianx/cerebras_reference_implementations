# Scripts to Generate TFRecords

## create_tf_records.py

This script generates TFRecords for both the Masked Language Model (MLM) and Next Sentence Prediction (NSP) pre-training tasks using the BERT model.

### Description

A high level overview of the implementation is as follows:

1. Given a list of raw text documents, generate raw examples by concatenating the two parts `tokens-a` and `tokens-b` as follows:

        [CLS] <tokens-a> [SEP] <tokens-b> [SEP]

    where:

    - `tokens-a` is a list of tokens taken from the current document. The list is of random length (less than `msl`).
    - `tokens-b` is a list of tokens chosen based on the randomly set `next_sentence_labels`. The list is of length `msl-len(<tokens-a>)- 3` (to account for [CLS] and [SEP] tokens).

    If `next_sentence_labels` is 1, (that is, if set to 1 with 0.5 probability):
    
            "tokens-b" is a list of tokens from sentences chosen randomly from different documents.
            
    else:
    
            "tokens-b" is a list of tokens taken from the same document and is a continuation of "tokens-a" in the document. The number of raw tokens depends on "short_sequence_prob" as well.

2. Mask the raw examples based on `max_predictions_per_seq` and `mask_whole_word` parameters.

3. Pad the masked example to `max_sequence_length` if less than MSL.

The TFRecords generated from this script can be used for pretraining using the dataloader script [BertTfRecordsProcessor.py](../BertTfRecordsProcessor.py). For more details, refer to [sentence_pair_processor.py](../../../data_processing/sentence_pair_processor.py) and [create_tfrecords.py](./create_tfrecords.py).


```bash
Usage: create_tfrecords.py [-h] --metadata_files METADATA_FILES
                           [METADATA_FILES ...]
                           [--multiple_docs_in_single_file]
                           [--multiple_docs_separator MULTIPLE_DOCS_SEPARATOR]
                           [--single_sentence_per_line]
                           [--input_files_prefix INPUT_FILES_PREFIX]
                           --vocab_file VOCAB_FILE [--split_num SPLIT_NUM]
                           [--do_lower_case] [--max_seq_length MAX_SEQ_LENGTH]
                           [--dupe_factor DUPE_FACTOR]
                           [--short_seq_prob SHORT_SEQ_PROB]
                           [--min_short_seq_length MIN_SHORT_SEQ_LENGTH]
                           [--masked_lm_prob MASKED_LM_PROB]
                           [--max_predictions_per_seq MAX_PREDICTIONS_PER_SEQ]
                           [--spacy_model SPACY_MODEL] [--mask_whole_word]
                           [--output_dir OUTPUT_DIR]
                           [--num_output_files NUM_OUTPUT_FILES] [--name NAME]
                           [--seed SEED]

Required arguments:
  --metadata_files METADATA_FILES [METADATA_FILES ...]
                      Path to text file containing a list of file names
                      corresponding to the raw input documents to be
                      processed and stored; Multiple metadata
                      files must be separated by a space (default: None).
  --vocab_file VOCAB_FILE
                      Path to the vocabulary (default: None).

Optional arguments:
  -h, --help            Show this help message and exit.
  --multiple_docs_in_single_file
                        Pass this flag when a single text file contains
                        multiple documents separated by
                        <multiple_docs_separator> (default: False).
  --multiple_docs_separator MULTIPLE_DOCS_SEPARATOR
                        String which separates multiple documents in a
                        single text file. If newline character,
                        pass `\n`. There can only
                        be one separator string for all the documents.
                        (default: )
  --single_sentence_per_line
                        Pass this flag when the document is already
                        split into sentences, with one sentence in
                        each line. There is no requirement for further
                        sentence segmentation of a document
                        (default: False).
  --input_files_prefix INPUT_FILES_PREFIX
                        Prefix to be added to paths of the input
                        files. For example, can be a directory where
                        aw data is stored if the paths are relative
                        (default: ).
  --split_num SPLIT_NUM
                        Number of input files to read at a  given
                        time, for processing. Default: 1000.
  --do_lower_case       Pass this flag to lower case the input text.
                        Must be True for uncased models and False
                        for cased models (default: False).
  --max_seq_length MAX_SEQ_LENGTH
                        Maximum sequence length (default: 128).
  --dupe_factor DUPE_FACTOR
                        Number of times to duplicate the input data (with
                        different masks) (default: 10).
  --short_seq_prob SHORT_SEQ_PROB
                        Probability of creating sequences that are
                        shorter than the maximum sequence length
                        (default: 0.1).
  --min_short_seq_length MIN_SHORT_SEQ_LENGTH
                        The minimum number of tokens to be present in an
                        exampleif short sequence probability > 0.If None,
                        defaults to 2 Allowed values are [2, max_seq_length -
                        3) (default: None)
  --masked_lm_prob MASKED_LM_PROB
                        Masked LM probability (default: 0.15).
  --max_predictions_per_seq MAX_PREDICTIONS_PER_SEQ
                        Maximum number of masked LM predictions per
                        sequence (default: 20).
  --spacy_model SPACY_MODEL
                        The spaCy model to load. Either a shortcut
                        link, a package name or a path
                        (default: en_core_web_sm).
  --mask_whole_word     Set to True to use whole word masking and
                        False to use per-WordPiece masking
                        (default: False).
  --output_dir OUTPUT_DIR
                        Directory where TFRecords will be stored
                        (default: "./tfrecords/").
  --num_output_files NUM_OUTPUT_FILES
                        TFRecords will be separated into the
                        specified number of files on disk
                        (default: 10).
  --name NAME           Name of the dataset, i.e., prefix to use
                        for TFRecord names (default: "examples").
  --seed SEED           Random seed (default: 0).    
```
