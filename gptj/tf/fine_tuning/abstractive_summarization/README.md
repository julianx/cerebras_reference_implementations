# Abstractive Summarization with GPTJ

## Data

Reference: https://blog.paperspace.com/generating-text-summaries-gpt-2/
Data from: https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail

1. Download finished files archive and unzip. Set the following variables: DATA_PATH, VOCAB_FILE, ENCODER_FILE, and OUTPUT_DIR. Then run for each of train, test, and val: `python create_tfrecords.py --data_file $DATA_PATH/[train|test|val].bin --vocab_file $VOCAB_FILE --encoder_file $ENCODER_FILE --max_seq_length 2048 --output_dir $OUTPUT_DIR/[train|test|val] --name [train|test|val]`

### Data Processing

Each example consists of an article and abstract. For examples with missing articles or missing abstracts, we skip the example. If the total sequence length is greater than MSL, we truncate the article. In the case that total sequence length is greater than MSL and the abstract length is greater than article length, we just skip the article. When running the steps above (defaults to 2048 MSL), we get the following number of examples in each dataset:

1. Train #: 287108. 119 examples were skipped, 114 of which were b/c one of the two was empty.
2. Val #: 13367. 1 was skipped due to length.
3. Test #: 11490. None skipped.

When using MSL 1024,  train set includes 287093 examples and validation set has 13367.

By default every sequence is structured as: [sos_id, article_token_ids, sep_id, abstract_token_ids, pad to MSL] and label is [article_token_ids, eos, abstract_token_ids, eos, pad to MSL], where sos_id=eos_id=pad_id and sep_id is distinct. This is in contrast to this reference:

https://github.com/SKRohit/Generating_Text_Summary_With_GPT2/blob/91963b12a59dc981f136f98df046f6dc584bd8a5/dataset.py#L49

where `content = data['article'] + self.tokenizer.encode(self.tokenizer.sep_token) + data['abstract']` followed by padding. In this reference, it doesn't seem like eos/sos is added.

We are also using a new SEP token (instead of trying to reuse a token from the original vocabulary). Because we have some extra unused vocab slots available in GPTJ, we use one of those (i.e. vocab size is set to 50400, includng for our pretraining, but there are only 50257 tokens in the standard encoder file we use. Thus, we can easily add a token without adjusting the vocabulary size).

Data processing is largely copied from Cerebras GPT2 data processing. When processing, while the pretraining tf record generation script (see `gpt2/input/create_tfrecords.py`) includes a `short_seq_prob` argument, we've removed this for now.

These examples are stored as tf records - each record has an input, label, and input mask (which is set to 0 for the mask). Our default is to consider loss over the article and abstract (i.e. the input mask has a 1 for every token that isn't a pad). The reference above only considers loss over the abstract. To enable this, one can re-generate the dataset with 0s in the mask for tokens corresponding to the article. Another approach would be to include article lengths in the tf record example and handle this in the data loader (not yet implemented).

Note that at the time of tfrecord creation, `inverted_mask=False`, hence true words have 1 and masks 0 (as we want). In `GPTTFRecordsProcessor.py` example["input_mask"] = tf.equal(example["input_mask"], 0) meaning that this is swapped. In loss calculation (`_compute_loss` in `GptJModel.py`), this is swapped again mask = 1 - features["input_mask"]. Ultimately, it is correctly applied, but goes through some gymnastics to get there.

## Model

The GPTJ finetuning model for abstractive summarization is the same as the GPTJ model.

For BERT style modeling, it is important to mask pads when computing attention. Here, note that all pads are always at the end of the sequence. Therefore, because attention is done with an auto regressive mask, no non-pad token will use pad tokens in attention. Ultimately, all computations involving pads will be ignored due to masking of padding tokens in the loss. If a user does not want to include the article tokens in the loss, this is best handled in the loss itself. Passing the mask into attention will stop tokens from attending to tokens in the article - wheras, we want the opposite, as summaries are based on the articles they are summarizing.

The current config `params_finetuning.yaml` is a placeholder for CS-x runs. To run this on GPU, you can reduce the size of a model (tested on GPU with 12 layers, 768 hidden size, 12 heads, 3072 filter size, 16 rotary dimensions, "tf_dynamic" and bsz 4).

Example run command: `python run.py --mode train_and_eval --params fine_tuning/abstractive_summarization/configs/[params] --model_dir small_test --variant gptj_6b`.

## Generation

1. Add the vocab and encoder files to the config. The generation file will automatically add these as well if they are not present.

```
  vocab_file: "/path/to/vocab.bpe"
  encoder_file: "/path/to/encoder.json"
```

2. The generating script `fine_tuning/abstractive_summarization/text_generation/generate.py` is adapted from the GPT-2 generating script. Note that this only works within monolith and is not GPU compatible.  Example: `python fine_tuning/abstractive_summarization/text_generation/generate.py --params [params] --ckpt_path [ckpt folder] --summary_length 10 --top_k 10 --top_p 8 --temperature 0.8`

The script automatically adds a seperator token as well.

## Evaluation

To evaluate on a checkpoint `python run.py --mode eval --params [params] --model_dir [model_dir] --checkpoint_path [path to checkpoint i.e. model.ckpt-9000]`.

## Run on GPU

1. Set batch size smaller and enable gradient accumulation.
2. Set loss scale to "tf_dynamic" instead of "dynamic"
3. If using distributed data parallel training, set distributed to True.
