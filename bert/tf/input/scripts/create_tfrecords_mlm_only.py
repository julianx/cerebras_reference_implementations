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

"""
Script that generates a dataset in tfrecords
format for BertModel.
"""
import argparse
import json
import logging
import os
import sys

import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))
from bert.data_processing.mlm_only_processor import data_generator
from bert.data_processing.utils import (
    count_total_documents,
    get_output_type_shapes,
)
from bert.tf.input.utils import (
    create_tf_train_example,
    create_unmasked_tokens_example,
)
from common.input.utils import check_and_create_output_dirs


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--metadata_files",
        type=str,
        required=True,
        nargs='+',
        help="path to text file containing a list of file names "
        "corresponding to the raw input documents to be "
        "processed and stored; can handle multiple metadata files "
        "separated by space",
    )
    parser.add_argument(
        "--multiple_docs_in_single_file",
        action="store_true",
        help="Pass this flag when a single text file contains multiple"
        " documents separated by <multiple_docs_separator>",
    )
    parser.add_argument(
        "--multiple_docs_separator",
        type=str,
        default="\n",
        help="String which separates multiple documents in a single text file. "
        "If newline character, pass \\n"
        "There can only be one separator string for all the documents.",
    )
    parser.add_argument(
        "--single_sentence_per_line",
        action="store_true",
        help="Pass this flag when the document is already split into sentences with"
        "one sentence in each line and there is no requirement for "
        "further sentence segmentation of a document ",
    )
    parser.add_argument(
        "--allow_cross_document_examples",
        action="store_true",
        help="Pass this flag when examples can cross document boundaries",
    )
    parser.add_argument(
        "--document_separator_token",
        type=str,
        default="[SEP]",
        help="If examples can span documents, "
        "use this separator to indicate separate tokens of current and next document",
    )
    parser.add_argument(
        "--overlap_size",
        type=int,
        default=None,
        help="overlap size for generating sequences from buffered data for mlm only sequences"
        "Defaults to None, which sets the overlap to max_seq_len/4.",
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=1e6,
        help="buffer_size number of elements to be processed at a time",
    )
    parser.add_argument(
        '--input_files_prefix',
        type=str,
        default="",
        help='prefix to be added to paths of the input files. '
        'For example, can be a directory where raw data is stored '
        'if the paths are relative',
    )
    parser.add_argument(
        "--vocab_file", type=str, required=True, help="path to vocabulary"
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="pass this flag to lower case the input text; should be "
        "True for uncased models and False for cased models",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="maximum sequence length",
    )
    parser.add_argument(
        "--dupe_factor",
        type=int,
        default=10,
        help="number of times to duplicate the input data (with "
        "different masks) if disable_masking is False",
    )
    parser.add_argument(
        "--short_seq_prob",
        type=float,
        default=0.1,
        help="probability of creating sequences which are shorter "
        "than the maximum sequence length",
    )
    parser.add_argument(
        "--min_short_seq_length",
        type=int,
        default=None,
        help="The minimum number of tokens to be present in an example if short sequence probability > 0."
        "If None, defaults to 2 + overlap_size"
        "Allowed values are between [2 + overlap_size, max_seq_length-2)",
    )
    parser.add_argument(
        "--disable_masking",
        action="store_true",
        help="If False, TFRecords will be stored with static masks. If True, "
        "masking will happen dynamically during training.",
    )
    parser.add_argument(
        "--masked_lm_prob",
        type=float,
        default=0.15,
        help="masked LM probability",
    )
    parser.add_argument(
        "--max_predictions_per_seq",
        type=int,
        default=20,
        help="maximum number of masked LM predictions per sequence",
    )
    parser.add_argument(
        "--spacy_model",
        type=str,
        default="en_core_web_sm",
        help="spaCy model to load, i.e. shortcut link, package name or path.",
    )
    parser.add_argument(
        "--mask_whole_word",
        action="store_true",
        help="whether to use whole word masking rather than per-WordPiece "
        "masking.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tfrecords/",
        help="directory where TFRecords will be stored. "
        "Defaults to ./tfrecords/'.'",
    )
    parser.add_argument(
        "--num_output_files",
        type=int,
        default=10,
        help="number of files on disk to separate tfrecords into. "
        "Defaults to 10.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="examples",
        help="name of the dataset; i.e. prefix to use for TFRecord names. "
        "Defaults to 'examples'.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="random seed. Defaults to 0.",
    )

    return parser.parse_args()


def create_tfrecords(
    metadata_files,
    vocab_file,
    do_lower_case,
    max_seq_length,
    short_seq_prob,
    disable_masking,
    mask_whole_word,
    max_predictions_per_seq,
    masked_lm_prob,
    dupe_factor,
    tfrecord_name_prefix,
    output_dir,
    num_output_files,
    overlap_size=None,
    min_short_seq_length=None,
    buffer_size=1e6,
    allow_cross_document_examples=False,
    document_separator_token="[SEP]",
    multiple_docs_in_single_file=False,
    multiple_docs_separator="\n",
    single_sentence_per_line=False,
    inverted_mask=False,
    seed=None,
    spacy_model="en_core_web_sm",
    input_files_prefix="",
):

    num_output_files = max(num_output_files, 1)

    output_files = [
        os.path.join(output_dir, f"{tfrecord_name_prefix}-{fidx+1}.tfrecords")
        for fidx in range(num_output_files)
    ]

    output_type_shapes = get_output_type_shapes(
        max_seq_length, max_predictions_per_seq, mlm_only=True
    )

    writers = []
    for output_file in output_files:
        writers.append(tf.io.TFRecordWriter(output_file))

    def _data_generator():
        return data_generator(
            metadata_files=metadata_files,
            vocab_file=vocab_file,
            do_lower=do_lower_case,
            disable_masking=disable_masking,
            mask_whole_word=mask_whole_word,
            max_seq_length=max_seq_length,
            max_predictions_per_seq=max_predictions_per_seq,
            masked_lm_prob=masked_lm_prob,
            dupe_factor=1 if disable_masking else dupe_factor,
            output_type_shapes=output_type_shapes,
            multiple_docs_in_single_file=multiple_docs_in_single_file,
            multiple_docs_separator=multiple_docs_separator,
            single_sentence_per_line=single_sentence_per_line,
            overlap_size=overlap_size,
            min_short_seq_length=min_short_seq_length,
            buffer_size=buffer_size,
            short_seq_prob=short_seq_prob,
            spacy_model=spacy_model,
            inverted_mask=inverted_mask,
            allow_cross_document_examples=allow_cross_document_examples,
            document_separator_token=document_separator_token,
            seed=seed,
            input_files_prefix=input_files_prefix,
        )

    writer_index = 0
    total_written = 0

    tf.compat.v1.logging.info("Writing instances to output files...")
    for output_file in output_files:
        tf.compat.v1.logging.info(f"  {output_file}")

    for example in _data_generator():
        if disable_masking:
            tf_example = create_unmasked_tokens_example(example)
        else:
            features, labels = example
            tf_example = create_tf_train_example(features, labels)
        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)
        total_written += 1
        if total_written % 10000 == 0:
            tf.compat.v1.logging.info(f"{total_written} examples written...")

    for writer in writers:
        writer.close()

    return total_written


def main():
    args = parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    check_and_create_output_dirs(args.output_dir, filetype="tfrecord")

    total_written = create_tfrecords(
        metadata_files=args.metadata_files,
        vocab_file=args.vocab_file,
        do_lower_case=args.do_lower_case,
        max_seq_length=args.max_seq_length,
        short_seq_prob=args.short_seq_prob,
        disable_masking=args.disable_masking,
        mask_whole_word=args.mask_whole_word,
        max_predictions_per_seq=args.max_predictions_per_seq,
        masked_lm_prob=args.masked_lm_prob,
        dupe_factor=args.dupe_factor,
        inverted_mask=False,
        seed=args.seed,
        spacy_model=args.spacy_model,
        overlap_size=args.overlap_size,
        min_short_seq_length=args.min_short_seq_length,
        buffer_size=args.buffer_size,
        allow_cross_document_examples=args.allow_cross_document_examples,
        document_separator_token=args.document_separator_token,
        multiple_docs_in_single_file=args.multiple_docs_in_single_file,
        multiple_docs_separator=args.multiple_docs_separator,
        single_sentence_per_line=args.single_sentence_per_line,
        input_files_prefix=args.input_files_prefix,
        tfrecord_name_prefix=args.name,
        output_dir=args.output_dir,
        num_output_files=args.num_output_files,
    )

    # store arguments used for tfrecords
    # generation into a json file
    params = vars(args)
    params["n_examples"] = total_written
    params["n_docs"] = count_total_documents(args.metadata_files)
    json_params_file = os.path.join(args.output_dir, "data_params.json")
    with open(json_params_file, 'w') as _fout:
        json.dump(params, _fout)

    tf.compat.v1.logging.info(f"Done! Wrote total of {total_written} examples.")


if __name__ == "__main__":
    main()
