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
Processor for handling TF records data for BERT
"""

import tensorflow as tf
from bert.tf.input.utils import (
    parse_raw_tfrecord,
)
from common.tf.input.TfRecordsProcessor import (
    TfRecordsProcessor,
)
from common.tf.input.utils import (
    bucketed_batch,
)


class BertTfRecordsProcessor(TfRecordsProcessor):
    """
    Creates dataset from pre-compiled TF records
    :param <dict> params: dict containing training input parameters for creating dataset
    """

    def __init__(self, params):
        super(BertTfRecordsProcessor, self).__init__(
            params["data_dir"],
            params["batch_size"],
            shuffle=params.get("shuffle", True),
            shuffle_seed=params.get("shuffle_seed", None),
            shuffle_buffer=params.get("shuffle_buffer", None),
            repeat=params.get("repeat", True),
            use_multiple_workers=params.get("use_multiple_workers", False),
            n_parallel_reads=params.get("n_parallel_reads", 4),
            map_before_batch=True,
        )
        self.max_sequence_length = params["max_sequence_length"]
        self.max_predictions_per_seq = params["max_predictions_per_seq"]
        self.input_pad_id = params.get("input_pad_id", None)
        self.segment_pad_id = params.get("segment_pad_id", None)
        self.mlm_pad_id = params.get("mlm_pad_id", None)
        self.scale_mlm_weights = params.get("scale_mlm_weights", False)
        self.sort_within_batch = params.get("sort_within_batch", False)

        # buckets must be either an integer or a list of boundaries of length
        # num_buckets - 1. If it is a list, it should exclude 0 and MSL. Buckets
        # are inclusive on the bottom and exclusive on the top
        self.buckets = params.get("buckets", 1)
        if isinstance(self.buckets, int):
            self.buckets = [
                int((i + 1) * self.max_sequence_length / self.buckets)
                for i in range(self.buckets - 1)
            ]

        self.mp_type = (
            tf.float16 if params.get("mixed_precision") else tf.float32
        )

    def batch_fn(self, dataset):
        if self.buckets:
            return bucketed_batch(
                dataset,
                element_length_func=lambda f, _: (
                    self.max_sequence_length - tf.reduce_sum(f["input_mask"])
                ),
                bucket_boundaries=self.buckets,
                batch_size=self.batch_size,
                no_padding=True,
                drop_remainder=True,
            )
        else:
            return dataset.batch(self.batch_size, drop_remainder=True)

    def map_fn(self, raw_record):
        """
        Parses a serialized protobuf example into a dictionary
        of input features and labels for BERT pretraining.
        """

        features, label = parse_raw_tfrecord(
            raw_record=raw_record,
            max_sequence_length=self.max_sequence_length,
            max_predictions_per_seq=self.max_predictions_per_seq,
            mp_type=self.mp_type,
        )
        label = tf.squeeze(label, axis=-1)

        if self.input_pad_id is not None:
            features["input_ids"] = tf.where(
                tf.cast(features["input_mask"], tf.bool),
                self.input_pad_id,
                features["input_ids"],
            )
        if self.segment_pad_id is not None:
            features["segment_ids"] = tf.where(
                tf.cast(features["input_mask"], tf.bool),
                self.segment_pad_id,
                features["segment_ids"],
            )
        if self.mlm_pad_id is not None:
            features["masked_lm_ids"] = tf.where(
                tf.cast(features["masked_lm_weights"], tf.bool),
                features["masked_lm_ids"],
                self.mlm_pad_id,
            )
        return features, label

    def post_batch_map_fn(self, features, label):
        """
        When appropriate, scale mlm weights by `batch_size / num_valid_tokens`.
        This is used to compute the correct scaling factor on the loss without
        running into precision issues. Intended for use in situations when the
        loss will be divided by `batch_size` at the time of computation.

        If desired, also sort elements within each batch by sequence length.
        """
        if self.scale_mlm_weights:
            mlm_weights = features["masked_lm_weights"]
            scale = self.batch_size / tf.reduce_sum(mlm_weights)
            features["masked_lm_weights"] = tf.cast(
                mlm_weights * scale, self.mp_type
            )

        if self.sort_within_batch:
            # sort the entries in the batch by sequence length
            lengths = tf.reduce_sum(1 - features["input_mask"], axis=-1)
            indices = tf.argsort(lengths)
            features = {
                key: tf.gather(features[key], indices) for key in features
            }
            label = tf.gather(label, indices)

        return features, label
