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
Accuracy metric for PyTorch.
"""
from common.pytorch.metrics.cb_metric import (
    CBMetric,
)


class AccuracyMetric(CBMetric):
    """Calculates accuracy from labels and predictions, the top-1 accuracy."""

    def init_state(self):
        self.reset_state()

    def update_on_host(self, labels, predictions, weights=None):
        labels = labels.detach().flatten()
        predictions = predictions.detach().flatten()
        correct_predictions = (labels == predictions).float()
        if weights is None:
            num_tokens = float(correct_predictions.numel())
        else:
            weights = weights.detach().flatten()
            correct_predictions = correct_predictions * weights
            num_tokens = float(weights.sum())

        self.total_correct_predictions += correct_predictions.sum()
        self.total_num_tokens += num_tokens

    def compute(self):
        """Returns the computed accuracy as a float."""
        return float(self.total_correct_predictions / self.total_num_tokens)

    def reset_state(self):
        self.total_correct_predictions = 0.0
        self.total_num_tokens = 0.0
