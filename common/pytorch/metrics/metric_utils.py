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

from torch import Tensor


def divide_no_nan(num: Tensor, denom: Tensor) -> Tensor:
    """
    Prevent zero division.
    Replicate the behavior of tf.math.divide_no_nan()
    """
    denom_zero_idx = denom == 0.0
    num[denom_zero_idx] = 0.0
    denom[denom_zero_idx] = 1.0
    return num / denom
