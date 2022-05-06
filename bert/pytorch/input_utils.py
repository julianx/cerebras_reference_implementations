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

import numpy as np

try:
    import cerebras.framework.torch.core.cb_model as cm
except ImportError:
    # use stand-in namespaces if cerebras.framework.torch is not importable
    from cerebras_reference_implementations.common.pytorch import cb_model as cm


def get_data_for_task(
    task_id,
    meta_data_values_cum_sum,
    num_examples_per_task,
    meta_data_values,
    meta_data_filenames,
):
    """
    Function to get distribute files with given number of examples such that each
    distributed task has access to exactly the same number of examples

    :param int task_id: integer id for a task.
    :param int meta_data_values_cum_sum: Cumulative sum of the file sizes in lines from meta data file.
    :param int num_examples_per_task: Number of the examples specified per slurm task.
      Equal to `batch_size` * `num_batch_per_task`.
    :param list[int] meta_data_values: List of the files sizes in lines in the meta data file.
    :param list[str] meta_data_filenames: List with file names in the meta data file.
    returns list of tuples of length 3. The tuple contains at
      - index 0: filepath.
      - index 1: number of examples to be considered for this task_id.
      - index 2: start index in the file from where these
                examples should be considered
        The list represents the files that should be
        considered for this task_id

    """
    files_in_task = []

    # file where the split starts
    file_start_idx = np.min(
        np.where(meta_data_values_cum_sum > task_id * num_examples_per_task)[0]
    )
    # Index in file from where the examples should be considered for this task
    start_idx = (
        task_id * num_examples_per_task
        - meta_data_values_cum_sum[file_start_idx - 1]
        # -1 since len(`meta_data_values_cum_sum`) = len(`meta_data_values`) + 1
    )

    # Number of examples to pick from this file.
    # We do a `min` to handle a case where the file has
    # examples > num_examples_per_task
    num_examples = min(
        meta_data_values[file_start_idx - 1] - start_idx, num_examples_per_task,
    )
    files_in_task.append(
        (
            meta_data_filenames[file_start_idx - 1],
            num_examples,
            start_idx,
        )  # (file_path, num_examples, start_index)
    )

    if num_examples != num_examples_per_task:
        # If the file has fewer number of examples than
        # `num_examples_per_task`, continue through files
        # till we reach our required number of examples.

        indices = np.where(
            meta_data_values_cum_sum > (task_id + 1) * num_examples_per_task
        )[0]
        if indices.size != 0:
            file_end_idx = np.min(indices)
        else:
            file_end_idx = len(meta_data_values_cum_sum)

        for i in range(file_start_idx + 1, file_end_idx):
            files_in_task.append(
                (
                    meta_data_filenames[i - 1],
                    meta_data_values[i - 1],
                    0,
                )  # (file_path, num_examples, start_index)
            )

        # If the number of examples needed to fulfill
        # `num_examples_per_task`, falls in between a file
        num_end_examples = (
            task_id + 1
        ) * num_examples_per_task - meta_data_values_cum_sum[file_end_idx - 1]
        if num_end_examples > 0:
            files_in_task.append(
                (
                    meta_data_filenames[file_end_idx - 1],
                    num_end_examples,
                    0,
                )  # (file_path, num_examples, start_index)
            )

    assert (
        sum([num_examples for _, num_examples, _ in files_in_task])
        == num_examples_per_task
    ), f"Incorrect number of examples in the split with task_id {task_id}"

    return files_in_task


def task_id():
    return cm.get_streaming_rank() if cm.is_streamer() else 0
