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

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from typing import Optional

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from transformers import GPTJConfig, GPTJForCausalLM


def get_ckpt_from_hf(save_path: str):
    """
    Save checkpoint from HuggingFace to local path

    Args:
        save_path (str): Path to save checkpoint for the Gpt-J model

    Returns:
        None
    """
    if not os.path.exists(save_path + "pytorch_model.bin"):
        model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
        model.save_pretrained(save_path)


def create_pt_model(debug: bool = False):
    """
    Create a Gpt-J model using transformers PreTrainedConfig and output model
    parameters to a text file

    Args:
        debug (bool): Enable debug for model creation

    Returns:
        None
    """
    config = GPTJConfig()
    model = GPTJForCausalLM(config)

    if debug:
        # print number of parameters for debuging
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        num_params = sum([p.numel() for p in model_parameters])
        print(f"Initialized model with param count: {num_params}")

    return model


def create_tf_var(tensor: np.ndarray, name: str, session: tf.compat.v1.Session):
    """
    Takes a tensor and creates a TensorFlow Variable with same shape and dtype,
    initialized as zeros

    Args:
        tensor (ndarray): Tensor to model as zeros
        name (str): Name to give the variable
        session (tf.Session): Session to create the variables

    Returns:
        Zero initialized TF Variable
    """
    tf_dtype = tf.dtypes.as_dtype(tensor.dtype)
    tf_var = tf.compat.v1.get_variable(
        dtype=tf_dtype,
        shape=tensor.shape,
        name=name,
        initializer=tf.zeros_initializer(),
    )
    session.run(tf.compat.v1.variables_initializer([tf_var]))
    session.run(tf_var)
    return tf_var


def map_embeddings_pt_to_tf(key: str):
    """
    Map embedding layer to weights.

    Args:
        key (str): key to map from PyTorch to TensorFlow

    Returns:
        A tuple (name, transpose), where name is the equivalent TensorFlow
        layer name, and transpose indicates if we need to transpose the weights
        when mapping
    """

    transpose = False
    dict_map = {"transformer.wte.weight": "input_embedding/embedding_weights"}
    return dict_map[key], transpose


def map_outputs_pt_to_tf(key: str, shared_embeddings: bool = True):
    """
    Map final layer norm and output head bias to weights. Output weights for
    sharedweight layer comes from emebdding layer mapping.

    Args:
        key (str): key to map from PyTorch to TensorFlow
        shared_embeddings (bool): whether the model is trained with shared
            embeddings. Defaults to `True`

    Returns:
        A tuple (name, transpose), where name is the equivalent TensorFlow
        layer name, and transpose indicates if we need to transpose the weights
        when mapping
    """

    transpose = False
    if key == "lm_head.weight":
        transpose = True

    dict_map = {
        "transformer.ln_f.weight": "post_decoder_layer_norm/post_decoder_layer_norm/gamma",
        "transformer.ln_f.bias": "post_decoder_layer_norm/post_decoder_layer_norm/beta",
    }

    embeddings_map = {}
    if shared_embeddings:
        embeddings_map = {
            "lm_head.bias": "bias",
        }
    else:
        embeddings_map = {
            "lm_head.weight": "lm_head/lm_head/kernel",
            "lm_head.bias": "lm_head/lm_head/bias",
        }
    dict_map.update(embeddings_map)

    return dict_map[key], transpose


def map_ln_pt_to_tf(key: str):
    """
    Map LayerNorm Layer in decoder to weights.

    Args:
        key (str): key to map from PyTorch to TensorFlow

    Returns:
        A tuple (name, transpose), where name is the equivalent TensorFlow
        layer name, and transpose indicates if we need to transpose the weights
        when mapping
    """
    transpose = False
    split_key = key.split('.')
    pt_layer_num = int(split_key[2])

    final_key_dict = {
        "bias": "beta",
        "weight": "gamma",
    }
    final_key = final_key_dict[split_key[-1]]

    tf_key = f"gptj_decoder/{pt_layer_num}/ln_1/ln_1/{final_key}"
    return tf_key, transpose


def map_attn_pt_to_tf(key: str):
    """
    Map Attention Block in decoder to weights.

    Args:
        key (str): key to map from PyTorch to TensorFlow

    Returns:
        A tuple (name, transpose), where name is the equivalent TensorFlow
        layer name, and transpose indicates if we need to transpose the weights
        when mapping
    """
    transpose = False
    if "weight" in key:
        transpose = True

    split_key = key.split('.')
    pt_layer_num = int(split_key[2])

    last_two = ".".join([split_key[-2], split_key[-1]])
    if last_two in ["attn.bias", "attn.masked_bias"]:
        return None, False

    dict_map = {
        f"transformer.h.{pt_layer_num}": f"gptj_decoder/{pt_layer_num}",
        "attn": "attn",
        "q_proj": "q_proj/q_proj",
        "k_proj": "k_proj/k_proj",
        "v_proj": "v_proj/v_proj",
        "weight": "kernel",
        "bias": "bias",
        "out_proj": "out_proj/out_proj",
    }

    for k in dict_map.keys():
        if k in key:
            key = key.replace(k, dict_map[k])

    tf_key = key.replace(".", "/")
    return tf_key, transpose


def map_mlp_pt_to_tf(key: str):
    """
    Map MLP Block in decoder to weights.

    Args:
        key (str): key to map from PyTorch to TensorFlow

    Returns:
        A tuple (name, transpose), where name is the equivalent TensorFlow
        layer name, and transpose indicates if we need to transpose the weights
        when mapping
    """

    transpose = False
    if "weight" in key:
        transpose = True

    split_key = key.split('.')
    pt_layer_num = int(split_key[2])

    final_key_dict = {"bias": "bias", "weight": "kernel"}
    final_key = final_key_dict[split_key[-1]]

    if "fc_in" in key:
        tf_layer_num = int(pt_layer_num * 2)
    elif "fc_out" in key:
        tf_layer_num = int(pt_layer_num * 2 + 1)

    tf_key = f"gptj_decoder/{pt_layer_num}/mlp/dense_layer_{tf_layer_num}/dense_layer_{tf_layer_num}/{final_key}"
    tf_key = tf_key.replace("_0", "")
    return tf_key, transpose


def map_decoder_pt_to_tf(key: str):
    """
    Map decoder to weights.

    Args:
        key (str): key to map from PyTorch to TensorFlow

    Returns:
        A tuple (name, transpose), where name is the equivalent TensorFlow
        layer name, and transpose indicates if we need to transpose the weights
        when mapping
    """

    if "ln_1" in key:
        output_ = map_ln_pt_to_tf(key)
    elif "attn" in key:
        output_ = map_attn_pt_to_tf(key)
    elif "mlp" in key:
        output_ = map_mlp_pt_to_tf(key)
    else:
        raise ValueError(
            f"expected key to have one of ln_1, attn, mlp substrings"
            + f" got {key} instead. Check that the model definition is correct!!"
        )

    return output_


def convert_pt_checkpoint_to_tf(
    pt_model: nn.Module,
    pt_ckpt_path: str,
    tf_ckpt_path: str,
    mapping_args: Optional[dict] = None,
):
    assert pt_model, "Expected a PyTorch model, got None"
    if not os.path.isfile(pt_ckpt_path):
        raise ValueError(
            "Expected file for checkpoint, pass in correct path for PyTorch"
            + " checkpoint to load"
        )

    checkpoint = torch.load(pt_ckpt_path, map_location=torch.device('cpu'))
    pt_model.load_state_dict(checkpoint)

    pt_weight_dict = pt_model.state_dict()
    weight_keys = sorted(list(pt_weight_dict.keys()))

    shared_embeddings = mapping_args["shared_embeddings"]
    state_dict_update = {}
    pt_to_tf_keys_mapping = {}

    # perform mapping
    for key in weight_keys:
        if "transformer.wte" in key:
            tf_name, transpose = map_embeddings_pt_to_tf(key)

        elif "transformer.h" in key:
            tf_name, transpose = map_decoder_pt_to_tf(key)

        elif "transformer.ln_f" or "lm_head" in key:
            if shared_embeddings and key == "lm_head.weight":
                print(
                    f"{key} not needed for this configuration."
                    + f" Continuing maping for next keys."
                )
                continue

            tf_name, transpose = map_outputs_pt_to_tf(key, shared_embeddings)

        if tf_name is None:
            print(
                f"{key} mapped as None. Ignore if this is desired behavior."
                + " Else exercise the checkpoint saving with caution!!"
            )
            continue

        try:
            val = pt_weight_dict[key]
            update_val = val.T if transpose else val
            update_val = torch.Tensor(update_val)
            # if valid tensor, insert to mapping
            state_dict_update[tf_name] = update_val
            pt_to_tf_keys_mapping[key] = tf_name
        except TypeError as e:
            print(
                f"tried to map {key} to {tf_name}, but got wrong TensorType"
                + f". Ignoring mapping for now, to be fixed later !!"
            )

    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as session:
        for tf_var_name in state_dict_update.keys():
            torch_tensor = state_dict_update[tf_var_name].numpy()
            if tf_var_name:
                tf_var = create_tf_var(
                    tensor=torch_tensor, name=tf_var_name, session=session
                )
                tf.keras.backend.set_value(tf_var, torch_tensor)
                tf_weight = session.run(tf_var)

                if not np.allclose(tf_weight, torch_tensor):
                    print(
                        f"Variable {tf_var_name} created with wrong shape"
                        + " {tf_var.shape}. Check execution workflow!!"
                    )

        saver = tf.compat.v1.train.Saver(tf.compat.v1.trainable_variables())
        saver.save(session, tf_ckpt_path)

    return pt_weight_dict, pt_to_tf_keys_mapping


def main():
    save_path = "/path/to/checkpoint/storage"
    get_ckpt_from_hf(save_path)

    pt_model = create_pt_model()
    pt_ckpt_path = save_path + "pytorch_model.bin"

    # get TF checkpoint for shared embeddings setting
    tf_ckpt_path = save_path + "se_ckpts/tf_model.ckpt"
    pt_weight_dict, pt_to_tf_keys_mapping = convert_pt_checkpoint_to_tf(
        pt_model, pt_ckpt_path, tf_ckpt_path, {"shared_embeddings": True}
    )

    # get TF checkpoint for non shared embeddings setting
    tf_ckpt_path = save_path + "non_se_ckpts/tf_model.ckpt"
    pt_weight_dict, pt_to_tf_keys_mapping = convert_pt_checkpoint_to_tf(
        pt_model, pt_ckpt_path, tf_ckpt_path, {"shared_embeddings": False}
    )


if __name__ == "__main__":
    main()
