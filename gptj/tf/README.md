# GPT-J

[GPT-J](https://github.com/kingoflolz/mesh-transformer-jax) is a GPT-like autoregressive language model created by a "decentralized grassroots collective of 
volunteer researchers" called [EleutherAI](https://www.eleuther.ai/). A canonical configuration of the model, 
GPT-J-6B, has 6B parameters and it is one of the largest open alternatives to OpenAI's GPT-3.
GPT-J-6B has been trained by EleutherAI on a dataset called [The Pile](https://pile.eleuther.ai/), 
carefully assembled and curated from a large number of text datasets from different domains. 
GPT-J-6B has been demonstrated to perform reasonably well on a number of natural language tasks "as-is", 
without any further training, in zero-shot setting. However, the capabilities of the model can be improved 
significantly for domain-specific tasks and datasets with fine-tuning. With our implementation of GPT-J 
it is now easy to load a publicly available GPT-J checkpoint and fine-tune this model on a single CS-2 
with a custom domain-specific or task-specific dataset.
      
The design of the GPT-J model is similar to GPT-3 with a few notable differences:
* GPT-J introduces parallel decoder architecture, when attention and feed-forward layers in decoder are 
computed in parallel and then the results are added, as opposed to computing them  sequenctially 
by feeding attentiou output into feedforward layer, as in standard transformer models. This arictectural 
change has been introduced by EleutherAI to achieve higher throughput with distributed training, as it decreases communication. 
With traditional design, residual attention with op-sharding requires one all-reduce in the forward pass and one in the backward pass [2]. 
By computing attention and feedforward layers in parallel, the results can be reduced locally before performing a single all-reduce. 
This leads to an averge 15% increase in throughput on traditional hardware without noticable impact on convergence.
* GPT-J model uses Rotary Position Embeedings as in [3], which is shown to result in better model quality 
in tasks with long texts. We use 25% rotary embeddings, as it is shown to get a good balance between 
computational efficiency and model performance (convergence) [1].
* GPT-J uses dense attention instead of efficient sparse attention used in GPT-3. EleutherAI stated that dense attention has been used 
for simplicity, as sparse attention would not have significantly improved throughput at this scale. 

Although most of these design choices have been made witha single purpose of improving throughput on traditional hardware 
and are not expected to impact throughput on the Cerebras CS-2, we replicate GPT-J design in our implementation to be able to leverage 
available trained weights. This reference implementation can be used to train and fine-tune the GPT-J-6B model with 
Weight Streaming execution mode on Cerebras hardware. 

To start continuous pre-training or fine-tuning from publicly avalable pre-trained weights, please follow 
instructions in [`checkpoint_utils`](checkpoint_utils) to download and convert the weights into the expected format.   


## References

1. [Mesh-Transformer-JAX: Model-Parallel Implementation of Transformer Language Model with JAX](https://github.com/kingoflolz/mesh-transformer-jax), May 2021.
2. [Megatron-lm: Training multi-billion parameter language models using model parallelism](https://arxiv.org/abs/1909.08053), September 2019.
3. [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864), April 2021.
