# FC-MNIST model

A simple multi-layer perceptron model composed of fully-connected layers
for performing handwriting recognition on the MNIST dataset.

### How to Run:

To run with the default configurations in `params.yaml`, run:
```bash
python run.py --params configs/params.yaml
```

### Structure

* `data.py`: Simple data input pipeline loading the TF MNIST dataset
* `model.py`: Model implementation
* `configs/params.yaml`: Example of a YAML configurations file
* `run.py`: Train script, performs train and eval
* `utils.py`: Misc helper scripts, including `get_params` which parses
the params dictionary from the YAML file

### Dataset and Input Pipeline

The MNIST dataset comes from a pre-made TF dataset. The train dataset
has a size of 60,000 and the eval dataset 10,000 images.
More information can be found on the
[Tensorflow website](https://www.tensorflow.org/datasets/catalog/mnist).
Each sample in the dataset is a black and white image of size 28x28, where
each pixel is an integer from 0 to 255 inclusive.

The first time that the input function is run, it will take some time
to download the entire dataset as a set of 10 `tfrecord` files, for a
total of 23MB.
The default behavior is to download the dataset to `~/tensorflow_datasets`,
although the `data_dir` can be customized (see the [TF documentation](
https://www.tensorflow.org/datasets/api_docs/python/tfds/load)
for more information).

Before running on CPU, `tfds` data should be downloaded using `prepare_data.py`. 

The input pipeline does minimal processing on this dataset. The dataset
returns one batch at a time, of the form:
```
inputs = (
    features = Tensor(size=(batch_size, 28*28), dtype=tf.floatX,
    labels = Tensor(size=(batch_size,), dtype=tf.int32,
)
```
Where here, `floatX = float32` if we are running in full precision and
`float16` if we are running in mixed precision.

### Model

The model is a 3-layer multi-layer perceptron. The first layer has hidden
size 500, the second 300, and the third layer has `num_classes` number of
hidden units (which here is 10). It then trains on a categorical cross entropy
loss. This structure is based on the survey of different structures on the
MNIST website (ref #3).
We also run it with a dropout rate of `0.02` and Adam Optimizer with
a learning rate of `1e-3`.

This model is able to achieve an accuracy of `~98.7%` after 100k steps
of training with a batch size of 256.
Evaluation accuracy here is measured as the `1 - top_one_error`, or
in other words is the fraction of the evaluation set for which the model
is able to predict the right digit on the first try.

### References

1. [Original MLP MNIST paper](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
2. [Tensorflow simple MNIST tutorial](
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/fully_connected_feed.py)
3. [MNIST website with wide survey of different parameters](
    http://yann.lecun.com/exdb/mnist/)
4. [Tensorflow Datasets MNIST documentation and code](
    https://www.tensorflow.org/datasets/catalog/mnist)
