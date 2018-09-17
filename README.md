# DeepSpeech-Keras 

Project DeepSpeech-Keras is an open source environment for interaction with 
the Speech-To-Text engines. 

```python
from deepspeech import load_model

files = ['to/test/sample.wav']
deepspeech = load_model(name='polish-model.bin')
sentences = deepspeech(files)
```

The project is designed to achieve three goals:
- easy understanding of the program structure
- easy interaction with pre-trained models
- easy to run new experiments

The key to achieve these goals is by using the high-level [Keras API](https://github.com/keras-team/keras). 


## Installation
Install DeepSpeech-Keras from PyPI:
```bash
pip install deepspeech-keras
```

Otherwise create a new environment, clone the code and install requirements:
```bash
python3 -m venv /path/to/new/virtual/environment
git clone https://github.com/rolczynski/DeepSpeech-Keras.git
pip install -r requirements.txt
```


## Getting started: one minute to interact

Model training from the scratch requires the heavy computation. You can make a use 
of pre-trained models (CPU support). Each published the pre-trained model has these attributes:

```
deepspeech.model            # trained Keras model
          .alphabet         # describe valid chars (Mozilla DeepSpeech format)
          .configuration    # parameters used during training 
          .language_model   # support decoding (optional)
```

### Keras model
The heart of the  _deepspeech_ object is the Keras model. You can take a use of all
available Keras functional API [methods](https://keras.io/models/model/#methods), 
e.g. _predict_on_batch_. If you get probabilities along characters, you would 
like to decode the most probable sequence of chars. This process can be boosted
by the language model.

```python
from deepspeech import audio, text, load_model

deepspeech = load_model(name='polish-model.bin')
files = ['to/test/sample.wav']
transcripts = ['this is a test']

X = audio.get_features(files)
y = text.get_batch_labels(transcripts, deepspeech.alphabet)

y_hat = deepspeech.model.predict_on_batch(X)
sentences = deepspeech.decode(y_hat) # also you could pass custom language model
# or simple:  X, y_hat, sentences = deepspeech(files, full=True)
```

### Tune pre-trained model
Rather than write your own train method from the scratch, you can use the _deepspeech.train_ method.
Firstly create new/modify `configuration.yaml` file, where you specify all training 
parameters. Inside the method are set: _generators_, _optimizer_, _ctc loss_, _callbacks_ ect.

```python
from deepspeech import load_model
from deepspeech.configuration import Configuration

deepspeech = load_model(name='polish-model.bin')
deepspeech.configuration = Configuration('new-configuration.yaml')
deepspeech.train()
deepspeech.save('path/to/my_model.bin')
```

### Eager execution
Eager execution makes development and debugging more interactive, e.g. you can check how 
activations are changing. Be careful, not all Keras API methods have been already implemented.
Unfortunately the [Eager Execution](https://www.tensorflow.org/guide/eager) is not supported for the training. 
The few attempts were made but still the multi gpu support is problematic. 
However there are available models which were converted to the eager mode after the training. 

```python
import tensorflow as tf
tf.enable_eager_execution()

...
deepspeech = load_model(name='polish-model-eager.bin')

for layer in deepspeech.model.layers:
    activation = layer(X)
    ... # at the end you get y_hat
```


## Create new model
If you have few GPU's and huge amount of data, you could start to create the new model.
Training uses the Tensorflow static Graph execution, mainly because this ease the multi 
GPU support. Keras code under TensorFlow are refactored (but not finished yet) 
and in the future version this project will be switch to  `tf.keras`.

### Experiments management
The training often requires few attempts. You can use the `consumer.py` to run experiments 
sequentially. Define your queue where you can specify base configuration file and custom
parameters (like hyper-parameters), e.g.:
```
experiments/configuration.yaml|model.parameters.rnn_sizes=[1024,1024]|exp_dir="experiments/new rnns"
```

### Multi GPU support
In this project we do data parallelism via [`multi_gpu_model`](https://keras.io/utils/#multi_gpu_model).
Replicates a model on different GPUs and merge results on CPU. This induces 
quasi-linear speedup on up to 8 GPUs. 


### Data generation
The feature extraction (on CPU) is in parallel to model training (on GPU). This 
is done via [`fit_generator`](https://keras.io/utils/#fit_generator).

However, the features should be cached and the multiprocessing can be turn off.


### CuDNN support
The model used during the training contains the fast LSTM implementation by 
NVIDIA Developers. The computation performance is significantly improved. 
This is only available if you have NVIDIA GPU's.


### Distributed training
Folk really rare use the distributed training. The architecture, which offers 
this feature, often suffers from the boilerplate code. This project 
can be extended to distributed system by more sophisticated tools 
[Horovod](https://github.com/uber/horovod), 
[Dist-Keras](https://github.com/cerndb/dist-keras)
[TensorFlow Estimators](https://www.tensorflow.org/api_docs/python/tf/keras/estimator/model_to_estimator) 
or check out the _Mozilla_ implementation.


## Pre-trained models
The calculations are in progress.


## Related work
Amazing work is already done. Check out the implementations 
[Mozilla DeepSpeech](https://github.com/mozilla/DeepSpeech) (TensorFlow), 
[deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch) (PyTorch) or 
even [KerasDeepSpeech](https://github.com/robmsmt/KerasDeepSpeech) (Keras). 

This project is complementary to projects listed above. It tries to be more
user-friendly for the newcomer users. 


## Contributing
Have a question? Like the tool? Don't like it? Open an issue and let's talk 
about it! Pull requests are appreciated!


### The computational resource is available thanks to
![Usage](http://www.indopolishedu.com/wp-content/uploads/2018/03/polish.png)
