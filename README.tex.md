
### Automatic Speech Recognition

The project aim is to distill and simplify the Automatic Speech Recognition.
You can load a ready-to-use pipeline with a pre-trained model as the starting point in your research.

```python
import automatic_speech_recognition as asr

file = 'to/test/sample.wav'  # sample rate 16 kHz, and 16 bit depth
sample = asr.utils.read_audio(file)
pipeline = asr.load('deepspeech2', lang='en')
sentences = pipeline.predict([sample])
```

Now, we have a single english model (thanks to [Open Seq2Seq](https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition.html#speech-recognition)).
The evaluation results on the LibriSpeech dev-clean are in the table.
To reference, the DeepSpeech (Mozilla) achieves around 7.5% WER, in contrast, the state-of-the-art (RWTH Aachen University) is 2.3% WER
(recent evaluation results can be found [here](https://paperswithcode.com/sota/speech-recognition-on-librispeech-test-clean))
Both of them, use the external language model to boost results.

$$
\begin{table}[h!]
\vspace*{10pt}
\centering
\begin{tabular}{l c c c}
 
\toprule
Model Name   &   Decoder  &   WER-dev   \\
\midrule
deepspeech2  &    greedy  &   6.71      \\ 
\bottomrule

\end{tabular}
\end{table}
$$

<br>


Shortly it turns out that the inference is not enough to truly start your own experiments.
Take a look at the [CTC Pipeline](automatic_speech_recognition/pipeline/ctc_pipeline.py).
The pipeline is responsible for connecting a neural network model 
with all non-differential transformations (features extraction or prediction decoding)
Components are independent.
You can adjust them to your needs e.g. use more sophisticated feature extraction,
different data augmentation, or add a language model decoder (static n-grams or huge transformers).

<br>


```python
import numpy as np
import automatic_speech_recognition as asr

dataset = asr.dataset.Audio.from_csv('train.csv', batch_size=32)
dev_dataset = asr.dataset.Audio.from_csv('dev.csv', batch_size=32)
alphabet = asr.text.Alphabet(lang='en')
features_extractor = asr.features.FilterBanks(
    features_num=160,
    winlen=0.02,
    winstep=0.01,
    winfunc=np.hanning
)
model = asr.model.get_deepspeech2(
    input_dim=160,
    output_dim=29,
    rnn_units=800,
    is_mixed_precision=True
)
optimizer = asr.optimizer.Adam(
    lr=1e-4,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8
)
decoder = asr.decoder.GreedyDecoder()
pipeline = asr.pipeline.CTCPipeline(
    alphabet, features_extractor, model, optimizer, decoder
)
pipeline.fit(dataset, dev_dataset, epochs=25)
pipeline.save('/checkpoint')

test_dataset = asr.dataset.Audio.from_csv('test.csv')
wer, cer = asr.evaluate.calculate_error_rates(pipeline, test_dataset)
print(f'WER: {wer}   CER: {cer}')
```

Benefit from the eager TensorFlow 2.0, and freely monitor the model weights, activations or gradients.
You can do much more like distribute the training using the [Strategy](https://www.tensorflow.org/guide/distributed_training),
or experiment with [mixed precision](https://www.tensorflow.org/api_docs/python/tf/keras/mixed_precision/experimental/Policy) policy.

<br>


#### Installation
You can use pip:
```bash
pip install automatic-speech-recognition
```
Otherwise clone the code and create a new environment via [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#):
```bash
git clone https://github.com/rolczynski/Automatic-Speech-Recognition.git
conda env create -f=environment.yml     # or use: environment-gpu.yml
conda activate Automatic-Speech-Recognition
```

<br>


#### References

The fundamental repositories:
- Baidu - [DeepSpeech2 - A PaddlePaddle implementation of DeepSpeech2 architecture for ASR](https://github.com/PaddlePaddle/DeepSpeech)
- NVIDIA - [Toolkit for efficient experimentation with Speech Recognition, Text2Speech and NLP](https://nvidia.github.io/OpenSeq2Seq)
- RWTH Aachen University - [The RWTH extensible training framework for universal recurrent neural networks](https://github.com/rwth-i6/returnn)
- TensorFlow - [The implementation of DeepSpeech2 model](https://github.com/tensorflow/models/tree/master/research/deep_speech)
- Mozilla - [DeepSpeech - A TensorFlow implementation of Baidu's DeepSpeech architecture](https://github.com/mozilla/DeepSpeech) 
- Espnet - [End-to-End Speech Processing Toolkit](https://github.com/espnet/espnet)
- Sean Naren - [Speech Recognition using DeepSpeech2](https://github.com/SeanNaren/deepspeech.pytorch)
Moreover, you can explore the GitHub using key phrases like `ASR`, `DeepSpeech`, or `Speech-To-Text`.
