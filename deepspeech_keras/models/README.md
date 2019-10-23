# Pretrained Models

### Polish lanugage
The polish model weights are available [here](https://storage.googleapis.com/deepspeech-keras-weights/weights.hdf5).
Please normalize each feature channel based on `features-stats.txt`.
The samples are clipped to 7 words long (otherwise the LSTM could explode).
This is an initial version. This model achieves 16.4% WER on held out [Clarin Dataset](https://github.com/danijel3/ClarinStudioKaldi). 
