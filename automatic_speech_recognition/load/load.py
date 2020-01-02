import os
import numpy as np
from .. import decoder
from .. import features
from .. import model
from .. import pipeline
from .. import text
from .. import utils


def load(name: str, lang: str, version=0.1) -> pipeline.Pipeline:
    if name == 'deepspeech2' and lang == 'en' and version == 0.1:
        return load_deepspeech2_en()
    raise ValueError('Specified model is not supported')


def weights(lang: str, name: str, version: float):
    """ Model weights are stored in the Google Cloud Bucket. """
    def closure(loader):
        """ The wrapper is required to run downloading after a call. """
        def wrapper() -> pipeline.Pipeline:
            bucket = 'automatic-speech-recognition'
            file_name = f'{lang}-{name}-weights-{version}.h5'
            remote_path = file_name
            local_path = f'{os.path.dirname(__file__)}/models/{file_name}'
            utils.maybe_download_from_bucket(bucket, remote_path, local_path)
            return loader(weights_path=local_path)
        return wrapper
    return closure


@weights(lang='en', name='deepspeech2', version=0.1)
def load_deepspeech2_en(weights_path: str) -> pipeline.CTCPipeline:
    deepspeech2 = model.get_deepspeech2(input_dim=160, output_dim=29)
    deepspeech2.load_weights(weights_path)
    alphabet_en = text.Alphabet(lang='en')
    spectrogram = features.Spectrogram(
        features_num=160,
        samplerate=16000,
        winlen=0.02,
        winstep=0.01,
        winfunc=np.hanning
    )
    greedy_decoder = decoder.GreedyDecoder()
    ctc_pipeline = pipeline.CTCPipeline(
        alphabet=alphabet_en,
        model=deepspeech2,
        optimizer=None,  # Inference mode
        decoder=greedy_decoder,
        features_extractor=spectrogram
    )
    return ctc_pipeline
