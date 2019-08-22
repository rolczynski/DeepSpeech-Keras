from typing import List

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from tabulate import tabulate

from decoding import BestPathDecoder, CTCDecoder
from text import Alphabet
from metric import get_metrics


ACTIVATION_PATH = '../scripts/evaluation-clarin-without-activations.hdf5'
ALPHABET = Alphabet(file_path='../models/pl/alphabet.txt')
LIMIT = -1  # Number of samples to use for evaluation (-1 for all)


def get_references(fname: str) -> pd.DataFrame:
    with pd.HDFStore(fname, mode='r') as store:
        return store['references']


def read_probabilities(fname: str, references: pd.DataFrame) -> List[np.ndarray]:
    with h5py.File(fname, mode='r') as store:
        output_index = 1
        return [store[f'outputs/{output_index}/{sample_id}'][:]
                for sample_id in tqdm(references.index)]


references = get_references(ACTIVATION_PATH)
probs = read_probabilities(ACTIVATION_PATH, references)
transcripts = references['transcript']


decoders = [
    ('Best-path decoding', BestPathDecoder(config={'alphabet': ALPHABET})),
    ('CTC decoding w/o lm', CTCDecoder(language_model=None, config={'alphabet': ALPHABET})),
]
results = []
for name, decoder in decoders:
    decoded_transcripts = [transcript for transcript, _ in decoder.batch_decode(probs[:LIMIT])]
    metrics = list(get_metrics(decoded_transcripts, transcripts[:LIMIT]))
    avg_cer = sum(metric.wer for metric in metrics)/len(metrics)
    avg_wer = sum(metric.cer for metric in metrics)/len(metrics)
    results.append((name, avg_cer, avg_wer))
print(tabulate(results, headers=['name', 'cer', 'wer'], tablefmt="grid"))



