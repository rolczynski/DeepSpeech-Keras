import os
from typing import List, Tuple
import urllib.request
import tarfile
import argparse
import logging
from pydub import AudioSegment
import pandas as pd
import tempfile

logging.basicConfig(level=logging.INFO)

datasets = {
    'dev': {
        'url': 'http://www.openslr.org/resources/12/dev-clean.tar.gz',
        'result_folder': 'libri_dev'
    },
    'test': {
        'url': 'http://www.openslr.org/resources/12/test-clean.tar.gz',
        'result_folder': 'libri_test'
    }
}


def read_transcript(filename: str) -> List[Tuple[str, str]]:
    """
    Reads rows and splits them by first space
    :param filename:
    :return:
    """
    with open(filename, 'r') as file:
        result = [tuple(line.strip().split(' ', 1).lower())
                  for line in file.readlines()]
    return result


def convert_flac_to_wav(source: str, dest: str):
    if os.path.isfile(dest):
        logging.info(
            f"Tried transfer {source} but file already exists {dest}")
    audio = AudioSegment.from_file(source, 'flac')
    audio.export(dest, 'wav')


def transfer_transcripted_audio(
        search_folder: str, dataset_folder: str) -> List[Tuple[str, str]]:
    """
    Recursive procedure goes over all files in search folder.
    If it meets *.trans.txt then it adds all of the files to
    dataset in dataset_folder.
    If it meets another folder then recursively executes on it.
    :param search_folder:
    :param dataset_folder:
    :return: transcripts
    """
    transcripts = []
    file_sizes = []
    for file in os.listdir(search_folder):
        if os.path.isdir(f'{search_folder}/{file}'):
            # Go recursive
            rec_transcripts, rec_filesizes = transfer_transcripted_audio(
                f'{search_folder}/{file}', dataset_folder)
            transcripts.extend(rec_transcripts)
            file_sizes.extend(rec_filesizes)
        elif file.endswith('.trans.txt'):
            # Read transcript and move each audio to final folder
            # converting in the same time
            current_transcript = read_transcript(
                f'{search_folder}/{file}')
            for audio_name, _ in current_transcript:
                convert_flac_to_wav(
                    f'{search_folder}/{audio_name}.flac',
                    f'{dataset_folder}/{audio_name}.wav')
                file_sizes.append(
                    os.path.getsize(f'{dataset_folder}/{audio_name}.wav'))
            transcripts.extend(current_transcript)
    return transcripts, file_sizes


def main(ds_name: str, audio_path_prefix: str):
    url = datasets[ds_name]['url']
    result_folder = datasets[ds_name]['result_folder']

    # tar_name = f'{ds_folder_name}.tar.gz'
    tar_file_name = url.split('/')[-1]
    audio_data_dir = tar_file_name.strip('.tar.gz')

    # Prepare main dataset
    if not os.path.isfile(tar_file_name):
        logging.info(f'Not found tar file {tar_file_name}.'
                     ' Downloading it.')
        urllib.request.urlretrieve(url, tar_file_name)
        logging.info(f'Successfully downloaded tar file.')

    with tempfile.TemporaryDirectory(prefix='.') as tmp_dir:
        logging.info(f'Extracting into {tmp_dir}')
        tar = tarfile.open(tar_file_name, "r:gz")
        tar.extractall(tmp_dir)
        tar.close()

        # Restructure dataset to our format
        if not os.path.isdir(result_folder):
            os.mkdir(result_folder)
            logging.info('Starting to convert all audio files and gather'
                         ' transcripts.')
            all_transcripts, all_filesizes = transfer_transcripted_audio(
                f'{tmp_dir}/LibriSpeech/{audio_data_dir}',
                result_folder)

    logging.info('Saving transcripts int data.csv.')
    transcripts_df = pd.DataFrame(
        all_transcripts, columns=['path', 'transcript'])
    transcripts_df['filesize'] = all_filesizes
    transcripts_df.path = audio_path_prefix + f"/{result_folder}/" +\
        transcripts_df.path + ".wav"
    transcripts_df.to_csv(f'{result_folder}/data.csv')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    parser = argparse.ArgumentParser(
        description='Prepare LibriSpeech data')
    parser.add_argument('--type', type=str,
                        help='which dataset to download',
                        default='dev',
                        choices=datasets.keys())
    parser.add_argument('--prefix', type=str,
                        help='prefix to add before all audio paths',
                        default='.')
    args = parser.parse_args()
    main(args.type, args.prefix)
