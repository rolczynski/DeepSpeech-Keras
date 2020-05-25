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
    'dev-clean': {
        'url': 'http://www.openslr.org/resources/12/dev-clean.tar.gz',
    },
    'dev-other': {
        'url': 'http://www.openslr.org/resources/12/dev-other.tar.gz',
    },
    'test-clean': {
        'url': 'http://www.openslr.org/resources/12/test-clean.tar.gz',
    },
    'test-other': {
        'url': 'http://www.openslr.org/resources/12/test-other.tar.gz',
    },
    'train-clean-100': {
        'url': 'http://www.openslr.org/resources/12/train-clean-100.tar.gz',
    },
    'train-clean-360': {
        'url': 'http://www.openslr.org/resources/12/train-clean-360.tar.gz',
    },
    'train-other-500': {
        'url': 'http://www.openslr.org/resources/12/train-other-500.tar.gz',
    }
}


def read_transcript(filename: str) -> List[Tuple[str, str]]:
    """
    Reads rows and splits them by first space
    :param filename:
    :return:
    """
    with open(filename, 'r') as file:
        result = [tuple(line.strip().lower().split(' ', 1))
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


def create_index_data(cwd: str, dataset_path: str) -> pd.DataFrame:
    """
    Creates the index file which is suitable for the pipeline.
    The file contains paths to audiofiles and the transcripts

    :param cwd: current work directory.
                All paths will be relative to this directory.
    :param dataset_path: path of the dataset. May be either absolute or
                relative
    :return: pandas DataFrame with path, transcript and file size
    """
    prefix = os.path.relpath(dataset_path, start=cwd)

    def walk_dirs(current_folder: str):
        file_paths = []
        transcripts = []
        file_sizes = []
        for item in os.listdir(current_folder):
            if os.path.isdir(os.path.join(current_folder, item)):
                # Go recursive
                (item_file_paths,
                 item_transcripts, item_filesizes) = walk_dirs(
                     os.path.join(current_folder, item))

                file_paths.extend(item_file_paths)
                transcripts.extend(item_transcripts)
                file_sizes.extend(item_filesizes)

            elif item.endswith('.trans.txt'):
                # Read transcript
                item_file_paths, item_transcripts = zip(*read_transcript(
                    os.path.join(current_folder, item)))
                for file_name in item_file_paths:
                    file_paths.append(
                        os.path.relpath(
                            os.path.join(
                                current_folder, f'{file_name}.wav'),
                            start=cwd)
                        )
                    file_sizes.append(
                        os.path.getsize(
                            os.path.join(
                                current_folder, f'{file_name}.wav')
                        )
                    )
                    transcripts.extend(item_transcripts)
        return file_paths, transcripts, file_sizes

    file_paths, transcripts, file_sizes = walk_dirs(dataset_path)
    index_data = pd.DataFrame(
        zip(file_paths, transcripts, file_sizes),
        columns=['path', 'transcript', 'file_size'])

    return index_data


def transcode_flac_wav_recursive(path, keep_original=False):
    """
    For any FLAC file found in path creates a corresponding WAV file
    :param path: entry point
    :param keep_original: if original FLAC file should be kept
    :return: None if successfull
    """
    for item in os.listdir(path):
        if os.path.isdir(os.path.join(path, item)):
            transcode_flac_wav_recursive(
                os.path.join(path, item), keep_original=keep_original)
        elif item.endswith('.flac'):
            convert_flac_to_wav(
                os.path.join(path, item),
                os.path.join(path, os.path.splitext(item)[0] + '.wav')
            )
            if not keep_original:
                os.remove(os.path.join(path, item))


def main(ds_name: str, cwd: str, dest_dir: str):
    """
    :param ds_name: name of the dataset to use
    :param cwd: current work dir, where index will be placed
    :param dest_dir: location where the dataset will be placed
    """
    url = datasets[ds_name]['url']

    tar_file_name = url.split('/')[-1]
    audio_data_dir = tar_file_name.split('.', 1)[0]

    # Prepare main dataset
    if not os.path.isdir(
            os.path.join(dest_dir, 'LibriSpeech', audio_data_dir)):
        if not os.path.isfile(tar_file_name):
            logging.info(f'Not found tar file {tar_file_name}.'
                         ' Downloading it.')
            urllib.request.urlretrieve(url, tar_file_name)
            logging.info(f'Successfully downloaded tar file.')

        logging.info(f'Extracting into {dest_dir}')
        tar = tarfile.open(tar_file_name, "r:gz")
        tar.extractall(dest_dir)
        tar.close()

    # Transcode FLAC to WAV and create an index
    logging.info('Transcoding FLAC files')
    transcode_flac_wav_recursive(
        os.path.join(dest_dir, 'LibriSpeech', audio_data_dir))

    logging.info('Generating index data')
    index_data = create_index_data(
        cwd, os.path.join(dest_dir, 'LibriSpeech', audio_data_dir))
    index_data.to_csv(os.path.join(cwd, f'{ds_name}-index.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare LibriSpeech data')
    parser.add_argument('--type', type=str,
                        help='which dataset to download',
                        default='dev-clean',
                        choices=datasets.keys())
    parser.add_argument('--dest', type=str,
                        help='where to place final dataset',
                        default='.')
    parser.add_argument('--index_top', type=str,
                        help='path relative to which all'
                             'audio paths will be indexed',
                        default='.')
    args = parser.parse_args()
    main(args.type, args.index_top, args.dest)
