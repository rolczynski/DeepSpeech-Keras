import numpy as np
import warnings
from typing import List


class Alphabet:
    """
    Read alphabet.txt, which is the list of valid characters. Alphabet has two
    special characters:
      - space: on the beginning
      - blank: default added as the last char

    This class is used to convert characters to labels and vice versa.
    """

    def __init__(self, file_path: str):
        self.size = 0
        self.blank_token = None
        self._label_to_str = []
        self._str_to_label = {}
        self.process_alphabet_file(file_path)

    def __contains__(self, char: str) -> bool:
        """ Check if char is in the Alphabet. """
        return char in self._str_to_label

    def string_from_label(self, label: int) -> str:
        """ Convert label to string. """
        return self._label_to_str[label]

    def label_from_string(self, string: str) -> int:
        """ Convert string to label. """
        return self._str_to_label[string]

    def process_alphabet_file(self, file_path: str):
        """ Read alphabet.txt file. """
        with open(file_path) as file:
            for line in file:
                if line.startswith('#'):
                    continue
                # Char can contain more than one letter
                char = line[:-1]  # remove the line ending
                self._label_to_str.append(char)
                self._str_to_label[char] = self.size
                self.size += 1
            # Blank token is added on the end
            self.blank_token = self.size - 1

    def get_batch_labels(self, transcripts: List[str]) -> np.ndarray:
        """ Convert batch transcripts to labels """
        batch_labels = [[self.label_from_string(c) for c in transcript if c in self]
                        for transcript in transcripts]

        max_len = max(map(len, batch_labels))
        default_value = self.blank_token

        for labels in batch_labels:
            remainder = [default_value] * (max_len - len(labels))
            labels.extend(remainder)

        return np.array(batch_labels)

    def get_batch_transcripts(self, sequences: np.ndarray) -> List[str]:
        """ Convert label sequences to transcripts. The `-1` also means the blank tag """
        return [''.join(self.string_from_label(char_label) for char_label in sequence
                        if char_label not in (-1, self.blank_token))
                for sequence in sequences]
