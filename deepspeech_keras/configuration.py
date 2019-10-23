import yaml
from typing import List, Dict


class Configuration:
    """
    Each experiment has own experiment.yaml configuration file. This
    configuration object passes through all methods which required
    additional parameters.
    """

    def __init__(self, file_path: str):
        """ All parameters saved in .yaml file convert to dot accessible """
        self._file_path = file_path
        self._data = self._read_yaml_file()
        self._check_file(required_keys=['features_extractor', 'model', 'callbacks', 'optimizer', 'decoder'])
        self.features_extractor = self._data.get('features_extractor')
        self.model = self._data.get('model')
        self.callbacks = self._data.get('callbacks')
        self.optimizer = self._data.get('optimizer')
        self.decoder = self._data.get('decoder')

    @property
    def data(self):
        return self._data

    def _read_yaml_file(self) -> Dict:
        """ Read YAML configuration file """
        with open(self._file_path, 'r') as stream:
            return yaml.load(stream)

    def _check_file(self, required_keys: List[str]):
        if not all(key in self._data for key in required_keys):
            raise KeyError(f'Configuration file should have all required keys: {required_keys}')
