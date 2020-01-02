import pathlib
import automatic_speech_recognition as asr
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
   name='automatic-speech-recognition',
   version=asr.__version__,  # Semantic: MAJOR, MINOR, and PATCH
   url='https://github.com/rolczynski/Automatic-Speech-Recognition',
   description='Distill the Automatic Speech Recognition (TensorFlow)',
   long_description=README,
   long_description_content_type="text/markdown",
   license="GNU",
   author='Rolczynski Rafal',
   author_email='rafal.rolczynski@gmail.com',
   include_package_data=True,
   packages=['automatic_speech_recognition'],
   install_requires=[
      'tensorflow>=2.0', 'pandas', 'tables', 'scipy',
      'google-cloud-storage',    # Load weights
      'python-speech-features'
   ],
   python_requires='~=3.7',
)
