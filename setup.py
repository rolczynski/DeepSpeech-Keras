from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='deepspeech_keras',
   version='1.0',
   description='Keras API on Deepspeech',
   long_description=long_description,
   license="GNU",
   author='Rolczynski Rafal',
   include_package_data=True,
   packages=['deepspeech_keras'],
)
