from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='automatic_speech_recognition',
   version='1.0',
   description='Keras API on Automatic Speech Recognition',
   long_description=long_description,
   license="GNU",
   author='Rolczynski Rafal',
   include_package_data=True,
   packages=['automatic_speech_recognition'],
   install_requires=['h5py', 'numpy', 'pandas', 'scipy', 'python_speech_features', 'pytest', 'tensorflow',
                     'testfixtures']
)
