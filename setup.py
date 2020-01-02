from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='automatic-speech-recognition',
   version='1.0',
   url='https://github.com/rolczynski/Automatic-Speech-Recognition',
   description='Distill the Automatic Speech Recognition in TensorFlow',
   long_description=long_description,
   license="GNU",
   author='Rolczynski Rafal',
   author_email='rafal.rolczynski@gmail.com',
   include_package_data=True,
   packages=['automatic_speech_recognition'],
   install_requires=[
      'tensorflow>=2.0', 'pandas', 'pytables',
      'google-cloud-storage',    # Load weights
      'python-speech-features>=0.6'
   ],
   python_requires='~=3.7',
)
