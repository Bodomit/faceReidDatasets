from distutils.core import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='FaceReidDatasets',
    version='0.1dev',
    packages=['faceReidDatasets'],
    license='GNU Affero General Public License v3.0',
    long_description=open('README.md').read(),
    install_requires=requirements
)