from distutils.core import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='freidds',
    version='0.1dev',
    packages=['freidds'],
    license='GNU Affero General Public License v3.0',
    long_description=open('README.md').read(),
    include_package_data=True,
    install_requires=requirements
)
