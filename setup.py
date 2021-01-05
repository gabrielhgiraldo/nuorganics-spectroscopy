from pathlib import Path
from setuptools import find_packages, setup

version = (Path(__file__).parent / "spectroscopy" / "VERSION").read_text()
README = (Path(__file__).parent / "README.MD").read_text()
setup(
    name='nuorganics-spectroscopy',
    version=version,
    packages=find_packages(exclude=('tests',)),
    description='spectroscopy models for nuorganics',
    long_description=README,
    long_description_content_type="text/markdown",
    author="Gabriel Giraldo-Wingler",
    author_email="Gabrielhgiraldo@gmail.com",
    install_requires=[
        'dash',
        'matplotlib',
        'pandas',
        'sklearn',
        'skorch',
        'torch'
    ],
    include_package_data=True
)