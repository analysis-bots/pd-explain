import os

from setuptools import setup, find_packages


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version():
    for line in read('src/pd_explain/__init__.py').splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


def get_long_description():
    with open('README.md', 'r') as fh:
        return fh.read()


setup(
    name='pd_explain',
    version=get_version(),
    description="Create explanation to dataframe",
    long_description=get_long_description(),  # Long description read from the readme file
    long_description_content_type="text/markdown",
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    project_urls={
        'Git': 'https://github.com/analysis-bots/pd-explain',
    },
    install_requires=[
        'wheel',
        'pandas>=1.4.2',
        'numpy>=1.20.3',
        'python-dotenv',
        'singleton-decorator',
        'matplotlib',
        'fedex-generator>=0.0.5',
    ]

)
