import os
import subprocess
import sys

from setuptools import setup, find_packages, Command


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


class UpdateDependenciesCommand(Command):
    description = 'Update dependencies to the latest versions'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'fedex-generator'])
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'cluster-explorer'])
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'external-explainers'])



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
        'pandas>=2.2.3',
        'numpy>=2.1.3',
        'python-dotenv',
        'singleton-decorator>=1.0.0',
        'matplotlib>=3.9.0',
        'fedex-generator>=1.0.6',
        'cluster-explorer>=1.0.2',
        'external-explainers>=1.0.1',
        'python-dotenv~=1.0.0',
        'openai~=1.66.0',
        'ipywidgets>=8.1.0',
        'together>=1.4.6',
        'openai>=1.66.5',
        'dill>=0.3.8',
    ],
    cmdclass = {
        'update_dependencies': UpdateDependenciesCommand,
    }
)
