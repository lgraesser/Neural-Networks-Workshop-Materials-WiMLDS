import sys
import os
from setuptools import setup
from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def run_tests(self):
        import shlex
        #import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)

# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

# the setup
setup(
    name='Neural-Networks-Workshop-Materials-WiMLDS',
    version='0.0.1',
    description='Materials for Neural Networks Workshop',
    long_description=read('README.md'),
    url='https://github.com/lgraesser/Neural-Networks-Workshop-Materials-WiMLDS.git',
    author='laura',
    author_email='lhgraesser@gmail.com',
    packages=[],
    install_requires=['h5py==2.6.0',
                      'numpy==1.11.1',
                      'scipy==0.18.0',
                      'matplotlib==1.5.2',
                      'theano',
                      'Keras>=1.1.0',
                      'git+https://github.com/lgraesser/Neural-Networks-Workshop-Materials-WiMLDS.git'],
    dependency_links=[],
    tests_require=['pytest'],
    cmdclass = {'test': PyTest},
)