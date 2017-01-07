import sys
import os
from setuptools import setup
from setuptools.command.test import test as TestCommand
from setuptools.command.install import install
reqs = open('./requirements.txt', 'r').read().split('\n')


class OverrideInstall(install):

    """
    Emulate sequential install of pip install -r requirements.txt
    To fix numpy bug in scipy, scikit in py2
    """

    def run(self):
        for req in reqs:
            if req:
                pip.main(["install", "-U", req])


# explicitly config
test_args = [
    '--cov-report=term',
    '--cov-report=html',
    '--cov=rl',
    'test'
]


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
    license='MIT',
    packages=[],
    install_requires=[],
    dependency_links=[],
    tests_require=['pytest'],
    test_suite='test',
    cmdclass={'test': PyTest, 'install': OverrideInstall}
)