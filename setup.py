#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
# import nltk

from setuptools import setup
from setuptools.command.install import install
from setuptools.command.test import test as TestCommand


nltk_packages = [
    "punkt",
    "stopwords",
]

requirements = [
    "nltk==3.0.2",
    "numpy==1.9.2",
]

test_requirements = [
    'pytest==2.7.0',
]


def nltk_downloader(packages):
    """Download the given NLTK packages."""
    import nltk

    for package in packages:
        nltk.download(package)


class CustomInstallCommand(install):

    """
    Customized setuptools install command - install required NLTK data.
    """

    def run(self):
        install.run(self)
        nltk_downloader(nltk_packages)


class DataMiningToolsTests(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


setup(
    name='datamining',
    author='Wael BEN ZID',
    packages=[
        'datamining',
    ],
    include_package_data=True,
    install_requires=requirements,
    license="MIT",
    zip_safe=False,
    test_suite='tests',
    tests_require=test_requirements,
    cmdclass={
        'install': CustomInstallCommand,
        'develop': CustomInstallCommand,
        'test': DataMiningToolsTests
    },
)
