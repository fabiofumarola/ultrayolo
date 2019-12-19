#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open('requirements.txt', 'r') as fh:
    requirements = fh.read().split('\n')

with open('requirements_dev.txt', 'r') as fh:
    test_requirements = fh.read().split('\n')

setup_requirements = ['pytest-runner', ]

setup(
    author="Fabio Fumarola",
    author_email='fabiofumarola@gmail.com',
    python_requires='>=3',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    name='tfyolo3',
    keywords=['tfyolo3'],
    version='1.0.0',
    license="Apache Software License 2.0",
    description="tensorflow 2 implemetation of yolo 3 with backbone extensions",
    long_description=readme + '\n\n' + history,
    entry_points={
        'console_scripts': [
            'tfyolo3=tfyolo3.cli:main',
        ],
    },
    # scripts
    scripts=[],
    install_requires=requirements,
    include_package_data=True,
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt', '*.rst', '*.json'],
    },
    packages=find_packages(include=['tfyolo3', 'tfyolo3.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/fabiofumarola/tfyolo3',
    zip_safe=False,
)
