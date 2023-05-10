#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

requirements = [
]

setup(
    author="Ryan Avery",
    author_email="ryan@developmentseed.org",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Cerulean",
    entry_points={
        "console_scripts": [
            "ceruleanml=pybayts.cli:main",
        ],
    },
    install_requires=requirements,
    license="MIT",
    # long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="pybayts",
    name="pybayts",
    packages=find_packages(include=["pybayts", "pybayts.*"]),
    test_suite="tests",
    url="https://github.com/developmentseed/pybayts",
    version="0.1.0",
    zip_safe=False,
)
