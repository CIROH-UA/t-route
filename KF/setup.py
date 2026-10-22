#!/usr/bin/env python

from setuptools import setup

setup(
    name="tx-fast-hydrology",
    version="0.1",
    description="Routing and data assimilation code for TxDOT FAST project.",
    long_description="Routing and data assimilation code for TxDOT FAST project.",
    long_description_content_type="text/x-rst",
    author="Matt Bartos",
    author_email="mdbartos@utexas.edu",
    url="https://future-water.org",
    packages=["tx_fast_hydrology"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Hydrology",
    ],
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "numba",
        "scipy",
    ],
    extras_require=dict(
        dev=["pytest", "pytest-cov"]
    ),
)
