"""Setup for the tradingapplication package."""

from setuptools import find_packages, setup

setup(
    name="tradingdmp",
    packages=find_packages("src/"),
    version="0.1.0",
    description="This repo defines core classes of our trading applications.",
    author="Julius Kittler, August Andersen",
    license="",
    python_requires=">=3.8"

)
