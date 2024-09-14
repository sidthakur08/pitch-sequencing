from setuptools import setup, find_packages

setup(
    name="pitch_sequencing",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src")
)