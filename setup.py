import os
import re
from setuptools import find_packages, setup


def get_version():
    """Reads version from dcs/__init__.py"""
    version_file = os.path.join(os.path.dirname(__file__), "dcs", "__init__.py")
    with open(version_file, "r") as f:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# Read the long description from README.md
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="dcs",
    version=get_version(),
    description="A package for detecting causality structures in time series",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Salar Nouri",
    author_email="salr.nouri@gmail.com",
    url="https://github.com/CMC-lab/dcs",
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    install_requires=["numpy>=1.26", "scipy>=1.14", "matplotlib>=3.9"],
    extras_require={
        "dev": ["pytest", "flake8", "black"],
    },
    package_data={
        "dcs": ["py.typed"],  # include py.typed in builds
    },
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.12",
)
