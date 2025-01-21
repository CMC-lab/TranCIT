from setuptools import setup, find_packages

setup(
    name='dcs',
    version='0.1.0',
    description='A package for detecting causality structures',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author='Salar Nouri',
    author_email='salr.nouri@gmail.com',
    url="https://github.com/sa-nouri/dcs",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy"
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.12',
)
