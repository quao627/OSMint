import setuptools
import os

# The directory containing this file
HERE = os.path.abspath(os.path.dirname(__file__))

# The text of the README file
with open(os.path.join(HERE, "README.md")) as fid:
    README = fid.read()

setuptools.setup(
    name="OSMint",
    version="0.0.1",
    author="Ao Qu & Anirudh Valiveru",
    author_email="qua@mit.edu",
    description="A python package for extracting signalized intersections from OpenStreetMap",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/quao627/OSMint",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=["OSMint"],
    include_package_data=True,
    install_requires=[],
)
