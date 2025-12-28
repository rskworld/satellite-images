"""
Satellite Image Dataset - Setup Script
Created by: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

Setup script for the satellite image dataset package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="satellite-image-dataset",
    version="1.0.0",
    author="RSK World",
    author_email="help@rskworld.in",
    description="Satellite imagery dataset with land cover classification and building detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://rskworld.in",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "rasterio>=1.2.0",
        "matplotlib>=3.4.0",
        "pillow>=8.3.0",
        "scikit-image>=0.18.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
        ],
    },
)

