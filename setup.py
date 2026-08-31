"""Setup script for PyroSpectra."""
from pathlib import Path
from setuptools import setup, find_packages

here = Path(__file__).parent
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="pyrospectra",
    version="2.0.0",
    author="Luke Richardson-Foulger, Martin Wooster, Jose Gomez-Dans, Mark Grosvenor",
    description=("Biomass burning emission factors from closed-path FTIR "
                 "time series spectra"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NAEO-KCL/PyroSpectra",
    packages=find_packages(),
    package_dir={"pyrospectra": "."},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0", "scipy>=1.7.0", "matplotlib>=3.4.0",
        "scikit-learn>=0.24.0", "pandas>=1.3.0", "joblib>=1.0.0", "tqdm>=4.62.0",
    ],
    extras_require={"lbl": ["radis>=0.12.0"], "test": ["pytest>=7.0"]},
    keywords="FTIR spectroscopy fire emissions biomass burning atmospheric chemistry",
)
