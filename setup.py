"""
Setup script for FTIR Fire Emissions Analysis package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().strip().split('\n')

setup(
    name="pyrospectra",
    version="1.0.0",
    author="Luke Richardson-Foulger, José Gómez-Dans",
    author_email="",  # Add email if desired
    description="Analysis toolkit for biomass burning emissions using closed-path FTIR spectroscopy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",  # Add repository URL if available
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",  # Adjust as needed
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    keywords="FTIR spectroscopy fire emissions biomass burning atmospheric chemistry",
    project_urls={
        "Documentation": "",  # Add if available
        "Source": "",  # Add repository URL
        "Bug Reports": "",  # Add issue tracker URL
    },
)
