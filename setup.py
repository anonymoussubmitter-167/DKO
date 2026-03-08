"""Setup script for DKO (Distribution Kernel Operators) research package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
with open("requirements.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("--"):
            requirements.append(line)

setup(
    name="dko",
    version="0.1.0",
    author="Anonymous",
    author_email="",
    description="Distribution Kernel Operators for Molecular Property Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anonymoussubmitter-167/DKO",
    packages=find_packages(exclude=["tests", "tests.*", "notebooks"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "jupyterlab>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dko-prepare=scripts.prepare_datasets:main",
            "dko-train=scripts.run_experiment:main",
            "dko-analyze=scripts.analyze_results:main",
        ],
    },
    include_package_data=True,
    package_data={
        "dko": ["configs/*.yaml", "configs/**/*.yaml"],
    },
)
