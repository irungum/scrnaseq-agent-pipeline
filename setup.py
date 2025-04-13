# setup.py

from setuptools import setup, find_packages

# Basic versioning - can be made more sophisticated later
VERSION = "0.1.0"
DESCRIPTION = "ScrnaSeq Agent: A pipeline orchestrator for scRNA-seq analysis."
# Attempt to read the long description from README.md
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        LONG_DESCRIPTION = fh.read()
except FileNotFoundError:
    LONG_DESCRIPTION = DESCRIPTION

setup(
    name="scrnaseq_agent",
    version=VERSION,
    author="MICHAEL IRUNGU", # Replace with your name/info
    author_email="<your_email@example.com>", # Replace with your email
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION, # Use content from README
    long_description_content_type="text/markdown", # Specify markdown format
    url="https://github.com/irungum/scrnaseq-agent-pipeline", # Replace with your actual repo URL
    packages=find_packages(exclude=["tests*", "docs*", "examples*", "new_data*", "output_*", "pipeline_*"]), # Exclude more patterns
    install_requires=[
        "scanpy>=1.9",
        "anndata>=0.8",
        "pandas>=1.5",
        "numpy>=1.21",
        "matplotlib>=3.5",
        "PyYAML>=6.0",
        "leidenalg>=0.9",        # Comma was needed here
        "igraph>=0.10",          # Comma was needed here
        "scikit-misc>=0.1.4",    # Comma was needed here
        "celltypist>=1.2"        # Added dependency, no comma after last item
        # Add scvi-tools >= 1.0 if/when re-integrated
    ],
    python_requires=">=3.9", # Specify minimum Python version
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License", # Choose/Confirm License
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    # --- Add Entry Point ---
    entry_points={
        'console_scripts': [
            'scrnaseq-agent=scrnaseq_agent.cli:main',
        ],
    }
    # --- End Entry Point ---
)