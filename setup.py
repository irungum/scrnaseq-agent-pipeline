# setup.py
import setuptools
import os

# --- Get the long description from the README file ---
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# --- Get the requirements from the requirements.txt file ---
# (You can also list them manually in install_requires, but reading from
# requirements.txt is common, especially if it mirrors environment.yml)
# For simplicity now, let's list key ones manually. Refine later if needed.
# If you heavily rely on environment.yml, you might skip install_requires
# or only put absolutely essential Python-only packages here.
# requirements = []
# with open('requirements.txt') as f:
#     requirements = f.read().splitlines()

setuptools.setup(
    name="scrnaseq-agent",  # How users install it (pip install scrnaseq-agent)
    version="0.1.0",       # Start with an initial version
    author="Irungu", # Replace with your name
    author_email="[wahomeirungu.m@gmail.com]", # Replace with your email
    description="An agentic framework for scRNA-seq analysis", # Short description
    long_description=long_description, # Long description read from README
    long_description_content_type="text/markdown", # Type of the long description
    url="[URL to your Git repository, if public]", # e.g., GitHub URL
    # --- Specify where the source code lives ---
    # find_packages() automatically finds all packages under the src directory (or root)
    packages=setuptools.find_packages(where=".", include=['scrnaseq_agent', 'scrnaseq_agent.*']),
    # package_dir={'': 'src'}, # Use this if your code was in a src/ directory
    # --- Specify dependencies ---
    install_requires=[
        "scanpy>=1.9",
        "anndata>=0.8",
        "numpy",
        "pandas",
        "matplotlib", # Keep it general, let user handle backends
        "seaborn",
        "pyyaml",
        "click", # Or argparse
        "python-igraph", # Often needed by scanpy
        "leidenalg", # Often needed by scanpy
        # Add boto3 if you plan S3 support within the 4 months
        # "boto3",
    ],
    # --- Specify Python version compatibility ---
    python_requires='>=3.9', # Be consistent with environment.yml
    # --- Entry points for command-line scripts ---
    # This tells pip to create executable scripts in the user's PATH
    # that call functions in your code.
    entry_points={
        'console_scripts': [
            'scrnaseq-agent=scrnaseq_agent.cli:main', # command = module:function
        ],
    },
    # --- Classifiers for PyPI ---
    # Provides metadata for PyPI search/filtering. Choose appropriate ones.
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License", # Or choose another license
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    # --- Include non-code files specified in MANIFEST.in ---
    # We might need this later for config files or example data
    # include_package_data=True,
    # Or specify manually:
    # package_data={
    #     'scrnaseq_agent': ['config/*.yaml'],
    # },
)

print("Finished setup.py") # Simple confirmation message