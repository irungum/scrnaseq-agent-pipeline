

# scrnaseq-agent-pipeline
=======
# scrnaseq_agent
=======
# scRNA-seq Agent Pipeline

## Overview

This project provides a modular, configurable, and tested command-line tool for running standard single-cell RNA-seq (scRNA-seq) analysis workflows using [Scanpy](https://scanpy.readthedocs.io/). It is designed to simplify the execution of common analysis steps for researchers, acting as an orchestrator for the underlying functions.

The current version implements the following core workflow:
1.  **Data Loading:** Handles 10x Genomics MTX directories and AnnData `.h5ad` files.
2.  **Quality Control:** Calculates standard QC metrics (gene counts, total counts, mitochondrial percentage) and filters cells based on user-defined thresholds. Generates QC violin plots (pre- and post-filtering).
3.  **Preprocessing:** Performs library size normalization, log1p transformation, identifies highly variable genes (HVGs) using the 'seurat_v3' method, subsets the data to HVGs, and scales the data.
4.  **Dimensionality Reduction:** Calculates Principal Component Analysis (PCA).
5.  **Clustering & Visualization Embedding:** Computes a nearest neighbor graph, performs Leiden clustering, and calculates a UMAP embedding.
6.  **Marker Gene Identification:** Finds differentially expressed genes (marker genes) for each cluster using the Wilcoxon rank-sum test by default. Can operate on raw or processed counts.
7.  **Cell Type Annotation (Optional):**
    *   **Marker Overlap:** Annotates clusters based on overlap with a user-provided JSON marker gene file (`sc.tl.marker_gene_overlap`).
    *   **CellTypist:** Performs automated cell type prediction using pre-trained models (`celltypist.annotate`).
8.  **Visualization:** Generates UMAP plots colored by cluster, QC metrics, or specified genes/annotations, plus DGE dot plots, stacked violin plots, and heatmaps.
9.  **Output:** Saves the final processed AnnData object containing all results and generated plots to a specified directory.

The pipeline is orchestrated by the `ScrnaSeqWorkflow` class (`scrnaseq_agent/agent.py`) and driven by a command-line interface (`scrnaseq_agent/cli.py`) configurable via arguments and a YAML file.

## Installation

**Prerequisites:**
*   Python >= 3.9
*   [Conda](https://docs.conda.io/en/latest/miniconda.html) (Recommended for managing complex dependencies)

**Steps:**

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/irungum/scrnaseq-agent-pipeline.git
    cd scrnaseq-agent-pipeline
    ```

2.  **Create and Activate Conda Environment:**
    ```bash
    # Create the environment (this might take a few minutes)
    conda create --name scrnaseq_agent_env python=3.10 -y
    conda activate scrnaseq_agent_env

    # Install dependencies using pip within the conda environment
    # (Using pip here as scikit-misc might be easier via pip)
    pip install scanpy pandas matplotlib pyyaml leidenalg python-igraph scikit-misc celltypist
    ```
    *(Note: You can also create an `environment.yml` file listing these dependencies for easier recreation)*

3.  **Install the Package (Editable Mode):** Install the `scrnaseq-agent` package itself in editable mode (`-e`) from the cloned directory root. This links the installed package to your source code.
    ```bash
    pip install -e .
    ```
    This step also creates the `scrnaseq-agent` command-line tool within your active environment.

## Usage

The pipeline is run using the `scrnaseq-agent` command.

**Required Arguments:**

*   `-i PATH`, `--input-path PATH`: Path to your input data (e.g., `./data/my_10x_dir/` or `./data/my_data.h5ad`).
*   `-o PATH`, `--output-dir PATH`: Path to the directory where results will be saved (e.g., `./results/my_run`).

**Configuration:**

Parameters are controlled using a combination of a YAML config file and command-line overrides.

*   **Config File (`-c CONFIG`):** Specify a YAML file containing parameters for different steps. See `config/default_config.yaml` for the structure and default values.
*   **CLI Arguments:** Arguments provided directly on the command line (e.g., `--leiden-resolution 0.8`, `--n-hvgs 3000`) will **override** any corresponding values set in the config file or the internal defaults.
*   **Precedence:** Command-Line Arguments > Config File > Internal Defaults.

**Example Runs:**

1.  **Run using default parameters from `config/default_config.yaml`:**
    ```bash
    scrnaseq-agent \
        -i ./test_data/filtered_gene_bc_matrices/hg19/ \
        -o ./pipeline_output_default \
        -c ./config/default_config.yaml \
        --output-prefix pbmc_default_run
    ```

2.  **Run using config file but override QC and clustering:**
    ```bash
    scrnaseq-agent \
        -i ./new_data/HumanOralAtlas_Palate.h5ad \
        -o ./pipeline_output_oral_atlas \
        -c ./config/default_config.yaml \
        --output-prefix OralAtlasRun \
        --max-pct-mito 15 \
        --leiden-resolution 0.6 \
        --no-dge-use-raw # Important if input .h5ad lacks aligned .raw
    ```

3.  **Run using CellTypist annotation:**
    ```bash
    scrnaseq-agent \
        -i ./test_data/filtered_gene_bc_matrices/hg19/ \
        -o ./pipeline_output_celltypist \
        -c ./config/default_config.yaml \
        --output-prefix pbmc_celltypist_annot \
        --annotation-tool celltypist \
        --celltypist-model Immune_All_Low.pkl \
        --celltypist-majority-voting
    ```

4.  **Run using Marker Overlap annotation:**
    ```bash
    # First, ensure config/pbmc_markers.json exists (or create your own)
    # Example content for config/pbmc_markers.json:
    # {
    #   "B_Cells": ["MS4A1", "CD79A"],
    #   "T_Cells": ["CD3D", "CD3E", "IL7R"],
    #   "NK": ["NKG7", "GNLY"],
    #   "Monocytes": ["CD14", "LYZ", "FCGR3A"]
    # }
    scrnaseq-agent \
        -i ./test_data/filtered_gene_bc_matrices/hg19/ \
        -o ./pipeline_output_marker_overlap \
        -c ./config/default_config.yaml \
        --output-prefix pbmc_marker_overlap \
        --annotation-tool marker_overlap \
        --marker-file config/pbmc_markers.json
    ```

*   Use `scrnaseq-agent --help` to see all available options and their defaults.

## Configuration File (`config/default_config.yaml`)

This file defines default parameters for each step. Key sections include: `qc`, `preprocessing`, `dimred`, `clustering`, `dge`, `annotation`, `plotting`.

**Important Parameters to Check/Modify per Dataset:**
*   `qc -> mito_prefix`: Match organism (e.g., `MT-` or `mt-`).
*   `qc -> min_genes`, `max_pct_mito`, etc.: Adjust based on QC plots for your specific data.
*   `annotation -> marker_file`: Provide path to your JSON marker file if using `marker_overlap`.
*   `clustering -> leiden_resolution`: Experiment with different values to find appropriate granularity.

## Output Files

The pipeline generates the following in the specified output directory (`-o`):
*   **`<output_prefix>_final.h5ad`**: The final AnnData object containing normalized data, HVGs, scaling (in `.X`), PCA (`.obsm['X_pca']`), neighbors graph (`.uns['neighbors']`, `.obsp`), clustering (`.obs['leiden']`), UMAP (`.obsm['X_umap']`), DGE results (`.uns`), annotations (`.obs`), and raw counts (in `.raw` if available).
*   **QC Plots:** `*_qc_violin_prefilt.png`, `*_qc_violin_postfilt.png`.
*   **UMAP Plots:** `*_umap_<feature>.png` (or other format).
*   **DGE Plots:** `*_dge_dotplot.png`, `*_dge_violin.png`, `*_dge_heatmap.png` (if enabled).

## Testing

This project uses `pytest`.

1.  Make sure you are in the activated conda environment where you installed the package (`pip install -e .`).
2.  Navigate to the project root directory.
3.  Run the tests:
    ```bash
    pytest -v
    ```

## License


This project is licensed under the MIT License - see the LICENSE file for details. 


