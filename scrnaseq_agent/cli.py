# scrnaseq_agent/cli.py

import argparse
import logging
import os
import sys
from pathlib import Path
import scanpy as sc
import yaml

# Import the Workflow class
from .agent import ScrnaSeqWorkflow # Requires agent.py to be in the same directory or package

# Setup logging
log = logging.getLogger("scrnaseq_agent.cli")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# --- Argument Parser Setup ---
def create_parser():
    parser = argparse.ArgumentParser(
        description="Run a standard scRNA-seq analysis pipeline (using PCA).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Input/Output Arguments ---
    parser.add_argument("-i", "--input-path", type=str, required=True, help="Path to input data (10x directory or .h5ad file).")
    parser.add_argument("-o", "--output-dir", type=str, required=True, help="Directory to save results (AnnData object and plots).")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to a YAML configuration file with pipeline parameters.")
    parser.add_argument("--output-prefix", type=str, help="Prefix for output files. Overrides config.")

    # --- Step Arguments ---
    # QC
    parser.add_argument("--mito-prefix", type=str, help="Mitochondrial gene prefix.")
    parser.add_argument("--min-genes", type=int, help="Min genes per cell.")
    parser.add_argument("--max-genes", type=int, help="Max genes per cell.")
    parser.add_argument("--min-counts", type=int, help="Min counts per cell.")
    parser.add_argument("--max-counts", type=int, help="Max counts per cell.")
    parser.add_argument("--max-pct-mito", type=float, help="Max mitochondrial percentage.")
    # Preprocessing
    parser.add_argument("--target-sum", type=float, help="Target sum for normalization.")
    parser.add_argument("--n-hvgs", type=int, help="Number of highly variable genes.")
    parser.add_argument("--hvg-flavor", type=str, choices=['seurat', 'cell_ranger', 'seurat_v3'], help="HVG selection flavor.")
    parser.add_argument("--scale-max-value", type=float, help="Max value for scaling.")
    # DimRed (PCA only for now)
    parser.add_argument("--n-pca-comps", type=int, help="Number of PCA components.")
    # Clustering
    parser.add_argument("--n-neighbors", type=int, help="Number of neighbors for graph.")
    parser.add_argument("--leiden-resolution", type=float, help="Leiden resolution.")
    parser.add_argument("--skip-umap", action='store_true', help="Skip UMAP calculation.")
    # DGE
    parser.add_argument("--dge-method", type=str, choices=['wilcoxon', 't-test', 'logreg'], help="DGE method.")
    parser.add_argument("--dge-corr-method", type=str, choices=['benjamini-hochberg', 'bonferroni'], help="DGE correction.")
    parser.add_argument("--dge-use-raw", action=argparse.BooleanOptionalAction, help="Use raw counts for DGE.")
    parser.add_argument("--dge-key", type=str, help="adata.uns key for DGE results.")
    # Annotation
    parser.add_argument(
        "--annotation-tool", type=str, choices=['marker_overlap', 'celltypist', 'none'],
        help="Annotation tool to use ('marker_overlap', 'celltypist', or 'none')."
    )
    parser.add_argument(
        "--marker-file", type=str,
        help="Path to JSON marker file (required if annotation-tool='marker_overlap')."
    )
    parser.add_argument(
        "--annotation-key", type=str,
        help="Base key for storing annotation results in adata.obs/uns."
    )
    parser.add_argument(
        "--annotation-method", type=str, choices=['overlap_count', 'overlap_coef', 'jaccard'], # Renamed from overlap-method
        help="Method for marker gene overlap (if tool='marker_overlap')."
    )
    parser.add_argument(
        "--celltypist-model", type=str,
        help="Name or path to CellTypist model (if tool='celltypist'). E.g., 'Immune_All_Low.pkl'."
    )
    parser.add_argument(
        "--celltypist-majority-voting", action=argparse.BooleanOptionalAction,
        help="Perform majority voting within clusters for CellTypist."
    )
    # Plotting
    parser.add_argument("--plot-umap-color", type=str, help="Comma-separated features for UMAP color.")
    parser.add_argument("--dge-n-genes", type=int, help="Number of genes in DGE plots.")
    parser.add_argument("--plot-dpi", type=int, help="DPI for plots.")
    parser.add_argument("--plot-format", type=str, choices=['png', 'pdf', 'svg'], help="Plot file format.")
    parser.add_argument("--run-qc-violin", action=argparse.BooleanOptionalAction, help="Generate QC violin plots.")
    parser.add_argument("--qc-violin-keys", type=str, help="Comma-separated obs keys for QC violin plot.")
    parser.add_argument("--run-dge-plots", action=argparse.BooleanOptionalAction, help="Generate all standard DGE plots.")
    parser.add_argument("--run-dge-heatmap", action=argparse.BooleanOptionalAction, help="Generate DGE heatmap plot specifically.")
    # Other
    parser.add_argument("--random-seed", type=int, help="Random seed for reproducibility.")

    return parser

# --- Parameter Loading and Precedence ---
def load_and_merge_params(args: argparse.Namespace) -> argparse.Namespace:
    """Loads config file and merges parameters with CLI args and defaults."""
    defaults = {
        'output_prefix': "scrnaseq_result",
        'mito_prefix': "MT-", 'min_genes': 200, 'max_genes': None, 'min_counts': None,
        'max_counts': None, 'max_pct_mito': 10.0, 'target_sum': 10000.0, 'n_hvgs': 2000,
        'hvg_flavor': "seurat_v3", 'scale_max_value': 10.0,
        'n_pca_comps': 50,
        'n_neighbors': 15, 'leiden_resolution': 1.0, 'skip_umap': False,
        'dge_method': "wilcoxon", 'dge_corr_method': "benjamini-hochberg", 'dge_use_raw': True,
        'dge_key': "rank_genes_groups",
        # Annotation Defaults
        'annotation_tool': 'none',
        'marker_file': None,
        'annotation_key': "cell_type_annotation",
        'annotation_method': "overlap_count", # Renamed from overlap_method
        'celltypist_model': "Immune_All_Low.pkl",
        'celltypist_majority_voting': False,
        # Plotting Defaults
        'plot_umap_color': "leiden,n_genes_by_counts,pct_counts_mt",
        'run_qc_violin': True,
        'qc_violin_keys': "n_genes_by_counts,total_counts,pct_counts_mt",
        'run_dge_plots': True,
        'dge_n_genes': 5,
        'run_dge_heatmap': True,
        'plot_dpi': 150,
        'plot_format': "png",
        'random_seed': 0
    }

    config_params = {}
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_file(): log.error(f"Config file not found: {config_path}"); sys.exit(1)
        try:
            with open(config_path, 'r') as f:
                config_yaml = yaml.safe_load(f)
                if config_yaml:
                    for section, params_in_section in config_yaml.items():
                        if isinstance(params_in_section, dict):
                             list_params = ['umap_color', 'qc_violin_keys']
                             for lp in list_params:
                                 if section == 'plotting' and lp in params_in_section and isinstance(params_in_section[lp], list):
                                      config_params[f'plot_{lp}'] = params_in_section.pop(lp)
                             config_params.update(params_in_section)
                        else: config_params[section] = params_in_section
            log.info(f"Loaded parameters from config file: {args.config}")
        except yaml.YAMLError as e: log.error(f"Error parsing config file {args.config}: {e}"); sys.exit(1)
        except Exception as e: log.error(f"Error reading config file {args.config}: {e}", exc_info=True); sys.exit(1)

    final_params = argparse.Namespace()
    cli_args_dict = vars(args)

    bool_action_keys = [
        'skip_umap', 'dge_use_raw', 'run_qc_violin', 'run_dge_plots',
        'run_dge_heatmap', 'celltypist_majority_voting'
    ]

    for key, default_value in defaults.items():
        cli_value = cli_args_dict.get(key)
        param_value = default_value

        config_value = config_params.get(key)
        if config_value is not None:
             param_value = None if str(config_value).lower() == 'null' else config_value

        if cli_value is not None:
             if key in bool_action_keys:
                 param_value = cli_value
             elif key == 'skip_umap': # store_true flag
                 param_value = cli_value
             else:
                 param_value = cli_value

        if key in ['plot_umap_color', 'qc_violin_keys'] and isinstance(param_value, str):
             param_value = [f.strip() for f in param_value.split(',') if f.strip()]
        elif key in ['max_genes', 'min_counts', 'max_counts', 'marker_file'] and param_value == '':
             param_value = None

        setattr(final_params, key, param_value)

    # Ensure required args are present
    final_params.input_path = args.input_path
    final_params.output_dir = args.output_dir
    # Set dimred_method implicitly for the agent (only PCA supported currently)
    final_params.dimred_method = 'pca'

    log.debug(f"Final parameters after merge: {vars(final_params)}")
    return final_params


# --- Main Pipeline Function (Simplified Version) ---
def run_pipeline(params): # Takes the merged parameters
    """Initializes and runs the ScrnaSeqWorkflow."""
    try:
        # Basic validation before starting agent
        if params.annotation_tool == 'marker_overlap' and not params.marker_file:
             log.error("Annotation tool is 'marker_overlap' but --marker-file was not provided.")
             # sys.exit(1) # Or let agent handle skipping
        if params.annotation_tool == 'celltypist' and not params.celltypist_model:
             log.error("Annotation tool is 'celltypist' but --celltypist-model was not provided.")
             sys.exit(1)

        workflow_agent = ScrnaSeqWorkflow(params)
        final_adata = workflow_agent.run()
        log.info(f"Workflow finished. Final AnnData object available.")

    except Exception as e:
         log.critical(f"Pipeline execution failed. See previous logs for details.")
         sys.exit(1) # Exit with error code

# --- Entry Point ---
def main():
    parser = create_parser()
    args = parser.parse_args() # Initial parse
    final_params = load_and_merge_params(args) # Load config and merge
    run_pipeline(final_params) # Run with final params

if __name__ == "__main__":
    main()