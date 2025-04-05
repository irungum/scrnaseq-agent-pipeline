# scrnaseq_agent/cli.py

import argparse
import logging
import os
import sys
from pathlib import Path
import scanpy as sc
import yaml

# Import our pipeline functions
from scrnaseq_agent.data.loader import load_data
from scrnaseq_agent.analysis.qc import calculate_qc_metrics, filter_cells_qc
from scrnaseq_agent.analysis.preprocess import normalize_log1p, select_hvg
from scrnaseq_agent.analysis.dimred import reduce_dimensionality
from scrnaseq_agent.analysis.clustering import perform_clustering
from scrnaseq_agent.analysis.dge import find_marker_genes
from scrnaseq_agent.analysis.annotation import annotate_cell_types, load_marker_dict
from scrnaseq_agent.visualization.plotting import (
    plot_umap,
    plot_qc_violin,
    plot_rank_genes_groups_dotplot,
    plot_rank_genes_groups_stacked_violin,
    plot_rank_genes_groups_heatmap
)

# Setup logging
log = logging.getLogger("scrnaseq_agent.cli")
# !!! FIX IS HERE: Replace ellipsis with actual config !!!
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
# !!! END FIX !!!

# --- Argument Parser Setup ---
def create_parser():
    parser = argparse.ArgumentParser(
        description="Run a standard scRNA-seq analysis pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Input/Output Arguments ---
    parser.add_argument("-i", "--input-path", type=str, required=True, help="Path to input data (10x directory or .h5ad file).")
    parser.add_argument("-o", "--output-dir", type=str, required=True, help="Directory to save results (AnnData object and plots).")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to a YAML configuration file with pipeline parameters.")
    parser.add_argument("--output-prefix", type=str, help="Prefix for output files (AnnData and plots). Overrides config file if set.")

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
    # DimRed
    parser.add_argument("--n-pca-comps", type=int, help="Number of PCA components.")
    # Clustering
    parser.add_argument("--n-neighbors", type=int, help="Number of neighbors for graph.")
    parser.add_argument("--leiden-resolution", type=float, help="Leiden resolution.")
    parser.add_argument("--skip-umap", action='store_true', help="Skip UMAP calculation.")
    # DGE
    parser.add_argument("--dge-method", type=str, choices=['wilcoxon', 't-test', 'logreg'], help="DGE method.")
    parser.add_argument("--dge-corr-method", type=str, choices=['benjamini-hochberg', 'bonferroni'], help="DGE correction method.")
    parser.add_argument("--dge-use-raw", action=argparse.BooleanOptionalAction, help="Use raw counts for DGE.")
    parser.add_argument("--dge-key", type=str, help="adata.uns key for DGE results.")
    # Annotation
    parser.add_argument("--marker-file", type=str, help="Path to JSON file containing marker genes dictionary for annotation.")
    parser.add_argument("--annotation-key", type=str, help="Base key for storing annotation results in adata.obs.")
    parser.add_argument("--annotation-method", type=str, choices=['overlap_count', 'overlap_coef', 'jaccard'], help="Method for marker gene overlap calculation.")
    # Plotting
    parser.add_argument("--plot-umap-color", type=str, help="Comma-separated features for UMAP color.")
    parser.add_argument("--dge-n-genes", type=int, help="Number of genes in DGE plots.") # Renamed from plot_dge_n_genes
    parser.add_argument("--plot-dpi", type=int, help="DPI for plots.")
    parser.add_argument("--plot-format", type=str, choices=['png', 'pdf', 'svg'], help="Plot file format.")
    parser.add_argument("--run-qc-violin", action=argparse.BooleanOptionalAction, help="Generate QC violin plots.")
    parser.add_argument("--qc-violin-keys", type=str, help="Comma-separated obs keys for QC violin plot.")
    parser.add_argument("--run-dge-plots", action=argparse.BooleanOptionalAction, help="Generate all standard DGE plots (dot, violin, heatmap).")
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
        'hvg_flavor': "seurat_v3", 'scale_max_value': 10.0, 'n_pca_comps': 50,
        'n_neighbors': 15, 'leiden_resolution': 1.0, 'skip_umap': False,
        'dge_method': "wilcoxon", 'dge_corr_method': "benjamini-hochberg", 'dge_use_raw': True,
        'dge_key': "rank_genes_groups",
        'marker_file': None,
        'annotation_key': "cell_type_annotation",
        'annotation_method': "overlap_count",
        'plot_umap_color': "leiden,n_genes_by_counts,pct_counts_mt",
        'run_qc_violin': True,
        'qc_violin_keys': "n_genes_by_counts,total_counts,pct_counts_mt",
        'run_dge_plots': True,
        'dge_n_genes': 5, # Renamed from plot_dge_n_genes
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
                    for section, params in config_yaml.items():
                        if isinstance(params, dict):
                             if section == 'plotting' and 'umap_color' in params and isinstance(params['umap_color'], list):
                                  config_params['plot_umap_color'] = params.pop('umap_color')
                             if section == 'plotting' and 'qc_violin_keys' in params and isinstance(params['qc_violin_keys'], list):
                                  config_params['qc_violin_keys'] = params.pop('qc_violin_keys')
                             config_params.update(params)
                        else: config_params[section] = params
            log.info(f"Loaded parameters from config file: {args.config}")
        except yaml.YAMLError as e: log.error(f"Error parsing config file {args.config}: {e}"); sys.exit(1)

    final_params = argparse.Namespace()
    cli_args_dict = vars(args)

    for key, default_value in defaults.items():
        cli_value = cli_args_dict.get(key)
        param_value = default_value

        config_value = config_params.get(key)
        if config_value is not None:
             param_value = config_value

        if cli_value is not None:
             # Handle boolean action flags carefully
             if key in ['skip_umap'] and not cli_value: pass
             elif key in ['run_qc_violin', 'run_dge_plots', 'run_dge_heatmap', 'dge_use_raw']: param_value = cli_value
             else: param_value = cli_value

        if key in ['plot_umap_color', 'qc_violin_keys'] and isinstance(param_value, str):
             param_value = [f.strip() for f in param_value.split(',') if f.strip()]

        setattr(final_params, key, param_value)

    final_params.input_path = args.input_path
    final_params.output_dir = args.output_dir

    log.debug(f"Final parameters after merge: {vars(final_params)}")
    return final_params


# --- Main Pipeline Function ---
def run_pipeline(args):
    """Orchestrates the scRNA-seq pipeline based on final parameters."""
    log.info("Starting scRNA-seq pipeline run...")
    log.info(f"Input path: {args.input_path}")
    log.info(f"Output directory: {args.output_dir}")
    log.info(f"Output prefix: {args.output_prefix}")
    log.info(f"Random seed: {args.random_seed}")
    log.info(f"QC Params: mito_prefix={args.mito_prefix}, min_genes={args.min_genes}, max_pct_mito={args.max_pct_mito}")
    log.info(f"Preprocessing Params: n_hvgs={args.n_hvgs}, flavor={args.hvg_flavor}")
    log.info(f"Clustering Params: n_neighbors={args.n_neighbors}, resolution={args.leiden_resolution}, skip_umap={args.skip_umap}")
    log.info(f"DGE Params: method={args.dge_method}, use_raw={args.dge_use_raw}, key={args.dge_key}")
    log.info(f"Annotation Params: marker_file={args.marker_file}, key={args.annotation_key}, method={args.annotation_method}")
    log.info(f"Plotting Params: umap_color={args.plot_umap_color}, run_qc_violin={args.run_qc_violin}, qc_violin_keys={args.qc_violin_keys}, run_dge_plots={args.run_dge_plots}, dge_n_genes={args.dge_n_genes}, run_dge_heatmap={args.run_dge_heatmap}, format={args.plot_format}")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log.info(f"Ensured output directory exists: {output_path}")

    sc.settings.figdir = str(output_path)
    sc.settings.verbosity = 3

    adata = None; adata_raw_filtered = None; adata_pre_filter = None

    try:
        # --- Pipeline Steps ---
        log.info("Step 1: Loading data...")
        adata = load_data(args.input_path)
        adata_pre_filter = adata.copy()
        log.info(f"Loaded data shape: {adata.shape}")

        log.info("Step 2: Calculating QC metrics...")
        calculate_qc_metrics(adata, mito_gene_prefix=args.mito_prefix, inplace=True)
        calculate_qc_metrics(adata_pre_filter, mito_gene_prefix=args.mito_prefix, inplace=True)
        log.info("QC calculation complete.")

        if args.run_qc_violin and args.qc_violin_keys:
             log.info("Plotting QC violin plots (pre-filtering)...")
             plot_qc_violin(
                 adata_pre_filter, keys=args.qc_violin_keys, output_dir=str(output_path),
                 file_prefix=f"{args.output_prefix}_qc_violin_prefilt",
                 file_format=args.plot_format, dpi=args.plot_dpi
             )

        log.info("Step 3: Filtering cells...")
        n_obs_before = adata.n_obs
        filter_cells_qc(
            adata, min_genes=args.min_genes, max_genes=args.max_genes,
            min_counts=args.min_counts, max_counts=args.max_counts,
            max_pct_mito=args.max_pct_mito, inplace=True
        )
        if adata.n_obs == 0: log.error("All cells filtered out!"); sys.exit(1)
        log.info(f"Filtering complete. Kept {adata.n_obs} / {n_obs_before} cells.")
        del adata_pre_filter

        if args.run_qc_violin and args.qc_violin_keys:
             log.info("Plotting QC violin plots (post-filtering)...")
             plot_qc_violin(
                 adata, keys=args.qc_violin_keys, output_dir=str(output_path),
                 file_prefix=f"{args.output_prefix}_qc_violin_postfilt",
                 file_format=args.plot_format, dpi=args.plot_dpi
             )

        log.info("Step 4: Storing filtered data state for .raw attribute...")
        adata_raw_placeholder = load_data(args.input_path)
        adata_raw_filtered = adata_raw_placeholder[adata.obs_names, :].copy()
        del adata_raw_placeholder
        log.info(f"Stored raw data for {adata_raw_filtered.n_obs} cells.")

        log.info("Step 5: Normalizing and log-transforming..."); normalize_log1p(adata, target_sum=args.target_sum, inplace=True)
        log.info("Step 6: Selecting highly variable genes..."); n_vars_before = adata.n_vars; select_hvg(adata, n_top_genes=args.n_hvgs, flavor=args.hvg_flavor, subset=True, inplace=True); log.info(f"Kept {adata.n_vars} / {n_vars_before} HVGs.")
        log.info("Step 7: Finalizing .raw attribute..."); adata.raw = adata_raw_filtered[:, adata.var_names].copy(); log.info(f"Set final .raw. Shape: {adata.raw.shape}"); del adata_raw_filtered
        log.info("Step 8: Scaling data..."); sc.pp.scale(adata, max_value=args.scale_max_value)
        log.info("Step 9: Performing PCA..."); reduce_dimensionality(adata, n_comps=args.n_pca_comps, random_state=args.random_seed, inplace=True)
        log.info("Step 10: Performing Neighbors, Clustering, and UMAP..."); cluster_key = 'leiden'; perform_clustering(adata, n_neighbors=args.n_neighbors, resolution=args.leiden_resolution, random_state=args.random_seed, leiden_key_added=cluster_key, calculate_umap=(not args.skip_umap), inplace=True)
        log.info("Step 11: Finding marker genes..."); find_marker_genes(adata, groupby=cluster_key, method=args.dge_method, corr_method=args.dge_corr_method, use_raw=args.dge_use_raw, key_added=args.dge_key)

        log.info("Step 12: Annotating cell types (optional)...")
        annotation_result_key = None
        if args.marker_file:
            marker_dict = load_marker_dict(args.marker_file)
            if marker_dict:
                annotate_cell_types(
                    adata, marker_dict=marker_dict, groupby=cluster_key,
                    rank_key=args.dge_key, annotation_key=args.annotation_key,
                    method=args.annotation_method
                )
                annotation_result_key = f"{args.annotation_key}_{args.annotation_method}"
                log.info(f"Annotation complete. Results in obs key like '{annotation_result_key}'.")
            else: log.warning("Failed to load markers. Skipping annotation.")
        else: log.info("No marker file provided (--marker-file). Skipping annotation step.")

        log.info("Step 13: Generating plots...")
        plot_features = args.plot_umap_color
        if annotation_result_key and annotation_result_key in adata.obs and annotation_result_key not in plot_features:
             log.info(f"Adding annotation '{annotation_result_key}' to UMAP plots.")
             plot_features.append(annotation_result_key)

        if not args.skip_umap and plot_features:
            log.info(f"Plotting UMAP colored by: {plot_features}")
            plot_umap(
                adata, color_by=plot_features, output_dir=str(output_path),
                file_prefix=f"{args.output_prefix}_umap",
                file_format=args.plot_format, dpi=args.plot_dpi
            )
        elif args.skip_umap: log.info("Skipping UMAP plots as UMAP calculation was skipped.")

        if args.run_dge_plots and args.dge_key in adata.uns:
            log.info("Plotting DGE results...")
            plot_rank_genes_groups_dotplot(
                adata, key=args.dge_key, n_genes=args.dge_n_genes,
                groupby=cluster_key, output_dir=str(output_path),
                file_prefix=f"{args.output_prefix}_dge_dotplot",
                file_format=args.plot_format, dpi=args.plot_dpi,
                use_raw=args.dge_use_raw
            )
            plot_rank_genes_groups_stacked_violin(
                adata, key=args.dge_key, n_genes=args.dge_n_genes,
                groupby=cluster_key, output_dir=str(output_path),
                file_prefix=f"{args.output_prefix}_dge_violin",
                file_format=args.plot_format, dpi=args.plot_dpi,
                use_raw=args.dge_use_raw
            )
            if args.run_dge_heatmap:
                plot_rank_genes_groups_heatmap(
                    adata, key=args.dge_key, n_genes=args.dge_n_genes,
                    groupby=cluster_key, output_dir=str(output_path),
                    file_prefix=f"{args.output_prefix}_dge_heatmap",
                    file_format=args.plot_format, dpi=args.plot_dpi,
                    use_raw=args.dge_use_raw, show_gene_labels=True
                )
        elif not args.run_dge_plots: log.info("Skipping DGE plots as requested.")
        else: log.warning(f"DGE results key '{args.dge_key}' not found. Skipping DGE plots.")
        log.info("Plot generation complete.")

        log.info("Step 14: Saving final AnnData object...")
        final_adata_path = output_path / f"{args.output_prefix}_final.h5ad"
        adata.write_h5ad(final_adata_path, compression="gzip")
        log.info(f"Final AnnData object saved to: {final_adata_path}")

    except FileNotFoundError as e: log.error(f"Input/Config file not found: {e}", exc_info=True); sys.exit(1)
    except KeyError as e: log.error(f"Missing expected data key: {e}.", exc_info=True); sys.exit(1)
    except ValueError as e: log.error(f"Value error during processing: {e}.", exc_info=True); sys.exit(1)
    except Exception as e: log.error(f"An unexpected error occurred: {e}", exc_info=True); sys.exit(1)

    log.info("Pipeline completed successfully!")


# --- Entry Point ---
def main():
    parser = create_parser()
    args = parser.parse_args()
    final_params = load_and_merge_params(args)
    run_pipeline(final_params)

if __name__ == "__main__":
    main()