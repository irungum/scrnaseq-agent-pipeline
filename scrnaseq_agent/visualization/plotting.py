# scrnaseq_agent/visualization/plotting.py

import scanpy as sc
import anndata as ad
import logging
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path # Ensure Path is imported

log = logging.getLogger(__name__)
# Basic config only runs if the script is executed directly.
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    import sys
    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# --- Helper to handle Scanpy saving ---
def _save_scanpy_plot(plot_func, plot_type, output_path, *args, **kwargs):
    """Internal helper to call scanpy plot func and handle file saving/renaming."""
    save_suffix = kwargs.pop('save', None)
    if not save_suffix:
         out_dir, filename = os.path.split(output_path)
         save_suffix = f"_{filename}"

    show = kwargs.pop('show', False)

    try:
        plot_func(*args, save=save_suffix, show=show, **kwargs)

        scanpy_save_filename = f"{plot_type}{save_suffix}"
        scanpy_saved_path = os.path.join(sc.settings.figdir, scanpy_save_filename)
        fallback_filename = f"{plot_type}_{save_suffix}"
        fallback_path = os.path.join(sc.settings.figdir, fallback_filename)

        found_path = None
        if os.path.exists(scanpy_saved_path): found_path = scanpy_saved_path
        elif os.path.exists(fallback_path):
            log.warning(f"Scanpy saved plot using fallback name convention: {fallback_path}")
            found_path = fallback_path
        else:
             msg = f"Scanpy did not save the plot to expected paths: '{scanpy_saved_path}' or '{fallback_path}'"
             log.error(msg)
             raise FileNotFoundError(msg)

        os.rename(found_path, output_path)
        log.info(f"Saved {plot_type} plot to {output_path}")
        plt.close()

    except Exception as e:
        msg = f"Failed during Scanpy plot generation/saving for {plot_type}: {e}"
        log.error(msg, exc_info=True)
        plt.close()
        if isinstance(e, FileNotFoundError): raise e
        else: raise RuntimeError(msg) from e


# --- Plotting Functions ---

def plot_umap(
    adata: ad.AnnData,
    color_by: list[str],
    output_dir: str,
    file_prefix: str = "umap",
    umap_key: str = 'X_umap',
    file_format: str = "png",
    dpi: int = 150,
    **kwargs
) -> None:
    """Generates and saves UMAP plots colored by specified features."""
    if not isinstance(adata, ad.AnnData): raise TypeError("adata must be AnnData")
    # !!! FIX: Match error message raised here !!!
    if umap_key not in adata.obsm: raise KeyError(f"UMAP key '{umap_key}' not found")
    # !!! END FIX !!!
    if not isinstance(color_by, list) or not color_by: raise ValueError("color_by must be non-empty list")
    if not output_dir: raise ValueError("output_dir must be provided")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log.info(f"Generating UMAP plots colored by: {', '.join(color_by)}")
    errors_occurred = []

    for feature in color_by:
        if feature not in adata.obs.columns and feature not in adata.var_names:
             log.warning(f"Feature '{feature}' not found. Skipping UMAP plot.")
             errors_occurred.append(feature)
             continue

        safe_feature = feature.replace('/', '_').replace('\\', '_')
        filename = f"{file_prefix}_{safe_feature}.{file_format}"
        output_path = os.path.join(output_dir, filename)
        save_suffix = f"_{file_prefix}_{safe_feature}.{file_format}"

        try:
            _save_scanpy_plot(sc.pl.umap, "umap", output_path, adata, color=feature, save=save_suffix, **kwargs)
        except Exception as e:
            errors_occurred.append(f"{feature}: {e}")

    if errors_occurred:
        log.warning(f"Some errors occurred during UMAP plotting for features: {errors_occurred}")


def plot_qc_violin(
    adata: ad.AnnData,
    keys: list[str],
    output_dir: str,
    file_prefix: str = "qc_violin",
    groupby: str | None = None,
    file_format: str = "png",
    dpi: int = 150,
    **kwargs
) -> None:
    """Generates and saves violin plots for QC metrics."""
    if not isinstance(adata, ad.AnnData): raise TypeError("adata must be AnnData")
    if not isinstance(keys, list) or not keys: raise ValueError("keys must be non-empty list")
    if not output_dir: raise ValueError("output_dir must be provided")

    missing_keys = [k for k in keys if k not in adata.obs]
    if missing_keys:
        log.warning(f"QC keys not found in adata.obs: {missing_keys}. Skipping violin plots for these.")
        keys = [k for k in keys if k in adata.obs]
        if not keys: log.error("No valid QC keys found to plot."); return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log.info(f"Generating QC violin plots for: {', '.join(keys)}")

    filename = f"{file_prefix}.{file_format}"
    output_path = os.path.join(output_dir, filename)
    save_suffix = f"_{file_prefix}.{file_format}"

    try:
        # Rotation=90 is valid for sc.pl.violin itself
        _save_scanpy_plot(
            sc.pl.violin, "violin", output_path,
            adata, keys=keys, groupby=groupby, rotation=90, save=save_suffix, **kwargs
        )
    except Exception as e:
        log.error(f"Failed to generate QC violin plot: {e}", exc_info=True)


def _extract_top_marker_genes(adata: ad.AnnData, key: str, n_genes: int) -> list[str]:
    """Helper to extract unique top N marker genes."""
    try:
        marker_genes_structured = adata.uns[key]['names']
        top_genes_flat = [gene for group_genes in marker_genes_structured[:n_genes] for gene in group_genes if isinstance(gene, str)]
        var_names_unique = list(dict.fromkeys(top_genes_flat))
        if not var_names_unique: raise ValueError("No valid marker gene names extracted.")
        log.debug(f"Extracted top {len(var_names_unique)} unique marker names.")
        return var_names_unique
    except KeyError: raise KeyError(f"Structure 'names' not found within adata.uns['{key}'].")
    except Exception as e:
        log.error(f"Failed to extract top {n_genes} genes from uns['{key}']: {e}", exc_info=True)
        raise ValueError(f"Error extracting markers from key '{key}'.") from e


def plot_rank_genes_groups_dotplot(
    adata: ad.AnnData,
    key: str,
    n_genes: int = 5,
    groupby: str | None = None,
    output_dir: str = ".",
    file_prefix: str = "dge_dotplot",
    file_format: str = "png",
    dpi: int = 150,
    **kwargs
) -> None:
    """Generates and saves a dotplot of marker genes."""
    if not isinstance(adata, ad.AnnData): raise TypeError("adata must be AnnData")
    if key not in adata.uns: raise KeyError(f"DGE key '{key}' not found")
    if not output_dir: raise ValueError("output_dir must be provided")

    groupby_used = groupby or adata.uns[key].get('params', {}).get('groupby')
    if not groupby_used or groupby_used not in adata.obs:
        raise ValueError(f"Invalid groupby key '{groupby_used}'.")

    var_names_to_plot = _extract_top_marker_genes(adata, key, n_genes)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log.info(f"Generating DGE dotplot for top {n_genes} genes per group.")

    filename = f"{file_prefix}.{file_format}"
    output_path = os.path.join(output_dir, filename)
    save_suffix = f"_{file_prefix}.{file_format}"

    try:
        _save_scanpy_plot(
            sc.pl.dotplot, "dotplot", output_path,
            adata, var_names=var_names_to_plot, groupby=groupby_used,
            save=save_suffix, **kwargs
        )
    except Exception as e:
        log.error(f"Failed to generate DGE dotplot: {e}", exc_info=True)


def plot_rank_genes_groups_stacked_violin(
    adata: ad.AnnData,
    key: str,
    n_genes: int = 5,
    groupby: str | None = None,
    output_dir: str = ".",
    file_prefix: str = "dge_stacked_violin",
    file_format: str = "png",
    dpi: int = 150,
    **kwargs
) -> None:
    """Generates and saves a stacked violin plot of marker genes."""
    if not isinstance(adata, ad.AnnData): raise TypeError("adata must be AnnData")
    if key not in adata.uns: raise KeyError(f"DGE key '{key}' not found")
    if not output_dir: raise ValueError("output_dir must be provided")

    groupby_used = groupby or adata.uns[key].get('params', {}).get('groupby')
    if not groupby_used or groupby_used not in adata.obs:
        raise ValueError(f"Invalid groupby key '{groupby_used}'.")

    var_names_to_plot = _extract_top_marker_genes(adata, key, n_genes)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log.info(f"Generating DGE stacked violin plot for top {n_genes} genes per group.")

    filename = f"{file_prefix}.{file_format}"
    output_path = os.path.join(output_dir, filename)
    save_suffix = f"_{file_prefix}.{file_format}"

    try:
        # !!! FIX: Remove rotation=90 from kwargs passed to helper !!!
         _save_scanpy_plot(
            sc.pl.stacked_violin, "stacked_violin", output_path,
            adata, var_names=var_names_to_plot, groupby=groupby_used,
            save=save_suffix, **kwargs # Remove rotation=90 here
         )
        # !!! END FIX !!!
    except Exception as e:
        log.error(f"Failed to generate DGE stacked_violin plot: {e}", exc_info=True)


def plot_rank_genes_groups_heatmap(
    adata: ad.AnnData,
    key: str,
    n_genes: int = 5,
    groupby: str | None = None,
    output_dir: str = ".",
    file_prefix: str = "dge_heatmap",
    file_format: str = "png",
    dpi: int = 150,
    **kwargs
) -> None:
    """Generates and saves a heatmap of marker genes."""
    if not isinstance(adata, ad.AnnData): raise TypeError("adata must be AnnData")
    if key not in adata.uns: raise KeyError(f"DGE key '{key}' not found")
    if not output_dir: raise ValueError("output_dir must be provided")

    groupby_used = groupby or adata.uns[key].get('params', {}).get('groupby')
    if not groupby_used or groupby_used not in adata.obs:
        raise ValueError(f"Invalid groupby key '{groupby_used}'.")

    var_names_to_plot = _extract_top_marker_genes(adata, key, n_genes)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log.info(f"Generating DGE heatmap for top {n_genes} genes per group.")

    filename = f"{file_prefix}.{file_format}"
    output_path = os.path.join(output_dir, filename)
    save_suffix = f"_{file_prefix}.{file_format}"

    try:
        _save_scanpy_plot(
            sc.pl.heatmap, "heatmap", output_path,
            adata, var_names=var_names_to_plot, groupby=groupby_used,
            save=save_suffix, **kwargs
        )
    except Exception as e:
        log.error(f"Failed to generate DGE heatmap: {e}", exc_info=True)

# --- Main execution block ---
# (Keep as is)
if __name__ == '__main__':
    # ... (Full main block from previous version) ...
     log.info("="*30)
     log.info("Running plotting.py directly for example")
     log.info("="*30)
     try:
         from scrnaseq_agent.data.loader import load_data
         from scrnaseq_agent.analysis.qc import calculate_qc_metrics, filter_cells_qc
         from scrnaseq_agent.analysis.preprocess import normalize_log1p, select_hvg
         from scrnaseq_agent.analysis.dimred import reduce_dimensionality
         from scrnaseq_agent.analysis.clustering import perform_clustering
         from scrnaseq_agent.analysis.dge import find_marker_genes
         from scrnaseq_agent.analysis.annotation import annotate_cell_types, load_marker_dict # Needed if testing annotation plots
         import scanpy as sc
     except ImportError: log.error("Could not import functions."); sys.exit(1)

     current_dir = os.path.dirname(__file__)
     project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
     TEST_10X_DIR = os.path.join(project_root, "test_data", "filtered_gene_bc_matrices", "hg19")
     PLOT_OUTPUT_DIR = os.path.join(project_root, "output_plots_full") # Use new dir

     if not os.path.exists(TEST_10X_DIR): log.warning("Test data not found."); sys.exit(0)

     log.info("\n--- Testing All Plotting Functions ---")
     adata_test = None; adata_original = None
     try:
         log.info("Running prerequisite pipeline steps...")
         adata_test = load_data(TEST_10X_DIR)
         adata_original = adata_test.copy()
         calculate_qc_metrics(adata_test, mito_gene_prefix="MT-", inplace=True)
         qc_keys_to_plot = ["n_genes_by_counts", "total_counts", "pct_counts_mt"]
         filter_cells_qc(adata_test, min_genes=200, max_pct_mito=15.0, inplace=True)
         normalize_log1p(adata_test, inplace=True)
         select_hvg(adata_test, n_top_genes=2000, subset=True, inplace=True)
         adata_test.raw = adata_original[adata_test.obs_names, adata_test.var_names].copy()
         sc.pp.scale(adata_test, max_value=10)
         reduce_dimensionality(adata_test, n_comps=50, inplace=True, random_state=0)
         perform_clustering(adata_test, resolution=0.8, random_state=0, calculate_umap=True, inplace=True)
         cluster_key='leiden'
         dge_key = 'rank_genes_groups_raw'
         find_marker_genes(adata_test, groupby=cluster_key, use_raw=True, key_added=dge_key)
         log.info("Prerequisite steps complete.")

         log.info(f"Generating plots in directory: {PLOT_OUTPUT_DIR}")
         sc.settings.figdir = PLOT_OUTPUT_DIR

         # a) QC Violin Plot
         plot_qc_violin(
             adata=adata_test, keys=qc_keys_to_plot, output_dir=PLOT_OUTPUT_DIR,
             file_prefix="pbmc3k_qc_violin_postfilt", groupby=cluster_key, dpi=100
         )

         # b) UMAP plots
         plot_umap(
             adata=adata_test, color_by=[cluster_key] + qc_keys_to_plot,
             output_dir=PLOT_OUTPUT_DIR, file_prefix="pbmc3k_umap", dpi=100
         )
         if 'MS4A1' in adata_test.var_names:
              plot_umap(
                  adata=adata_test, color_by=['MS4A1'], output_dir=PLOT_OUTPUT_DIR,
                  file_prefix="pbmc3k_gene_umap", use_raw=False, cmap='viridis', dpi=100
              )

         # c) DGE Plots
         plot_rank_genes_groups_dotplot(
              adata=adata_test, key=dge_key, n_genes=4, groupby=cluster_key,
              output_dir=PLOT_OUTPUT_DIR, file_prefix="pbmc3k_dge_dotplot",
              dpi=100, use_raw=True
         )
         plot_rank_genes_groups_stacked_violin(
              adata=adata_test, key=dge_key, n_genes=4, groupby=cluster_key,
              output_dir=PLOT_OUTPUT_DIR, file_prefix="pbmc3k_dge_violin",
              dpi=100, use_raw=True
         )
         plot_rank_genes_groups_heatmap(
              adata=adata_test, key=dge_key, n_genes=4, groupby=cluster_key,
              output_dir=PLOT_OUTPUT_DIR, file_prefix="pbmc3k_dge_heatmap",
              dpi=100, use_raw=True, show_gene_labels=True
         )

         log.info(f"Plotting complete. Check '{PLOT_OUTPUT_DIR}'.")

     except Exception as e:
         log.error(f"\nError during plotting example run: {e}", exc_info=True)

     log.info("\nFinished plotting.py direct run example.")