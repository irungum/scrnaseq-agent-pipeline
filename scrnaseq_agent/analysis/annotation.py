# scrnaseq_agent/analysis/annotation.py

import scanpy as sc
import anndata as ad
import logging
import pandas as pd
import json # For potentially loading marker files
import os # Added for main block example

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

def load_marker_dict(file_path: str) -> dict | None:
    """Loads a marker gene dictionary from a JSON file."""
    if not file_path:
        log.warning("No marker file path provided.")
        return None
    if not os.path.exists(file_path):
        log.error(f"Marker file not found: {file_path}")
        return None

    try:
        with open(file_path, 'r') as f:
            marker_dict = json.load(f)
        log.info(f"Successfully loaded marker dictionary from {file_path}")
        # Basic validation
        if not isinstance(marker_dict, dict):
             log.error(f"Marker file '{file_path}' content is not a dictionary.")
             return None
        if not all(isinstance(k, str) and isinstance(v, list) and all(isinstance(g, str) for g in v) for k, v in marker_dict.items()):
            log.error(f"Marker file '{file_path}' does not contain a valid dictionary structure of cell_type:[gene_list].")
            return None
        return marker_dict
    except json.JSONDecodeError as e:
        log.error(f"Error decoding JSON marker file {file_path}: {e}")
        return None
    except Exception as e:
        log.error(f"An unexpected error occurred loading marker file {file_path}: {e}", exc_info=True)
        return None


def annotate_cell_types(
    adata: ad.AnnData,
    marker_dict: dict,
    groupby: str,
    rank_key: str = 'rank_genes_groups',
    annotation_key: str = 'marker_gene_overlap_annotation',
    method: str = 'overlap_count',
    **kwargs
) -> None:
    """
    Performs basic cell type annotation based on marker gene overlap.

    Wraps scanpy.tl.marker_gene_overlap. Requires rank_genes_groups results.
    Adds annotation scores to `adata.obs` based on the overlap between cluster
    markers and the provided known marker dictionary.

    Note: scanpy.tl.marker_gene_overlap might not add the output column if no
    significant overlaps are found, depending on the method and data.

    Args:
        adata: Annotated data matrix with clustering and DGE results.
        marker_dict: Dictionary where keys are cell type names and values are
                     lists of known marker genes.
        groupby: The key in `adata.obs` used for clustering/grouping in DGE.
        rank_key: Key in `adata.uns` where rank_genes_groups results are stored.
                  Defaults to 'rank_genes_groups'.
        annotation_key: Base key for storing the annotation results in `adata.obs`.
                        Scanpy will append the method name (e.g., _overlap_count).
                        Defaults to 'marker_gene_overlap_annotation'.
        method: Method for calculating overlap score ('overlap_count', 'overlap_coef', 'jaccard').
                Defaults to 'overlap_count'.
        **kwargs: Additional keyword arguments passed to `sc.tl.marker_gene_overlap`.

    Returns:
        None. Results are added to `adata.obs`.

    Raises:
        TypeError: If input `adata` is not an AnnData object or `marker_dict` is invalid.
        KeyError: If `groupby` key or `rank_key` is missing.
        RuntimeError: If the underlying scanpy function fails unexpectedly.
    """
    if not isinstance(adata, ad.AnnData):
        raise TypeError("Input 'adata' must be an AnnData object.")
    if not isinstance(marker_dict, dict) or not marker_dict:
         raise TypeError("Input 'marker_dict' must be a non-empty dictionary.")
    if groupby not in adata.obs:
        raise KeyError(f"Group key '{groupby}' not found in adata.obs.")
    if rank_key not in adata.uns:
        raise KeyError(f"Rank genes groups key '{rank_key}' not found in adata.uns.")

    log.info(f"Performing marker gene overlap annotation for groups '{groupby}' using results '{rank_key}'.")
    log.info(f"Using method '{method}'. Annotation key base: '{annotation_key}'.")

    # Store current obs columns to check if new one is added
    original_obs_cols = set(adata.obs.columns)

    try:
        sc.tl.marker_gene_overlap(
            adata,
            marker_dict,
            key=rank_key,
            method=method,
            key_added=annotation_key,
            **kwargs
        )

        # Verify if *any* new column starting with annotation_key was added
        new_cols = set(adata.obs.columns) - original_obs_cols
        added_expected_format = any(col.startswith(f"{annotation_key}_") for col in new_cols)

        if not added_expected_format:
            log.warning(f"sc.tl.marker_gene_overlap finished, but no column starting with '{annotation_key}_' was added to adata.obs. No significant overlap might have been found.")
        else:
            log.info(f"Marker gene overlap annotation complete. Results added to adata.obs (check columns starting with '{annotation_key}_').")

    except Exception as e:
        log.error(f"An error occurred during marker gene overlap annotation: {e}", exc_info=True)
        raise RuntimeError(f"Failed during marker gene overlap annotation: {e}") from e

    return None


# Example direct execution block
if __name__ == '__main__':
    log.info("="*30)
    log.info("Running annotation.py directly for example")
    log.info("="*30)

    pbmc_markers_example = {
        'CD4 T cells': ['IL7R', 'CD3D', 'CD4'],
        'CD8 T cells': ['CD8A', 'CD3D', 'CD8B'],
        'NK cells': ['GNLY', 'NKG7', 'KLRF1', 'KLRD1'], # Added KLRD1
        'B cells': ['MS4A1', 'CD79A', 'CD19'],
        'Monocytes': ['CD14', 'LYZ', 'FCGR3A', 'MS4A7', 'S100A8', 'S100A9'], # Added S100A8/9
        'Dendritic Cells': ['FCER1A', 'CST3'],
        'Megakaryocytes': ['PPBP']
    }
    example_marker_file = "example_pbmc_markers.json"
    try:
        with open(example_marker_file, 'w') as f:
            json.dump(pbmc_markers_example, f, indent=2)
        log.info(f"Saved example marker dict to {example_marker_file}")
    except Exception:
        log.warning(f"Could not save example marker file {example_marker_file}")

    try:
        from scrnaseq_agent.data.loader import load_data
        from scrnaseq_agent.analysis.qc import calculate_qc_metrics, filter_cells_qc
        from scrnaseq_agent.analysis.preprocess import normalize_log1p, select_hvg
        from scrnaseq_agent.analysis.dimred import reduce_dimensionality
        from scrnaseq_agent.analysis.clustering import perform_clustering
        from scrnaseq_agent.analysis.dge import find_marker_genes
        import scanpy as sc
    except ImportError:
        log.error("Could not import required functions."); sys.exit(1)

    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    TEST_10X_DIR = os.path.join(project_root, "test_data", "filtered_gene_bc_matrices", "hg19")

    if not os.path.exists(TEST_10X_DIR):
        log.warning("Skipping direct run example (test data not found)."); sys.exit(0)

    log.info("\n--- Testing Cell Type Annotation ---")
    adata_test = None
    adata_original = None
    try:
        log.info("Running pipeline steps up to DGE...")
        adata_test = load_data(TEST_10X_DIR)
        adata_original = adata_test.copy()
        calculate_qc_metrics(adata_test, mito_gene_prefix="MT-", inplace=True)
        filter_cells_qc(adata_test, min_genes=200, max_pct_mito=15.0, inplace=True)
        normalize_log1p(adata_test, inplace=True)
        select_hvg(adata_test, n_top_genes=2000, subset=True, inplace=True)
        adata_test.raw = adata_original[adata_test.obs_names, adata_test.var_names].copy()
        sc.pp.scale(adata_test, max_value=10)
        reduce_dimensionality(adata_test, n_comps=50, inplace=True, random_state=0)
        perform_clustering(adata_test, resolution=0.8, random_state=0, calculate_umap=False, inplace=True)
        cluster_key = 'leiden'
        dge_key = 'rank_genes_groups_raw'
        find_marker_genes(adata_test, groupby=cluster_key, use_raw=True, key_added=dge_key)
        log.info("Prerequisite steps complete.")

        loaded_markers = load_marker_dict(example_marker_file)

        if loaded_markers:
             log.info("Performing annotation...")
             annotation_base_key='pbmc_annotation'
             annotation_method='overlap_count'
             annotate_cell_types(
                 adata=adata_test,
                 marker_dict=loaded_markers,
                 groupby=cluster_key,
                 rank_key=dge_key,
                 annotation_key=annotation_base_key,
                 method=annotation_method
             )
             log.info("Annotation function executed.")
             log.info(f"adata.obs columns: {list(adata_test.obs.columns)}")

             annotation_result_key = f"{annotation_base_key}_{annotation_method}"
             if annotation_result_key in adata_test.obs:
                 log.info(f"Annotation results head for key '{annotation_result_key}':")
                 print(adata_test.obs[[cluster_key, annotation_result_key]].head())
                 log.info(f"\nCounts per assigned type in each original cluster:")
                 # Use observed=False for older pandas/scanpy compatibility if needed
                 print(pd.crosstab(adata_test.obs[cluster_key], adata_test.obs[annotation_result_key], dropna=False))
             else:
                 log.warning(f"Expected annotation result key '{annotation_result_key}' not found.")
        else:
             log.error("Failed to load marker dictionary, skipping annotation step.")

    except Exception as e:
        log.error(f"\nError during annotation example run: {e}", exc_info=True)
    finally:
         if os.path.exists(example_marker_file):
             try: os.remove(example_marker_file); log.info(f"Removed example marker file: {example_marker_file}")
             except Exception: log.warning(f"Could not remove example marker file: {example_marker_file}")

    log.info("\nFinished annotation.py direct run example.")