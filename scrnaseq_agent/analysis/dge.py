# scrnaseq_agent/analysis/dge.py

import scanpy as sc
import anndata as ad
import logging
import pandas as pd # For potential result formatting later

log = logging.getLogger(__name__)
# Basic config only runs if the script is executed directly.
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # Add parent directory to sys.path for relative imports in direct run
    import sys, os
    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

def find_marker_genes(
    adata: ad.AnnData,
    groupby: str,
    method: str = 'wilcoxon',
    corr_method: str = 'benjamini-hochberg',
    use_raw: bool | None = None,
    key_added: str = 'rank_genes_groups',
    **kwargs
) -> None:
    """
    Performs differential gene expression analysis to find marker genes.

    Wraps scanpy.tl.rank_genes_groups. This function identifies genes that are
    differentially expressed in each group defined by `groupby` compared to
    all other cells. Results are stored inplace in `adata.uns[key_added]`.

    Args:
        adata: The annotated data matrix (must contain cluster labels in .obs).
        groupby: The key in `adata.obs` that contains the group labels (e.g.,
                 'leiden', 'cell_type').
        method: The statistical method to use ('wilcoxon', 't-test', 'logreg').
                Defaults to 'wilcoxon'.
        corr_method: Method for multiple testing correction ('benjamini-hochberg',
                     'bonferroni'). Defaults to 'benjamini-hochberg'.
        use_raw: Whether to use `adata.raw.X` for the analysis. If None (default),
                 tries to use `.raw` if it exists, otherwise uses `adata.X`.
                 It's often recommended to use raw (unscaled, non-log) counts
                 for DGE.
        key_added: Key under which the results dictionary is stored in `adata.uns`.
                   Defaults to 'rank_genes_groups'.
        **kwargs: Additional keyword arguments passed directly to
                  `sc.tl.rank_genes_groups` (e.g., `n_genes`, `pts`).

    Returns:
        None. Results are added to `adata.uns`.

    Raises:
        TypeError: If input `adata` is not an AnnData object.
        KeyError: If `groupby` key is not found in `adata.obs`.
        ValueError: If `use_raw=True` but `adata.raw` is None.
        RuntimeError: If the underlying scanpy function fails.
    """
    if not isinstance(adata, ad.AnnData):
        raise TypeError("Input 'adata' must be an AnnData object.")
    if groupby not in adata.obs:
        raise KeyError(f"Group key '{groupby}' not found in adata.obs.")

    log.info(f"Finding marker genes using '{method}' method for groups in '{groupby}'.")
    log.info(f"Correction method: '{corr_method}'. Results key: '{key_added}'.")

    # Handle use_raw logic carefully
    if use_raw is None:
        use_raw_calc = adata.raw is not None
        log.info(f"'use_raw' is None, automatically setting to {use_raw_calc} based on presence of adata.raw.")
    else:
        use_raw_calc = use_raw
        if use_raw_calc and adata.raw is None:
            raise ValueError("Argument 'use_raw' was set to True, but adata.raw is None.")
        log.info(f"Using {'adata.raw.X' if use_raw_calc else 'adata.X'} for calculation based on 'use_raw'={use_raw}.")

    try:
        # Perform DGE using scanpy's function
        # Note: rank_genes_groups always modifies adata.uns inplace.
        sc.tl.rank_genes_groups(
            adata,
            groupby=groupby,
            method=method,
            corr_method=corr_method,
            use_raw=use_raw_calc,
            key_added=key_added,
            **kwargs
        )

        # Verify that results were added
        if key_added not in adata.uns:
            raise RuntimeError(f"sc.tl.rank_genes_groups finished but '{key_added}' not found in adata.uns.")

        # Check for essential result structure (names, pvals_adj)
        if 'names' not in adata.uns[key_added] or 'pvals_adj' not in adata.uns[key_added]:
             log.warning(f"Results in adata.uns['{key_added}'] might be incomplete (missing 'names' or 'pvals_adj').")

        log.info(f"Marker gene analysis completed. Results stored in adata.uns['{key_added}'].")

    except Exception as e:
        log.error(f"An error occurred during marker gene identification: {e}", exc_info=True)
        raise RuntimeError(f"Failed during marker gene identification: {e}") from e

    # No return value as operation is inplace on adata.uns
    return None

# Example direct execution block
# Example direct execution block
if __name__ == '__main__':
    log.info("="*30)
    log.info("Running dge.py directly for example")
    log.info("="*30)

    # Need functions from previous steps
    try:
        from scrnaseq_agent.data.loader import load_data
        from scrnaseq_agent.analysis.qc import calculate_qc_metrics, filter_cells_qc
        from scrnaseq_agent.analysis.preprocess import normalize_log1p, select_hvg
        from scrnaseq_agent.analysis.dimred import reduce_dimensionality
        from scrnaseq_agent.analysis.clustering import perform_clustering
        import scanpy as sc # Need pp.scale and potentially get.rank_genes_groups_df
    except ImportError:
        log.error("Could not import required functions. Ensure running from project root "
                  "or environment is correctly set up.")
        import sys
        sys.exit(1)

    # Define path relative to this file's expected location
    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    TEST_10X_DIR = os.path.join(project_root, "test_data", "filtered_gene_bc_matrices", "hg19")

    if not os.path.exists(TEST_10X_DIR):
        log.warning(f"Test data directory not found: {TEST_10X_DIR}")
        log.warning("Skipping direct run example.")
        import sys
        sys.exit(0)

    # --- Run full pipeline up to DGE ---
    log.info("\n--- Testing Marker Gene Identification ---")
    adata_test = None
    adata_original = None # To store original raw data
    try:
        log.info("1. Loading data...")
        adata_test = load_data(TEST_10X_DIR)
        adata_original = adata_test.copy() # Store the original raw data state
        log.info(f"Loaded data shape: {adata_test.shape}. Copied original state.")

        log.info("2. Calculating QC...")
        calculate_qc_metrics(adata_test, mito_gene_prefix="MT-", inplace=True)
        log.info("3. Filtering cells...")
        filter_cells_qc(adata_test, min_genes=200, max_pct_mito=15.0, inplace=True)
        log.info(f"Shape after QC: {adata_test.shape}.")
        if adata_test.n_obs == 0: raise ValueError("Filtered all cells")

        log.info("4. Normalizing and Log1p (on main adata)...")
        normalize_log1p(adata_test, inplace=True)
        log.info("5. Selecting HVGs (subsetting main adata)...")
        select_hvg(adata_test, n_top_genes=2000, subset=True, inplace=True)
        log.info(f"Shape after HVG selection: {adata_test.shape}")

        # ---> Create .raw attribute AFTER all filtering <---
        log.info("Creating final .raw attribute from original data matching current cell/gene selection...")
        # Select the cells and genes from the original data based on the final state of adata_test
        adata_test.raw = adata_original[adata_test.obs_names, adata_test.var_names].copy()
        log.info(f"Set final .raw attribute. Final .raw shape: {adata_test.raw.shape}")
        # --- End .raw creation ---

        log.info("6. Scaling data (on main adata)...")
        sc.pp.scale(adata_test, max_value=10)
        log.info("Data scaled.")

        log.info("7. Reducing dimensionality (PCA)...")
        reduce_dimensionality(adata_test, n_comps=50, inplace=True, random_state=0)
        log.info(f"PCA completed.")

        log.info("8. Performing Clustering & UMAP...")
        perform_clustering(
            adata_test,
            resolution=0.8, # Same resolution as before
            random_state=0,
            calculate_umap=True, # Keep UMAP for context if needed later
            inplace=True
        )
        log.info(f"Clustering & UMAP completed.")
        cluster_key = 'leiden' # The default key used by perform_clustering
        if cluster_key not in adata_test.obs:
            raise KeyError(f"Clustering key '{cluster_key}' not found after clustering step.")

        # --- 9. Find Marker Genes ---
        log.info(f"Finding marker genes for '{cluster_key}' clusters using .raw data...")
        find_marker_genes(
            adata=adata_test,
            groupby=cluster_key,
            method='wilcoxon',
            use_raw=True, # Use the raw counts stored earlier
            key_added='rank_genes_groups_raw' # Use a specific key for clarity
        )
        log.info("Marker gene function executed.")
        log.info(f"adata.uns keys: {list(adata_test.uns.keys())}")

        # Display top markers for a few clusters
        if 'rank_genes_groups_raw' in adata_test.uns:
            log.info("Top markers per cluster (from .raw data):")
            # Use scanpy helper to create a DataFrame for easier viewing
            # Need to specify the key we used
            try:
                marker_df = sc.get.rank_genes_groups_df(adata_test, group=None, key='rank_genes_groups_raw')
                # Print top 2 markers for first 3 groups/clusters
                groups_to_show = adata_test.obs[cluster_key].cat.categories[:3]
                if not groups_to_show.empty:
                     print(marker_df[marker_df['group'].isin(groups_to_show)].groupby('group').head(2))
                else:
                     log.warning("No cluster groups found to display markers for.")
            except Exception as df_err:
                log.warning(f"Could not generate DataFrame view of markers: {df_err}")
        else:
            log.warning("Marker gene results key 'rank_genes_groups_raw' not found in adata.uns.")


    except Exception as e:
        log.error(f"\nError during DGE test run: {e}", exc_info=True)

    log.info("\nFinished dge.py direct run example.")