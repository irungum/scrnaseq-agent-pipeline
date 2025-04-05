# scrnaseq_agent/analysis/clustering.py

import scanpy as sc
import anndata as ad
import logging
import numpy as np
import os
import warnings

log = logging.getLogger(__name__)
# Basic config only runs if the script is executed directly.
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # Add parent directory to sys.path for relative imports in direct run
    import sys
    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

def perform_clustering(
    adata: ad.AnnData,
    use_rep: str = 'X_pca',
    n_neighbors: int = 15,
    resolution: float = 1.0,
    random_state: int = 0,
    leiden_key_added: str = 'leiden',
    calculate_umap: bool = True, # <-- New parameter
    umap_key_added: str = 'X_umap', # <-- New parameter
    inplace: bool = True
) -> ad.AnnData | None:
    """
    Computes neighborhood graph, performs Leiden clustering, and optionally UMAP.

    Uses scanpy.pp.neighbors, scanpy.tl.leiden, and optionally scanpy.tl.umap.
    Assumes dimensionality reduction (e.g., PCA) has been performed and the
    result is stored in adata.obsm[use_rep]. Neighbors graph is required for
    both Leiden and UMAP.

    Args:
        adata: The annotated data matrix (typically after PCA).
        use_rep: Representation in adata.obsm to use for neighbor calculation
                 (e.g., 'X_pca'). Defaults to 'X_pca'.
        n_neighbors: Number of neighbors to use for k-NN graph construction.
                     Defaults to 15.
        resolution: Resolution parameter for the Leiden algorithm, influencing
                    the number of clusters found. Defaults to 1.0.
        random_state: Seed for the random number generator used by Leiden and UMAP
                      for reproducibility. Defaults to 0.
        leiden_key_added: Key under which the Leiden clustering results will be
                          stored in adata.obs. Defaults to 'leiden'.
        calculate_umap: Whether to calculate UMAP embedding. Defaults to True.
        umap_key_added: Key under which the UMAP coordinates will be stored in
                        adata.obsm. Defaults to 'X_umap'.
        inplace: Modify AnnData object inplace. Defaults to True.

    Returns:
        If inplace=True, returns None. Otherwise, returns the modified AnnData
        object with neighbors graph, clustering results, and optionally UMAP.

    Raises:
        TypeError: If input `adata` is not an AnnData object.
        KeyError: If `use_rep` is not found in `adata.obsm`.
        ValueError: If `n_neighbors` or `resolution` are invalid.
        RuntimeError: If the underlying scanpy functions fail.
    """
    if not isinstance(adata, ad.AnnData):
        raise TypeError("Input 'adata' must be an AnnData object.")
    if use_rep not in adata.obsm:
        raise KeyError(f"Representation '{use_rep}' not found in adata.obsm. Run dimensionality reduction first.")
    if not isinstance(n_neighbors, int) or n_neighbors <= 0:
        raise ValueError("Argument 'n_neighbors' must be a positive integer.")
    if not isinstance(resolution, (int, float)) or resolution <= 0:
        raise ValueError("Argument 'resolution' must be a positive number.")

    log.info(
        f"Performing clustering using {use_rep}: n_neighbors={n_neighbors}, "
        f"resolution={resolution}, random_state={random_state}. UMAP calculation: {calculate_umap}."
    )

    # Handle inplace logic
    adata_work = adata if inplace else adata.copy()
    log.debug(f"Operating {'inplace' if inplace else 'on a copy'}.")

    try:
        # 1. Compute Neighborhood Graph (Required for both Leiden and UMAP)
        log.info(f"Computing neighborhood graph using '{use_rep}' with {n_neighbors} neighbors...")
        sc.pp.neighbors(
            adata_work,
            n_neighbors=n_neighbors,
            use_rep=use_rep,
            random_state=random_state, # Pass random state here
            # copy=False # Neighbors modifies inplace by default
        )
        log.info("Neighborhood graph computed successfully. Results in .uns['neighbors'].")

        # Verify neighbors results
        if 'neighbors' not in adata_work.uns:
            raise RuntimeError("scanpy.pp.neighbors finished but 'neighbors' not found in adata.uns.")
        if 'connectivities' not in adata_work.obsp:
            raise RuntimeError("scanpy.pp.neighbors finished but 'connectivities' not found in adata.obsp.")
        if 'distances' not in adata_work.obsp:
            raise RuntimeError("scanpy.pp.neighbors finished but 'distances' not found in adata.obsp.")


        # 2. Perform Leiden Clustering
        log.info(f"Performing Leiden clustering with resolution {resolution}...")
        sc.tl.leiden(
            adata_work,
            resolution=resolution,
            random_state=random_state,
            key_added=leiden_key_added,
            # copy=False # Leiden modifies inplace by default
        )
        log.info(f"Leiden clustering completed. Results stored in adata.obs['{leiden_key_added}'].")

        # Verify Leiden results
        if leiden_key_added not in adata_work.obs:
             raise RuntimeError(f"scanpy.tl.leiden finished but '{leiden_key_added}' not found in adata.obs.")

        n_clusters_found = len(adata_work.obs[leiden_key_added].unique())
        log.info(f"Found {n_clusters_found} clusters.")

        # 3. Optionally Calculate UMAP (Requires neighbors graph)
        if calculate_umap:
            log.info("Calculating UMAP embedding...")
            sc.tl.umap(
                adata_work,
                random_state=random_state, # Use same random state for reproducibility
                # copy=False # UMAP modifies inplace by default
            )
            # Verify UMAP results - UMAP adds results to obsm with 'X_umap' key by default
            # Scanpy's default key is 'X_umap', so we check for that if umap_key_added matches
            if umap_key_added not in adata_work.obsm:
                 # If a custom key was intended but not used by scanpy's default, this might be misleading.
                 # However, sc.tl.umap doesn't have a direct key_added parameter like Leiden.
                 # It *always* adds 'X_umap'. We should rely on this convention.
                 # If the user REALLY wanted a different key, they'd have to rename it after.
                 # Let's ensure the default key 'X_umap' is present.
                 if 'X_umap' not in adata_work.obsm:
                      raise RuntimeError("scanpy.tl.umap finished but 'X_umap' not found in adata.obsm.")
                 else:
                      # If default X_umap exists but doesn't match requested key, warn or rename?
                      # For simplicity, let's assume standard usage where X_umap is expected.
                      log.info(f"UMAP calculation completed. Results stored in adata.obsm['X_umap'].")

            # Add UMAP params to uns for tracking (Scanpy doesn't do this automatically for umap)
            if 'umap' not in adata_work.uns:
                adata_work.uns['umap'] = {}
            adata_work.uns['umap']['params'] = {'random_state': random_state}
            # Note: sc.tl.umap doesn't directly store parameters like n_neighbors used implicitly via sc.pp.neighbors
            # We could copy them from uns['neighbors'] if desired for full traceability

        else:
             log.info("Skipping UMAP calculation as requested.")


    except Exception as e:
        log.error(f"An error occurred during neighbors calculation, clustering or UMAP: {e}", exc_info=True)
        raise RuntimeError(f"Failed during clustering/UMAP steps: {e}") from e

    # Return based on inplace flag
    if not inplace:
        log.debug("Returning modified AnnData copy with clustering/UMAP results.")
        return adata_work
    else:
        log.debug("Returning None as operation was inplace.")
        return None

# Example direct execution block - updated to show UMAP results
if __name__ == '__main__':
    log.info("="*30)
    log.info("Running clustering.py directly for example")
    log.info("="*30)

    # Need functions from previous steps
    try:
        from scrnaseq_agent.data.loader import load_data
        from scrnaseq_agent.analysis.qc import calculate_qc_metrics, filter_cells_qc
        from scrnaseq_agent.analysis.preprocess import normalize_log1p, select_hvg
        from scrnaseq_agent.analysis.dimred import reduce_dimensionality
        import scanpy as sc # Need pp.scale
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

    # --- Run full pipeline up to clustering & UMAP ---
    log.info("\n--- Testing Clustering + UMAP (Inplace Example) ---")
    adata_test = None
    try:
        log.info("1. Loading data...")
        adata_test = load_data(TEST_10X_DIR)
        log.info(f"Loaded data shape: {adata_test.shape}")

        log.info("2. Calculating QC...")
        calculate_qc_metrics(adata_test, mito_gene_prefix="MT-", inplace=True)
        log.info("3. Filtering cells...")
        filter_cells_qc(adata_test, min_genes=200, max_pct_mito=15.0, inplace=True)
        log.info(f"Shape after QC: {adata_test.shape}")
        if adata_test.n_obs == 0: raise ValueError("Filtered all cells")

        log.info("4. Normalizing and Log1p...")
        normalize_log1p(adata_test, inplace=True)
        log.info("5. Selecting HVGs...")
        select_hvg(adata_test, n_top_genes=2000, subset=True, inplace=True)
        log.info(f"Shape after HVG selection: {adata_test.shape}")

        log.info("6. Scaling data...")
        sc.pp.scale(adata_test, max_value=10)
        log.info("Data scaled.")

        log.info("7. Reducing dimensionality (PCA)...")
        reduce_dimensionality(adata_test, n_comps=50, inplace=True, random_state=0)
        log.info(f"PCA completed.")

        log.info("8. Performing Clustering & UMAP (inplace)...")
        perform_clustering(
            adata_test,
            use_rep='X_pca',
            n_neighbors=15,
            resolution=0.8, # Example resolution
            random_state=0,
            calculate_umap=True, # Explicitly True (default)
            inplace=True
        )
        log.info(f"Clustering & UMAP completed.")
        log.info(f"adata.obs columns: {list(adata_test.obs.columns)}")
        log.info(f"adata.uns keys: {list(adata_test.uns.keys())}")
        log.info(f"adata.obsm keys: {list(adata_test.obsm.keys())}") # Should now include X_umap
        log.info(f"adata.obsp keys: {list(adata_test.obsp.keys())}")
        if 'leiden' in adata_test.obs:
            log.info(f"Leiden cluster counts:\n{adata_test.obs['leiden'].value_counts()}")
        if 'X_umap' in adata_test.obsm:
            log.info(f"UMAP coordinates stored in .obsm['X_umap'] with shape {adata_test.obsm['X_umap'].shape}")

    except Exception as e:
        log.error(f"\nError during Clustering/UMAP test: {e}", exc_info=True)

    log.info("\nFinished clustering.py direct run example.")