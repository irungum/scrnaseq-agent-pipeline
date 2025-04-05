# scrnaseq_agent/analysis/dimred.py

import scanpy as sc
import anndata as ad
import logging
import numpy as np
import os
import warnings # <--- Import added

log = logging.getLogger(__name__)
# Basic config only runs if the script is executed directly.
# In package use, the main script should configure logging.
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

# Function definition consistent with preprocess.py and qc.py
def reduce_dimensionality(
    adata: ad.AnnData,
    n_comps: int = 50,
    random_state: int = 0,
    inplace: bool = True
) -> ad.AnnData | None:
    """
    Performs principal component analysis (PCA) to reduce the dimensionality.

    Uses scanpy.tl.pca. Stores PCA results in adata.obsm['X_pca'] and
    related info (variance, variance ratio, loadings) in adata.uns['pca']
    and adata.varm['PCs']. Assumes data has been preprocessed
    (normalized, log1p, scaled).

    Args:
        adata: The annotated data matrix (typically after scaling).
               Requires `adata.X` to be present.
        n_comps: Number of principal components to compute. Defaults to 50.
               Must be less than min(n_obs, n_vars).
        random_state: Random seed for reproducibility of the SVD solver.
                      Defaults to 0.
        inplace: Modify AnnData object inplace. Defaults to True.

    Returns:
        If inplace=True, returns None. Otherwise, returns the modified AnnData
        object with PCA results.

    Raises:
        TypeError: If input `adata` is not an AnnData object.
        ValueError: If `n_comps` is not a positive integer or cannot be adjusted.
        AttributeError: If `adata.X` is not present or not suitable (e.g., None).
        RuntimeError: If the underlying scanpy PCA function fails.
    """
    if not isinstance(adata, ad.AnnData):
        raise TypeError("Input 'adata' must be an AnnData object.")
    if not isinstance(n_comps, int) or n_comps <= 0:
        raise ValueError("Argument 'n_comps' must be a positive integer.")

    log.info(f"Performing PCA with n_comps={n_comps}, random_state={random_state}...")

    # Check for adata.X *before* potentially copying
    if adata.X is None:
         raise AttributeError("Cannot perform PCA: AnnData object does not have a suitable '.X' attribute.")

    # Check if n_comps is feasible - Scanpy does this but an early check is good
    min_dim = min(adata.shape)
    if n_comps >= min_dim:
        # PCA requires n_comps < min(n_obs, n_vars)
        adjusted_n_comps = min_dim - 1
        if adjusted_n_comps <= 0:
             raise ValueError(f"Cannot compute PCA. Input data has shape {adata.shape}, "
                              f"requiring n_comps < {min_dim}, but minimum is 1.")

        warning_message = (
            f"Requested n_comps ({n_comps}) >= smallest dimension ({min_dim}). "
            f"Adjusting n_comps to {adjusted_n_comps}."
        )
        # Use warnings.warn so pytest.warns can catch it
        warnings.warn(warning_message, UserWarning, stacklevel=2)
        log.warning(warning_message) # Keep logging the warning as well

        n_comps = adjusted_n_comps # Use the adjusted value

    # Handle inplace logic: Work on a copy if not inplace
    adata_work = adata if inplace else adata.copy()
    log.debug(f"Operating {'inplace' if inplace else 'on a copy'}.")

    try:
        # Perform PCA using scanpy's tools function
        # Ensure copy=False so it modifies adata_work directly
        # zero_center=True is standard for PCA
        sc.tl.pca(
            adata_work,
            n_comps=n_comps, # Pass the potentially adjusted n_comps
            svd_solver='arpack', # Robust solver for typical scRNA-seq data
            random_state=random_state,
            zero_center=True,
            copy=False # MUST be False to modify adata_work
        )

        # Verify that Scanpy added the expected fields
        if 'X_pca' not in adata_work.obsm:
             raise RuntimeError("PCA calculation finished but 'X_pca' not found in adata.obsm.")
        if 'pca' not in adata_work.uns or 'variance_ratio' not in adata_work.uns['pca']:
             raise RuntimeError("PCA calculation finished but variance info not found in adata.uns['pca'].")
        if 'PCs' not in adata_work.varm:
            # This might happen if n_comps is very small, log a warning rather than error
            log.warning("PCA calculation finished but 'PCs' (loadings) not found in adata.varm.")

        log.info(f"PCA completed successfully. Results in .obsm['X_pca'] "
                 f"(shape: {adata_work.obsm['X_pca'].shape}), .uns['pca'], and potentially .varm['PCs'].")

    except ValueError as ve:
        # Catch potential value errors from scanpy (e.g., related to n_comps, data type)
        log.error(f"ValueError during PCA execution: {ve}", exc_info=True)
        raise ValueError(f"Input value error during PCA (check data scaling/type?): {ve}") from ve
    except Exception as e:
        # Catch any other unexpected errors during the scanpy call
        log.error(f"An unexpected error occurred during PCA: {e}", exc_info=True)
        raise RuntimeError(f"Failed to perform PCA due to an unexpected error: {e}") from e

    # Return based on inplace flag
    if not inplace:
        log.debug("Returning modified AnnData copy.")
        return adata_work
    else:
        log.debug("Returning None as operation was inplace.")
        # Explicitly return None for clarity
        return None

# Example direct execution block matching style in preprocess.py/qc.py
if __name__ == '__main__':
    log.info("="*30)
    log.info("Running dimred.py directly for example")
    log.info("="*30)

    # Need loader and preprocess functions for a realistic example
    try:
        from scrnaseq_agent.data.loader import load_data
        from scrnaseq_agent.analysis.preprocess import normalize_log1p, select_hvg
        from scrnaseq_agent.analysis.qc import calculate_qc_metrics, filter_cells_qc # Assuming QC needed first
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

    # --- Inplace Example ---
    log.info("\n--- Testing Dimensionality Reduction (Inplace) ---")
    adata_test = None
    try:
        log.info("1. Loading data...")
        adata_test = load_data(TEST_10X_DIR)
        log.info(f"Loaded data shape: {adata_test.shape}")

        log.info("2. Calculating QC...")
        # Use 'MT-' prefix for human data as used in the test fixture
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
        # Scaling is crucial before PCA
        sc.pp.scale(adata_test, max_value=10)
        log.info("Data scaled.")

        log.info("7. Reducing dimensionality (PCA, inplace)...")
        reduce_dimensionality(adata_test, n_comps=50, inplace=True, random_state=0)
        log.info(f"PCA completed. Final AnnData shape: {adata_test.shape}") # Shape unchanged by PCA itself
        log.info(f"adata.obsm keys: {list(adata_test.obsm.keys())}")
        log.info(f"adata.uns keys: {list(adata_test.uns.keys())}")
        if 'pca' in adata_test.uns and 'variance_ratio' in adata_test.uns['pca']:
            log.info(f"First 5 PCA variance ratios: {adata_test.uns['pca']['variance_ratio'][:5]}")
        else:
            log.warning("PCA variance ratio info not found in adata.uns['pca']")

    except Exception as e:
        log.error(f"\nError during Inplace Dimensionality Reduction test: {e}", exc_info=True)


    # --- Non-inplace Example ---
    log.info("\n--- Testing Dimensionality Reduction (Non-Inplace) ---")
    adata_orig = None
    try:
        log.info("1. Loading data...")
        adata_orig = load_data(TEST_10X_DIR)
        log.info(f"Loaded data shape: {adata_orig.shape}")
        # Keep original safe for comparison, apply steps to copies or results
        adata_qc = adata_orig.copy()
        calculate_qc_metrics(adata_qc, mito_gene_prefix="MT-", inplace=True)
        adata_filt = filter_cells_qc(adata_qc, min_genes=200, max_pct_mito=15.0, inplace=False)
        log.info(f"Shape after QC (non-inplace): {adata_filt.shape}")
        adata_norm = normalize_log1p(adata_filt, inplace=False)
        adata_hvg = select_hvg(adata_norm, n_top_genes=2000, subset=True, inplace=False)
        log.info(f"Shape after HVG (non-inplace): {adata_hvg.shape}")

        log.info("6. Scaling data (on copy)...")
        # Important: Scale the object that will be input to PCA
        adata_scaled = adata_hvg.copy() # Explicit copy before scaling if needed downstream
        sc.pp.scale(adata_scaled, max_value=10)
        log.info("Data scaled.")

        log.info("7. Reducing dimensionality (PCA, non-inplace)...")
        adata_pca = reduce_dimensionality(adata_scaled, n_comps=30, inplace=False, random_state=0)

        log.info(f"PCA completed. Returned new AnnData object: {isinstance(adata_pca, ad.AnnData)}")
        if adata_pca:
            log.info(f"New AnnData shape: {adata_pca.shape}") # Shape doesn't change here
            log.info(f"New AnnData .obsm keys: {list(adata_pca.obsm.keys())}")
            log.info(f"New AnnData .uns keys: {list(adata_pca.uns.keys())}")
            if 'pca' in adata_pca.uns and 'variance_ratio' in adata_pca.uns['pca']:
                log.info(f"New AnnData first 5 PCA variance ratios: {adata_pca.uns['pca']['variance_ratio'][:5]}")
            else:
                log.warning("PCA variance ratio info not found in returned adata.uns['pca']")

        log.info(f"Original scaled data has PCA results? {'X_pca' in adata_scaled.obsm}") # Should be False

    except Exception as e:
        log.error(f"\nError during Non-Inplace Dimensionality Reduction test: {e}", exc_info=True)

    log.info("\nFinished dimred.py direct run example.")