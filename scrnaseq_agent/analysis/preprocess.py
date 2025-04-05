# scrnaseq_agent/analysis/preprocess.py

import scanpy as sc
import anndata as ad
import logging
import numpy as np # For log1p
import os

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) # Basic config for direct runs

def normalize_log1p(
    adata: ad.AnnData,
    target_sum: float | None = 1e4, # Normalize to counts per 10,000 by default
    inplace: bool = True
) -> ad.AnnData | None:
    """
    Normalizes counts per cell to target_sum and log1p transforms the data.

    Uses scanpy.pp.normalize_total and scanpy.pp.log1p.
    Stores normalized, log1p data in adata.X.

    Args:
        adata: The annotated data matrix (typically after QC filtering).
               Assumes raw counts are in adata.X or adata.raw.X if applicable.
        target_sum: Total counts per cell after normalization. If None, library sizes
                    are just scaled to the median library size. Defaults to 1e4.
        inplace: Modify AnnData object inplace. Defaults to True.

    Returns:
        If inplace=True, returns None. Otherwise, returns the modified AnnData object.
    """
    if not isinstance(adata, ad.AnnData):
        raise TypeError("Input must be an AnnData object.")

    log.info(f"Normalizing total counts per cell to target_sum={target_sum} and log1p transforming.")

    adata_copy = adata if inplace else adata.copy()

    # Check if data looks like raw counts (often integers, non-negative)
    is_int_dtype = hasattr(adata_copy.X, 'dtype') and np.issubdtype(adata_copy.X.dtype, np.integer)
    has_neg_values = False
    # Check for negative values (needs iteration for sparse matrix)
    if hasattr(adata_copy.X, 'min'): # Dense array or some sparse formats
        try:
            if adata_copy.X.min() < 0:
                has_neg_values = True
        except Exception: # Handle cases where min might not be defined or fails
             pass
    # Add check for sparse matrices more carefully if needed

    if has_neg_values or (not is_int_dtype and hasattr(adata_copy.X, 'data') and not np.allclose(np.modf(adata_copy.X.data)[0], 0)):
         log.warning("Data in adata.X does not look like raw counts (e.g., negative values or non-integers found). "
                     "Normalization and log1p transformation assume raw counts.")

    try:
        # Normalize counts per cell
        sc.pp.normalize_total(adata_copy, target_sum=target_sum, inplace=True) # Modifies inplace or on copy

        # Log-transform the data
        sc.pp.log1p(adata_copy) # Modifies inplace or on copy

        log.info("Normalization and log1p transformation complete.")

    except Exception as e:
        log.error(f"Error during normalization/log1p: {e}", exc_info=True)
        raise RuntimeError(f"Failed to normalize/log1p data: {e}") from e

    if not inplace:
        return adata_copy
    else:
        return None


def select_hvg(
    adata: ad.AnnData,
    n_top_genes: int | None = 3000,
    flavor: str = 'seurat_v3',
    subset: bool = True,
    inplace: bool = True
) -> ad.AnnData | None:
    """
    Selects Highly Variable Genes (HVGs).

    Uses scanpy.pp.highly_variable_genes. Stores HVG information in adata.var.
    Optionally subsets the AnnData object to only include HVGs.

    Args:
        adata: The annotated data matrix (typically after normalization and log1p).
        n_top_genes: Number of highly variable genes to select. Defaults to 3000.
                     Set to None to use other criteria like min/max mean/dispersion.
        flavor: Method for HVG selection ('seurat', 'cell_ranger', 'seurat_v3').
                'seurat_v3' is often recommended. Defaults to 'seurat_v3'.
        subset: If True, subset the AnnData object to only the selected HVGs.
                Defaults to True.
        inplace: Modify AnnData object inplace (adds info to adata.var and potentially subsets).
                 Defaults to True.

    Returns:
        If inplace=True, returns None.
        If inplace=False, returns a new AnnData object with HVG info in .var
        (and potentially subsetted if subset=True).

    Raises:
        ValueError: If flavor is invalid or n_top_genes is invalid.
        ImportError: If a required dependency for the flavor is missing (e.g., scikit-misc).
        Exception: Re-raises other exceptions caught from scanpy.
    """
    if not isinstance(adata, ad.AnnData):
        raise TypeError("Input must be an AnnData object.")
    if n_top_genes is not None and n_top_genes <= 0:
        raise ValueError("n_top_genes must be a positive integer or None.")

    log.info(f"Selecting highly variable genes (flavor='{flavor}', n_top_genes={n_top_genes}).")

    if inplace:
        adata_to_modify = adata
        try:
            # Run HVG selection, adding info to adata.var
            sc.pp.highly_variable_genes(
                adata_to_modify,
                flavor=flavor,
                n_top_genes=n_top_genes,
                inplace=True,  # Add results to .var inplace
                subset=False   # Do *not* subset yet
            )
            n_hvgs = int(adata_to_modify.var['highly_variable'].sum()) # Ensure integer
            log.info(f"Identified {n_hvgs} highly variable genes.")

            # Now, apply subsetting if requested, modifying the original adata
            if subset:
                log.info("Subsetting AnnData to selected HVGs.")
                adata_to_modify._inplace_subset_var(adata_to_modify.var['highly_variable']) # Use inplace subset method
                log.info(f"AnnData shape after HVG subsetting: {adata_to_modify.shape}")

            return None # Consistent return for inplace=True

        except Exception as e: # Catch any exception from scanpy
            log.error(f"Error during HVG selection: {e}", exc_info=True)
            raise e # *** Re-raise the original exception ***

    else: # inplace=False
        adata_copy = adata.copy()
        try:
             # Run HVG, adding info to the copy's .var
             sc.pp.highly_variable_genes(
                 adata_copy,
                 flavor=flavor,
                 n_top_genes=n_top_genes,
                 inplace=True, # Modifies adata_copy.var
                 subset=False
             )
             n_hvgs = int(adata_copy.var['highly_variable'].sum()) # Ensure integer
             log.info(f"Identified {n_hvgs} highly variable genes (on copy).")

             if subset:
                 log.info("Subsetting copy to selected HVGs.")
                 # Subset the copy and return the result
                 adata_subset = adata_copy[:, adata_copy.var['highly_variable']].copy()
                 log.info(f"Subsetted copy shape: {adata_subset.shape}")
                 return adata_subset
             else:
                 # Return the copy with HVG info in .var but not subsetted
                 return adata_copy

        except Exception as e: # Catch any exception from scanpy
            log.error(f"Error during HVG selection (on copy): {e}", exc_info=True)
            raise e # *** Re-raise the original exception ***


# --- Example Usage (for testing) ---
if __name__ == '__main__':
    # --- Setup ---
    try:
        # Try relative import first (works when run with python -m)
        from ..data.loader import load_data
        from .qc import calculate_qc_metrics, filter_cells_qc
    except ImportError:
        # Fallback for direct execution (less ideal but helpful)
        import sys
        print("Note: Running preprocess.py directly, attempting fallback imports.")
        # Assuming preprocess.py is in analysis/, need to go up two levels for project root
        module_dir = os.path.dirname(__file__)
        project_root = os.path.abspath(os.path.join(module_dir, '..', '..'))
        sys.path.insert(0, project_root) # Add project root to path
        from scrnaseq_agent.data.loader import load_data
        from scrnaseq_agent.analysis.qc import calculate_qc_metrics, filter_cells_qc

    # Define paths relative to project root
    module_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(module_dir, '..', '..'))
    TEST_10X_DIR = os.path.join(project_root, "test_data/filtered_gene_bc_matrices/hg19/")

    print("--- Testing Normalization and HVG Selection (Inplace) ---")
    adata_test = None
    try:
        if not os.path.exists(TEST_10X_DIR):
            print(f"SKIPPING test, 10x directory not found: {TEST_10X_DIR}")
        else:
            # 1. Load Data
            adata_test = load_data(TEST_10X_DIR)
            print(f"Loaded data shape: {adata_test.shape}")

            # 2. Calculate QC
            calculate_qc_metrics(adata_test, mito_gene_prefix="MT-", inplace=True)

            # 3. Filter QC
            print("Filtering cells...")
            filter_cells_qc(adata_test, min_genes=500, max_pct_mito=10.0, inplace=True)
            print(f"Shape after filtering: {adata_test.shape}")
            if adata_test.n_obs == 0:
                 raise ValueError("Filtering removed all cells, cannot proceed.")

            # 4. Normalize and Log1p (inplace)
            print("Normalizing and Log1p transforming (inplace)...")
            normalize_log1p(adata_test, target_sum=1e4, inplace=True)
            print("Data type after norm/log1p:", adata_test.X.dtype)
            print("Sum of first 5 rows/cols after norm/log1p:", adata_test.X[:5, :5].sum()) # Should not be 0.0 if transformed

            # 5. Select HVGs (inplace, subset=True)
            print(f"Selecting HVGs (inplace, subset=True, n=2000)...")
            select_hvg(adata_test, n_top_genes=2000, flavor='seurat_v3', subset=True, inplace=True)
            print(f"Shape after HVG selection: {adata_test.shape}")
            assert adata_test.n_vars == 2000, f"Expected 2000 genes after HVG selection, found {adata_test.n_vars}"
            assert 'highly_variable' in adata_test.var.columns
            assert adata_test.var['highly_variable'].all(), "Not all remaining vars marked as highly_variable"

    except Exception as e:
        print(f"\nError during Inplace Preprocessing test: {e}")
        import traceback
        traceback.print_exc()


    # --- Test non-inplace versions ---
    print("\n--- Testing non-inplace versions ---")
    adata_orig_for_noninplace = None
    adata_filt = None
    adata_norm = None
    adata_hvg_subset = None
    adata_hvg_nosubset = None
    try:
        if not os.path.exists(TEST_10X_DIR):
             print(f"SKIPPING non-inplace tests, 10x directory not found: {TEST_10X_DIR}")
        else:
            adata_orig_for_noninplace = load_data(TEST_10X_DIR)
            calculate_qc_metrics(adata_orig_for_noninplace, mito_gene_prefix="MT-", inplace=True)
            # Create the filtered object that serves as input for non-inplace norm/hvg
            adata_filt = filter_cells_qc(adata_orig_for_noninplace, min_genes=500, max_pct_mito=10.0, inplace=False)
            if adata_filt.n_obs == 0:
                raise ValueError("Filtering removed all cells in non-inplace test setup.")

            adata_filt_sum_before_norm = adata_filt.X.sum()

            # Test norm/log1p non-inplace
            print("\nTesting norm/log1p (non-inplace)...")
            adata_norm = normalize_log1p(adata_filt, inplace=False)
            assert adata_norm is not adata_filt, "normalize_log1p(inplace=False) did not return a new object"
            assert np.allclose(adata_filt.X.sum(), adata_filt_sum_before_norm), "adata_filt modified by normalize_log1p(inplace=False)"
            print("Shape after norm (non-inplace):", adata_norm.shape)
            assert not np.allclose(adata_norm.X.sum(), adata_filt_sum_before_norm), "Normalized data sum matches original raw sum"
            assert adata_norm.X.dtype == np.float32, "Normalized data type is not float32"

            # Test HVG non-inplace, subset=True
            print("\nTesting HVG subset (non-inplace)...")
            adata_norm_sum_before_hvg = adata_norm.X.sum() # Check this isn't modified
            adata_hvg_subset = select_hvg(adata_norm, n_top_genes=2000, subset=True, inplace=False)
            assert adata_hvg_subset is not adata_norm, "select_hvg(subset=True, inplace=False) did not return new object"
            assert np.allclose(adata_norm.X.sum(), adata_norm_sum_before_hvg), "adata_norm modified by select_hvg(subset=True, inplace=False)"
            assert adata_norm.shape == adata_filt.shape, "adata_norm shape changed unexpectedly"
            assert adata_hvg_subset.n_vars == 2000, "Non-inplace HVG subset has wrong var count"
            print("Shape after HVG subset (non-inplace):", adata_hvg_subset.shape)

            # Test HVG non-inplace, subset=False
            print("\nTesting HVG no subset (non-inplace)...")
            adata_norm_sum_before_hvg_nosub = adata_norm.X.sum() # Check this isn't modified
            adata_hvg_nosubset = select_hvg(adata_norm, n_top_genes=2000, subset=False, inplace=False)
            assert adata_hvg_nosubset is not adata_norm, "select_hvg(subset=False, inplace=False) did not return new object"
            assert np.allclose(adata_norm.X.sum(), adata_norm_sum_before_hvg_nosub), "adata_norm modified by select_hvg(subset=False, inplace=False)"
            assert adata_hvg_nosubset.shape == adata_norm.shape, "Non-inplace HVG no-subset changed shape"
            assert 'highly_variable' in adata_hvg_nosubset.var, "HVG info not added in non-inplace, no-subset"
            assert adata_hvg_nosubset.var['highly_variable'].sum() == 2000, "Wrong HVG count in non-inplace, no-subset"
            print("Shape after HVG nosubset (non-inplace):", adata_hvg_nosubset.shape)

    except Exception as e:
        print(f"\nError during Non-inplace Preprocessing test: {e}")
        import traceback
        traceback.print_exc()