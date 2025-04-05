# scrnaseq_agent/analysis/qc.py

import scanpy as sc
import anndata as ad
import logging
import numpy as np
import os # Make sure OS is imported

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) # Ensure basic config for direct runs


def calculate_qc_metrics(
    adata: ad.AnnData,
    mito_gene_prefix: str = "mt-",
    inplace: bool = True
) -> ad.AnnData | None:
    """
    Calculates standard QC metrics using scanpy.

    Adds the following to adata.obs:
        - 'n_genes_by_counts', 'total_counts'
        - 'total_counts_mt', 'pct_counts_mt' (if mito genes found)
    Adds relevant metrics to adata.var.

    Args:
        adata: The annotated data matrix.
        mito_gene_prefix: Prefix for mitochondrial genes. Defaults to "mt-". Use "MT-" for human.
        inplace: Modify AnnData object inplace. Defaults to True.

    Returns:
        If inplace=True, returns None. Otherwise, returns the modified AnnData object.
    """
    if not isinstance(adata, ad.AnnData):
        raise TypeError("Input must be an AnnData object.")

    log.info(f"Calculating QC metrics. Identifying mitochondrial genes with prefix: '{mito_gene_prefix}'")

    adata_copy = adata if inplace else adata.copy()

    # --- Calculate basic metrics first (always needed) ---
    # Pass an empty list [] instead of None
    try:
        sc.pp.calculate_qc_metrics(
            adata_copy,
            qc_vars=[], # *** CHANGE HERE: Use empty list instead of None ***
            percent_top=None,
            log1p=False,
            inplace=True
        )
        log.info("Calculated default QC metrics (genes/cell, counts/cell, etc.).")
    except Exception as e:
        log.error(f"Error calculating default QC metrics: {e}", exc_info=True)
        raise RuntimeError(f"Failed to calculate default QC metrics: {e}") from e


    # --- Now handle mitochondrial metrics specifically ---
    adata_copy.var['mt'] = adata_copy.var_names.str.startswith(mito_gene_prefix)
    n_mt_genes = np.sum(adata_copy.var['mt'])

    if n_mt_genes > 0:
        log.info(f"Found {n_mt_genes} mitochondrial genes. Calculating MT percentages.")
        # Call calculate_qc_metrics *again* but specifically for 'mt'
        try:
            sc.pp.calculate_qc_metrics(
                adata_copy,
                qc_vars=['mt'], # Pass the list ['mt'] here
                percent_top=None,
                log1p=False,
                inplace=True # Operate on the same adata_copy
            )
            log.info("Calculated mitochondrial QC metrics.")
        except Exception as e:
            log.error(f"Error calculating mitochondrial QC metrics: {e}", exc_info=True)
            raise RuntimeError(f"Failed to calculate mitochondrial QC metrics: {e}") from e
    else:
        log.warning(f"No mitochondrial genes found using prefix '{mito_gene_prefix}'. "
                    f"Columns 'total_counts_mt' and 'pct_counts_mt' may not be added or will be zero.")
        # Ensure the columns exist, filled with 0
        if 'total_counts_mt' not in adata_copy.obs.columns:
            adata_copy.obs['total_counts_mt'] = 0.0
        if 'pct_counts_mt' not in adata_copy.obs.columns:
            adata_copy.obs['pct_counts_mt'] = 0.0

    log.info("Finished QC metrics calculation step.")

    if not inplace:
        return adata_copy
    else:
        return None
# Add this function inside scrnaseq_agent/analysis/qc.py

def filter_cells_qc(
    adata: ad.AnnData,
    min_genes: int | None = 200,
    max_genes: int | None = None,
    min_counts: int | None = None,
    max_counts: int | None = None,
    max_pct_mito: float | None = 5.0,
    inplace: bool = True
) -> ad.AnnData | None:
    """
    Filters cells based on calculated QC metrics.

    Assumes `calculate_qc_metrics` has been run previously, adding
    'n_genes_by_counts', 'total_counts', and 'pct_counts_mt' to adata.obs.

    Args:
        adata: The annotated data matrix with QC metrics.
        min_genes: Minimum number of genes expressed required for a cell.
                   Defaults to 200. Set to None to disable.
        max_genes: Maximum number of genes expressed allowed for a cell.
                   Used to filter potential doublets. Defaults to None (disabled).
        min_counts: Minimum total counts required for a cell.
                    Defaults to None (disabled).
        max_counts: Maximum total counts allowed for a cell.
                    Used to filter potential doublets. Defaults to None (disabled).
        max_pct_mito: Maximum percentage of mitochondrial counts allowed for a cell.
                      Defaults to 5.0. Set to None to disable.
        inplace: Whether to modify the AnnData object inplace (subsets cells).
                 Defaults to True.

    Returns:
        If inplace=True, returns None. Otherwise, returns the filtered AnnData object.

    Raises:
        KeyError: If required QC columns ('n_genes_by_counts', 'total_counts',
                  'pct_counts_mt') are missing in adata.obs.
        ValueError: If thresholds are illogical (e.g., min > max).
    """
    required_cols = []
    if min_genes is not None or max_genes is not None:
        required_cols.append('n_genes_by_counts')
    if min_counts is not None or max_counts is not None:
        required_cols.append('total_counts')
    if max_pct_mito is not None:
        required_cols.append('pct_counts_mt')

    missing_cols = [col for col in required_cols if col not in adata.obs.columns]
    if missing_cols:
        raise KeyError(
            f"Missing required QC columns in adata.obs: {missing_cols}. "
            "Run calculate_qc_metrics first."
        )

    # --- Input Validation ---
    if min_genes is not None and max_genes is not None and min_genes > max_genes:
        raise ValueError(f"min_genes ({min_genes}) cannot be greater than max_genes ({max_genes}).")
    if min_counts is not None and max_counts is not None and min_counts > max_counts:
         raise ValueError(f"min_counts ({min_counts}) cannot be greater than max_counts ({max_counts}).")
    if max_pct_mito is not None and (max_pct_mito < 0 or max_pct_mito > 100):
        raise ValueError(f"max_pct_mito ({max_pct_mito}) must be between 0 and 100.")

    adata_orig = adata # Keep reference for logging if inplace=True
    if not inplace:
        adata = adata.copy() # Work on a copy if not inplace

    n_obs_start = adata.n_obs
    log.info(f"Starting filtering with {n_obs_start} cells.")

    # Apply filters sequentially using scanpy's functions
    if min_genes is not None:
        sc.pp.filter_cells(adata, min_genes=min_genes)
        log.info(f"Applied filter: min_genes = {min_genes}. Cells remaining: {adata.n_obs}")

    if max_genes is not None:
        sc.pp.filter_cells(adata, max_genes=max_genes)
        log.info(f"Applied filter: max_genes = {max_genes}. Cells remaining: {adata.n_obs}")

    if min_counts is not None:
        sc.pp.filter_cells(adata, min_counts=min_counts)
        log.info(f"Applied filter: min_counts = {min_counts}. Cells remaining: {adata.n_obs}")

    if max_counts is not None:
        sc.pp.filter_cells(adata, max_counts=max_counts)
        log.info(f"Applied filter: max_counts = {max_counts}. Cells remaining: {adata.n_obs}")

    if max_pct_mito is not None:
        # Scanpy's filter_cells doesn't directly support max_pct_mito
        # We apply this filter using boolean indexing
        mito_filter = adata.obs['pct_counts_mt'] < max_pct_mito
        if inplace:
             # Need to apply filter to the original object if inplace
             adata_orig._inplace_subset_obs(mito_filter)
             log.info(f"Applied filter: max_pct_mito = {max_pct_mito}. Cells remaining: {adata_orig.n_obs}")
             adata = adata_orig # Point adata back to original for correct final count if inplace
        else:
             adata = adata[mito_filter, :]
             log.info(f"Applied filter: max_pct_mito = {max_pct_mito}. Cells remaining: {adata.n_obs}")


    n_obs_end = adata.n_obs
    log.info(f"Filtering complete. Kept {n_obs_end} cells out of {n_obs_start} ({n_obs_end/n_obs_start*100:.2f}%).")

    if not inplace:
        return adata
    else:
        # If inplace, adata_orig was modified directly, return None
        return None



# --- Example Usage (for testing) ---
# The __main__ block remains the same
# --- Example Usage (for testing) ---
if __name__ == '__main__':
    # --- Setup ---
    try:
        # Try relative import first (works when run with python -m)
        from ..data.loader import load_data
    except ImportError:
        # Fallback for direct execution (less ideal but helpful)
        import sys
        print("Note: Running qc.py directly, attempting fallback import for loader.")
        # Assuming qc.py is in analysis/, need to go up two levels for project root
        module_dir = os.path.dirname(__file__)
        project_root = os.path.abspath(os.path.join(module_dir, '..', '..'))
        sys.path.insert(0, project_root) # Add project root to path
        from scrnaseq_agent.data.loader import load_data

    # Define paths relative to project root
    module_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(module_dir, '..', '..'))
    TEST_H5AD_PATH = os.path.join(project_root, "test_data/pbmc3k.h5ad")
    TEST_10X_DIR = os.path.join(project_root, "test_data/filtered_gene_bc_matrices/hg19/")

    # --- Test H5AD QC Calculation ---
    print("--- Testing QC calculation on H5AD data ---")
    adata_h5ad = None # Initialize to None
    try:
        if not os.path.exists(TEST_H5AD_PATH):
             print(f"SKIPPING H5AD test, file not found: {TEST_H5AD_PATH}")
        else:
            adata_h5ad = load_data(TEST_H5AD_PATH)
            print(f"Original adata_h5ad.obs columns: {adata_h5ad.obs.columns.tolist()}")
            calculate_qc_metrics(adata_h5ad, mito_gene_prefix="MT-", inplace=True)
            print(f"New adata_h5ad.obs columns: {adata_h5ad.obs.columns.tolist()}")
            print("adata_h5ad.obs head after QC calculation:")
            print(adata_h5ad.obs.head())
            assert 'pct_counts_mt' in adata_h5ad.obs.columns
            print("MT gene check (var['mt'].sum()):", adata_h5ad.var['mt'].sum())
            print("First 5 pct_counts_mt values:", adata_h5ad.obs['pct_counts_mt'].head().tolist())
    except Exception as e:
        print(f"Error during H5AD QC test: {e}")
        import traceback
        traceback.print_exc()


    # --- Test 10x QC Calculation AND Filtering ---
    print("\n--- Testing QC calculation and Filtering on 10x data ---")
    adata_10x = None # Initialize to None
    try:
        if not os.path.exists(TEST_10X_DIR):
            print(f"SKIPPING 10x test, directory not found: {TEST_10X_DIR}")
        else:
            # --- Load and Calculate QC ---
            adata_10x = load_data(TEST_10X_DIR)
            print(f"Original adata_10x.obs columns: {adata_10x.obs.columns.tolist()}")
            calculate_qc_metrics(adata_10x, mito_gene_prefix="MT-", inplace=True)
            print(f"New adata_10x.obs columns after QC calc: {adata_10x.obs.columns.tolist()}")
            print("MT gene check (var['mt'].sum()):", adata_10x.var['mt'].sum())
            print(f"Shape after QC calc: {adata_10x.shape}")

            # --- Test Filtering (Inplace) ---
            # Now that adata_10x exists and has QC metrics, test filtering
            print("\n--- Testing QC Filtering on 10x data (inplace) ---")
            adata_10x_copy = adata_10x.copy() # Use a copy for the inplace filtering test
            print(f"Shape before inplace filtering: {adata_10x_copy.shape}")
            filter_cells_qc(
                adata_10x_copy,
                min_genes=500,     # Example threshold
                max_pct_mito=10.0, # Example threshold
                inplace=True
            )
            print(f"Shape after inplace filtering: {adata_10x_copy.shape}")
            # Ensure the copy was actually modified
            assert adata_10x_copy.shape[0] < adata_10x.shape[0], "Inplace filter did not reduce cell count"

            # --- Test Filtering (Returning New Object) ---
            print("\n--- Testing QC Filtering on 10x data (returning new object) ---")
            print(f"Original object shape before non-inplace call: {adata_10x.shape}")
            adata_filtered = filter_cells_qc(
                adata_10x,         # Pass original object (which has QC metrics already)
                min_genes=500,
                max_pct_mito=10.0,
                inplace=False      # Get a new object back
            )
            print(f"Original object shape after non-inplace call: {adata_10x.shape}") # Should be unchanged
            print(f"Returned filtered object shape: {adata_filtered.shape}")
            # Ensure a new object was returned and it's smaller
            assert adata_filtered is not adata_10x
            assert adata_filtered.shape[0] < adata_10x.shape[0], "Returned filter object did not reduce cell count"
            assert adata_filtered.shape[0] == adata_10x_copy.shape[0], "Inplace and non-inplace filter results differ"


    except Exception as e:
        # This except block now correctly handles errors from loading, QC calc, OR filtering tests for 10x
        print(f"\nError during 10x QC/Filtering tests: {e}")
        import traceback
        traceback.print_exc()