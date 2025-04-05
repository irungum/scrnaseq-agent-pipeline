# tests/test_qc.py

import pytest
import anndata as ad
import os
import numpy as np
import logging # *** ADD THIS IMPORT ***

# Import the function to be tested AND the loader needed for setup
from scrnaseq_agent.analysis.qc import calculate_qc_metrics
from scrnaseq_agent.data.loader import load_data

# --- Test Data Paths ---
# Use paths relative to the project root directory
TEST_H5AD_PATH = "test_data/pbmc3k.h5ad"
TEST_10X_DIR = "test_data/filtered_gene_bc_matrices/hg19/" # Adjust if needed

# --- Fixtures to load data once for multiple tests ---

@pytest.fixture(scope="module") # scope="module" loads data only once per test file
def adata_h5ad():
    """Loads the H5AD test data."""
    if not os.path.exists(TEST_H5AD_PATH):
        pytest.skip(f"Test H5AD file not found at {TEST_H5AD_PATH}")
    return load_data(TEST_H5AD_PATH)

@pytest.fixture(scope="module")
def adata_10x():
    """Loads the 10x test data."""
    if not os.path.exists(TEST_10X_DIR):
        pytest.skip(f"Test 10x directory not found at {TEST_10X_DIR}")
    return load_data(TEST_10X_DIR)


# --- Test Functions ---

def test_qc_h5ad_runs_and_adds_cols(adata_h5ad):
    """Tests that QC calculation runs on H5AD and adds expected columns."""
    adata = adata_h5ad.copy() # Work on a copy to avoid modifying fixture
    original_obs_cols = set(adata.obs.columns)
    original_var_cols = set(adata.var.columns)

    # Run QC calculation (use correct prefix "MT-" for human pbmc3k)
    calculate_qc_metrics(adata, mito_gene_prefix="MT-", inplace=True)

    new_obs_cols = set(adata.obs.columns)
    new_var_cols = set(adata.var.columns)

    # Check specific columns were added to obs
    expected_obs_additions = {'n_genes_by_counts', 'total_counts', 'total_counts_mt', 'pct_counts_mt'}
    assert expected_obs_additions.issubset(new_obs_cols), "Expected QC columns not added to adata.obs"

    # Check specific column added to var
    assert 'mt' in new_var_cols, "'mt' column not added to adata.var"
    # Check other var columns are also present (names might vary slightly by scanpy version)
    assert 'n_cells_by_counts' in new_var_cols
    assert 'mean_counts' in new_var_cols
    assert 'pct_dropout_by_counts' in new_var_cols
    assert 'total_counts' in new_var_cols # Note: 'total_counts' is also in var

def test_qc_10x_runs_and_adds_cols(adata_10x):
    """Tests that QC calculation runs on 10x data and adds expected columns."""
    adata = adata_10x.copy() # Work on a copy
    original_obs_cols = set(adata.obs.columns)
    original_var_cols = set(adata.var.columns)

    calculate_qc_metrics(adata, mito_gene_prefix="MT-", inplace=True)

    new_obs_cols = set(adata.obs.columns)
    new_var_cols = set(adata.var.columns)

    expected_obs_additions = {'n_genes_by_counts', 'total_counts', 'total_counts_mt', 'pct_counts_mt'}
    assert expected_obs_additions.issubset(new_obs_cols), "Expected QC columns not added to adata.obs"

    assert 'mt' in new_var_cols, "'mt' column not added to adata.var"
    assert 'n_cells_by_counts' in new_var_cols
    assert 'mean_counts' in new_var_cols
    assert 'pct_dropout_by_counts' in new_var_cols
    assert 'total_counts' in new_var_cols

def test_qc_h5ad_mt_values_correct(adata_h5ad):
    """Tests the mitochondrial calculation specifically for the H5AD case (no MT- genes)."""
    adata = adata_h5ad.copy()
    calculate_qc_metrics(adata, mito_gene_prefix="MT-", inplace=True)

    # For this specific H5AD, we expect no 'MT-' genes
    assert adata.var['mt'].sum() == 0, "Expected 0 'MT-' genes in pbmc3k H5AD var names"
    assert 'pct_counts_mt' in adata.obs.columns, "'pct_counts_mt' column missing"
    # Check that all values in the column are indeed 0
    assert np.all(adata.obs['pct_counts_mt'] == 0), "Expected all pct_counts_mt to be 0"

def test_qc_10x_mt_values_correct(adata_10x):
    """Tests the mitochondrial calculation specifically for the 10x case."""
    adata = adata_10x.copy()
    calculate_qc_metrics(adata, mito_gene_prefix="MT-", inplace=True)

    # For pbmc3k 10x data, we expect 13 'MT-' genes
    assert adata.var['mt'].sum() == 13, "Expected 13 'MT-' genes in pbmc3k 10x data"
    assert 'pct_counts_mt' in adata.obs.columns
    # Check that values are reasonable (e.g., between 0 and 100)
    assert adata.obs['pct_counts_mt'].min() >= 0
    assert adata.obs['pct_counts_mt'].max() <= 100 # Max could be higher in stressed cells, but 100 is safe upper bound
    # Check at least some non-zero values exist
    assert adata.obs['pct_counts_mt'].sum() > 0, "Expected some non-zero MT percentages"

def test_qc_wrong_prefix_warning(adata_10x, caplog):
    """Tests that a warning is logged if the wrong prefix finds no genes."""
    adata = adata_10x.copy()
    # Use a prefix that definitely won't match
    with caplog.at_level(logging.WARNING): # This line needs 'logging' defined
        calculate_qc_metrics(adata, mito_gene_prefix="WRONG_PREFIX-", inplace=True)

    assert "No mitochondrial genes found using prefix 'WRONG_PREFIX-'" in caplog.text
    # Check that pct_counts_mt is still added and is 0
    assert 'pct_counts_mt' in adata.obs.columns
    assert np.all(adata.obs['pct_counts_mt'] == 0)

def test_qc_inplace_false(adata_10x):
    """Tests that inplace=False returns a new object and leaves original unchanged."""
    adata_original = adata_10x.copy()
    original_obs_cols = set(adata_original.obs.columns)

    # Run with inplace=False
    adata_modified = calculate_qc_metrics(adata_original, mito_gene_prefix="MT-", inplace=False)

    # Check original is unchanged
    assert set(adata_original.obs.columns) == original_obs_cols

    # Check modified object is different and has new columns
    assert adata_modified is not adata_original
    assert isinstance(adata_modified, ad.AnnData)
    expected_obs_additions = {'n_genes_by_counts', 'total_counts', 'total_counts_mt', 'pct_counts_mt'}
    assert expected_obs_additions.issubset(set(adata_modified.obs.columns))

def test_qc_invalid_input():
    """Tests that passing non-AnnData input raises TypeError."""
    with pytest.raises(TypeError):
        calculate_qc_metrics("not an anndata object")
# Add these test functions to tests/test_qc.py

# Import the new function
from scrnaseq_agent.analysis.qc import filter_cells_qc

# --- Tests for filter_cells_qc ---

# Use the existing adata_10x fixture which loads the data

def test_filter_cells_inplace(adata_10x):
    """Tests filtering works inplace."""
    # Calculate QC first on a copy specific to this test
    adata = adata_10x.copy()
    calculate_qc_metrics(adata, mito_gene_prefix="MT-", inplace=True)
    n_obs_start = adata.n_obs

    # Apply filtering inplace
    filter_cells_qc(
        adata,
        min_genes=500,
        max_pct_mito=10.0,
        inplace=True
    )

    # Check cells were removed
    assert adata.n_obs < n_obs_start
    # Check against expected number based on manual run (can be fragile)
    assert adata.n_obs == 2481, "Unexpected cell count after filtering"

def test_filter_cells_return_copy(adata_10x):
    """Tests filtering returns a new object when inplace=False."""
     # Calculate QC first on a copy specific to this test
    adata_orig = adata_10x.copy()
    calculate_qc_metrics(adata_orig, mito_gene_prefix="MT-", inplace=True)
    n_obs_start = adata_orig.n_obs

    # Apply filtering returning new object
    adata_filtered = filter_cells_qc(
        adata_orig,
        min_genes=500,
        max_pct_mito=10.0,
        inplace=False
    )

    # Check original is unchanged
    assert adata_orig.n_obs == n_obs_start
    # Check filtered is new object and has fewer cells
    assert adata_filtered is not adata_orig
    assert isinstance(adata_filtered, ad.AnnData)
    assert adata_filtered.n_obs < n_obs_start
    assert adata_filtered.n_obs == 2481, "Unexpected cell count after filtering"

def test_filter_cells_no_filters(adata_10x):
    """Tests that no cells are filtered if all thresholds are None."""
    adata = adata_10x.copy()
    calculate_qc_metrics(adata, mito_gene_prefix="MT-", inplace=True)
    n_obs_start = adata.n_obs

    filter_cells_qc(
        adata,
        min_genes=None,
        max_genes=None,
        min_counts=None,
        max_counts=None,
        max_pct_mito=None,
        inplace=True
    )
    assert adata.n_obs == n_obs_start, "Cells were filtered when no thresholds were set"

def test_filter_cells_missing_qc_cols(adata_10x):
    """Tests that KeyError is raised if QC cols are missing."""
    adata = adata_10x.copy() # Use raw data without QC metrics
    with pytest.raises(KeyError, match="Missing required QC columns"):
        filter_cells_qc(adata, min_genes=500) # Try to filter without n_genes_by_counts

    with pytest.raises(KeyError, match="Missing required QC columns"):
        filter_cells_qc(adata, max_pct_mito=10.0) # Try to filter without pct_counts_mt

def test_filter_cells_invalid_thresholds(adata_10x):
    """Tests that ValueError is raised for illogical thresholds."""
    adata = adata_10x.copy()
    calculate_qc_metrics(adata, mito_gene_prefix="MT-", inplace=True)

    with pytest.raises(ValueError, match="min_genes .* cannot be greater than max_genes"):
        filter_cells_qc(adata, min_genes=1000, max_genes=500)

    with pytest.raises(ValueError, match="max_pct_mito .* must be between 0 and 100"):
        filter_cells_qc(adata, max_pct_mito=101)

    with pytest.raises(ValueError, match="max_pct_mito .* must be between 0 and 100"):
        filter_cells_qc(adata, max_pct_mito=-5)

# Add more tests for min/max counts if desired