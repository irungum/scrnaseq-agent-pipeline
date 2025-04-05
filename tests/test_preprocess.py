# tests/test_preprocess.py

import pytest
import anndata as ad
import os
import numpy as np
import scanpy as sc # Needed for comparison sometimes

# Import functions to be tested
from scrnaseq_agent.analysis.preprocess import normalize_log1p, select_hvg
# Import setup functions
from scrnaseq_agent.data.loader import load_data
from scrnaseq_agent.analysis.qc import calculate_qc_metrics, filter_cells_qc

# --- Test Data Paths ---
TEST_10X_DIR = "test_data/filtered_gene_bc_matrices/hg19/" # Adjust if needed

# --- Fixture to get filtered data (run QC and Filter once) ---
@pytest.fixture(scope="module") # scope="module" loads data only once per test file
def adata_filtered_10x():
    """Loads 10x data, calculates QC, and filters it."""
    if not os.path.exists(TEST_10X_DIR):
        pytest.skip(f"Test 10x directory not found at {TEST_10X_DIR}")
    adata = load_data(TEST_10X_DIR)
    calculate_qc_metrics(adata, mito_gene_prefix="MT-", inplace=True)
    # Use default filter values or define specific ones for consistency
    filter_cells_qc(
        adata,
        min_genes=200, # Use default min_genes
        max_pct_mito=10.0, # Be specific about mito filter used in tests
        inplace=True
    )
    # Important: Return a copy so modifications in tests don't affect other tests using this fixture
    return adata.copy()


# --- Tests for normalize_log1p ---

def test_normalize_log1p_inplace(adata_filtered_10x):
    """Tests normalization works inplace."""
    adata = adata_filtered_10x.copy() # Work on copy
    sum_before = adata.X.sum()
    dtype_before = adata.X.dtype

    result = normalize_log1p(adata, inplace=True)

    assert result is None, "Inplace function should return None"
    assert not np.allclose(adata.X.sum(), sum_before), "Data sum did not change after normalization"
    assert adata.X.dtype == np.float32, "Data type not float32 after log1p"
    # Check if counts are scaled (approx target_sum, but log1p makes direct check hard)
    # Check non-zero elements changed, safer for sparse
    if hasattr(adata.X, 'data'): # Check sparse data array
        # Check if *any* non-zero value changed - might be subtle with log1p+norm
        # This is not a perfect check but better than checking the corner slice
        assert not np.allclose(adata.X.data, adata_filtered_10x.X.data), "Sparse data values seem unchanged"
    elif hasattr(adata.X, 'toarray'): # Dense array check
         assert not np.allclose(adata.X, adata_filtered_10x.X), "Dense data values seem unchanged"


def test_normalize_log1p_return_copy(adata_filtered_10x):
    """Tests normalization returns a new object when inplace=False."""
    adata_orig = adata_filtered_10x # Use fixture directly
    sum_before = adata_orig.X.sum()

    adata_norm = normalize_log1p(adata_orig, inplace=False)

    assert adata_norm is not adata_orig, "Did not return a new object"
    assert isinstance(adata_norm, ad.AnnData)
    # Check original is unchanged
    assert np.allclose(adata_orig.X.sum(), sum_before), "Original data modified"
    # Check new object is transformed
    assert not np.allclose(adata_norm.X.sum(), sum_before), "Returned data sum unchanged"
    assert adata_norm.X.dtype == np.float32, "Returned data type not float32"


# --- Tests for select_hvg ---

def test_select_hvg_inplace_subset(adata_filtered_10x):
    """Tests HVG selection works inplace with subset=True."""
    adata = adata_filtered_10x.copy()
    normalize_log1p(adata, inplace=True) # Need normalized data for HVG
    n_vars_start = adata.n_vars

    n_hvgs = 2000
    result = select_hvg(adata, n_top_genes=n_hvgs, subset=True, inplace=True)

    assert result is None, "Inplace function should return None"
    assert adata.n_vars == n_hvgs, f"Expected {n_hvgs} vars after subsetting"
    assert adata.n_vars < n_vars_start, "Number of variables did not decrease"
    assert 'highly_variable' in adata.var.columns
    assert adata.var['highly_variable'].all()

def test_select_hvg_inplace_nosubset(adata_filtered_10x):
    """Tests HVG selection works inplace with subset=False."""
    adata = adata_filtered_10x.copy()
    normalize_log1p(adata, inplace=True)
    n_vars_start = adata.n_vars

    n_hvgs = 2000
    result = select_hvg(adata, n_top_genes=n_hvgs, subset=False, inplace=True)

    assert result is None, "Inplace function should return None"
    assert adata.n_vars == n_vars_start, "Shape changed unexpectedly with subset=False"
    assert 'highly_variable' in adata.var.columns
    assert adata.var['highly_variable'].sum() == n_hvgs

def test_select_hvg_return_copy_subset(adata_filtered_10x):
    """Tests HVG selection returns subsetted copy when inplace=False, subset=True."""
    adata_orig = adata_filtered_10x.copy()
    normalize_log1p(adata_orig, inplace=True)
    n_vars_start = adata_orig.n_vars
    sum_before = adata_orig.X.sum()

    n_hvgs = 2000
    adata_subset = select_hvg(adata_orig, n_top_genes=n_hvgs, subset=True, inplace=False)

    # Check original unchanged
    assert adata_orig.n_vars == n_vars_start
    assert np.allclose(adata_orig.X.sum(), sum_before)
    assert 'highly_variable' not in adata_orig.var or not adata_orig.var['highly_variable'].any()

    # Check returned subset
    assert adata_subset is not adata_orig
    assert isinstance(adata_subset, ad.AnnData)
    assert adata_subset.n_vars == n_hvgs
    assert 'highly_variable' in adata_subset.var.columns
    assert adata_subset.var['highly_variable'].all()

def test_select_hvg_return_copy_nosubset(adata_filtered_10x):
    """Tests HVG selection returns annotated copy when inplace=False, subset=False."""
    adata_orig = adata_filtered_10x.copy()
    normalize_log1p(adata_orig, inplace=True)
    n_vars_start = adata_orig.n_vars
    sum_before = adata_orig.X.sum()

    n_hvgs = 2000
    adata_annotated = select_hvg(adata_orig, n_top_genes=n_hvgs, subset=False, inplace=False)

    # Check original unchanged
    assert adata_orig.n_vars == n_vars_start
    assert np.allclose(adata_orig.X.sum(), sum_before)
    assert 'highly_variable' not in adata_orig.var or not adata_orig.var['highly_variable'].any()

    # Check returned annotated copy
    assert adata_annotated is not adata_orig
    assert isinstance(adata_annotated, ad.AnnData)
    assert adata_annotated.n_vars == n_vars_start # Shape unchanged
    assert 'highly_variable' in adata_annotated.var.columns
    assert adata_annotated.var['highly_variable'].sum() == n_hvgs


def test_select_hvg_missing_dependency(adata_filtered_10x, mocker):
    """ Tests that ImportError is raised if skmisc is missing (mocked). """
    adata = adata_filtered_10x.copy()
    normalize_log1p(adata, inplace=True)

    # Mock the main scanpy function to raise ImportError
    mocker.patch('scanpy.pp.highly_variable_genes',
                 side_effect=ImportError("Mock skmisc missing"))

    # Now test if our wrapper function correctly propagates this
    with pytest.raises(ImportError, match="Mock skmisc missing"):
         select_hvg(adata, n_top_genes=1000, flavor='seurat_v3')


def test_select_hvg_invalid_flavor(adata_filtered_10x):
     """ Tests ValueError for invalid flavor """
     adata = adata_filtered_10x.copy()
     normalize_log1p(adata, inplace=True)
     # Check that the call *inside* pytest.raises actually raises ValueError
     # Our wrapper now re-raises the original error from scanpy
     with pytest.raises(ValueError, match='`flavor` needs to be'):
         select_hvg(adata, n_top_genes=1000, flavor='invalid_flavor_name')