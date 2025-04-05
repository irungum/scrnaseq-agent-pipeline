# tests/test_dimred.py


import pytest
import anndata as ad
import numpy as np
import scanpy as sc
from scrnaseq_agent.analysis.dimred import reduce_dimensionality
import logging
import os
import pandas as pd # Make sure pandas is imported

# Configure logging for tests (can be silenced or adjusted)
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

# --- Fixtures ---

@pytest.fixture(scope="module") # Use module scope for potentially expensive preprocessing
def preprocessed_adata_for_pca() -> ad.AnnData:
    """
    Provides a realistic, preprocessed AnnData object suitable for PCA testing.
    Loads data, applies QC, normalization, HVG selection, and scaling.
    Returns None if test data is not found.
    Uses 'MT-' as the mitochondrial gene prefix.
    """
    # Construct path relative to the test file location
    test_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(test_dir, '..'))
    TEST_10X_DIR = os.path.join(project_root, "test_data", "filtered_gene_bc_matrices", "hg19")

    if not os.path.exists(TEST_10X_DIR):
        log.warning(f"Test data not found at {TEST_10X_DIR}, skipping PCA tests requiring it.")
        pytest.skip(f"Test data not found: {TEST_10X_DIR}")
        return None

    log.info("Setting up preprocessed AnnData for PCA tests...")
    # Load data (use var_names='gene_symbols' to match 'MT-')
    adata = sc.read_10x_mtx(TEST_10X_DIR, var_names='gene_symbols', cache=True)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    # --- CORRECTED QC STEPS ---
    # 1. Identify mitochondrial genes (using 'MT-' prefix)
    adata.var['mt'] = adata.var_names.str.startswith('MT-')

    # 2. Calculate QC metrics using the 'mt' column we just created
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    # 3. Apply filtering (example thresholds)
    sc.pp.filter_cells(adata, min_genes=200)
    # Filter by pct_counts_mt using the column calculated above
    adata = adata[adata.obs.pct_counts_mt < 15, :].copy()
    sc.pp.filter_genes(adata, min_cells=3)
    # --- END CORRECTED QC ---

    # Continue with standard preprocessing
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat_v3', subset=True)
    sc.pp.scale(adata, max_value=10) # Scaling is important for PCA

    # Ensure X is dense float for PCA
    if hasattr(adata.X, "toarray"):
        adata.X = adata.X.toarray()
    adata.X = adata.X.astype(np.float32)

    log.info(f"Preprocessed AnnData ready for PCA tests, shape: {adata.shape}")
    return adata

# --- Test Cases ---

def test_reduce_dimensionality_success_inplace(preprocessed_adata_for_pca):
    """Test basic PCA execution with inplace=True."""
    if preprocessed_adata_for_pca is None: pytest.skip("Test data not available")
    adata = preprocessed_adata_for_pca.copy() # Work on a copy within the test
    n_comps_test = 30
    random_seed = 42

    result = reduce_dimensionality(
        adata,
        n_comps=n_comps_test,
        random_state=random_seed,
        inplace=True
    )

    assert result is None, "Should return None when inplace=True"
    assert 'X_pca' in adata.obsm, "'X_pca' should be added to adata.obsm"
    assert isinstance(adata.obsm['X_pca'], np.ndarray), "'X_pca' should be a numpy array"
    assert adata.obsm['X_pca'].shape == (adata.n_obs, n_comps_test), f"Shape should be (n_obs, {n_comps_test})"
    assert np.issubdtype(adata.obsm['X_pca'].dtype, np.floating), "'X_pca' dtype should be float"
    assert 'pca' in adata.uns, "'pca' should be added to adata.uns"
    assert 'variance_ratio' in adata.uns['pca'], "'variance_ratio' should be in adata.uns['pca']"
    assert len(adata.uns['pca']['variance_ratio']) == n_comps_test
    assert 'PCs' in adata.varm, "'PCs' (loadings) should be in adata.varm"
    assert adata.varm['PCs'].shape == (adata.n_vars, n_comps_test)


def test_reduce_dimensionality_success_not_inplace(preprocessed_adata_for_pca):
    """Test basic PCA execution with inplace=False."""
    if preprocessed_adata_for_pca is None: pytest.skip("Test data not available")
    adata_orig = preprocessed_adata_for_pca.copy()
    n_comps_test = 40
    random_seed = 42

    adata_new = reduce_dimensionality(
        adata_orig,
        n_comps=n_comps_test,
        random_state=random_seed,
        inplace=False
    )

    assert isinstance(adata_new, ad.AnnData), "Should return an AnnData object"
    assert adata_new is not adata_orig, "Returned object should be a different instance"
    assert 'X_pca' in adata_new.obsm, "'X_pca' should be in returned adata.obsm"
    assert adata_new.obsm['X_pca'].shape == (adata_new.n_obs, n_comps_test)
    assert 'pca' in adata_new.uns, "'pca' should be in returned adata.uns"
    assert 'PCs' in adata_new.varm

    assert 'X_pca' not in adata_orig.obsm, "Original adata.obsm should NOT contain 'X_pca'"
    assert 'pca' not in adata_orig.uns, "Original adata.uns should NOT contain 'pca'"


def test_reduce_dimensionality_invalid_input_type():
    """Test error handling for incorrect input object type."""
    with pytest.raises(TypeError, match="Input 'adata' must be an AnnData object"):
        reduce_dimensionality([1, 2, 3])


def test_reduce_dimensionality_invalid_n_comps(preprocessed_adata_for_pca):
    """Test error handling for invalid n_comps values."""
    if preprocessed_adata_for_pca is None: pytest.skip("Test data not available")
    adata = preprocessed_adata_for_pca[:10, :50].copy() # Smaller subset

    with pytest.raises(ValueError, match="Argument 'n_comps' must be a positive integer"):
        reduce_dimensionality(adata.copy(), n_comps=0)
    with pytest.raises(ValueError, match="Argument 'n_comps' must be a positive integer"):
        reduce_dimensionality(adata.copy(), n_comps=-10)
    with pytest.raises(ValueError, match="Argument 'n_comps' must be a positive integer"):
        reduce_dimensionality(adata.copy(), n_comps=10.5)


def test_reduce_dimensionality_missing_x():
    """Test error handling when adata.X is missing (is None)."""
    # Create AnnData without X or set X to None explicitly
    adata = ad.AnnData(obs=pd.DataFrame(index=[f"cell_{i}" for i in range(10)]),
                       var=pd.DataFrame(index=[f"gene_{i}" for i in range(50)]))
    adata.X = None # Explicitly set X to None
    with pytest.raises(AttributeError, match="Cannot perform PCA: AnnData object does not have a suitable '.X' attribute."):
        reduce_dimensionality(adata)


def test_reduce_dimensionality_n_comps_too_large_warning(preprocessed_adata_for_pca):
    """Test warning and adjustment when n_comps >= min(n_obs, n_vars)."""
    if preprocessed_adata_for_pca is None: pytest.skip("Test data not available")
    # Create a small adata where adjustment is needed
    adata = preprocessed_adata_for_pca[:20, :15].copy() # n_obs=20, n_vars=15 -> min_dim=15
    min_dim = min(adata.shape)
    large_n_comps = min_dim + 5 # e.g., 20

    # Expect a warning and successful run with adjusted n_comps = min_dim - 1
    with pytest.warns(UserWarning, match=f"Adjusting n_comps to {min_dim - 1}"):
         result = reduce_dimensionality(adata, n_comps=large_n_comps, inplace=True)

    assert result is None
    assert 'X_pca' in adata.obsm
    expected_n_comps = min_dim - 1 # e.g., 14
    assert adata.obsm['X_pca'].shape == (adata.n_obs, expected_n_comps)
    assert len(adata.uns['pca']['variance_ratio']) == expected_n_comps


def test_reduce_dimensionality_reproducibility(preprocessed_adata_for_pca):
    """Test that the random_state ensures reproducible results."""
    if preprocessed_adata_for_pca is None: pytest.skip("Test data not available")
    adata1 = preprocessed_adata_for_pca.copy()
    adata2 = preprocessed_adata_for_pca.copy()
    n_comps_test = 25
    random_seed = 123

    reduce_dimensionality(adata1, n_comps=n_comps_test, random_state=random_seed, inplace=True)
    reduce_dimensionality(adata2, n_comps=n_comps_test, random_state=random_seed, inplace=True)

    np.testing.assert_array_almost_equal(
        adata1.obsm['X_pca'],
        adata2.obsm['X_pca'],
        decimal=5, err_msg="X_pca results differ for same random_state"
    )
    np.testing.assert_array_almost_equal(
        adata1.varm['PCs'],
        adata2.varm['PCs'],
        decimal=5, err_msg="PCs (loadings) differ for same random_state"
    )

    # Test different seed produces different results
    adata3 = preprocessed_adata_for_pca.copy()
    reduce_dimensionality(adata3, n_comps=n_comps_test, random_state=random_seed + 1, inplace=True)
    diff = np.sum(np.abs(adata1.obsm['X_pca'] - adata3.obsm['X_pca']))
    assert diff > 1e-5, "PCA results should differ for different random_states"

# Need pandas for the missing X test fixture creation
import pandas as pd 