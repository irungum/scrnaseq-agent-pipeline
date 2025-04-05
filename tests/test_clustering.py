# tests/test_clustering.py

import pytest
import anndata as ad
import numpy as np
import scanpy as sc
import pandas as pd # Required for cluster counts comparison
from scrnaseq_agent.analysis.dimred import reduce_dimensionality
from scrnaseq_agent.analysis.clustering import perform_clustering
import logging
import os

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

# --- Fixtures ---

@pytest.fixture(scope="module")
def adata_with_pca(preprocessed_adata_for_pca) -> ad.AnnData | None:
    """
    Provides AnnData object after QC, preprocessing, scaling, and PCA.
    Ready for clustering tests. Skips if base fixture is skipped.
    """
    if preprocessed_adata_for_pca is None:
        pytest.skip("Base preprocessed data not available for PCA step.")
        return None

    adata_pca = preprocessed_adata_for_pca.copy()
    log.info("Running PCA on preprocessed data for clustering fixture...")
    reduce_dimensionality(adata_pca, n_comps=50, inplace=True, random_state=0)

    if 'X_pca' not in adata_pca.obsm:
         pytest.fail("Fixture setup failed: X_pca not found after running reduce_dimensionality.")
    if 'pca' not in adata_pca.uns:
         pytest.fail("Fixture setup failed: pca not found in uns after running reduce_dimensionality.")

    log.info(f"AnnData with PCA ready for clustering tests, shape: {adata_pca.shape}")
    return adata_pca


# --- Test Cases ---

def test_clustering_success_inplace(adata_with_pca):
    """Test basic clustering and UMAP (default) execution with inplace=True."""
    if adata_with_pca is None: pytest.skip("PCA data not available")
    adata = adata_with_pca.copy() # Work on a copy
    n_neighbors_test = 10
    resolution_test = 0.5
    random_seed = 42
    key_added = 'leiden_test'
    umap_key = 'X_umap' # Default UMAP key

    result = perform_clustering(
        adata,
        use_rep='X_pca',
        n_neighbors=n_neighbors_test,
        resolution=resolution_test,
        random_state=random_seed,
        leiden_key_added=key_added,
        calculate_umap=True, # Explicitly testing the default
        inplace=True
    )

    assert result is None, "Should return None when inplace=True"
    # Check neighbors results
    assert 'neighbors' in adata.uns
    assert 'connectivities' in adata.obsp
    assert 'distances' in adata.obsp
    if 'params' in adata.uns['neighbors']:
         assert adata.uns['neighbors']['params']['n_neighbors'] == n_neighbors_test

    # Check Leiden results
    assert key_added in adata.obs
    assert isinstance(adata.obs[key_added].dtype, pd.CategoricalDtype)
    assert len(adata.obs[key_added].cat.categories) > 0
    assert key_added in adata.uns # Check for params entry
    if key_added in adata.uns and 'params' in adata.uns[key_added]:
        assert adata.uns[key_added]['params']['resolution'] == resolution_test

    # --- Check UMAP results (as calculate_umap=True by default/explicit) ---
    assert umap_key in adata.obsm, f"'{umap_key}' should be added to .obsm"
    assert adata.obsm[umap_key].shape == (adata.n_obs, 2), f"'{umap_key}' should have shape (n_obs, 2)"
    assert 'umap' in adata.uns, "'.uns['umap']' should be added"
    if 'umap' in adata.uns and 'params' in adata.uns['umap']:
         assert adata.uns['umap']['params']['random_state'] == random_seed


def test_clustering_success_not_inplace(adata_with_pca):
    """Test basic clustering and UMAP (default) execution with inplace=False."""
    if adata_with_pca is None: pytest.skip("PCA data not available")
    adata_orig = adata_with_pca.copy()
    n_neighbors_test = 20
    resolution_test = 1.2
    random_seed = 42
    key_added = 'leiden_new'
    umap_key = 'X_umap'

    # Store original state markers
    orig_obs_cols = set(adata_orig.obs.columns)
    orig_uns_keys = set(adata_orig.uns.keys())
    orig_obsm_keys = set(adata_orig.obsm.keys()) # Check obsm for UMAP
    orig_obsp_keys = set(adata_orig.obsp.keys())

    adata_new = perform_clustering(
        adata_orig,
        use_rep='X_pca',
        n_neighbors=n_neighbors_test,
        resolution=resolution_test,
        random_state=random_seed,
        leiden_key_added=key_added,
        calculate_umap=True, # Explicitly testing default
        inplace=False
    )

    # --- Assertions on returned object ---
    assert isinstance(adata_new, ad.AnnData)
    assert adata_new is not adata_orig
    # Check neighbors results
    assert 'neighbors' in adata_new.uns
    assert 'connectivities' in adata_new.obsp
    assert 'distances' in adata_new.obsp
    # Check Leiden results
    assert key_added in adata_new.obs
    assert isinstance(adata_new.obs[key_added].dtype, pd.CategoricalDtype)
    # --- Check UMAP results ---
    assert umap_key in adata_new.obsm
    assert adata_new.obsm[umap_key].shape == (adata_new.n_obs, 2)
    assert 'umap' in adata_new.uns

    # --- Assertions on original object (should be unchanged) ---
    assert set(adata_orig.obs.columns) == orig_obs_cols
    assert set(adata_orig.uns.keys()) == orig_uns_keys
    assert set(adata_orig.obsm.keys()) == orig_obsm_keys # Ensure X_umap NOT added here
    assert set(adata_orig.obsp.keys()) == orig_obsp_keys


def test_clustering_skip_umap(adata_with_pca):
    """Test that UMAP is skipped when calculate_umap=False."""
    if adata_with_pca is None: pytest.skip("PCA data not available")
    adata = adata_with_pca.copy()
    umap_key = 'X_umap'

    result = perform_clustering(
        adata,
        use_rep='X_pca',
        calculate_umap=False, # Explicitly skip UMAP
        inplace=True
    )

    assert result is None
    # Check Leiden results are still present
    assert 'leiden' in adata.obs # Default leiden key
    assert 'neighbors' in adata.uns
    # --- Check UMAP results are ABSENT ---
    assert umap_key not in adata.obsm, f"'{umap_key}' should NOT be in .obsm when calculate_umap=False"
    assert 'umap' not in adata.uns, "'.uns['umap']' should NOT be added when calculate_umap=False"


def test_clustering_reproducibility(adata_with_pca):
    """Test that random_state ensures reproducible clustering AND UMAP."""
    if adata_with_pca is None: pytest.skip("PCA data not available")
    adata1 = adata_with_pca.copy()
    adata2 = adata_with_pca.copy()
    adata3 = adata_with_pca.copy()
    random_seed = 123
    key = 'leiden_rep'
    umap_key = 'X_umap'

    # Run 1 & 2 with same seed
    perform_clustering(adata1, random_state=random_seed, leiden_key_added=key, calculate_umap=True, inplace=True)
    perform_clustering(adata2, random_state=random_seed, leiden_key_added=key, calculate_umap=True, inplace=True)

    # Run 3 with different seed
    perform_clustering(adata3, random_state=random_seed + 1, leiden_key_added=key, calculate_umap=True, inplace=True)

    # Assert Leiden results 1 & 2 are identical
    pd.testing.assert_series_equal(
        adata1.obs[key],
        adata2.obs[key],
        check_names=False,
        check_categorical=True,
    )
    # Assert UMAP results 1 & 2 are identical (using numpy for array comparison)
    np.testing.assert_array_almost_equal(
        adata1.obsm[umap_key],
        adata2.obsm[umap_key],
        decimal=5, # UMAP can have small float differences
        err_msg="UMAP results should be identical for the same random_state"
    )


    # Assert Leiden results 1 & 3 are different
    assert not adata1.obs[key].equals(adata3.obs[key]), \
        "Leiden results should differ for different random_states"
    # Assert UMAP results 1 & 3 are different
    umap_diff = np.sum(np.abs(adata1.obsm[umap_key] - adata3.obsm[umap_key]))
    assert umap_diff > 1e-5, "UMAP results should differ for different random_states"


def test_clustering_missing_representation(adata_with_pca):
    """Test error if use_rep is missing."""
    if adata_with_pca is None: pytest.skip("PCA data not available")
    adata = adata_with_pca.copy()
    if 'X_nonexistent' in adata.obsm:
        del adata.obsm['X_nonexistent']
    with pytest.raises(KeyError, match="Representation 'X_nonexistent' not found"):
        perform_clustering(adata, use_rep='X_nonexistent')


def test_clustering_invalid_input_type():
    """Test error handling for incorrect input object type."""
    with pytest.raises(TypeError, match="Input 'adata' must be an AnnData object"):
        perform_clustering([1, 2, 3])


def test_clustering_invalid_params(adata_with_pca):
    """Test error handling for invalid parameter values."""
    if adata_with_pca is None: pytest.skip("PCA data not available")
    adata = adata_with_pca.copy()
    with pytest.raises(ValueError, match="Argument 'n_neighbors' must be a positive integer"):
        perform_clustering(adata.copy(), n_neighbors=0)
    with pytest.raises(ValueError, match="Argument 'n_neighbors' must be a positive integer"):
        perform_clustering(adata.copy(), n_neighbors=-5)
    with pytest.raises(ValueError, match="Argument 'resolution' must be a positive number"):
        perform_clustering(adata.copy(), resolution=0)
    with pytest.raises(ValueError, match="Argument 'resolution' must be a positive number"):
        perform_clustering(adata.copy(), resolution=-1.0)


# Import fixture from test_dimred
try:
    from .test_dimred import preprocessed_adata_for_pca
except ImportError:
    log.warning("Could not directly import preprocessed_adata_for_pca fixture.")
    pass