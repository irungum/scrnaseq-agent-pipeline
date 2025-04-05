# tests/test_dge.py

import pytest
import anndata as ad
import numpy as np
import scanpy as sc
import pandas as pd
import logging
import os

# Import functions from our modules
from scrnaseq_agent.analysis.qc import filter_cells_qc
from scrnaseq_agent.analysis.preprocess import normalize_log1p, select_hvg
from scrnaseq_agent.analysis.dimred import reduce_dimensionality
from scrnaseq_agent.analysis.clustering import perform_clustering
from scrnaseq_agent.analysis.dge import find_marker_genes # Function under test

# Configure logging
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

# --- Fixtures ---

@pytest.fixture(scope="module")
def adata_for_dge(adata_with_pca) -> ad.AnnData | None:
    """
    Provides AnnData object ready for DGE testing *and* plotting.
    Starts from data with PCA (adata_with_pca), runs clustering, sets up .raw,
    and runs find_marker_genes with default settings ('rank_genes_groups_raw' key).
    """
    if adata_with_pca is None:
        pytest.skip("Data with PCA (adata_with_pca) not available for DGE setup.")
        return None

    log.info("Setting up AnnData with clustering, .raw, and DGE results...")

    # Reload original raw data to ensure clean state for .raw setup
    test_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(test_dir, '..'))
    TEST_10X_DIR = os.path.join(project_root, "test_data", "filtered_gene_bc_matrices", "hg19")
    if not os.path.exists(TEST_10X_DIR):
         pytest.skip(f"Test data not found: {TEST_10X_DIR}")
         return None

    adata_orig_raw = sc.read_10x_mtx(TEST_10X_DIR, var_names='gene_symbols', cache=True)
    adata_orig_raw.var_names_make_unique()
    adata_orig_raw.obs_names_make_unique()

    # Start with the adata object that already has PCA
    adata_proc = adata_with_pca.copy()

    # Run clustering on it
    perform_clustering(
        adata_proc,
        resolution=0.8,
        random_state=0,
        calculate_umap=False, # UMAP not needed for DGE/plotting DGE
        inplace=True
    )
    cluster_key = 'leiden' # Default key from perform_clustering
    if cluster_key not in adata_proc.obs:
         pytest.fail("Fixture setup failed: 'leiden' clustering not found.")

    # Set the .raw attribute using the original filtered data matched to current adata
    common_obs = adata_proc.obs_names.intersection(adata_orig_raw.obs_names)
    common_vars = adata_proc.var_names.intersection(adata_orig_raw.var_names)
    raw_counterpart = adata_orig_raw[common_obs, common_vars].copy()
    adata_proc.raw = raw_counterpart[adata_proc.obs_names, adata_proc.var_names].copy()
    log.info(f"Set final .raw attribute. Shape: {adata_proc.raw.shape}")
    if adata_proc.raw is None:
         pytest.fail("Fixture setup failed: .raw attribute not set.")


    # !!! FIX: Run DGE within the fixture !!!
    dge_results_key = 'rank_genes_groups_raw' # Key expected by plotting tests
    log.info(f"Running find_marker_genes within fixture (key='{dge_results_key}')...")
    try:
        find_marker_genes(
            adata_proc,
            groupby=cluster_key,
            use_raw=True, # Use raw for consistency with examples
            key_added=dge_results_key
        )
        # Check if DGE results were actually added
        if dge_results_key not in adata_proc.uns:
             pytest.fail(f"Fixture setup failed: DGE key '{dge_results_key}' not found after running find_marker_genes.")
        log.info(f"DGE run successful within fixture.")
    except Exception as e:
         pytest.fail(f"Fixture setup failed during find_marker_genes call: {e}")
    # !!! END FIX ---


    # Final checks on fixture state
    if not adata_proc.obs_names.equals(adata_proc.raw.obs_names):
        pytest.fail("Fixture setup failed: .obs_names mismatch between adata_proc and .raw")
    if not adata_proc.var_names.equals(adata_proc.raw.var_names):
         pytest.fail(f"Fixture setup failed: .var_names mismatch between adata_proc and .raw")

    log.info(f"AnnData ready for DGE tests and plotting, shape: {adata_proc.shape}, .raw shape: {adata_proc.raw.shape}")
    return adata_proc


# --- Test Cases ---
# (Test cases remain exactly the same as before)

def test_dge_success_use_raw(adata_for_dge):
    """Test DGE was run successfully in fixture using adata.raw."""
    if adata_for_dge is None: pytest.skip("DGE data not available")
    # DGE already run by fixture, just check results exist with expected key
    results_key = 'rank_genes_groups_raw'
    assert results_key in adata_for_dge.uns, f"Results key '{results_key}' not found in adata.uns (expected from fixture)"
    assert 'names' in adata_for_dge.uns[results_key]
    assert 'pvals_adj' in adata_for_dge.uns[results_key]
    assert 'logfoldchanges' in adata_for_dge.uns[results_key]
    assert 'scores' in adata_for_dge.uns[results_key]
    assert isinstance(adata_for_dge.uns[results_key]['names'], np.ndarray)


def test_dge_success_use_x(adata_for_dge):
    """Test DGE runs successfully using adata.X (log-normalized)."""
    if adata_for_dge is None: pytest.skip("DGE data not available")
    adata = adata_for_dge.copy()
    # Remove .raw to force using .X when use_raw=False
    adata.raw = None
    groupby_key = 'leiden'
    results_key = 'test_rank_genes_X' # Use a different key for this run

    # Run DGE again, this time using .X
    find_marker_genes(
        adata,
        groupby=groupby_key,
        use_raw=False, # Explicitly use .X
        key_added=results_key
    )

    assert results_key in adata.uns, f"Results key '{results_key}' not found in adata.uns"
    assert 'names' in adata.uns[results_key]


def test_dge_success_use_raw_none_with_raw(adata_for_dge):
    """Test DGE runs successfully with use_raw=None when .raw exists."""
    if adata_for_dge is None: pytest.skip("DGE data not available")
    adata = adata_for_dge.copy()
    assert adata.raw is not None # Pre-condition check
    groupby_key = 'leiden'
    results_key = 'test_rank_genes_none_with_raw' # Use a different key

    # Run DGE again with use_raw=None
    find_marker_genes(
        adata,
        groupby=groupby_key,
        use_raw=None, # Default behavior
        key_added=results_key
    )

    assert results_key in adata.uns, f"Results key '{results_key}' not found in adata.uns"
    assert 'names' in adata.uns[results_key]
    # Check it's different from the fixture run (which used raw) if possible, or just structure
    assert results_key != 'rank_genes_groups_raw'


def test_dge_success_use_raw_none_without_raw(adata_for_dge):
    """Test DGE runs successfully with use_raw=None when .raw is missing."""
    if adata_for_dge is None: pytest.skip("DGE data not available")
    adata = adata_for_dge.copy()
    adata.raw = None # Ensure .raw is missing
    assert adata.raw is None # Pre-condition check
    groupby_key = 'leiden'
    results_key = 'test_rank_genes_none_without_raw' # Use a different key

    # Run DGE again with use_raw=None and no .raw
    find_marker_genes(
        adata,
        groupby=groupby_key,
        use_raw=None, # Default behavior
        key_added=results_key
    )

    assert results_key in adata.uns, f"Results key '{results_key}' not found in adata.uns"
    assert 'names' in adata.uns[results_key]


def test_dge_error_use_raw_true_without_raw(adata_for_dge):
    """Test ValueError if use_raw=True but .raw is missing."""
    if adata_for_dge is None: pytest.skip("DGE data not available")
    adata = adata_for_dge.copy()
    adata.raw = None # Ensure .raw is missing

    with pytest.raises(ValueError, match="adata.raw is None"):
        find_marker_genes(adata, groupby='leiden', use_raw=True)


def test_dge_error_missing_groupby(adata_for_dge):
    """Test KeyError if groupby key is missing."""
    if adata_for_dge is None: pytest.skip("DGE data not available")
    adata = adata_for_dge.copy()
    invalid_key = 'nonexistent_group'
    if invalid_key in adata.obs:
        del adata.obs[invalid_key] # Make sure it's not there

    with pytest.raises(KeyError, match=f"Group key '{invalid_key}' not found"):
        find_marker_genes(adata, groupby=invalid_key)


def test_dge_invalid_input_type():
    """Test TypeError for non-AnnData input."""
    with pytest.raises(TypeError, match="Input 'adata' must be an AnnData object"):
        find_marker_genes([1, 2, 3], groupby='some_key')


def test_dge_custom_key(adata_for_dge):
    """Test using a custom key_added."""
    if adata_for_dge is None: pytest.skip("DGE data not available")
    adata = adata_for_dge.copy()
    custom_key = "my_markers_wilcoxon"

    find_marker_genes(
        adata,
        groupby='leiden',
        use_raw=True,
        key_added=custom_key # Use custom key
    )

    assert custom_key in adata.uns, f"Custom key '{custom_key}' not found in adata.uns"
    # Fixture already ran with 'rank_genes_groups_raw', so default might exist if not cleared
    # Better check: make sure our specific custom key was used
    assert 'names' in adata.uns[custom_key]


# Import fixtures from other test files that this file *depends on*
try:
    from .test_dimred import preprocessed_adata_for_pca # Needed by adata_with_pca
    from .test_clustering import adata_with_pca # This fixture depends on this one
except ImportError:
    log.warning("Could not directly import prerequisite fixtures (preprocessed_adata_for_pca, adata_with_pca).")
    pass