# tests/test_loader.py

import pytest
import anndata as ad
import os
import shutil # For creating temporary test directories if needed

# Import the function to be tested
from scrnaseq_agent.data.loader import load_data

# --- Fixtures (optional but good practice for setup/teardown) ---
# Define paths to your test data - adjust if needed
# Use absolute paths or paths relative to the project root where pytest runs
TEST_H5AD_PATH = "test_data/pbmc3k.h5ad"
TEST_10X_DIR = "test_data/filtered_gene_bc_matrices/hg19/" # Might need adjustment based on actual extract
INVALID_PATH = "non_existent_dir/non_existent_file.h5ad"
README_PATH = "README.md" # Assumes README.md is in project root

# Check if test data exists - skip tests if not found
# Note: Pytest skips are evaluated *before* fixtures run
h5ad_exists = os.path.exists(TEST_H5AD_PATH)
tenx_dir_exists = os.path.exists(TEST_10X_DIR)


# --- Test Functions ---

@pytest.mark.skipif(not h5ad_exists, reason=f"Test H5AD file not found at {TEST_H5AD_PATH}")
def test_load_h5ad_success():
    """Tests successful loading of an H5AD file."""
    adata = load_data(TEST_H5AD_PATH)
    assert isinstance(adata, ad.AnnData), "Loaded object is not an AnnData instance"
    assert adata.n_obs > 0, "AnnData object has no observations (cells)"
    assert adata.n_vars > 0, "AnnData object has no variables (genes)"
    # Add more specific checks if needed (e.g., expected shape, key layers/obs/var)
    # assert adata.shape == (2638, 1838), "Unexpected shape for pbmc3k h5ad" # Example

@pytest.mark.skipif(not tenx_dir_exists, reason=f"Test 10x directory not found at {TEST_10X_DIR}")
def test_load_10x_dir_success():
    """Tests successful loading of a 10x MTX directory."""
    # Adjust path if tar extraction resulted in a different subfolder name
    # e.g., if it extracted to test_data/filtered_gene_bc_matrices/hg19/
    # then TEST_10X_DIR should point there.
    adata = load_data(TEST_10X_DIR)
    assert isinstance(adata, ad.AnnData), "Loaded object is not an AnnData instance"
    assert adata.n_obs > 0, "AnnData object has no observations (cells)"
    assert adata.n_vars > 0, "AnnData object has no variables (genes)"
    # Add more specific checks if needed
    # assert adata.shape == (2700, 32738), "Unexpected shape for pbmc3k 10x" # Example

def test_load_invalid_path_raises_error():
    """Tests that loading a non-existent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_data(INVALID_PATH)

def test_load_wrong_file_type_raises_error():
    """Tests that loading an unsupported file type raises ValueError."""
    # Ensure the README file exists for this test to be valid
    if not os.path.exists(README_PATH):
        pytest.skip(f"README file not found at {README_PATH}, skipping test.")

    with pytest.raises(ValueError, match="Unrecognized file format"):
        load_data(README_PATH)

def test_load_non_string_path_raises_error():
    """Tests that passing a non-string path raises TypeError."""
    with pytest.raises(TypeError):
        load_data(12345) # Pass an integer instead of a string


# --- Optional: More advanced tests ---
# - Test caching behavior if implemented
# - Test loading from S3 (once implemented)
# - Test edge cases (empty files, corrupted files - may need specific test data)