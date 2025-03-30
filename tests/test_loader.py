"""
Tests for the data loader module
"""

import pytest
import scanpy as sc
import numpy as np
from scrnaseq_agent.data.loader import load_data

def test_load_data(tmp_path):
    """Test the load_data function"""
    # Create a simple test dataset
    adata = sc.AnnData(
        X=np.random.rand(10, 5),
        obs={'cell_type': ['A'] * 5 + ['B'] * 5},
        var={'gene_names': [f'gene_{i}' for i in range(5)]}
    )
    
    # Save as h5ad
    test_file = tmp_path / "test.h5ad"
    adata.write_h5ad(test_file)
    
    # Test loading
    loaded_adata = load_data(str(test_file))
    assert loaded_adata.shape == adata.shape
    assert list(loaded_adata.obs['cell_type']) == list(adata.obs['cell_type']) 