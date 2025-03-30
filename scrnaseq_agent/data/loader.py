"""
Data loading functionality for single-cell RNA-seq data
"""

import scanpy as sc
import anndata
from typing import Union, Optional

def load_data(
    file_path: str,
    file_type: Optional[str] = None,
    **kwargs
) -> anndata.AnnData:
    """
    Load single-cell RNA-seq data from various file formats.
    
    Args:
        file_path: Path to the data file
        file_type: Type of file (optional, will be inferred if not provided)
        **kwargs: Additional arguments passed to the loading function
    
    Returns:
        AnnData object containing the loaded data
    """
    if file_type is None:
        file_type = file_path.split('.')[-1].lower()
    
    if file_type == 'h5ad':
        return sc.read_h5ad(file_path)
    elif file_type in ['mtx', 'matrix']:
        return sc.read_mtx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}") 