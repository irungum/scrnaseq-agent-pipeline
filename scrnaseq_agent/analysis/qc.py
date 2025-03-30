"""
Quality control functions for single-cell RNA-seq data
"""

import scanpy as sc
import numpy as np
from typing import Optional

def calculate_qc_metrics(
    adata: sc.AnnData,
    min_genes: Optional[int] = None,
    max_genes: Optional[int] = None,
    min_cells: Optional[int] = None,
    max_cells: Optional[int] = None,
    min_counts: Optional[int] = None,
    max_counts: Optional[int] = None,
    percent_mt: Optional[float] = None
) -> sc.AnnData:
    """
    Calculate and filter cells based on QC metrics.
    
    Args:
        adata: AnnData object containing the data
        min_genes: Minimum number of genes expressed
        max_genes: Maximum number of genes expressed
        min_cells: Minimum number of cells expressing a gene
        max_cells: Maximum number of cells expressing a gene
        min_counts: Minimum total counts per cell
        max_counts: Maximum total counts per cell
        percent_mt: Maximum percentage of mitochondrial genes
    
    Returns:
        Filtered AnnData object
    """
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    
    # Apply filters if specified
    if min_genes is not None:
        sc.pp.filter_cells(adata, min_genes=min_genes)
    if max_genes is not None:
        sc.pp.filter_cells(adata, max_genes=max_genes)
    if min_cells is not None:
        sc.pp.filter_genes(adata, min_cells=min_cells)
    if max_cells is not None:
        sc.pp.filter_genes(adata, max_cells=max_cells)
    if min_counts is not None:
        sc.pp.filter_cells(adata, min_counts=min_counts)
    if max_counts is not None:
        sc.pp.filter_cells(adata, max_counts=max_counts)
    if percent_mt is not None:
        sc.pp.filter_cells(adata, max_percent_mt=percent_mt)
    
    return adata 