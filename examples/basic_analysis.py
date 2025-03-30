"""
Example script demonstrating basic usage of scrnaseq_agent
"""

import scanpy as sc
from scrnaseq_agent.data.loader import load_data
from scrnaseq_agent.analysis.qc import calculate_qc_metrics

def main():
    # Load data
    adata = load_data("path/to/your/data.h5ad")
    
    # Perform QC
    adata = calculate_qc_metrics(
        adata,
        min_genes=200,
        max_genes=5000,
        min_cells=3,
        percent_mt=20
    )
    
    # Basic preprocessing
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    sc.pp.scale(adata, max_value=10)
    
    # PCA
    sc.tl.pca(adata, svd_solver='arpack')
    
    # Compute neighborhood graph
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    
    # UMAP
    sc.tl.umap(adata)
    
    # Clustering
    sc.tl.leiden(adata)
    
    # Save results
    adata.write_h5ad("results.h5ad")

if __name__ == "__main__":
    main() 