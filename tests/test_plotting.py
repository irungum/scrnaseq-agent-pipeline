# tests/test_plotting.py

import pytest
import anndata as ad
import numpy as np
import scanpy as sc
import pandas as pd
import os
import shutil
import logging
from pathlib import Path

# Functions under test and needed for fixtures
from scrnaseq_agent.visualization.plotting import (
    plot_umap,
    plot_qc_violin,
    plot_rank_genes_groups_dotplot,
    plot_rank_genes_groups_stacked_violin,
    plot_rank_genes_groups_heatmap
)
# Imports needed for the fixtures defined in THIS file
from scrnaseq_agent.analysis.clustering import perform_clustering
from scrnaseq_agent.analysis.qc import calculate_qc_metrics

# Configure logging
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

# --- Fixtures ---

# Fixture providing data processed up to *clustering* (including UMAP and QC cols)
@pytest.fixture(scope="module")
def adata_with_clustering_umap(adata_with_pca) -> ad.AnnData | None:
    """Provides AnnData processed up to clustering and UMAP."""
    if adata_with_pca is None:
        pytest.skip("Base data for clustering/UMAP setup not available.")
        return None
    adata = adata_with_pca.copy()
    log.info("Running Clustering/UMAP for plotting fixture...")
    qc_keys_needed = ["n_genes_by_counts", "total_counts", "pct_counts_mt"]
    if not all(k in adata.obs for k in qc_keys_needed):
        log.warning("QC columns missing in fixture input, recalculating...")
        calculate_qc_metrics(adata, mito_gene_prefix="MT-", inplace=True) # Assumes 'MT-'

    perform_clustering(
        adata, resolution=0.8, random_state=0, calculate_umap=True, inplace=True
    )
    if 'X_umap' not in adata.obsm or 'leiden' not in adata.obs:
        pytest.fail("Fixture setup failed: UMAP or Leiden results missing.")
    return adata

# Fixture providing data processed up to *DGE*
# (Relies on import from test_dge.py via pytest mechanism)


# --- Test Cases ---

# == UMAP Plot Tests ==
def test_plot_umap_success(adata_with_clustering_umap, tmp_path):
    """Test successful generation of UMAP plots for valid features."""
    if adata_with_clustering_umap is None: pytest.skip("UMAP data not available")
    adata = adata_with_clustering_umap
    output_dir = tmp_path / "plotting_output_success"
    file_prefix = "test_success"
    features_to_plot = ['leiden', 'n_genes_by_counts']
    test_gene = 'MS4A1' if 'MS4A1' in adata.var_names else None
    if test_gene: features_to_plot.append(test_gene)

    sc.settings.figdir = str(tmp_path)
    plot_umap(
        adata=adata, color_by=features_to_plot, output_dir=str(output_dir),
        file_prefix=file_prefix, file_format="png", dpi=50
    )
    assert output_dir.is_dir()
    for feature in features_to_plot:
         safe_feature_name = feature.replace('/', '_').replace('\\', '_')
         expected_file = output_dir / f"{file_prefix}_{safe_feature_name}.png"
         assert expected_file.is_file(), f"Expected UMAP file '{expected_file}' not created."
         assert expected_file.stat().st_size > 0

def test_plot_umap_creates_dir(adata_with_clustering_umap, tmp_path):
    """Test that the output directory is created if it doesn't exist for UMAP."""
    if adata_with_clustering_umap is None: pytest.skip("UMAP data not available")
    adata = adata_with_clustering_umap
    output_dir = tmp_path / "new_plotting_dir_umap"
    assert not output_dir.is_dir()
    sc.settings.figdir = str(tmp_path)
    plot_umap(adata=adata, color_by=['leiden'], output_dir=str(output_dir), file_prefix="test_dir", file_format="pdf")
    assert output_dir.is_dir()
    expected_file = output_dir / "test_dir_leiden.pdf"
    assert expected_file.is_file()

def test_plot_umap_missing_umap_key(adata_with_clustering_umap):
    """Test error if umap_key is missing for UMAP plot."""
    if adata_with_clustering_umap is None: pytest.skip("UMAP data not available")
    adata = adata_with_clustering_umap.copy()
    if 'X_umap' in adata.obsm: del adata.obsm['X_umap']
    # Match exact error message from plotting.py
    with pytest.raises(KeyError, match="UMAP key 'X_umap' not found"):
        plot_umap(adata, color_by=['leiden'], output_dir="dummy_dir")

def test_plot_umap_invalid_inputs(adata_with_clustering_umap, tmp_path):
    """Test error handling for invalid inputs for UMAP plot."""
    if adata_with_clustering_umap is None: pytest.skip("UMAP data not available")
    adata = adata_with_clustering_umap
    with pytest.raises(ValueError, match="color_by must be non-empty list"):
        plot_umap(adata, color_by=[], output_dir=str(tmp_path))
    with pytest.raises(ValueError, match="color_by must be non-empty list"):
        plot_umap(adata, color_by='leiden', output_dir=str(tmp_path))
    with pytest.raises(ValueError, match="output_dir must be provided"):
        plot_umap(adata, color_by=['leiden'], output_dir="")

def test_plot_umap_feature_not_found(adata_with_clustering_umap, tmp_path, caplog):
    """Test that missing features are skipped with a warning for UMAP plot."""
    if adata_with_clustering_umap is None: pytest.skip("UMAP data not available")
    adata = adata_with_clustering_umap
    output_dir = tmp_path / "plotting_output_missing_umap"
    file_prefix = "test_missing"
    valid_feature = 'leiden'
    invalid_feature = 'feature_does_not_exist_abc123'
    features_to_plot = [valid_feature, invalid_feature]
    sc.settings.figdir = str(tmp_path)
    with caplog.at_level(logging.WARNING):
        plot_umap(adata=adata, color_by=features_to_plot, output_dir=str(output_dir), file_prefix=file_prefix)
    expected_valid_file = output_dir / f"{file_prefix}_{valid_feature}.png"
    assert expected_valid_file.is_file()
    expected_invalid_file = output_dir / f"{file_prefix}_{invalid_feature}.png"
    assert not expected_invalid_file.is_file()
    assert f"Feature '{invalid_feature}' not found" in caplog.text

# == QC Violin Plot Tests ==
def test_plot_qc_violin_success(adata_with_clustering_umap, tmp_path):
    """Test successful generation of QC violin plots."""
    if adata_with_clustering_umap is None: pytest.skip("Clustered data not available")
    adata = adata_with_clustering_umap
    output_dir = tmp_path / "plotting_qc_violin"
    file_prefix = "test_qc_violin"
    keys_to_plot = ["n_genes_by_counts", "total_counts", "pct_counts_mt"]
    if not all(k in adata.obs for k in keys_to_plot): pytest.fail(f"Fixture missing keys: {keys_to_plot}")
    sc.settings.figdir = str(tmp_path)
    plot_qc_violin(
        adata=adata, keys=keys_to_plot, output_dir=str(output_dir),
        file_prefix=file_prefix, file_format="png", dpi=50, groupby='leiden'
    )
    assert output_dir.is_dir()
    expected_file = output_dir / f"{file_prefix}.png"
    assert expected_file.is_file(), f"Expected QC violin file '{expected_file}' not created."
    assert expected_file.stat().st_size > 0

def test_plot_qc_violin_missing_keys(adata_with_clustering_umap, tmp_path, caplog):
    """Test warning and partial plotting if some keys are missing."""
    if adata_with_clustering_umap is None: pytest.skip("Clustered data not available")
    adata = adata_with_clustering_umap.copy()
    output_dir = tmp_path / "plotting_qc_violin_missing"
    valid_key = "total_counts"
    invalid_key = "nonexistent_qc_metric"
    keys_to_plot = [valid_key, invalid_key]
    if invalid_key in adata.obs: del adata.obs[invalid_key]
    if valid_key not in adata.obs: pytest.fail(f"Fixture missing expected QC key '{valid_key}'")
    sc.settings.figdir = str(tmp_path)
    with caplog.at_level(logging.WARNING):
        plot_qc_violin(adata=adata, keys=keys_to_plot, output_dir=str(output_dir))
    expected_file = output_dir / "qc_violin.png"
    assert expected_file.is_file(), "Plot file should still be created for valid keys."
    assert f"QC keys not found in adata.obs: ['{invalid_key}']" in caplog.text

# == DGE Plot Tests ==
def test_plot_dge_dotplot_success(adata_for_dge, tmp_path):
    """Test successful generation of DGE dotplot."""
    if adata_for_dge is None: pytest.skip("DGE data not available")
    dge_key = 'rank_genes_groups_raw'
    if dge_key not in adata_for_dge.uns: pytest.fail(f"Fixture missing DGE results '{dge_key}'")
    adata = adata_for_dge
    output_dir = tmp_path / "plotting_output_dotplot"
    file_prefix = "test_dotplot"
    sc.settings.figdir = str(tmp_path)
    plot_rank_genes_groups_dotplot(
        adata=adata, key=dge_key, n_genes=3, groupby='leiden',
        output_dir=str(output_dir), file_prefix=file_prefix, dpi=50
    )
    assert output_dir.is_dir()
    expected_file = output_dir / f"{file_prefix}.png"
    assert expected_file.is_file(), f"Expected dotplot file '{expected_file}' not created."
    assert expected_file.stat().st_size > 0

def test_plot_dge_stacked_violin_success(adata_for_dge, tmp_path):
    """Test successful generation of DGE stacked violin plot."""
    if adata_for_dge is None: pytest.skip("DGE data not available")
    dge_key = 'rank_genes_groups_raw'
    if dge_key not in adata_for_dge.uns: pytest.fail(f"Fixture missing DGE results '{dge_key}'")
    adata = adata_for_dge
    output_dir = tmp_path / "plotting_output_violin"
    file_prefix = "test_violin"
    sc.settings.figdir = str(tmp_path)
    plot_rank_genes_groups_stacked_violin(
        adata=adata, key=dge_key, n_genes=3, groupby='leiden',
        output_dir=str(output_dir), file_prefix=file_prefix, file_format="pdf", dpi=50
    )
    assert output_dir.is_dir()
    expected_file = output_dir / f"{file_prefix}.pdf"
    assert expected_file.is_file(), f"Expected violin file '{expected_file}' was not created."
    assert expected_file.stat().st_size > 0

def test_plot_dge_heatmap_success(adata_for_dge, tmp_path):
    """Test successful generation of DGE heatmap plot."""
    if adata_for_dge is None: pytest.skip("DGE data not available")
    dge_key = 'rank_genes_groups_raw'
    if dge_key not in adata_for_dge.uns: pytest.fail(f"Fixture missing DGE results '{dge_key}'")
    adata = adata_for_dge
    output_dir = tmp_path / "plotting_output_heatmap"
    file_prefix = "test_heatmap"
    sc.settings.figdir = str(tmp_path)
    plot_rank_genes_groups_heatmap(
        adata=adata, key=dge_key, n_genes=3, groupby='leiden',
        output_dir=str(output_dir), file_prefix=file_prefix, dpi=50,
        show_gene_labels=False
    )
    assert output_dir.is_dir()
    expected_file = output_dir / f"{file_prefix}.png"
    assert expected_file.is_file(), f"Expected heatmap file '{expected_file}' was not created."
    assert expected_file.stat().st_size > 0

# --- Fixture Import Block ---
try:
    from .test_dimred import preprocessed_adata_for_pca
    from .test_clustering import adata_with_pca
    from .test_dge import adata_for_dge
except ImportError:
    log.warning("Could not directly import prerequisite fixtures.")
    pass