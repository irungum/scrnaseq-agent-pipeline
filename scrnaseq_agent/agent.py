# scrnaseq_agent/agent.py

import logging
import anndata as ad
from pathlib import Path
import scanpy as sc
import sys
import pandas as pd # Needed for checking obs columns

# Import pipeline step functions
from .data.loader import load_data
from .analysis.qc import calculate_qc_metrics, filter_cells_qc
from .analysis.preprocess import normalize_log1p, select_hvg
from .analysis.dimred import reduce_dimensionality
from .analysis.clustering import perform_clustering
from .analysis.dge import find_marker_genes
# Updated annotation import
from .analysis.annotation import (
    annotate_cell_types,
    load_marker_dict,
    annotate_celltypist # Import new function
)
from .visualization.plotting import (
    plot_umap,
    plot_qc_violin,
    plot_rank_genes_groups_dotplot,
    plot_rank_genes_groups_stacked_violin,
    plot_rank_genes_groups_heatmap
)

log = logging.getLogger(__name__)

class ScrnaSeqWorkflow:
    """Orchestrates a standard scRNA-seq analysis workflow using PCA."""
    def __init__(self, params):
        """Initializes the workflow orchestrator."""
        self.params = params
        self.adata = None
        self._adata_original_raw = None
        self._adata_pre_filter = None
        self.output_dir = Path(self.params.output_dir)
        self.prefix = self.params.output_prefix
        self.cluster_key = 'leiden' # Default key for clustering results
        self.annotation_result_key = None # Stores the primary key of the annotation result

        log.info("ScrnaSeqWorkflow initialized.")
        log.debug(f"Workflow parameters: {vars(self.params)}")
        required_attrs = ['input_path', 'output_dir', 'output_prefix']
        for attr in required_attrs:
            if not hasattr(self.params, attr):
                 raise ValueError(f"Initialization failed: Missing required parameter '{attr}'.")
        # Initial check for annotation parameter consistency
        if hasattr(params, 'annotation_tool'):
             if params.annotation_tool == 'celltypist' and hasattr(params, 'celltypist_majority_voting') and params.celltypist_majority_voting:
                  log.info(f"CellTypist majority voting requested. Will use cluster key '{self.cluster_key}' after clustering.")
             if params.annotation_tool == 'marker_overlap' and (not hasattr(params, 'marker_file') or not params.marker_file):
                  log.warning("Annotation tool set to 'marker_overlap' but --marker-file is missing. Annotation will be skipped.")


    def run(self):
        """Executes the full scRNA-seq pipeline sequentially using PCA."""
        log.info(f"Starting workflow run: {self.prefix}")
        try:
            self._setup_environment()          # Step 0
            self._load_data()                  # Step 1
            self._run_qc()                     # Step 2
            self._filter_cells()               # Step 3
            self._finalize_raw()               # Step 4
            # --- Pipeline Order Changed Here ---
            self._normalize_log_hvg()          # Step 5, 6
            self._annotate()                   # Step 7 (before scale)
            self._scale_data()                 # Step 8
            self._run_pca_dimred()             # Step 9
            self._cluster_and_umap()           # Step 10
            self._find_markers()               # Step 11
            self._plot_results()               # Step 12
            self._save_results()               # Step 13
            # --- End Changed Order ---

            log.info(f"Workflow run '{self.prefix}' completed successfully.")
            return self.adata

        except Exception as e:
            log.error(f"Workflow run '{self.prefix}' failed: {e}", exc_info=True)
            raise # Re-raise the exception for CLI/caller to handle

    def _setup_environment(self):
        """Sets up Scanpy settings and output directory."""
        log.debug("Setting up environment...")
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            sc.settings.figdir = str(self.output_dir)
            sc.settings.verbosity = 3 # info, warnings, errors, hints
            log.info(f"Output directory set to: {self.output_dir}")
        except OSError as e:
            log.error(f"Failed to create output directory '{self.output_dir}': {e}")
            raise

    def _load_data(self):
        """Loads initial data and stores original raw copy."""
        log.info("Step 1: Loading data...")
        self.adata = load_data(self.params.input_path)
        self._adata_original_raw = self.adata.copy()
        self._adata_pre_filter = self.adata.copy()
        log.info(f"Loaded data shape: {self.adata.shape}. Stored initial copy.")

    def _run_qc(self):
        """Calculates QC metrics and plots pre-filter violins."""
        if self.adata is None: raise RuntimeError("adata not loaded before running QC.")
        log.info("Step 2: Calculating QC metrics...")
        calculate_qc_metrics(self.adata, mito_gene_prefix=self.params.mito_prefix, inplace=True)
        if self._adata_pre_filter is not None:
             calculate_qc_metrics(self._adata_pre_filter, mito_gene_prefix=self.params.mito_prefix, inplace=True)
        log.info("QC calculation complete.")

        if self.params.run_qc_violin and self.params.qc_violin_keys and self._adata_pre_filter is not None:
             log.info("Plotting QC violin plots (pre-filtering)...")
             plot_qc_violin(
                 self._adata_pre_filter, keys=self.params.qc_violin_keys, output_dir=str(self.output_dir),
                 file_prefix=f"{self.prefix}_qc_violin_prefilt",
                 file_format=self.params.plot_format, dpi=self.params.plot_dpi
             )

    def _filter_cells(self):
        """Filters cells based on QC metrics and plots post-filter violins."""
        if self.adata is None: raise RuntimeError("adata not loaded before filtering.")
        log.info("Step 3: Filtering cells...")
        n_obs_before = self.adata.n_obs
        filter_cells_qc(
            self.adata, min_genes=self.params.min_genes, max_genes=self.params.max_genes,
            min_counts=self.params.min_counts, max_counts=self.params.max_counts,
            max_pct_mito=self.params.max_pct_mito, inplace=True
        )
        if self.adata.n_obs == 0: raise ValueError("All cells filtered out!")
        log.info(f"Filtering complete. Kept {self.adata.n_obs} / {n_obs_before} cells.")
        self._adata_pre_filter = None # Clear pre-filter copy

        if self.params.run_qc_violin and self.params.qc_violin_keys:
             log.info("Plotting QC violin plots (post-filtering)...")
             plot_qc_violin(
                 self.adata, keys=self.params.qc_violin_keys, output_dir=str(self.output_dir),
                 file_prefix=f"{self.prefix}_qc_violin_postfilt",
                 file_format=self.params.plot_format, dpi=self.params.plot_dpi
                 # No groupby needed here as clusters don't exist yet
             )

    def _finalize_raw(self):
         """Sets the .raw attribute using original counts for filtered cells."""
         log.info("Step 4: Setting .raw attribute...")
         if self._adata_original_raw is None:
              log.warning("Original data copy missing. Cannot set .raw.")
              self.adata.raw = None; return
         if self.adata is None: raise RuntimeError("adata object missing.")
         try:
              # Keep all original genes in .raw, just subset cells
              self.adata.raw = self._adata_original_raw[self.adata.obs_names, :].copy()
              log.info(f"Set .raw attribute. Shape: {self.adata.raw.shape}")
         except Exception as e:
              log.error(f"Failed to set .raw attribute: {e}", exc_info=True)
              self.adata.raw = None
         finally: self._adata_original_raw = None # Clear copy

    def _normalize_log_hvg(self):
        """Runs normalization, log1p, and HVG selection."""
        if self.adata is None: raise RuntimeError("adata not loaded.")
        log.info("Step 5: Normalizing and log-transforming...")
        normalize_log1p(self.adata, target_sum=self.params.target_sum, inplace=True)
        log.info("Step 6: Selecting highly variable genes (subsetting)...")
        n_vars_before = self.adata.n_vars
        select_hvg(self.adata, n_top_genes=self.params.n_hvgs, flavor=self.params.hvg_flavor, subset=True, inplace=True)
        log.info(f"Kept {self.adata.n_vars} / {n_vars_before} HVGs.")
        if self.adata.raw is not None: log.info(f".raw attribute retained shape {self.adata.raw.shape}")

    def _annotate(self): # Step 7
        """Runs the selected cell type annotation method."""
        if self.adata is None: raise RuntimeError("adata not available for Annotation.")
        log.info(f"Step 7: Annotating cell types using tool: '{self.params.annotation_tool}'...")
        self.annotation_result_key = None # Reset

        # Cluster key check (only relevant if needed by a method like MV or marker overlap)
        cluster_exists = self.cluster_key in self.adata.obs

        if self.params.annotation_tool == 'marker_overlap':
            log.warning("Marker overlap annotation skipped: Requires DGE results which run later in this workflow order.")
            return # Skip this method

        elif self.params.annotation_tool == 'celltypist':
            log.info("Running CellTypist annotation on log-normalized data...")
            cluster_key_mv = self.cluster_key if self.params.celltypist_majority_voting else None
            mv_flag_for_call = self.params.celltypist_majority_voting

            # Check if clusters exist *only if* majority voting is requested
            if mv_flag_for_call and not cluster_exists:
                 log.warning(f"Cannot perform CellTypist majority voting: cluster key '{self.cluster_key}' not found yet. Running per-cell prediction only.")
                 mv_flag_for_call = False
                 cluster_key_mv = None # Ensure key is None if voting disabled

            try:
                # !!! FIX: Ensure `layer` argument is NOT passed !!!
                annotate_celltypist(
                    adata=self.adata,
                    model_name=self.params.celltypist_model,
                    majority_voting=mv_flag_for_call, # Use adjusted flag
                    output_key_prefix=self.params.annotation_key,
                    cluster_key_for_voting=cluster_key_mv # Pass None if MV is off
                    # No layer='X' argument here
                )
                # !!! END FIX !!!

                # Determine primary result key
                pred_labels_key = f"{self.params.annotation_key}_predicted_labels"
                mv_labels_key = f"{self.params.annotation_key}_majority_voting"
                if mv_flag_for_call and mv_labels_key in self.adata.obs:
                     self.annotation_result_key = mv_labels_key
                elif pred_labels_key in self.adata.obs:
                     self.annotation_result_key = pred_labels_key
                else: self.annotation_result_key = None
                log.info(f"CellTypist annotation complete. Primary results key: '{self.annotation_result_key}'.")

            except ImportError as e: log.error(f"CellTypist import failed: {e}.")
            except Exception as e: log.error(f"CellTypist annotation failed: {e}", exc_info=True)

        elif self.params.annotation_tool is None or str(self.params.annotation_tool).lower() == 'none':
            log.info("Annotation tool set to None. Skipping annotation step.")
        else:
            log.warning(f"Unknown annotation tool: '{self.params.annotation_tool}'. Skipping.")

    def _scale_data(self): # Step 8
        if self.adata is None: raise RuntimeError("adata not loaded.")
        log.info("Step 8: Scaling data (HVGs)...")
        sc.pp.scale(self.adata, max_value=self.params.scale_max_value)
        log.info("Scaling complete.")

    def _run_pca_dimred(self): # Step 9
        if self.adata is None: raise RuntimeError("adata not available.")
        log.info("Step 9: Performing PCA...")
        reduce_dimensionality(
            self.adata, n_comps=self.params.n_pca_comps,
            random_state=self.params.random_seed, inplace=True
        )
        log.info("PCA complete.")

    def _cluster_and_umap(self): # Step 10
        if self.adata is None: raise RuntimeError("adata not available.")
        pca_key = 'X_pca'
        if pca_key not in self.adata.obsm: raise RuntimeError(f"PCA key '{pca_key}' not found.")
        log.info(f"Step 10: Performing Neighbors, Clustering, and UMAP using '{pca_key}'...")
        perform_clustering(
            self.adata, use_rep=pca_key, n_neighbors=self.params.n_neighbors,
            resolution=self.params.leiden_resolution, random_state=self.params.random_seed,
            leiden_key_added=self.cluster_key, calculate_umap=(not self.params.skip_umap),
            inplace=True
        )
        log.info("Clustering and UMAP calculation complete.")

    def _find_markers(self): # Step 11
        if self.adata is None: raise RuntimeError("adata not available.")
        if self.cluster_key not in self.adata.obs: log.warning(f"Cluster key '{self.cluster_key}' not found. Skipping DGE."); return
        log.info("Step 11: Finding marker genes...")
        find_marker_genes(
            self.adata, groupby=self.cluster_key, method=self.params.dge_method,
            corr_method=self.params.dge_corr_method, use_raw=self.params.dge_use_raw,
            key_added=self.params.dge_key
        )
        log.info("Marker gene identification complete.")

    def _plot_results(self): # Step 12
        if self.adata is None: raise RuntimeError("adata not available.")
        log.info("Step 12: Generating plots...")

        # UMAP Plots
        plot_features = self.params.plot_umap_color[:] # Copy
        if self.cluster_key in self.adata.obs and self.cluster_key not in plot_features: plot_features.insert(0, self.cluster_key)
        if self.annotation_result_key and self.annotation_result_key in self.adata.obs and self.annotation_result_key not in plot_features:
            log.info(f"Adding annotation '{self.annotation_result_key}' to UMAP plots.")
            plot_features.append(self.annotation_result_key)

        if not self.params.skip_umap and 'X_umap' in self.adata.obsm and plot_features:
            log.info(f"Plotting UMAP colored by: {plot_features}")
            try:
                # !!! FIX: Remove copy=False !!!
                plot_umap(
                    self.adata, color_by=plot_features, output_dir=str(self.output_dir),
                    file_prefix=f"{self.prefix}_umap", file_format=self.params.plot_format,
                    dpi=self.params.plot_dpi # Removed copy=False
                )
            except Exception as e: log.error(f"Failed generating UMAP plots: {e}", exc_info=True)
        elif self.params.skip_umap: log.info("Skipping UMAP plots.")
        elif 'X_umap' not in self.adata.obsm: log.warning("X_umap not found. Skipping UMAP plots.")

        # DGE Plots
        if self.params.run_dge_plots and self.params.dge_key in self.adata.uns:
            log.info("Plotting DGE results...")
            dge_plot_kwargs = dict(
                 key=self.params.dge_key, n_genes=self.params.dge_n_genes,
                 groupby=self.cluster_key, output_dir=str(self.output_dir),
                 file_format=self.params.plot_format, dpi=self.params.plot_dpi,
                 use_raw=self.params.dge_use_raw
            )
            try: plot_rank_genes_groups_dotplot(self.adata, file_prefix=f"{self.prefix}_dge_dotplot", **dge_plot_kwargs)
            except Exception as e: log.error(f"Failed generating DGE dotplot: {e}", exc_info=True)
            try: plot_rank_genes_groups_stacked_violin(self.adata, file_prefix=f"{self.prefix}_dge_violin", **dge_plot_kwargs)
            except Exception as e: log.error(f"Failed generating DGE stacked violin: {e}", exc_info=True)
            if self.params.run_dge_heatmap:
                try: plot_rank_genes_groups_heatmap(self.adata, file_prefix=f"{self.prefix}_dge_heatmap", show_gene_labels=True, **dge_plot_kwargs)
                except Exception as e: log.error(f"Failed generating DGE heatmap: {e}", exc_info=True)
        elif not self.params.run_dge_plots: log.info("Skipping DGE plots as requested.")
        else: log.warning(f"DGE key '{self.params.dge_key}' not found. Skipping DGE plots.")
        log.info("Plot generation complete.")

    def _save_results(self): # Step 13
        if self.adata is None: raise RuntimeError("No AnnData object to save.")
        log.info("Step 13: Saving final AnnData object...")
        final_adata_path = self.output_dir / f"{self.prefix}_final.h5ad"
        try:
            for col in self.adata.obs.select_dtypes(include='category').columns:
                 if len(self.adata.obs[col].cat.categories) > 500: log.warning(f"Obs column '{col}' has >500 categories.")
            self.adata.write_h5ad(final_adata_path, compression="gzip")
            log.info(f"Final AnnData object saved to: {final_adata_path}")
        except Exception as e: log.error(f"Failed to save final AnnData: {e}", exc_info=True); raise