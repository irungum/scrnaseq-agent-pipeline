# scrnaseq_agent/analysis/annotation.py

import scanpy as sc
import anndata as ad
import logging
import pandas as pd
import json
import os
import sys

# --- CellTypist Import Handling ---
try:
    import celltypist
    from celltypist import models
    CELLTYPIST_INSTALLED = True
except ImportError:
    celltypist = None
    models = None
    CELLTYPIST_INSTALLED = False
# --- END ---

log = logging.getLogger(__name__)
# Basic config only runs if the script is executed directly.
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    if project_root not in sys.path: sys.path.insert(0, project_root)


def load_marker_dict(file_path: str) -> dict | None:
    """Loads a marker gene dictionary from a JSON file."""
    if not file_path: log.warning("No marker file path provided."); return None
    try:
        with open(file_path, 'r') as f: marker_dict = json.load(f)
        log.info(f"Loaded marker dictionary from {file_path}")
        if not isinstance(marker_dict, dict) or \
           not all(isinstance(k, str) and isinstance(v, list) and all(isinstance(g, str) for g in v) for k, v in marker_dict.items()):
            log.error(f"Marker file '{file_path}' invalid format."); return None
        return marker_dict
    except FileNotFoundError: log.error(f"Marker file not found: {file_path}"); return None
    except json.JSONDecodeError as e: log.error(f"Error decoding JSON {file_path}: {e}"); return None
    except Exception as e: log.error(f"Error loading marker file {file_path}: {e}", exc_info=True); return None


def annotate_cell_types(
    adata: ad.AnnData,
    marker_dict: dict,
    groupby: str,
    rank_key: str = 'rank_genes_groups',
    annotation_key: str = 'marker_gene_overlap_annotation',
    method: str = 'overlap_count',
    **kwargs
) -> None:
    """Performs basic cell type annotation based on marker gene overlap."""
    if not isinstance(adata, ad.AnnData): raise TypeError("Input 'adata' must be an AnnData object.")
    if not isinstance(marker_dict, dict) or not marker_dict: raise TypeError("Input 'marker_dict' must be a non-empty dictionary.")
    if groupby not in adata.obs: raise KeyError(f"Group key '{groupby}' not found in adata.obs.")
    if rank_key not in adata.uns: raise KeyError(f"Rank genes groups key '{rank_key}' not found in adata.uns.")

    log.info(f"Performing marker overlap annotation for groups '{groupby}' using results '{rank_key}'.")
    log.info(f"Using method '{method}'. Annotation key base: '{annotation_key}'.")
    try:
        sc.tl.marker_gene_overlap(
            adata, marker_dict, key=rank_key, method=method,
            key_added=annotation_key, **kwargs
        )
        expected_output_key = f"{annotation_key}_{method}"
        if expected_output_key not in adata.obs:
            log.warning(f"sc.tl.marker_gene_overlap finished, but key '{expected_output_key}' not found in adata.obs.")
        log.info(f"Marker overlap annotation complete. Results may be in adata.obs (e.g., '{expected_output_key}').")
    except Exception as e:
        log.error(f"Error during marker overlap annotation: {e}", exc_info=True)
        raise RuntimeError(f"Failed marker overlap annotation: {e}") from e
    return None


# --- Updated CellTypist Function ---
def annotate_celltypist(
    adata: ad.AnnData,
    model_name: str = "Immune_All_Low.pkl",
    majority_voting: bool = False,
    output_key_prefix: str = "celltypist",
    cluster_key_for_voting: str | None = None,
    **kwargs
) -> None:
    """Performs cell type annotation using CellTypist."""
    if not CELLTYPIST_INSTALLED: raise ImportError("celltypist not installed.")
    if not isinstance(adata, ad.AnnData): raise TypeError("Input 'adata' must be AnnData.")
    if majority_voting and (cluster_key_for_voting is None or cluster_key_for_voting not in adata.obs):
         raise ValueError(f"Majority voting requires valid 'cluster_key_for_voting'.") # Added period

    log.info(f"Performing CellTypist annotation using model: {model_name}")
    log.info(f"Majority voting: {majority_voting}. Output prefix: {output_key_prefix}")
    try:
        log.debug(f"Loading CellTypist model '{model_name}'...")
        model = models.Model.load(model=model_name)
        log.debug("CellTypist model loaded.")

        over_clustering_param = adata.obs[cluster_key_for_voting].astype(str) if majority_voting else None

        # Run prediction
        predictions = celltypist.annotate(
            adata, model=model, majority_voting=majority_voting,
            over_clustering=over_clustering_param, **kwargs
        )

        # --- FIX: Correctly handle assignment from predictions object ---
        pred_output = predictions.predicted_labels # Main output object

        # Always try to add per-cell predicted labels
        label_col_name = f"{output_key_prefix}_predicted_labels"
        try:
            # If pred_output is DataFrame, 'predicted_labels' column holds the labels
            if isinstance(pred_output, pd.DataFrame) and 'predicted_labels' in pred_output.columns:
                adata.obs[label_col_name] = pred_output['predicted_labels'].astype('category')
            # If pred_output is Series (e.g., MV=False), use it directly
            elif isinstance(pred_output, pd.Series):
                adata.obs[label_col_name] = pred_output.astype('category')
            else:
                log.warning(f"Could not assign per-cell predicted labels from CellTypist output type: {type(pred_output)}")
            if label_col_name in adata.obs:
                 log.info(f"Added '{label_col_name}' to adata.obs.")
        except Exception as e:
             log.error(f"Failed to assign '{label_col_name}': {e}")


        # Add confidence scores if present
        conf_col_name = f"{output_key_prefix}_conf_score"
        if hasattr(predictions, 'conf_score') and isinstance(predictions.conf_score, pd.Series):
             try:
                  adata.obs[conf_col_name] = predictions.conf_score
                  log.info(f"Added '{conf_col_name}' to adata.obs.")
             except Exception as e:
                  log.error(f"Failed to assign '{conf_col_name}': {e}")


        # Add majority voting results if requested and present
        mv_col_name = f"{output_key_prefix}_majority_voting"
        if majority_voting:
             # Check within the DataFrame (if MV=True, pred_output should be DataFrame)
             if isinstance(pred_output, pd.DataFrame) and 'majority_voting' in pred_output.columns:
                  try:
                       adata.obs[mv_col_name] = pred_output['majority_voting'].astype('category')
                       log.info(f"Added '{mv_col_name}' to adata.obs.")
                  except Exception as e:
                       log.error(f"Failed to assign '{mv_col_name}': {e}")
             # Check the predictions object attribute as fallback
             elif hasattr(predictions, 'majority_voting') and isinstance(predictions.majority_voting, pd.Series):
                  try:
                       adata.obs[mv_col_name] = predictions.majority_voting.astype('category')
                       log.info(f"Added '{mv_col_name}' (from attribute) to adata.obs.")
                  except Exception as e:
                       log.error(f"Failed to assign '{mv_col_name}' from attribute: {e}")
             else:
                  log.warning(f"Majority voting requested but results column/attribute not found in prediction object.")
        # --- END FIX ---

        log.info("CellTypist annotation complete.")
    except Exception as e:
        log.error(f"An error occurred during CellTypist annotation: {e}", exc_info=True)
        raise RuntimeError(f"CellTypist annotation failed: {e}") from e
    return None
# --- Example __main__ block ---
if __name__ == '__main__':
    # ... (Keep the example execution block as it was) ...
     log.info("="*30)
     log.info("Running annotation.py directly for example")
     log.info("="*30)
     pbmc_markers_example = {
         'CD4 T cells': ['IL7R', 'CD3D', 'CD4'], 'CD8 T cells': ['CD8A', 'CD3D', 'CD8B'],
         'NK cells': ['GNLY', 'NKG7', 'KLRF1'], 'B cells': ['MS4A1', 'CD79A', 'CD19'],
         'Monocytes': ['CD14', 'LYZ', 'FCGR3A', 'MS4A7'], 'Dendritic Cells': ['FCER1A', 'CST3'],
         'Megakaryocytes': ['PPBP']
     }
     example_marker_file = "example_pbmc_markers.json"
     try:
         with open(example_marker_file, 'w') as f: json.dump(pbmc_markers_example, f, indent=2)
         log.info(f"Saved example marker dict to {example_marker_file}")
     except Exception: log.warning(f"Could not save example marker file {example_marker_file}")

     try:
         from scrnaseq_agent.data.loader import load_data
         from scrnaseq_agent.analysis.qc import calculate_qc_metrics, filter_cells_qc
         from scrnaseq_agent.analysis.preprocess import normalize_log1p, select_hvg
         from scrnaseq_agent.analysis.dimred import reduce_dimensionality
         from scrnaseq_agent.analysis.clustering import perform_clustering
         from scrnaseq_agent.analysis.dge import find_marker_genes
         import scanpy as sc
     except ImportError: log.error("Could not import functions."); sys.exit(1)

     current_dir = os.path.dirname(__file__)
     project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
     TEST_10X_DIR = os.path.join(project_root, "test_data", "filtered_gene_bc_matrices", "hg19")
     if not os.path.exists(TEST_10X_DIR): log.warning("Skipping"); sys.exit(0)

     adata_test = None; adata_original = None
     try:
         log.info("Running pipeline steps up to DGE...")
         adata_test = load_data(TEST_10X_DIR)
         adata_original = adata_test.copy()
         calculate_qc_metrics(adata_test, mito_gene_prefix="MT-", inplace=True)
         filter_cells_qc(adata_test, min_genes=200, max_pct_mito=15.0, inplace=True)
         normalize_log1p(adata_test, inplace=True)
         select_hvg(adata_test, n_top_genes=2000, subset=True, inplace=True)
         adata_test.raw = adata_original[adata_test.obs_names, adata_test.var_names].copy()
         sc.pp.scale(adata_test, max_value=10)
         reduce_dimensionality(adata_test, n_comps=50, inplace=True, random_state=0)
         perform_clustering(adata_test, resolution=0.8, random_state=0, calculate_umap=False, inplace=True)
         cluster_key = 'leiden'; dge_key = 'rank_genes_groups_raw'
         find_marker_genes(adata_test, groupby=cluster_key, use_raw=True, key_added=dge_key)
         log.info("Prerequisite steps complete.")

         log.info("\n--- Testing Marker Gene Overlap Annotation ---")
         loaded_markers = load_marker_dict(example_marker_file)
         if loaded_markers:
              annotate_cell_types(adata=adata_test, marker_dict=loaded_markers, groupby=cluster_key, rank_key=dge_key, annotation_key='pbmc_overlap', method='overlap_count')
              log.info("Marker overlap function executed.")
              overlap_res_key = 'pbmc_overlap_overlap_count'
              if overlap_res_key in adata_test.obs: print(f"Overlap results head ({overlap_res_key}):\n{adata_test.obs[[cluster_key, overlap_res_key]].head()}")

         log.info("\n--- Testing CellTypist Annotation ---")
         if CELLTYPIST_INSTALLED:
             try:
                 annotate_celltypist(adata=adata_test, model_name="Immune_All_Low.pkl", majority_voting=True, cluster_key_for_voting=cluster_key, output_key_prefix="ct_immune")
                 log.info("CellTypist annotation executed.")
                 log.info(f"adata.obs columns now: {list(adata_test.obs.columns)}")
                 mv_key = 'ct_immune_majority_voting'; pred_key = 'ct_immune_predicted_labels'
                 if mv_key in adata_test.obs: print(f"\nCellTypist MV Results:\n{pd.crosstab(adata_test.obs[cluster_key], adata_test.obs[mv_key])}")
                 elif pred_key in adata_test.obs: print(f"\nCellTypist Predicted Labels head:\n{adata_test.obs[pred_key].head()}")
             except Exception as celltypist_err: log.error(f"CellTypist failed in example: {celltypist_err}", exc_info=True)
         else: log.warning("CellTypist not installed, skipping example.")
     except Exception as e: log.error(f"\nError during annotation run: {e}", exc_info=True)
     finally:
          if os.path.exists(example_marker_file):
              try: os.remove(example_marker_file)
              except Exception: log.warning(f"Could not remove example marker file.")
     log.info("\nFinished annotation.py direct run example.")