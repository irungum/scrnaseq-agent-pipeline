# scrnaseq_agent/analysis/dimred.py

import scanpy as sc
import anndata as ad
import logging
import numpy as np
import os
import warnings
from pathlib import Path

# --- scVI Import Handling (Keep for future reference, but commented) ---
# try:
#     import scvi
#     SCVI_INSTALLED = True
# except ImportError:
#     scvi = None
#     SCVI_INSTALLED = False
# --- END ---


log = logging.getLogger(__name__)
# Basic config only runs if the script is executed directly.
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    import sys
    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# --- PCA Function (Keep as is) ---
def reduce_dimensionality(
    adata: ad.AnnData,
    n_comps: int = 50,
    random_state: int = 0,
    inplace: bool = True
) -> ad.AnnData | None:
    """
    Performs principal component analysis (PCA) to reduce the dimensionality.

    Uses scanpy.tl.pca. Stores PCA results in adata.obsm['X_pca'] and
    related info (variance, variance ratio, loadings) in adata.uns['pca']
    and adata.varm['PCs']. Assumes data has been preprocessed
    (normalized, log1p, scaled).

    Args:
        adata: The annotated data matrix (typically after scaling).
               Requires `adata.X` to be present.
        n_comps: Number of principal components to compute. Defaults to 50.
               Must be less than min(n_obs, n_vars).
        random_state: Random seed for reproducibility of the SVD solver.
                      Defaults to 0.
        inplace: Modify AnnData object inplace. Defaults to True.

    Returns:
        If inplace=True, returns None. Otherwise, returns the modified AnnData
        object with PCA results.

    Raises:
        TypeError: If input `adata` is not an AnnData object.
        ValueError: If `n_comps` is not a positive integer or cannot be adjusted.
        AttributeError: If `adata.X` is not present or not suitable (e.g., None).
        RuntimeError: If the underlying scanpy PCA function fails.
    """
    if not isinstance(adata, ad.AnnData):
        raise TypeError("Input 'adata' must be an AnnData object.")
    if not isinstance(n_comps, int) or n_comps <= 0:
        raise ValueError("Argument 'n_comps' must be a positive integer.")

    log.info(f"Performing PCA with n_comps={n_comps}, random_state={random_state}...")

    if adata.X is None:
         raise AttributeError("Cannot perform PCA: AnnData object does not have a suitable '.X' attribute.")

    min_dim = min(adata.shape)
    if n_comps >= min_dim:
        adjusted_n_comps = min_dim - 1
        if adjusted_n_comps <= 0:
             raise ValueError(f"Cannot compute PCA. Input data has shape {adata.shape}, "
                              f"requiring n_comps < {min_dim}, but minimum is 1.")

        warning_message = (
            f"Requested n_comps ({n_comps}) >= smallest dimension ({min_dim}). "
            f"Adjusting n_comps to {adjusted_n_comps}."
        )
        warnings.warn(warning_message, UserWarning, stacklevel=2)
        log.warning(warning_message)
        n_comps = adjusted_n_comps

    adata_work = adata if inplace else adata.copy()
    log.debug(f"Operating {'inplace' if inplace else 'on a copy'}.")

    try:
        sc.tl.pca(
            adata_work, n_comps=n_comps, svd_solver='arpack',
            random_state=random_state, zero_center=True, copy=False
        )

        if 'X_pca' not in adata_work.obsm: raise RuntimeError("PCA calc finished but 'X_pca' not found.")
        if 'pca' not in adata_work.uns or 'variance_ratio' not in adata_work.uns['pca']: raise RuntimeError("PCA calc finished but variance info not found.")
        if 'PCs' not in adata_work.varm: log.warning("PCA calc finished but 'PCs' (loadings) not found.")

        log.info(f"PCA completed. Results in .obsm['X_pca'] ({adata_work.obsm['X_pca'].shape}), .uns['pca'], .varm['PCs'].")

    except ValueError as ve: log.error(f"ValueError during PCA: {ve}", exc_info=True); raise ValueError(f"Input value error during PCA: {ve}") from ve
    except Exception as e: log.error(f"Unexpected error during PCA: {e}", exc_info=True); raise RuntimeError(f"Failed PCA: {e}") from e

    if not inplace: return adata_work
    else: return None

# --- scVI FUNCTION (Commented Out) ---
# def run_scvi_latent(
#     adata: ad.AnnData,
#     n_latent: int = 10,
#     max_epochs: int = 400,
#     save_model_dir: str | None = None,
#     output_key: str = 'X_scvi',
#     use_gpu: bool = False,
#     batch_key: str | None = None,
#     **train_kwargs
# ) -> None:
#     """ Computes latent representation using scVI. Modifies adata inplace. """
#     if not SCVI_INSTALLED: raise ImportError("scvi-tools not installed.")
#     if not isinstance(adata, ad.AnnData): raise TypeError("Input 'adata' must be an AnnData object.")
#     if adata.raw is None: raise AttributeError("scVI requires raw counts, but adata.raw is None.")
#
#     log.info(f"Running scVI for latent space generation (n_latent={n_latent})...")
#     # Data checks
#     if hasattr(adata.raw.X, 'data'):
#         if np.any(adata.raw.X.data < 0) or not np.all(np.modf(adata.raw.X.data)[0] == 0): log.warning("Data in adata.raw.X does not appear to be integer counts.")
#     elif np.any(adata.raw.X < 0) or not np.all(np.modf(adata.raw.X)[0] == 0): log.warning("Data in adata.raw.X does not appear to be integer counts.")
#
#     try:
#         log.debug("Setting up scvi-tools model...")
#         scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key)
#         log.debug("scvi-tools setup complete.")
#         model = scvi.model.SCVI(adata, n_latent=n_latent)
#         if use_gpu: accelerator, devices = "auto", "auto"; log.info("Attempting GPU training.")
#         else: accelerator, devices = "cpu", 1; log.info("Using CPU training.")
#         final_train_kwargs = {"accelerator": accelerator, "devices": devices, **train_kwargs}
#         log.info(f"Training scVI model for max {max_epochs} epochs...")
#         model.train(max_epochs=max_epochs, **final_train_kwargs)
#         log.info("scVI training complete.")
#         latent_rep = model.get_latent_representation()
#         adata.obsm[output_key] = latent_rep
#         log.info(f"scVI latent space stored in adata.obsm['{output_key}'] shape: {latent_rep.shape}")
#         if save_model_dir:
#              save_path = Path(save_model_dir); save_path.mkdir(parents=True, exist_ok=True)
#              model_save_path = save_path / "scvi_model"
#              model.save(str(model_save_path), overwrite=True)
#              log.info(f"Trained scVI model saved to {model_save_path}")
#     except Exception as e: log.error(f"Error during scVI processing: {e}", exc_info=True); raise RuntimeError(f"Failed scVI: {e}") from e

# --- Main block ---
if __name__ == '__main__':
     log.info("dimred.py script contains PCA function.")
     # if not SCVI_INSTALLED:
     #     log.warning("scvi-tools not installed, scVI functionality unavailable.")