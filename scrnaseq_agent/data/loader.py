# scrnaseq_agent/data/loader.py

import scanpy as sc
import anndata as ad
import os
import logging

# Set up basic logging
# TODO: Integrate with a more robust logging setup later (perhaps in utils)
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) # Set level for basic feedback

def load_data(data_path: str, cache: bool = False) -> ad.AnnData:
    """
    Loads single-cell RNA sequencing data into an AnnData object.

    Supports:
        - 10x Genomics MTX directory (containing matrix.mtx.gz, features.tsv.gz, barcodes.tsv.gz)
        - AnnData (.h5ad) file

    Args:
        data_path: Path to the data file or directory (local path initially).
        cache: Whether to cache the loaded AnnData object (uses scanpy's cache).
               Defaults to False.

    Returns:
        An AnnData object containing the loaded data.

    Raises:
        FileNotFoundError: If the data_path does not exist.
        ValueError: If the data format is not recognized or loading fails.
        TypeError: If data_path is not a string.
    """
    log.info(f"Attempting to load data from: {data_path}")

    if not isinstance(data_path, str):
        raise TypeError(f"Expected data_path to be a string, but got {type(data_path)}")

    # Expand user path (e.g., ~/) and check if path exists
    expanded_path = os.path.expanduser(data_path)
    if not os.path.exists(expanded_path):
        raise FileNotFoundError(f"Data path not found: {expanded_path}")

    try:
        # Check if it's a directory (likely 10x MTX)
        if os.path.isdir(expanded_path):
            log.info("Detected directory, attempting to load as 10x MTX format.")
            # Check for expected 10x files (basic check)
            mtx_file = os.path.join(expanded_path, 'matrix.mtx.gz')
            features_file = os.path.join(expanded_path, 'features.tsv.gz') # or genes.tsv
            barcodes_file = os.path.join(expanded_path, 'barcodes.tsv.gz')

            # Allow for older 'genes.tsv' naming
            if not os.path.exists(features_file):
                 features_file_alt = os.path.join(expanded_path, 'genes.tsv.gz')
                 if os.path.exists(features_file_alt):
                     features_file = features_file_alt
                 else:
                     # Try unzipped versions too? For now, let scanpy handle it.
                     pass # Let scanpy try and raise the error if files are truly missing


            # Let scanpy.read_10x_mtx handle the actual file existence check inside dir
            adata = sc.read_10x_mtx(
                expanded_path,
                var_names='gene_symbols', # Standard for 10x
                cache=cache
            )
            log.info(f"Successfully loaded 10x MTX data. Shape: {adata.shape}")
            # Ensure unique var names (important downstream)
            adata.var_names_make_unique()
            return adata

        # Check if it's an H5AD file
        elif os.path.isfile(expanded_path) and expanded_path.lower().endswith(".h5ad"):
            log.info("Detected .h5ad file, attempting to load.")
            adata = sc.read_h5ad(expanded_path)
            log.info(f"Successfully loaded .h5ad file. Shape: {adata.shape}")
            # Ensure unique var names (can happen if saved incorrectly)
            adata.var_names_make_unique()
            return adata

        # Check if it's an MTX file directly (less common for 10x structure)
        elif os.path.isfile(expanded_path) and expanded_path.lower().endswith(".mtx.gz"):
             log.warning("Detected .mtx.gz file directly. Requires corresponding features/barcodes files in the same directory.")
             # This case requires more assumptions or parameters about where feature/barcode files are.
             # For now, let's raise an error, guiding the user to provide the directory.
             raise ValueError("Loading a single .mtx.gz file is ambiguous. Please provide the path to the directory containing matrix.mtx.gz, features.tsv.gz, and barcodes.tsv.gz.")

        else:
            raise ValueError(f"Unrecognized file format or path type: {expanded_path}. Expecting a directory (for 10x MTX) or an .h5ad file.")

    except FileNotFoundError as e:
        # Catch specific errors from scanpy if files inside dir are missing
        log.error(f"File not found during loading process: {e}")
        raise FileNotFoundError(f"Required file missing within {expanded_path}: {e}") from e
    except Exception as e:
        log.error(f"Failed to load data from {expanded_path}: {e}", exc_info=True) # Log traceback
        raise ValueError(f"An error occurred during data loading: {e}") from e


# --- Example Usage (for testing within this file) ---
if __name__ == '__main__':
    # Create dummy data or point to real test data
    # IMPORTANT: Replace with actual paths to *your* test data
    test_h5ad_path = "/Volumes/PortableSSD/scrnaseq_agent/test_data/pbmc3k.h5ad"  # H5AD file
    test_10x_dir = "/Volumes/PortableSSD/scrnaseq_agent/test_data/filtered_gene_bc_matrices/hg19/" # 10x directory with matrix.mtx, genes.tsv, barcodes.tsv
    invalid_path = "path/that/does/not/exist"
    text_file_path = "README.md" # Example of wrong file type

    # Test loading H5AD
    try:
        if os.path.exists(test_h5ad_path):
            print(f"\n--- Testing H5AD load: {test_h5ad_path} ---")
            adata_h5ad = load_data(test_h5ad_path)
            print("H5AD load successful:")
            print(adata_h5ad)
        else:
            print(f"\nSkipping H5AD test, path not found: {test_h5ad_path}")
    except Exception as e:
        print(f"H5AD load failed: {e}")

    # Test loading 10x Directory
    try:
        if os.path.exists(test_10x_dir):
            print(f"\n--- Testing 10x MTX load: {test_10x_dir} ---")
            adata_10x = load_data(test_10x_dir)
            print("10x MTX load successful:")
            print(adata_10x)
        else:
            print(f"\nSkipping 10x test, path not found: {test_10x_dir}")
    except Exception as e:
        print(f"10x MTX load failed: {e}")

    # Test invalid path
    try:
        print(f"\n--- Testing invalid path: {invalid_path} ---")
        load_data(invalid_path)
    except FileNotFoundError as e:
        print(f"Caught expected error for invalid path: {e}")
    except Exception as e:
        print(f"Caught unexpected error for invalid path: {e}")

    # Test wrong file type
    try:
        print(f"\n--- Testing wrong file type: {text_file_path} ---")
        load_data(text_file_path)
    except ValueError as e:
         print(f"Caught expected error for wrong file type: {e}")
    except Exception as e:
        print(f"Caught unexpected error for wrong file type: {e}")
