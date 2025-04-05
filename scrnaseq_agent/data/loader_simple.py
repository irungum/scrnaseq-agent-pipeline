# scrnaseq_agent/data/loader_simple.py
# A simplified version of loader.py that doesn't require scanpy or anndata

import os
import logging

# Set up basic logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # Set level for basic feedback

def check_data_path(data_path: str) -> bool:
    """
    Checks if a data path exists and what type of data it contains.
    
    Args:
        data_path: Path to the data file or directory.
        
    Returns:
        True if the path exists, False otherwise.
    """
    log.info(f"Checking data path: {data_path}")
    
    if not isinstance(data_path, str):
        log.error(f"Expected data_path to be a string, but got {type(data_path)}")
        return False
    
    # Expand user path (e.g., ~/) and check if path exists
    expanded_path = os.path.expanduser(data_path)
    if not os.path.exists(expanded_path):
        log.error(f"Data path not found: {expanded_path}")
        return False
    
    # Check if it's a directory (likely 10x MTX)
    if os.path.isdir(expanded_path):
        log.info(f"Path is a directory: {expanded_path}")
        
        # Check for expected 10x files (basic check)
        mtx_files = [
            os.path.join(expanded_path, 'matrix.mtx.gz'),
            os.path.join(expanded_path, 'matrix.mtx')
        ]
        features_files = [
            os.path.join(expanded_path, 'features.tsv.gz'),
            os.path.join(expanded_path, 'features.tsv'),
            os.path.join(expanded_path, 'genes.tsv.gz'),
            os.path.join(expanded_path, 'genes.tsv')
        ]
        barcodes_files = [
            os.path.join(expanded_path, 'barcodes.tsv.gz'),
            os.path.join(expanded_path, 'barcodes.tsv')
        ]
        
        # Log which files are found
        for file_list, file_type in [
            (mtx_files, "Matrix"), 
            (features_files, "Features/Genes"), 
            (barcodes_files, "Barcodes")
        ]:
            for file_path in file_list:
                if os.path.exists(file_path):
                    log.info(f"Found {file_type} file: {file_path}")
                    break
            else:
                log.warning(f"No {file_type} file found in expected locations")
        
        return True
    
    # Check if it's an H5AD file
    elif os.path.isfile(expanded_path) and expanded_path.lower().endswith(".h5ad"):
        log.info(f"Path is an H5AD file: {expanded_path}")
        return True
    
    # Check if it's an MTX file directly
    elif os.path.isfile(expanded_path) and expanded_path.lower().endswith(".mtx.gz"):
        log.info(f"Path is an MTX file: {expanded_path}")
        return True
    
    # Other file type
    else:
        log.warning(f"Unrecognized file format or path type: {expanded_path}")
        return True  # Path exists but format is unknown


# --- Example Usage (for testing within this file) ---
if __name__ == '__main__':
    # Create dummy data or point to real test data
    test_h5ad_path = "/Volumes/PortableSSD/scrnaseq_agent/test_data/pbmc3k.h5ad"  # H5AD file
    test_10x_dir = "/Volumes/PortableSSD/scrnaseq_agent/test_data/filtered_gene_bc_matrices/hg19/"  # 10x directory
    invalid_path = "path/that/does/not/exist"
    text_file_path = "README.md"  # Example of wrong file type
    
    # Test checking H5AD
    print(f"\n--- Testing H5AD path: {test_h5ad_path} ---")
    check_data_path(test_h5ad_path)
    
    # Test checking 10x Directory
    print(f"\n--- Testing 10x MTX path: {test_10x_dir} ---")
    check_data_path(test_10x_dir)
    
    # Test invalid path
    print(f"\n--- Testing invalid path: {invalid_path} ---")
    check_data_path(invalid_path)
    
    # Test wrong file type
    print(f"\n--- Testing wrong file type: {text_file_path} ---")
    check_data_path(text_file_path)
