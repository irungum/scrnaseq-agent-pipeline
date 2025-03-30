<<<<<<< HEAD
# scrnaseq-agent-pipeline
=======
# scrnaseq_agent

A Python package for automating single-cell RNA-seq analysis workflows.

## Features

- Data loading and validation
- Quality control and filtering
- Preprocessing and normalization
- Dimensionality reduction
- Clustering and visualization
- Command-line interface for workflow automation

## Installation

### Using pip

```bash
pip install scrnaseq_agent
```

### Using conda

```bash
conda env create -f environment.yml
conda activate scrnaseq_agent
```

### Development installation

```bash
git clone https://github.com/yourusername/scrnaseq_agent.git
cd scrnaseq_agent
pip install -e .
```

## Usage

### Command-line interface

```bash
scrnaseq-agent run-analysis config/default_config.yaml output/
```

### Python API

```python
from scrnaseq_agent.data.loader import load_data
from scrnaseq_agent.analysis.qc import calculate_qc_metrics

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
```

See the [examples](examples/) directory for more detailed usage examples.

## Configuration

The package uses YAML configuration files to specify analysis parameters. See [config/default_config.yaml](config/default_config.yaml) for an example configuration.

## Testing

Run the test suite:

```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
>>>>>>> c328447 (Initial project structure and environment setup)
