"""
Command-line interface for scrnaseq_agent
"""

import click
import yaml
from pathlib import Path
from typing import Dict, Any

@click.group()
def cli():
    """scrnaseq_agent - A tool for single-cell RNA-seq analysis automation"""
    pass

@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
def run_analysis(config_file: str, output_dir: str):
    """Run the complete analysis pipeline using a configuration file"""
    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # TODO: Implement the analysis pipeline
    click.echo(f"Running analysis with config: {config_file}")
    click.echo(f"Output will be saved to: {output_dir}")

if __name__ == '__main__':
    cli() 