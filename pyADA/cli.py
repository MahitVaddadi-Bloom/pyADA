"""
Command-line interface for pyADA (Python Applicability Domain Analyzer).
"""

import click
import pandas as pd
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
import json
import sys

from .pyADA import Smetrics, Similarity, ApplicabilityDomain, LeverageAD
from .utils import (
    validate_fingerprint_data,
    calculate_similarity_matrix,
    export_results,
    load_molecular_data,
)

console = Console()

__version__ = "1.2.0"


@click.group()
@click.version_option(version=__version__, prog_name="pyADA")
def main():
    """
    pyADA - Python Applicability Domain Analyzer
    
    A cheminformatics package for performing Applicability Domain analysis
    of molecular fingerprints based on similarity calculations.
    """
    pass


@main.command()
@click.argument('train_file', type=click.Path(exists=True))
@click.argument('test_file', type=click.Path(exists=True))
@click.option('--similarity', '-s', 
              type=click.Choice(['tanimoto', 'dice', 'cosine', 'euclidean']), 
              default='tanimoto',
              help='Similarity metric to use')
@click.option('--threshold', '-t', type=float, default=0.7,
              help='Similarity threshold for AD analysis')
@click.option('--output', '-o', type=click.Path(), 
              help='Output file for results (CSV/JSON)')
@click.option('--output-format', type=click.Choice(['csv', 'json', 'table']),
              default='table', help='Output format')
@click.option('--fingerprint-cols', '-f', help='Comma-separated fingerprint column names or range (e.g., "1-1024")')
@click.option('--activity-col', '-a', help='Activity column name')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def analyze(train_file, test_file, similarity, threshold, output, output_format, 
           fingerprint_cols, activity_col, verbose):
    """
    Perform Applicability Domain analysis on molecular fingerprints.
    
    TRAIN_FILE: CSV file containing training set fingerprints
    TEST_FILE: CSV file containing test set fingerprints
    
    Examples:
        pyada analyze train.csv test.csv --similarity tanimoto --threshold 0.7
        pyada analyze train.csv test.csv -s dice -t 0.8 -o results.csv --output-format csv
        pyada analyze train.csv test.csv -f "fp_1,fp_2,fp_3" -a "activity"
    """
    console.print(f"[bold blue]pyADA Applicability Domain Analysis[/bold blue]")
    console.print(f"Similarity metric: {similarity}")
    console.print(f"Threshold: {threshold}")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            
            # Load data
            task = progress.add_task("Loading training data...", total=None)
            train_data = load_molecular_data(train_file, fingerprint_cols, activity_col)
            progress.update(task, description="Loading test data...")
            test_data = load_molecular_data(test_file, fingerprint_cols, activity_col)
            
            progress.update(task, description="Validating data...")
            validate_fingerprint_data(train_data['fingerprints'])
            validate_fingerprint_data(test_data['fingerprints'])
            
            progress.update(task, description="Calculating applicability domain...")
            
            # Perform AD analysis
            ad_analyzer = ApplicabilityDomain()
            
            # Choose similarity metric
            sim_calculator = Similarity()
            if similarity == 'tanimoto':
                sim_func = sim_calculator.tanimoto_similarity
            elif similarity == 'dice':
                sim_func = sim_calculator.dice_similarity
            elif similarity == 'cosine':
                sim_func = sim_calculator.cosine_similarity
            elif similarity == 'euclidean':
                sim_func = sim_calculator.euclidean_similarity
            
            # Calculate similarities
            similarities = []
            train_fps = train_data['fingerprints']
            test_fps = test_data['fingerprints']
            
            for test_fp in test_fps:
                max_sim = 0
                for train_fp in train_fps:
                    sim = sim_func(test_fp, train_fp)
                    max_sim = max(max_sim, sim)
                similarities.append(max_sim)
            
            # Create results
            results = {
                'similarities': similarities,
                'in_domain': [sim >= threshold for sim in similarities],
                'threshold': threshold,
                'similarity_metric': similarity,
                'total_molecules': len(similarities),
                'in_domain_count': sum(sim >= threshold for sim in similarities),
                'out_domain_count': sum(sim < threshold for sim in similarities)
            }
            
            progress.update(task, description="Analysis complete!")
        
        # Display results
        if output_format == 'table' or not output:
            display_results_table(results)
        
        # Export results
        if output:
            export_results(results, output, output_format)
            console.print(f"[green]Results saved to: {output}[/green]")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


@main.command()
@click.argument('fingerprint1')
@click.argument('fingerprint2')
@click.option('--similarity', '-s', 
              type=click.Choice(['tanimoto', 'dice', 'cosine', 'euclidean']), 
              default='tanimoto',
              help='Similarity metric to use')
@click.option('--format', '-f', type=click.Choice(['binary', 'list']), 
              default='binary',
              help='Fingerprint format (binary string or comma-separated list)')
def similarity(fingerprint1, fingerprint2, similarity, format):
    """
    Calculate similarity between two molecular fingerprints.
    
    Examples:
        pyada similarity "110101" "111001" --similarity tanimoto
        pyada similarity "1,1,0,1,0,1" "1,1,1,0,0,1" --format list --similarity dice
    """
    try:
        # Parse fingerprints
        if format == 'binary':
            fp1 = [int(x) for x in fingerprint1]
            fp2 = [int(x) for x in fingerprint2]
        else:  # list format
            fp1 = [int(x.strip()) for x in fingerprint1.split(',')]
            fp2 = [int(x.strip()) for x in fingerprint2.split(',')]
        
        # Calculate similarity
        sim_calculator = Similarity()
        if similarity == 'tanimoto':
            sim_value = sim_calculator.tanimoto_similarity(fp1, fp2)
        elif similarity == 'dice':
            sim_value = sim_calculator.dice_similarity(fp1, fp2)
        elif similarity == 'cosine':
            sim_value = sim_calculator.cosine_similarity(fp1, fp2)
        elif similarity == 'euclidean':
            sim_value = sim_calculator.euclidean_similarity(fp1, fp2)
        
        console.print(f"[bold green]{similarity.title()} Similarity: {sim_value:.4f}[/bold green]")
        
    except Exception as e:
        console.print(f"[red]Error calculating similarity: {e}[/red]")
        sys.exit(1)


@main.command()
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--fingerprint-cols', '-f', help='Comma-separated fingerprint column names or range')
@click.option('--output-format', type=click.Choice(['table', 'json']),
              default='table', help='Output format')
def validate(data_file, fingerprint_cols, output_format):
    """
    Validate molecular fingerprint data.
    
    Examples:
        pyada validate data.csv
        pyada validate data.csv --fingerprint-cols "fp_1,fp_2,fp_3"
    """
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Loading and validating data...", total=None)
            
            data = load_molecular_data(data_file, fingerprint_cols)
            fingerprints = data['fingerprints']
            
            # Validation
            is_valid, issues = validate_fingerprint_data(fingerprints, return_issues=True)
            
            progress.update(task, description="Validation complete!")
        
        # Display results
        validation_results = {
            'file': str(data_file),
            'total_molecules': len(fingerprints),
            'fingerprint_length': len(fingerprints[0]) if fingerprints else 0,
            'is_valid': is_valid,
            'issues': issues
        }
        
        if output_format == 'table':
            display_validation_table(validation_results)
        else:
            console.print(json.dumps(validation_results, indent=2))
            
    except Exception as e:
        console.print(f"[red]Error during validation: {e}[/red]")
        sys.exit(1)


@main.command()
@click.option('--show-dependencies', is_flag=True, help='Show dependency versions')
def info(show_dependencies):
    """Display pyADA version and system information."""
    console.print(f"[bold blue]pyADA v{__version__}[/bold blue]")
    console.print(f"Python: {sys.version.split()[0]}")
    console.print(f"Platform: {sys.platform}")
    
    if show_dependencies:
        console.print("\n[bold]Dependencies:[/bold]")
        deps = {
            'numpy': np.__version__,
            'pandas': pd.__version__,
        }
        
        try:
            import scipy
            deps['scipy'] = scipy.__version__
        except ImportError:
            deps['scipy'] = 'Not installed'
            
        try:
            import sklearn
            deps['scikit-learn'] = sklearn.__version__
        except ImportError:
            deps['scikit-learn'] = 'Not installed'
            
        try:
            import plotly
            deps['plotly'] = plotly.__version__
        except ImportError:
            deps['plotly'] = 'Not installed'
        
        for name, version in deps.items():
            console.print(f"  {name}: {version}")


def display_results_table(results):
    """Display AD analysis results in a table format."""
    table = Table(title="Applicability Domain Analysis Results")
    
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Similarity Metric", results['similarity_metric'].title())
    table.add_row("Threshold", f"{results['threshold']:.3f}")
    table.add_row("Total Molecules", str(results['total_molecules']))
    table.add_row("In Domain", str(results['in_domain_count']))
    table.add_row("Out of Domain", str(results['out_domain_count']))
    table.add_row("Domain Coverage", f"{results['in_domain_count']/results['total_molecules']*100:.1f}%")
    
    console.print(table)
    
    # Show similarity distribution
    similarities = results['similarities']
    if similarities:
        console.print(f"\n[bold]Similarity Statistics:[/bold]")
        console.print(f"  Mean: {np.mean(similarities):.3f}")
        console.print(f"  Median: {np.median(similarities):.3f}")
        console.print(f"  Min: {np.min(similarities):.3f}")
        console.print(f"  Max: {np.max(similarities):.3f}")


def display_validation_table(results):
    """Display validation results in a table format."""
    table = Table(title="Data Validation Results")
    
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("File", results['file'])
    table.add_row("Total Molecules", str(results['total_molecules']))
    table.add_row("Fingerprint Length", str(results['fingerprint_length']))
    table.add_row("Valid", "✓" if results['is_valid'] else "✗")
    
    if results['issues']:
        table.add_row("Issues", str(len(results['issues'])))
    
    console.print(table)
    
    if results['issues']:
        console.print("\n[bold red]Issues Found:[/bold red]")
        for issue in results['issues']:
            console.print(f"  • {issue}")


if __name__ == "__main__":
    main()