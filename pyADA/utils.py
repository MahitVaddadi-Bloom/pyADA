"""
Utility functions for pyADA CLI and data processing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import csv
from typing import List, Dict, Any, Tuple, Optional, Union


def load_molecular_data(
    file_path: Union[str, Path], 
    fingerprint_cols: Optional[str] = None,
    activity_col: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load molecular fingerprint data from CSV file.
    
    Args:
        file_path: Path to CSV file
        fingerprint_cols: Comma-separated column names or range (e.g., "1-1024")
        activity_col: Activity column name
        
    Returns:
        Dictionary containing fingerprints and optionally activities
    """
    df = pd.read_csv(file_path)
    
    # Determine fingerprint columns
    if fingerprint_cols:
        if '-' in fingerprint_cols:
            # Range format like "1-1024" or "fp_1-fp_100"
            parts = fingerprint_cols.split('-')
            if len(parts) == 2:
                start_str, end_str = parts
                # Handle numeric ranges
                if start_str.isdigit() and end_str.isdigit():
                    start, end = int(start_str), int(end_str)
                    fp_cols = [str(i) for i in range(start, end + 1)]
                # Handle column name ranges like "fp_1-fp_100"
                elif start_str.startswith('fp_') and end_str.startswith('fp_'):
                    start_num = int(start_str.split('_')[1])
                    end_num = int(end_str.split('_')[1])
                    fp_cols = [f'fp_{i}' for i in range(start_num, end_num + 1)]
                else:
                    raise ValueError(f"Invalid range format: {fingerprint_cols}")
            else:
                raise ValueError(f"Invalid range format: {fingerprint_cols}")
        else:
            # Comma-separated format
            fp_cols = [col.strip() for col in fingerprint_cols.split(',')]
    else:
        # Auto-detect numeric columns (assume they are fingerprints)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if activity_col and activity_col in numeric_cols:
            numeric_cols.remove(activity_col)
        fp_cols = numeric_cols
    
    # Extract fingerprints
    if not fp_cols:
        raise ValueError("No fingerprint columns found")
        
    # Ensure all fingerprint columns exist
    missing_cols = [col for col in fp_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Fingerprint columns not found: {missing_cols}")
    
    fingerprints = df[fp_cols].values.tolist()
    
    result = {
        'fingerprints': fingerprints,
        'columns': fp_cols,
        'total_molecules': len(fingerprints)
    }
    
    # Add activity data if specified
    if activity_col:
        if activity_col not in df.columns:
            raise ValueError(f"Activity column '{activity_col}' not found")
        result['activities'] = df[activity_col].tolist()
        result['activity_column'] = activity_col
    
    return result


def validate_fingerprint_data(
    fingerprints: List[List], 
    return_issues: bool = False
) -> Union[bool, Tuple[bool, List[str]]]:
    """
    Validate molecular fingerprint data.
    
    Args:
        fingerprints: List of fingerprint vectors
        return_issues: Whether to return list of issues
        
    Returns:
        Boolean validity or tuple of (validity, issues)
    """
    issues = []
    
    if not fingerprints:
        issues.append("No fingerprints provided")
        return (False, issues) if return_issues else False
    
    # Check consistency
    first_length = len(fingerprints[0])
    for i, fp in enumerate(fingerprints):
        if len(fp) != first_length:
            issues.append(f"Fingerprint {i} has inconsistent length: {len(fp)} vs {first_length}")
    
    # Check for valid binary values (assuming binary fingerprints)
    for i, fp in enumerate(fingerprints):
        for j, bit in enumerate(fp):
            if not isinstance(bit, (int, float)) or bit not in [0, 1]:
                if len(issues) < 10:  # Limit number of issues reported
                    issues.append(f"Invalid value at fingerprint {i}, position {j}: {bit}")
                elif len(issues) == 10:
                    issues.append("... (more validation errors truncated)")
                    break
        if len(issues) > 10:
            break
    
    is_valid = len(issues) == 0
    return (is_valid, issues) if return_issues else is_valid


def calculate_similarity_matrix(
    fingerprints1: List[List], 
    fingerprints2: List[List],
    similarity_func: callable
) -> np.ndarray:
    """
    Calculate similarity matrix between two sets of fingerprints.
    
    Args:
        fingerprints1: First set of fingerprints
        fingerprints2: Second set of fingerprints  
        similarity_func: Similarity calculation function
        
    Returns:
        2D numpy array of similarities
    """
    n1, n2 = len(fingerprints1), len(fingerprints2)
    similarity_matrix = np.zeros((n1, n2))
    
    for i, fp1 in enumerate(fingerprints1):
        for j, fp2 in enumerate(fingerprints2):
            similarity_matrix[i, j] = similarity_func(fp1, fp2)
    
    return similarity_matrix


def export_results(
    results: Dict[str, Any], 
    output_path: Union[str, Path],
    format: str = 'csv'
) -> None:
    """
    Export analysis results to file.
    
    Args:
        results: Results dictionary
        output_path: Output file path
        format: Output format ('csv', 'json')
    """
    output_path = Path(output_path)
    
    if format == 'csv':
        # Create DataFrame with results
        df_data = {
            'molecule_id': range(len(results['similarities'])),
            'similarity': results['similarities'],
            'in_domain': results['in_domain']
        }
        df = pd.DataFrame(df_data)
        df.to_csv(output_path, index=False)
        
    elif format == 'json':
        # Convert numpy types to Python types for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                json_results[key] = value.item()
            else:
                json_results[key] = value
                
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
    
    else:
        raise ValueError(f"Unsupported format: {format}")


def calculate_ad_statistics(
    similarities: List[float], 
    threshold: float
) -> Dict[str, Any]:
    """
    Calculate Applicability Domain statistics.
    
    Args:
        similarities: List of similarity values
        threshold: AD threshold
        
    Returns:
        Dictionary of statistics
    """
    similarities = np.array(similarities)
    in_domain = similarities >= threshold
    
    stats = {
        'threshold': threshold,
        'total_molecules': len(similarities),
        'in_domain_count': int(np.sum(in_domain)),
        'out_domain_count': int(np.sum(~in_domain)),
        'domain_coverage': float(np.mean(in_domain)),
        'mean_similarity': float(np.mean(similarities)),
        'median_similarity': float(np.median(similarities)),
        'min_similarity': float(np.min(similarities)),
        'max_similarity': float(np.max(similarities)),
        'std_similarity': float(np.std(similarities))
    }
    
    return stats


def optimize_threshold(
    train_similarities: List[float],
    test_similarities: List[float], 
    train_activities: List[float],
    test_activities: List[float],
    threshold_range: Tuple[float, float] = (0.1, 0.9),
    n_steps: int = 50
) -> Dict[str, Any]:
    """
    Optimize AD threshold based on prediction performance.
    
    Args:
        train_similarities: Training set similarities
        test_similarities: Test set similarities
        train_activities: Training set activities
        test_activities: Test set activities
        threshold_range: Range of thresholds to test
        n_steps: Number of threshold steps
        
    Returns:
        Dictionary with optimization results
    """
    from .pyADA import Smetrics
    
    metrics = Smetrics()
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_steps)
    results = []
    
    for threshold in thresholds:
        # Calculate domain coverage
        in_domain = np.array(test_similarities) >= threshold
        coverage = np.mean(in_domain)
        
        if np.sum(in_domain) == 0:  # No molecules in domain
            continue
            
        # Calculate performance metrics for molecules in domain
        in_domain_pred = np.array(test_activities)[in_domain]
        in_domain_true = np.array(test_activities)[in_domain]  # Placeholder - would need actual predictions
        
        # For demonstration, calculate based on coverage and similarity
        quality_score = coverage * np.mean(np.array(test_similarities)[in_domain])
        
        results.append({
            'threshold': threshold,
            'coverage': coverage,
            'in_domain_count': int(np.sum(in_domain)),
            'quality_score': quality_score
        })
    
    # Find optimal threshold (balance between coverage and quality)
    if not results:
        return {'optimal_threshold': 0.7, 'results': []}
        
    results_df = pd.DataFrame(results)
    optimal_idx = results_df['quality_score'].idxmax()
    optimal_threshold = results_df.loc[optimal_idx, 'threshold']
    
    return {
        'optimal_threshold': optimal_threshold,
        'results': results,
        'optimization_summary': results_df.loc[optimal_idx].to_dict()
    }


def create_fingerprint_from_smiles(
    smiles: str, 
    fingerprint_type: str = 'morgan',
    n_bits: int = 1024
) -> List[int]:
    """
    Generate molecular fingerprint from SMILES.
    
    Note: This requires RDKit to be installed.
    
    Args:
        smiles: SMILES string
        fingerprint_type: Type of fingerprint ('morgan', 'maccs', 'topological')
        n_bits: Number of bits for fingerprint
        
    Returns:
        Binary fingerprint as list
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        if fingerprint_type == 'morgan':
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
        elif fingerprint_type == 'maccs':
            fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
        elif fingerprint_type == 'topological':
            fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=n_bits)
        else:
            raise ValueError(f"Unsupported fingerprint type: {fingerprint_type}")
            
        return list(fp)
        
    except ImportError:
        raise ImportError("RDKit is required for SMILES to fingerprint conversion. Install with: pip install rdkit")


def batch_similarity_calculation(
    query_fingerprints: List[List],
    reference_fingerprints: List[List],
    similarity_func: callable,
    batch_size: int = 1000
) -> List[float]:
    """
    Calculate maximum similarities for query molecules against reference set in batches.
    
    Args:
        query_fingerprints: Query fingerprints
        reference_fingerprints: Reference fingerprints
        similarity_func: Similarity function
        batch_size: Batch size for processing
        
    Returns:
        List of maximum similarities for each query
    """
    max_similarities = []
    
    for i in range(0, len(query_fingerprints), batch_size):
        batch_queries = query_fingerprints[i:i + batch_size]
        batch_max_sims = []
        
        for query_fp in batch_queries:
            max_sim = 0.0
            for ref_fp in reference_fingerprints:
                sim = similarity_func(query_fp, ref_fp)
                max_sim = max(max_sim, sim)
            batch_max_sims.append(max_sim)
            
        max_similarities.extend(batch_max_sims)
    
    return max_similarities


def generate_sample_data(
    n_molecules: int = 100,
    n_bits: int = 1024,
    activity_range: Tuple[float, float] = (0.0, 10.0),
    output_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Generate sample molecular fingerprint data for testing.
    
    Args:
        n_molecules: Number of molecules to generate
        n_bits: Number of fingerprint bits
        activity_range: Range for activity values
        output_path: Optional path to save CSV
        
    Returns:
        DataFrame with sample data
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate random binary fingerprints
    fingerprints = np.random.randint(0, 2, size=(n_molecules, n_bits))
    
    # Generate correlated activities (some correlation with fingerprint density)
    fp_density = np.mean(fingerprints, axis=1)
    activities = (
        activity_range[0] + 
        (activity_range[1] - activity_range[0]) * fp_density +
        np.random.normal(0, 0.5, n_molecules)
    )
    activities = np.clip(activities, activity_range[0], activity_range[1])
    
    # Create DataFrame
    data = {}
    for i in range(n_bits):
        data[f'fp_{i+1}'] = fingerprints[:, i]
    data['activity'] = activities
    data['molecule_id'] = [f'mol_{i+1:04d}' for i in range(n_molecules)]
    
    df = pd.DataFrame(data)
    
    if output_path:
        df.to_csv(output_path, index=False)
    
    return df