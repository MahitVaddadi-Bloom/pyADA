"""
NumPy compatibility and enhancement utilities for pyADA.
"""

import numpy as np
import warnings
from typing import Union, List, Tuple, Any
from packaging import version


def check_numpy_compatibility():
    """Check NumPy version compatibility and provide warnings if needed."""
    try:
        numpy_version = version.parse(np.__version__)
        
        if numpy_version >= version.parse("2.0.0"):
            # NumPy 2.x compatibility
            return True
        elif numpy_version >= version.parse("1.20.0"):
            # NumPy 1.20+ compatibility
            return True
        else:
            warnings.warn(
                f"NumPy version {np.__version__} detected. "
                "pyADA is optimized for NumPy >= 1.20.0. "
                "Consider upgrading for better performance and compatibility.",
                UserWarning
            )
            return False
    except Exception:
        warnings.warn(
            "Could not determine NumPy version. Some features may not work correctly.",
            UserWarning
        )
        return False


def ensure_numpy_array(data: Union[List, np.ndarray], dtype=None) -> np.ndarray:
    """
    Ensure data is a NumPy array with proper dtype.
    
    Args:
        data: Input data (list or array)
        dtype: Desired NumPy dtype
        
    Returns:
        NumPy array
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=dtype)
    elif dtype is not None and data.dtype != dtype:
        data = data.astype(dtype)
    
    return data


def validate_fingerprint_array(fingerprints: Union[List, np.ndarray]) -> Tuple[bool, str]:
    """
    Validate fingerprint data format and compatibility.
    
    Args:
        fingerprints: Fingerprint data
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Convert to numpy array
        fp_array = ensure_numpy_array(fingerprints)
        
        # Check dimensions
        if fp_array.ndim != 2:
            return False, f"Fingerprints must be 2D array, got {fp_array.ndim}D"
        
        # Check for empty array
        if fp_array.size == 0:
            return False, "Empty fingerprint array"
        
        # Check for consistent fingerprint lengths
        if fp_array.shape[1] == 0:
            return False, "Fingerprints have zero length"
        
        # Check for valid binary values (0 or 1)
        unique_vals = np.unique(fp_array)
        if not all(val in [0, 1] for val in unique_vals):
            return False, f"Fingerprints must contain only 0 and 1, found values: {unique_vals}"
        
        return True, ""
        
    except Exception as e:
        return False, f"Error validating fingerprints: {str(e)}"


def optimize_similarity_calculation(
    fp1: Union[List, np.ndarray], 
    fp2: Union[List, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimize fingerprint arrays for similarity calculations.
    
    Args:
        fp1: First fingerprint
        fp2: Second fingerprint
        
    Returns:
        Tuple of optimized numpy arrays
    """
    # Ensure numpy arrays with int dtype for better performance
    fp1 = ensure_numpy_array(fp1, dtype=np.int8)
    fp2 = ensure_numpy_array(fp2, dtype=np.int8)
    
    # Validate same length
    if len(fp1) != len(fp2):
        raise ValueError(f"Fingerprints must have same length: {len(fp1)} vs {len(fp2)}")
    
    return fp1, fp2


def batch_tanimoto_similarity(
    query_fps: Union[List, np.ndarray],
    reference_fps: Union[List, np.ndarray]
) -> np.ndarray:
    """
    Vectorized Tanimoto similarity calculation for better performance.
    
    Args:
        query_fps: Query fingerprints (n_queries x n_bits)
        reference_fps: Reference fingerprints (n_refs x n_bits)
        
    Returns:
        Similarity matrix (n_queries x n_refs)
    """
    query_fps = ensure_numpy_array(query_fps, dtype=np.int8)
    reference_fps = ensure_numpy_array(reference_fps, dtype=np.int8)
    
    # Ensure 2D arrays
    if query_fps.ndim == 1:
        query_fps = query_fps.reshape(1, -1)
    if reference_fps.ndim == 1:
        reference_fps = reference_fps.reshape(1, -1)
    
    # Calculate intersection (AND operation)
    # Using broadcasting: query_fps[:, None, :] & reference_fps[None, :, :]
    intersection = np.sum(
        query_fps[:, None, :] & reference_fps[None, :, :], 
        axis=2
    )
    
    # Calculate union
    union = np.sum(
        query_fps[:, None, :] | reference_fps[None, :, :], 
        axis=2
    )
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        similarities = intersection / union
        similarities = np.nan_to_num(similarities, nan=0.0)
    
    return similarities


def batch_dice_similarity(
    query_fps: Union[List, np.ndarray],
    reference_fps: Union[List, np.ndarray]
) -> np.ndarray:
    """
    Vectorized Dice similarity calculation.
    
    Args:
        query_fps: Query fingerprints (n_queries x n_bits)
        reference_fps: Reference fingerprints (n_refs x n_bits)
        
    Returns:
        Similarity matrix (n_queries x n_refs)
    """
    query_fps = ensure_numpy_array(query_fps, dtype=np.int8)
    reference_fps = ensure_numpy_array(reference_fps, dtype=np.int8)
    
    # Ensure 2D arrays
    if query_fps.ndim == 1:
        query_fps = query_fps.reshape(1, -1)
    if reference_fps.ndim == 1:
        reference_fps = reference_fps.reshape(1, -1)
    
    # Calculate intersection
    intersection = np.sum(
        query_fps[:, None, :] & reference_fps[None, :, :], 
        axis=2
    )
    
    # Calculate sums
    query_sums = np.sum(query_fps, axis=1)[:, None]
    ref_sums = np.sum(reference_fps, axis=1)[None, :]
    
    # Dice coefficient: 2 * |A ∩ B| / (|A| + |B|)
    denominator = query_sums + ref_sums
    
    with np.errstate(divide='ignore', invalid='ignore'):
        similarities = (2 * intersection) / denominator
        similarities = np.nan_to_num(similarities, nan=0.0)
    
    return similarities


def batch_cosine_similarity(
    query_fps: Union[List, np.ndarray],
    reference_fps: Union[List, np.ndarray]
) -> np.ndarray:
    """
    Vectorized Cosine similarity calculation.
    
    Args:
        query_fps: Query fingerprints (n_queries x n_bits)
        reference_fps: Reference fingerprints (n_refs x n_bits)
        
    Returns:
        Similarity matrix (n_queries x n_refs)
    """
    query_fps = ensure_numpy_array(query_fps, dtype=np.float32)
    reference_fps = ensure_numpy_array(reference_fps, dtype=np.float32)
    
    # Ensure 2D arrays
    if query_fps.ndim == 1:
        query_fps = query_fps.reshape(1, -1)
    if reference_fps.ndim == 1:
        reference_fps = reference_fps.reshape(1, -1)
    
    # Calculate dot product
    dot_product = np.dot(query_fps, reference_fps.T)
    
    # Calculate norms
    query_norms = np.linalg.norm(query_fps, axis=1)[:, None]
    ref_norms = np.linalg.norm(reference_fps, axis=1)[None, :]
    
    # Cosine similarity
    denominator = query_norms * ref_norms
    
    with np.errstate(divide='ignore', invalid='ignore'):
        similarities = dot_product / denominator
        similarities = np.nan_to_num(similarities, nan=0.0)
    
    return similarities


def memory_efficient_similarity(
    query_fps: Union[List, np.ndarray],
    reference_fps: Union[List, np.ndarray],
    similarity_metric: str = 'tanimoto',
    batch_size: int = 1000
) -> np.ndarray:
    """
    Memory-efficient similarity calculation for large datasets.
    
    Args:
        query_fps: Query fingerprints
        reference_fps: Reference fingerprints
        similarity_metric: Type of similarity ('tanimoto', 'dice', 'cosine')
        batch_size: Batch size for processing
        
    Returns:
        Maximum similarities for each query
    """
    query_fps = ensure_numpy_array(query_fps)
    reference_fps = ensure_numpy_array(reference_fps)
    
    if query_fps.ndim == 1:
        query_fps = query_fps.reshape(1, -1)
    if reference_fps.ndim == 1:
        reference_fps = reference_fps.reshape(1, -1)
    
    n_queries = query_fps.shape[0]
    max_similarities = np.zeros(n_queries)
    
    # Choose similarity function
    if similarity_metric == 'tanimoto':
        sim_func = batch_tanimoto_similarity
    elif similarity_metric == 'dice':
        sim_func = batch_dice_similarity
    elif similarity_metric == 'cosine':
        sim_func = batch_cosine_similarity
    else:
        raise ValueError(f"Unsupported similarity metric: {similarity_metric}")
    
    # Process in batches
    for i in range(0, n_queries, batch_size):
        end_idx = min(i + batch_size, n_queries)
        batch_queries = query_fps[i:end_idx]
        
        # Calculate similarities for this batch
        batch_similarities = sim_func(batch_queries, reference_fps)
        
        # Get maximum similarity for each query in batch
        max_similarities[i:end_idx] = np.max(batch_similarities, axis=1)
    
    return max_similarities


def numpy_array_to_fingerprints(arr: np.ndarray) -> List[List[int]]:
    """
    Convert NumPy array to list of fingerprint lists.
    
    Args:
        arr: NumPy array of fingerprints
        
    Returns:
        List of fingerprint lists
    """
    if arr.ndim == 1:
        return arr.astype(int).tolist()
    else:
        return arr.astype(int).tolist()


def fingerprints_to_numpy_array(fingerprints: List[List]) -> np.ndarray:
    """
    Convert list of fingerprints to NumPy array.
    
    Args:
        fingerprints: List of fingerprint lists
        
    Returns:
        NumPy array
    """
    return np.array(fingerprints, dtype=np.int8)


# Initialize compatibility check on import
_numpy_compatible = check_numpy_compatibility()