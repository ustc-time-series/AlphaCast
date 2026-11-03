from __future__ import annotations

from typing import List, Tuple
import numpy as np

from ..utils.time import ClusterEntry


def zscore(x: np.ndarray) -> np.ndarray:
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma == 0:
        return np.zeros_like(x)
    return (x - mu) / (sigma + 1e-8)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(float)
    b = b.astype(float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the Euclidean distance between two vectors."""
    a = a.astype(float)
    b = b.astype(float)
    return float(np.linalg.norm(a - b))


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the dynamic time warping distance."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float('inf')
    
    # Create the DTW matrix
    dtw_matrix = np.full((n + 1, m + 1), float('inf'))
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i-1] - b[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],      # insertion
                dtw_matrix[i, j-1],      # deletion
                dtw_matrix[i-1, j-1]     # match
            )
    
    return dtw_matrix[n, m] / max(n, m)  # Normalize by sequence length


def trend_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute trend similarity."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    
    # Compute first-order differences (trend)
    diff_a = np.diff(a)
    diff_b = np.diff(b)
    
    # Compute trend direction consistency
    signs_a = np.sign(diff_a)
    signs_b = np.sign(diff_b)
    
    # Trend direction consistency
    direction_consistency = np.mean(signs_a == signs_b)
    
    # Trend strength similarity
    strength_a = np.std(diff_a) if len(diff_a) > 0 else 0
    strength_b = np.std(diff_b) if len(diff_b) > 0 else 0
    
    if strength_a + strength_b == 0:
        strength_similarity = 1.0
    else:
        strength_similarity = 1.0 - abs(strength_a - strength_b) / (strength_a + strength_b)
    
    return 0.7 * direction_consistency + 0.3 * strength_similarity


def volatility_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute volatility similarity."""
    vol_a = np.std(a) if len(a) > 0 else 0
    vol_b = np.std(b) if len(b) > 0 else 0
    
    if vol_a + vol_b == 0:
        return 1.0
    
    return 1.0 - abs(vol_a - vol_b) / (vol_a + vol_b)


def pattern_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute pattern similarity based on autocorrelation."""
    if len(a) < 3 or len(b) < 3:
        return 0.0
    
    # Compute the autocorrelation function
    def autocorr(x, max_lag=None):
        if max_lag is None:
            max_lag = min(len(x) // 4, 10)
        
        x_centered = x - np.mean(x)
        autocorrs = []
        
        for lag in range(1, max_lag + 1):
            if lag >= len(x):
                break
            corr = np.corrcoef(x_centered[:-lag], x_centered[lag:])[0, 1]
            autocorrs.append(corr if not np.isnan(corr) else 0.0)
        
        return np.array(autocorrs)
    
    autocorr_a = autocorr(a)
    autocorr_b = autocorr(b)
    
    if len(autocorr_a) == 0 or len(autocorr_b) == 0:
        return 0.0
    
    # Align sequence lengths
    min_len = min(len(autocorr_a), len(autocorr_b))
    autocorr_a = autocorr_a[:min_len]
    autocorr_b = autocorr_b[:min_len]
    
    # Compute similarity between autocorrelation signatures
    return cosine_similarity(autocorr_a, autocorr_b)


def comprehensive_similarity(query: np.ndarray, candidate: np.ndarray) -> float:
    """Composite similarity metric that combines several time-series features."""
    
    # Standardize the input series
    query_norm = zscore(query)
    candidate_norm = zscore(candidate)
    
    # 1. Shape similarity (cosine similarity)
    shape_sim = cosine_similarity(query_norm, candidate_norm)
    
    # 2. Dynamic time warping similarity
    dtw_dist = dtw_distance(query_norm, candidate_norm)
    dtw_sim = 1.0 / (1.0 + dtw_dist)  # Convert distance into similarity
    
    # 3. Trend similarity
    trend_sim = trend_similarity(query, candidate)
    
    # 4. Volatility similarity
    vol_sim = volatility_similarity(query, candidate)
    
    # 5. Pattern similarity (autocorrelation)
    pattern_sim = pattern_similarity(query, candidate)
    
    # Weighted combination (adjust weights empirically if needed)
    weights = {
        'shape': 0.25,      # Shape similarity
        'dtw': 0.25,        # Dynamic time warping similarity
        'trend': 0.20,      # Trend similarity
        'volatility': 0.15, # Volatility similarity
        'pattern': 0.15     # Pattern similarity
    }
    
    total_sim = (
        weights['shape'] * max(0, shape_sim) +
        weights['dtw'] * dtw_sim +
        weights['trend'] * trend_sim +
        weights['volatility'] * vol_sim +
        weights['pattern'] * max(0, pattern_sim)
    )
    
    return total_sim


def top1_most_similar(query: np.ndarray, candidates: List[Tuple[np.ndarray, str]]) -> Tuple[str, float]:
    """
    Enhanced model selection that uses the composite similarity metric,
    combining shape, trend, volatility, and pattern features.
    """
    if len(candidates) == 0:
        return "SeasonalNaive", 0.0
    
    best_model = None
    best_sim = -1.0
    
    # Track similarity scores per model for debugging
    model_scores = {}
    
    for window_vec, model_name in candidates:
        try:
            # Use the composite similarity metric
            sim = comprehensive_similarity(query, window_vec)
            
            # Collect the score
            if model_name not in model_scores:
                model_scores[model_name] = []
            model_scores[model_name].append(sim)
            
            if sim > best_sim:
                best_sim = sim
                best_model = model_name
                
        except Exception as e:
            # Fall back to cosine similarity if the composite metric fails
            sim = cosine_similarity(zscore(query), zscore(window_vec))
            if sim > best_sim:
                best_sim = sim
                best_model = model_name
    
    return best_model or "SeasonalNaive", best_sim

def top1_most_similar_neighbor(query: np.ndarray, candidates: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enhanced neighbor selection that relies on the composite similarity metric.
    """
    if len(candidates) == 0:
        return query, query  # Return the query as a default
    
    best_lookback = None
    best_pred = None
    best_sim = -1.0
    
    for lookback, pred in candidates:
        try:
            # Use the composite similarity metric
            sim = comprehensive_similarity(query, lookback)
            
            if sim > best_sim:
                best_sim = sim
                best_lookback = lookback
                best_pred = pred
                
        except Exception as e:
            # Fall back to the original cosine similarity approach on failure
            sim = cosine_similarity(zscore(query), zscore(lookback))
            if sim > best_sim:
                best_sim = sim
                best_lookback = lookback
                best_pred = pred
    
    # If no suitable neighbor is found, return the query itself
    if best_lookback is None or best_pred is None:
        return query, query
    
    return best_lookback, best_pred

def top1_most_similar_cluster(query: np.ndarray, candidates: List[ClusterEntry]) -> ClusterEntry:
    best_cluster = None
    best_sim = float("inf")
    for cluster in candidates:
        sim = euclidean_distance(query, np.asarray(cluster.window, dtype=float))
        if sim < best_sim:
            best_sim = sim
            best_cluster = cluster
    return best_cluster or candidates[0]
