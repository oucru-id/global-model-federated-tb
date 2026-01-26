#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Matrix Completion: Soft Impute with anchor-guided.
"""

import pandas as pd
import numpy as np
import argparse
import json
import os
import warnings
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
warnings.filterwarnings('ignore')

try:
    from fancyimpute import SoftImpute
    FANCYIMPUTE_AVAILABLE = True
except ImportError:
    FANCYIMPUTE_AVAILABLE = False

try:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    SKLEARN_ITERATIVE_AVAILABLE = True
except ImportError:
    SKLEARN_ITERATIVE_AVAILABLE = False

try:
    from sklearn.impute import KNNImputer
    SKLEARN_KNN_AVAILABLE = True
except ImportError:
    SKLEARN_KNN_AVAILABLE = False


def load_distance_matrix(filepath):
    df = pd.read_csv(filepath, sep='\t', index_col=0)
    df = (df + df.T) / 2
    np.fill_diagonal(df.values, 0)
    return df


def identify_anchors_in_matrix(df, anchor_list):
    return [a for a in anchor_list if a in df.index and a in df.columns]


def calculate_correction_factor(matrices, anchors, primary_anchor):
    correction_factors = []
    
    common_anchors = set(anchors)
    for df in matrices:
        present = set(identify_anchors_in_matrix(df, anchors))
        common_anchors = common_anchors.intersection(present)
    
    common_anchors = list(common_anchors)
    
    if len(common_anchors) < 2:
        print("Less than 2 common anchors found.")
        return [1.0] * len(matrices)
    
    if primary_anchor not in common_anchors:
        primary_anchor = common_anchors[0]
        print(f"Using {primary_anchor} as primary anchor")
    
    ref_matrix = matrices[0]
    correction_factors.append(1.0)
    
    for i, df in enumerate(matrices[1:], 1):
        factors = []
        for a1 in common_anchors:
            for a2 in common_anchors:
                if a1 >= a2:
                    continue
                ref_dist = ref_matrix.loc[a1, a2]
                local_dist = df.loc[a1, a2]
                if local_dist > 0 and ref_dist > 0:
                    factors.append(ref_dist / local_dist)
        
        if factors:
            correction_factors.append(np.median(factors))
        else:
            correction_factors.append(1.0)
    
    return correction_factors


def apply_correction(df, factor):
    if factor == 1.0:
        return df.copy()
    
    corrected = df * factor
    np.fill_diagonal(corrected.values, 0)
    return corrected


def create_incomplete_matrix(matrices, lab_samples, anchors):
    all_samples = set()
    for samples in lab_samples:
        all_samples.update(samples)
    all_samples = sorted(list(all_samples))
    
    n = len(all_samples)
    
    global_matrix = pd.DataFrame(
        np.full((n, n), np.nan),
        index=all_samples,
        columns=all_samples
    )
    
    np.fill_diagonal(global_matrix.values, 0)
    
    known_mask = pd.DataFrame(
        np.zeros((n, n), dtype=bool),
        index=all_samples,
        columns=all_samples
    )
    np.fill_diagonal(known_mask.values, True)
    
    sample_to_lab = {}
    for lab_idx, samples in enumerate(lab_samples):
        for sample in samples:
            if sample not in sample_to_lab:
                sample_to_lab[sample] = []
            sample_to_lab[sample].append(lab_idx)
    
    for s1 in all_samples:
        for s2 in all_samples:
            if s1 == s2:
                continue
            
            values = []
            for lab_idx, df in enumerate(matrices):
                if s1 in df.index and s2 in df.index:
                    values.append(df.loc[s1, s2])
            
            if values:
                global_matrix.loc[s1, s2] = np.mean(values)
                known_mask.loc[s1, s2] = True
    
    return global_matrix, sample_to_lab, known_mask


def enforce_metric_properties_balanced(matrix, known_mask, max_iterations=5):

    matrix = np.maximum(matrix, 0)
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0)
    
    n = matrix.shape[0]
    known = known_mask.values if hasattr(known_mask, 'values') else known_mask
    
    for iteration in range(max_iterations):
        changes = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if known[i, j]:
                    continue
                
                upper_bounds = []
                lower_bounds = []
                
                for k in range(n):
                    if k == i or k == j:
                        continue
                    
                    d_ik = matrix[i, k]
                    d_kj = matrix[k, j]
                    
                    upper_bounds.append(d_ik + d_kj)
                    
                    lower_bounds.append(abs(d_ik - d_kj))
                
                if upper_bounds and lower_bounds:
                    upper = min(upper_bounds)
                    lower = max(lower_bounds)
                    
                    current = matrix[i, j]
                    
                    if current > upper:
                        new_val = upper
                        changes += 1
                    elif current < lower:
                        new_val = lower
                        changes += 1
                    else:
                        new_val = current
                    
                    matrix[i, j] = new_val
                    matrix[j, i] = new_val
        
        if changes == 0:
            break
        print(f"  Iteration {iteration + 1}: {changes} adjustments")
    
    return matrix


def impute_anchor_guided(incomplete_matrix, known_mask, anchors, sample_to_lab):

    print("Using Anchor-Guided Imputation")
    
    matrix = incomplete_matrix.copy()
    samples = list(matrix.index)
    known = known_mask.copy()
    
    present_anchors = [a for a in anchors if a in samples]
    print(f"  Using {len(present_anchors)} anchors for guided imputation")
    
    for i, s1 in enumerate(samples):
        for j, s2 in enumerate(samples):
            if i >= j:
                continue
            
            if not np.isnan(matrix.loc[s1, s2]):
                continue
            
            estimates = []
            weights = []
            
            for anchor in present_anchors:
                d1 = matrix.loc[s1, anchor]
                d2 = matrix.loc[s2, anchor]
                
                if not np.isnan(d1) and not np.isnan(d2):
                    upper = d1 + d2
                    lower = abs(d1 - d2)
                    
                    weight = 1.0 / (1.0 + min(d1, d2))
                    
                    estimate = (upper + lower) / 2 + (upper - lower) * 0.3
                    
                    estimates.append(estimate)
                    weights.append(weight)
            
            if estimates:
                weights = np.array(weights)
                weights = weights / weights.sum()
                imputed = np.average(estimates, weights=weights)
                
                matrix.loc[s1, s2] = imputed
                matrix.loc[s2, s1] = imputed
    
    known_values = matrix.values[known.values & (matrix.values > 0)]
    if len(known_values) > 0:
        mean_dist = np.mean(known_values)
        matrix = matrix.fillna(mean_dist)
    
    return matrix


def impute_softimpute_improved(incomplete_matrix, known_mask):
    print("Using SoftImpute")
    
    matrix_values = incomplete_matrix.values.copy()
    
    known_vals = matrix_values[~np.isnan(matrix_values) & (matrix_values > 0)]
    if len(known_vals) > 0:
        init_val = np.mean(known_vals)
    else:
        init_val = 1000
    
    nan_mask = np.isnan(matrix_values)
    matrix_values[nan_mask] = init_val
    
    imputer = SoftImpute(max_iters=200, convergence_threshold=0.0001, verbose=False)
    completed = imputer.fit_transform(matrix_values)
    
    return pd.DataFrame(
        completed,
        index=incomplete_matrix.index,
        columns=incomplete_matrix.columns
    )


def impute_knn_improved(incomplete_matrix, known_mask, n_neighbors=5):
    print(f"Using KNN Imputer (k={n_neighbors})")
    
    matrix_values = incomplete_matrix.values.copy()
    
    n_samples = matrix_values.shape[0]
    known_per_row = (~np.isnan(matrix_values)).sum(axis=1)
    min_known = known_per_row.min()
    
    effective_k = min(n_neighbors, max(1, min_known - 1))
    print(f"  Effective k: {effective_k}")
    
    imputer = KNNImputer(n_neighbors=effective_k, weights='distance')
    completed = imputer.fit_transform(matrix_values)
    
    return pd.DataFrame(
        completed,
        index=incomplete_matrix.index,
        columns=incomplete_matrix.columns
    )


def validate_and_report(original_matrix, completed_matrix, known_mask, anchors):
    samples = list(completed_matrix.index)
    present_anchors = [a for a in anchors if a in samples]
    
    anchor_errors = []
    for i, a1 in enumerate(present_anchors):
        for a2 in present_anchors[i+1:]:
            if known_mask.loc[a1, a2]:
                orig = original_matrix.loc[a1, a2]
                comp = completed_matrix.loc[a1, a2]
                if orig > 0:
                    error = abs(comp - orig) / orig
                    anchor_errors.append(error)
    
    if anchor_errors:
        print(f"  Anchor distance preservation error: {np.mean(anchor_errors)*100:.2f}%")
    
    known_vals = original_matrix.values[known_mask.values & ~np.isnan(original_matrix.values)]
    imputed_vals = completed_matrix.values[~known_mask.values]
    
    if len(known_vals) > 0 and len(imputed_vals) > 0:
        print(f"  Known distances: mean={np.mean(known_vals):.1f}, std={np.std(known_vals):.1f}")
        print(f"  Imputed distances: mean={np.mean(imputed_vals):.1f}, std={np.std(imputed_vals):.1f}")
        
        if np.mean(imputed_vals) < np.mean(known_vals) * 0.7:
            print("  WARNING: Imputed distances appear underestimated!")
        elif np.mean(imputed_vals) > np.mean(known_vals) * 1.5:
            print("  WARNING: Imputed distances appear overestimated!")


def impute_cross_site_distances(matrices, lab_samples, anchors, primary_anchor, method='auto'):
    
    incomplete_matrix, sample_to_lab, known_mask = create_incomplete_matrix(
        matrices, lab_samples, anchors
    )
    
    n_total = len(incomplete_matrix)
    n_missing = np.isnan(incomplete_matrix.values).sum()
    n_known = n_total * n_total - n_missing
    pct_known = 100 * n_known / (n_total * n_total)
    
    print(f"Matrix size: {n_total} x {n_total}")
    print(f"Known entries: {n_known} ({pct_known:.1f}%)")
    print(f"Missing entries: {n_missing} ({100-pct_known:.1f}%)")
    
    if method == 'auto':
        if pct_known > 70:
            method = 'knn'
        elif FANCYIMPUTE_AVAILABLE:
            method = 'softimpute'
        elif len(anchors) >= 3:
            method = 'anchor_guided'
        else:
            method = 'knn' if SKLEARN_KNN_AVAILABLE else 'anchor_guided'
        print(f"Auto-selected method: {method}")
    
    if method == 'softimpute' and FANCYIMPUTE_AVAILABLE:
        global_matrix = impute_softimpute_improved(incomplete_matrix, known_mask)
    elif method == 'knn' and SKLEARN_KNN_AVAILABLE:
        global_matrix = impute_knn_improved(incomplete_matrix, known_mask)
    elif method == 'anchor_guided':
        global_matrix = impute_anchor_guided(incomplete_matrix, known_mask, anchors, sample_to_lab)
    else:
        global_matrix = impute_anchor_guided(incomplete_matrix, known_mask, anchors, sample_to_lab)
    
    print("Enforcing metric properties")
    completed_values = enforce_metric_properties_balanced(
        global_matrix.values.copy(), 
        known_mask,
        max_iterations=3
    )
    
    global_matrix = pd.DataFrame(
        completed_values,
        index=global_matrix.index,
        columns=global_matrix.columns
    )
    
    validate_and_report(incomplete_matrix, global_matrix, known_mask, anchors)
    
    return global_matrix, method


def merge_matrices(matrix_files, anchors, primary_anchor, imputation_method='auto'):
    matrices = [load_distance_matrix(f) for f in matrix_files]
    
    print(f"Loaded {len(matrices)} matrices")
    for i, df in enumerate(matrices):
        print(f"  Matrix {i}: {len(df)} samples")
        print(f"    Distance range: {df.values[df.values > 0].min():.0f} - {df.values.max():.0f}")
    
    correction_factors = calculate_correction_factor(matrices, anchors, primary_anchor)
    print(f"Correction factors: {correction_factors}")
    
    corrected_matrices = [
        apply_correction(df, factor) 
        for df, factor in zip(matrices, correction_factors)
    ]
    
    lab_samples = [set(df.index) for df in corrected_matrices]
    
    global_matrix, method_used = impute_cross_site_distances(
        corrected_matrices, 
        lab_samples, 
        anchors,
        primary_anchor,
        method=imputation_method
    )
    
    nonzero_vals = global_matrix.values[global_matrix.values > 0]
    
    report = {
        "num_labs": len(matrices),
        "input_files": [os.path.basename(f) for f in matrix_files],
        "correction_factors": correction_factors,
        "anchors_used": anchors,
        "primary_anchor": primary_anchor,
        "total_samples": len(global_matrix),
        "samples_per_lab": [len(s) for s in lab_samples],
        "unique_samples_per_lab": [len(s - set(anchors)) for s in lab_samples],
        "imputation_method": method_used,
        "max_distance": float(global_matrix.values.max()),
        "min_nonzero_distance": float(nonzero_vals.min()) if len(nonzero_vals) > 0 else 0,
        "mean_distance": float(nonzero_vals.mean()) if len(nonzero_vals) > 0 else 0,
        "median_distance": float(np.median(nonzero_vals)) if len(nonzero_vals) > 0 else 0,
    }
    
    return global_matrix, report, lab_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrices', nargs='+', required=True)
    parser.add_argument('--anchors', type=str, required=True)
    parser.add_argument('--primary-anchor', type=str, required=True)
    parser.add_argument('--output', type=str, default="global_distance_matrix.tsv")
    parser.add_argument('--report', type=str, default="correction_report.json")
    parser.add_argument('--mapping', type=str, default="samples_mapping.json")
    parser.add_argument('--method', type=str, default='auto',
                        choices=['auto', 'softimpute', 'knn', 'anchor_guided'])
    args = parser.parse_args()
    
    anchors = [a.strip() for a in args.anchors.split(',')]
    
    print("=" * 60)
    print("Federated Matrix Merger")
    print("=" * 60)
    
    global_matrix, report, lab_samples = merge_matrices(
        args.matrices, anchors, args.primary_anchor, args.method
    )
    
    global_matrix.to_csv(args.output, sep='\t')
    print(f"\nGlobal matrix saved to {args.output}")
    
    with open(args.report, 'w') as f:
        json.dump(report, f, indent=2)
    
    mapping = {
        "lab_samples": {f"lab_{i}": list(samples) for i, samples in enumerate(lab_samples)},
        "anchors": anchors,
        "all_samples": list(global_matrix.index)
    }
    with open(args.mapping, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Total samples: {report['total_samples']}")
    print(f"  Distance range: {report['min_nonzero_distance']:.0f} - {report['max_distance']:.0f}")
    print(f"  Mean/Median: {report['mean_distance']:.0f} / {report['median_distance']:.0f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
