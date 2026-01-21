#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Matrix Completion methods being explored:
1. SoftImpute (Nuclear Norm Minimization) (1)
2. Iterative Imputer (MICE) (2)
3. KNN Imputer (k-Nearest Neighbors) (3)
4. Triangle Inequality (4)
"""

import pandas as pd
import numpy as np
import argparse
import json
import os
import warnings
warnings.filterwarnings('ignore')

try:
    from fancyimpute import SoftImpute, IterativeImputer as FancyIterativeImputer
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
        print("Less than 2 anchors found.")
        return [1.0] * len(matrices)
    
    if primary_anchor not in common_anchors:
        primary_anchor = common_anchors[0]
        print(f"Using {primary_anchor} as primary anchor")
    
    ref_matrix = matrices[0]
    ref_distances = {}
    for anchor in common_anchors:
        if anchor != primary_anchor:
            ref_distances[anchor] = ref_matrix.loc[primary_anchor, anchor]
    
    correction_factors.append(1.0)
    
    for i, df in enumerate(matrices[1:], 1):
        factors = []
        for anchor, ref_dist in ref_distances.items():
            if anchor in df.index and primary_anchor in df.index:
                local_dist = df.loc[primary_anchor, anchor]
                if local_dist > 0:
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
    
    sample_to_lab = {}
    for lab_idx, samples in enumerate(lab_samples):
        for sample in samples:
            sample_to_lab[sample] = lab_idx
    
    for lab_idx, df in enumerate(matrices):
        samples_in_lab = [s for s in df.index if s in all_samples]
        for s1 in samples_in_lab:
            for s2 in samples_in_lab:
                if s1 in global_matrix.index and s2 in global_matrix.columns:
                    global_matrix.loc[s1, s2] = df.loc[s1, s2]
    
    return global_matrix, sample_to_lab


def enforce_metric_properties(matrix):

    matrix = np.maximum(matrix, 0)
    
    matrix = (matrix + matrix.T) / 2
    
    np.fill_diagonal(matrix, 0)
    
    n = matrix.shape[0]
    max_iterations = 10
    
    for iteration in range(max_iterations):
        violations = 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(n):
                    if k == i or k == j:
                        continue
                    
                    upper_bound = matrix[i, k] + matrix[k, j]
                    if matrix[i, j] > upper_bound:
                        matrix[i, j] = upper_bound
                        matrix[j, i] = upper_bound
                        violations += 1
        
        if violations == 0:
            break
    
    return matrix


def impute_softimpute(incomplete_matrix):

    print("Using SoftImpute")
    
    matrix_values = incomplete_matrix.values.copy()
    
    imputer = SoftImpute(max_iters=100, verbose=False)
    completed = imputer.fit_transform(matrix_values)
    
    completed = enforce_metric_properties(completed)
    
    return pd.DataFrame(
        completed,
        index=incomplete_matrix.index,
        columns=incomplete_matrix.columns
    )


def impute_iterative(incomplete_matrix):

    print("Using Iterative Imputer (MICE)")
    
    matrix_values = incomplete_matrix.values.copy()
    
    if SKLEARN_ITERATIVE_AVAILABLE:
        imputer = IterativeImputer(
            max_iter=50,
            random_state=42,
            initial_strategy='mean'
        )
        completed = imputer.fit_transform(matrix_values)
    elif FANCYIMPUTE_AVAILABLE:
        imputer = FancyIterativeImputer(n_iter=50, verbose=False)
        completed = imputer.fit_transform(matrix_values)
    else:
        raise ImportError("No iterative imputer available")
    
    completed = enforce_metric_properties(completed)
    
    return pd.DataFrame(
        completed,
        index=incomplete_matrix.index,
        columns=incomplete_matrix.columns
    )


def impute_knn(incomplete_matrix, n_neighbors=5):

    print(f"Using KNN Imputer (k={n_neighbors})")
    
    matrix_values = incomplete_matrix.values.copy()
    
    imputer = KNNImputer(n_neighbors=n_neighbors)
    completed = imputer.fit_transform(matrix_values)
    
    completed = enforce_metric_properties(completed)
    
    return pd.DataFrame(
        completed,
        index=incomplete_matrix.index,
        columns=incomplete_matrix.columns
    )


def impute_triangle_inequality(incomplete_matrix, anchors):

    print("Using Triangle Inequality")
    
    matrix = incomplete_matrix.copy()
    samples = list(matrix.index)
    
    for i, s1 in enumerate(samples):
        for j, s2 in enumerate(samples):
            if i >= j:
                continue
            
            if np.isnan(matrix.loc[s1, s2]):
                estimates = []
                
                for anchor in anchors:
                    if anchor in samples:
                        d1 = matrix.loc[s1, anchor]
                        d2 = matrix.loc[s2, anchor]
                        
                        if not np.isnan(d1) and not np.isnan(d2):
                            estimates.append(d1 + d2)
                
                if estimates:
                    imputed = min(estimates)
                    matrix.loc[s1, s2] = imputed
                    matrix.loc[s2, s1] = imputed
    
    mean_dist = matrix.values[~np.isnan(matrix.values)].mean()
    matrix = matrix.fillna(mean_dist)
    
    completed = enforce_metric_properties(matrix.values)
    
    return pd.DataFrame(
        completed,
        index=incomplete_matrix.index,
        columns=incomplete_matrix.columns
    )


def impute_cross_site_distances(matrices, lab_samples, anchors, primary_anchor, method='auto'):

    incomplete_matrix, sample_to_lab = create_incomplete_matrix(
        matrices, lab_samples, anchors
    )
    
    n_total = len(incomplete_matrix)
    n_missing = np.isnan(incomplete_matrix.values).sum()
    n_known = n_total * n_total - n_missing
    
    print(f"Matrix size: {n_total} x {n_total}")
    print(f"Known entries: {n_known} ({100*n_known/(n_total*n_total):.1f}%)")
    print(f"Missing entries: {n_missing} ({100*n_missing/(n_total*n_total):.1f}%)")
    
    if method == 'auto':
        if FANCYIMPUTE_AVAILABLE:
            method = 'softimpute'
        elif SKLEARN_ITERATIVE_AVAILABLE:
            method = 'iterative'
        elif SKLEARN_KNN_AVAILABLE:
            method = 'knn'
        else:
            method = 'triangle'
        print(f"Auto-selected method: {method}")
    
    if method == 'softimpute' and FANCYIMPUTE_AVAILABLE:
        global_matrix = impute_softimpute(incomplete_matrix)
    elif method == 'iterative' and (SKLEARN_ITERATIVE_AVAILABLE or FANCYIMPUTE_AVAILABLE):
        global_matrix = impute_iterative(incomplete_matrix)
    elif method == 'knn' and SKLEARN_KNN_AVAILABLE:
        global_matrix = impute_knn(incomplete_matrix)
    else:
        global_matrix = impute_triangle_inequality(incomplete_matrix, anchors)
    
    global_matrix = pd.DataFrame(
        enforce_metric_properties(global_matrix.values),
        index=global_matrix.index,
        columns=global_matrix.columns
    )
    
    return global_matrix, method


def merge_matrices(matrix_files, anchors, primary_anchor, imputation_method='auto'):
    matrices = [load_distance_matrix(f) for f in matrix_files]
    
    print(f"Loaded {len(matrices)} matrices")
    for i, df in enumerate(matrices):
        print(f"  Matrix {i}: {len(df)} samples")
    
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
        "min_nonzero_distance": float(global_matrix.values[global_matrix.values > 0].min()) if (global_matrix.values > 0).any() else 0,
        "mean_distance": float(global_matrix.values[global_matrix.values > 0].mean()) if (global_matrix.values > 0).any() else 0,
        "fancyimpute_available": FANCYIMPUTE_AVAILABLE,
        "sklearn_iterative_available": SKLEARN_ITERATIVE_AVAILABLE,
        "sklearn_knn_available": SKLEARN_KNN_AVAILABLE
    }
    
    return global_matrix, report, lab_samples


def main():
    parser = argparse.ArgumentParser(
        description="Merge distance matrices from multiple labs using matrix completion"
    )
    parser.add_argument('--matrices', nargs='+', required=True,
                        help="Input distance matrix files")
    parser.add_argument('--anchors', type=str, required=True,
                        help="Comma-separated list of anchor samples")
    parser.add_argument('--primary-anchor', type=str, required=True,
                        help="Primary anchor sample for normalization")
    parser.add_argument('--output', type=str, default="global_distance_matrix.tsv",
                        help="Output file for global matrix")
    parser.add_argument('--report', type=str, default="correction_report.json",
                        help="Output file for correction report")
    parser.add_argument('--mapping', type=str, default="samples_mapping.json",
                        help="Output file for sample mapping")
    parser.add_argument('--method', type=str, default='auto',
                        choices=['auto', 'softimpute', 'iterative', 'knn', 'triangle'],
                        help="Imputation method (default: auto)")
    args = parser.parse_args()
    
    anchors = [a.strip() for a in args.anchors.split(',')]
    
    print("=" * 60)
    print("Federated Matrix Merger with Matrix Completion")
    print("=" * 60)
    
    global_matrix, report, lab_samples = merge_matrices(
        args.matrices, anchors, args.primary_anchor, args.method
    )
    
    global_matrix.to_csv(args.output, sep='\t')
    print(f"\nGlobal distance matrix saved to {args.output}")
    
    with open(args.report, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Correction report saved to {args.report}")
    
    mapping = {
        "lab_samples": {f"lab_{i}": list(samples) for i, samples in enumerate(lab_samples)},
        "anchors": anchors,
        "all_samples": list(global_matrix.index)
    }
    with open(args.mapping, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"Sample mapping saved to {args.mapping}")
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Total samples: {report['total_samples']}")
    print(f"  Imputation method: {report['imputation_method']}")
    print(f"  Distance range: {report['min_nonzero_distance']:.1f} - {report['max_distance']:.1f}")
    print(f"  Mean distance: {report['mean_distance']:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()