#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


def calculate_sample_bias(samples, all_known_distances, present_anchors, original_matrix):

    sample_bias = {}
    print("\n  Calculating Per-Sample Bias (PSBC)")

    sidx = {s: i for i, s in enumerate(samples)}
    orig_arr = original_matrix.values
    anchor_idxs = [sidx[a] for a in present_anchors if a in sidx]

    for s in samples:
        biases = []
        known_neighbors = all_known_distances.get(s, {})
        si = sidx[s]

        for k, known_dist in known_neighbors.items():
            if known_dist < 500:
                continue

            ki = sidx[k]
            min_anchor_sum = float('inf')
            for ai in anchor_idxs:
                d1 = orig_arr[si, ai]
                d2 = orig_arr[ki, ai]
                if not np.isnan(d1) and not np.isnan(d2) and d1 > 0 and d2 > 0:
                    dist_sum = d1 + d2
                    if dist_sum < min_anchor_sum:
                        min_anchor_sum = dist_sum

            if min_anchor_sum < float('inf'):
                bias = min_anchor_sum - known_dist
                biases.append(bias)

        if len(biases) >= 3:
            median_bias = np.median(biases)
            if median_bias > 100:
                sample_bias[s] = median_bias
                if s in ["ERR3588243", "ERR8170876_ont", "ERR4830684", "ERR8170873_ont", "ERR3806818"]:
                    print(f"    Bias detected for {s}: {median_bias:.1f} (from {len(biases)} far neighbors)")
        elif len(biases) > 0:
             mean_bias = np.mean(biases)
             if mean_bias > 100:
                 sample_bias[s] = mean_bias
                 if s in ["ERR3588243", "ERR8170876_ont", "ERR4830684"]:
                     print(f"    Bias detected for {s}: {mean_bias:.1f} (from {len(biases)} far neighbor)")

    print(f"  Detected systematic bias for {len(sample_bias)} samples")
    return sample_bias


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
    sample_idx = {s: i for i, s in enumerate(all_samples)}

    gm_arr = np.full((n, n), np.nan)
    np.fill_diagonal(gm_arr, 0)

    sample_to_lab = {}
    for lab_idx, samples in enumerate(lab_samples):
        for sample in samples:
            if sample not in sample_to_lab:
                sample_to_lab[sample] = []
            sample_to_lab[sample].append(lab_idx)

    sum_arr = np.zeros((n, n))
    count_arr = np.zeros((n, n), dtype=int)

    for lab_idx, df in enumerate(matrices):
        lab_samples_in_global = [s for s in df.index if s in sample_idx]
        if not lab_samples_in_global:
            continue
        local_indices = np.array([sample_idx[s] for s in lab_samples_in_global])
        local_vals = df.loc[lab_samples_in_global, lab_samples_in_global].values
        ix = np.ix_(local_indices, local_indices)
        valid = ~np.isnan(local_vals) & (local_vals >= 0)
        sum_arr[ix] += np.where(valid, local_vals, 0)
        count_arr[ix] += valid.astype(int)

    has_values = count_arr > 0
    gm_arr[has_values] = sum_arr[has_values] / count_arr[has_values]
    np.fill_diagonal(gm_arr, 0)
    km_arr = has_values | np.eye(n, dtype=bool)

    global_matrix = pd.DataFrame(gm_arr, index=all_samples, columns=all_samples)
    known_mask = pd.DataFrame(km_arr, index=all_samples, columns=all_samples)

    return global_matrix, sample_to_lab, known_mask


def diagnose_imputation_coverage(incomplete_matrix, known_mask, anchors, sample_to_lab):

    samples = list(incomplete_matrix.index)
    present_anchors = [a for a in anchors if a in samples]

    print("\n" + "=" * 60)
    print("Imputation Coverage")
    print("=" * 60)

    lab_counts = {}
    for sample, labs in sample_to_lab.items():
        for lab in labs:
            lab_counts[lab] = lab_counts.get(lab, 0) + 1
    print(f"\nSamples per lab: {lab_counts}")

    sidx = {s: i for i, s in enumerate(samples)}
    km_arr = known_mask.values
    im_arr = incomplete_matrix.values
    n = len(samples)

    print(f"\nAnchor coverage:")
    for anchor in present_anchors:
        ai = sidx[anchor]
        known_count = int(km_arr[:, ai].sum())
        non_nan_count = int((~np.isnan(im_arr[:, ai])).sum())
        print(f"  {anchor}: known_mask={known_count}, non_nan={non_nan_count}/{n}")

    anchor_idxs = np.array([sidx[a] for a in present_anchors]) if present_anchors else np.array([], dtype=int)

    nan_mask = np.isnan(im_arr)
    upper_tri = np.triu(np.ones((n, n), dtype=bool), k=1)
    missing_mask = nan_mask & upper_tri
    missing_i, missing_j = np.where(missing_mask)

    can_triangulate = 0
    cannot_triangulate = 0

    if len(missing_i) > 0 and len(anchor_idxs) > 0:
        anchor_dists = im_arr[:, anchor_idxs]
        anchor_valid = ~np.isnan(anchor_dists) & (anchor_dists > 0)
        both_valid = anchor_valid[missing_i] & anchor_valid[missing_j]
        has_anchor = both_valid.any(axis=1)
        can_triangulate = int(has_anchor.sum())
        cannot_triangulate = int((~has_anchor).sum())

        cannot_idxs = np.where(~has_anchor)[0]
        for ci in cannot_idxs[:5]:
            s1, s2 = samples[missing_i[ci]], samples[missing_j[ci]]
            print(f"\n  Cannot triangulate: {s1} and {s2}")
            for anchor in present_anchors:
                ai = sidx[anchor]
                d1 = im_arr[missing_i[ci], ai]
                d2 = im_arr[missing_j[ci], ai]
                print(f"    {anchor}: {s1}={d1}, {s2}={d2}")

    print(f"\nMissing pairs: {len(missing_i)}")
    print(f"  Can triangulate: {can_triangulate}")
    print(f"  Cannot triangulate: {cannot_triangulate}")
    print("=" * 60 + "\n")

    return can_triangulate, cannot_triangulate


def enforce_metric_properties_balanced(matrix, known_mask, close_detected_mask=None, max_iterations=10):

    matrix = np.maximum(matrix, 0)
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0)

    n = matrix.shape[0]
    known = known_mask.values if hasattr(known_mask, 'values') else known_mask

    if close_detected_mask is not None:
        close_detected = close_detected_mask.values if hasattr(close_detected_mask, 'values') else close_detected_mask
        protected = known | close_detected
    else:
        protected = known

    original_protected = matrix.copy()

    protected_count = protected.sum()
    print(f"  Protected values: {protected_count}")

    upper_tri = np.triu(np.ones((n, n), dtype=bool), k=1)
    adjustable = upper_tri & ~protected

    for iteration in range(max_iterations):
        upper = np.full((n, n), np.inf)
        lower = np.zeros((n, n))
        positive = matrix > 0

        for k in range(n):
            d_ik = matrix[:, k]  
            d_kj = matrix[k, :]  
            valid_ik = positive[:, k]
            valid_kj = positive[k, :]
            valid = valid_ik[:, None] & valid_kj[None, :] 
            valid[k, :] = False
            valid[:, k] = False
            sum_k = d_ik[:, None] + d_kj[None, :]
            diff_k = np.abs(d_ik[:, None] - d_kj[None, :])
            np.minimum(upper, np.where(valid, sum_k, np.inf), out=upper)
            np.maximum(lower, np.where(valid, diff_k, 0.0), out=lower)

        has_bounds = upper < np.inf
        over_upper = adjustable & has_bounds & (matrix > upper)
        under_lower = adjustable & has_bounds & (matrix < lower) & ~over_upper

        changes = int(over_upper.sum() + under_lower.sum())
        total_adjustment = 0.0

        if over_upper.any():
            old_vals = matrix[over_upper]
            new_vals = old_vals * 0.3 + upper[over_upper] * 0.7
            total_adjustment += np.sum(np.abs(new_vals - old_vals))
            matrix[over_upper] = new_vals

        if under_lower.any():
            old_vals = matrix[under_lower]
            new_vals = old_vals * 0.3 + lower[under_lower] * 0.7
            total_adjustment += np.sum(np.abs(new_vals - old_vals))
            matrix[under_lower] = new_vals

        i_idx, j_idx = np.triu_indices(n, k=1)
        matrix[j_idx, i_idx] = matrix[i_idx, j_idx]

        avg_adjustment = total_adjustment / max(changes, 1)
        print(f"  Iteration {iteration + 1}: {changes} adjustments, avg change: {avg_adjustment:.2f}")

        if changes == 0 or avg_adjustment < 0.5:
            break

    protected_upper = upper_tri & protected
    matrix[protected_upper] = original_protected[protected_upper]
    i_idx, j_idx = np.triu_indices(n, k=1)
    matrix[j_idx, i_idx] = matrix[i_idx, j_idx]

    return matrix


def mds_cross_lab_estimation(incomplete_matrix, known_mask, sample_to_lab, matrices_for_mds=None):
    from scipy.linalg import orthogonal_procrustes

    samples = list(incomplete_matrix.index)
    n = len(samples)
    sidx = {s: i for i, s in enumerate(samples)}
    mat = incomplete_matrix.values.copy()

    num_labs = max(lab for labs in sample_to_lab.values() for lab in labs) + 1
    lab_sample_sets = [set() for _ in range(num_labs)]
    for s, labs in sample_to_lab.items():
        for lab in labs:
            lab_sample_sets[lab].add(s)

    shared = set(samples)
    for ls in lab_sample_sets:
        shared &= ls
    shared = sorted(shared)
    print(f"\n  MDS estimation: {len(shared)} shared alignment points across {num_labs} labs")

    if len(shared) < 4:
        print("  MDS: insufficient shared samples, skipping")
        return {}

    n_dims = min(len(shared) - 1, 10)
    n_dims = max(n_dims, 3)

    def classical_mds(D, k):
        n_pts = D.shape[0]
        H = np.eye(n_pts) - np.ones((n_pts, n_pts)) / n_pts
        B = -0.5 * H @ (D ** 2) @ H
        eigvals, eigvecs = np.linalg.eigh(B)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        pos_dims = np.maximum(eigvals[:k], 0)
        return eigvecs[:, :k] * np.sqrt(pos_dims)

    lab_embeddings = []
    for lab_i in range(num_labs):
        lab_samps = sorted(lab_sample_sets[lab_i])
        if len(lab_samps) < 5:
            lab_embeddings.append(None)
            continue
        lab_idx = [sidx[s] for s in lab_samps]
        D = mat[np.ix_(lab_idx, lab_idx)].copy()
        D = np.nan_to_num(D, nan=0)
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0)
        D = np.maximum(D, 0)
        coords = classical_mds(D, n_dims)
        lab_embeddings.append((coords, lab_samps))
        print(f"    Lab {lab_i}: {len(lab_samps)} samples -> {n_dims}D MDS")

    ref_lab = None
    for i, emb in enumerate(lab_embeddings):
        if emb is not None:
            ref_lab = i
            break
    if ref_lab is None:
        return {}

    ref_coords, ref_samples = lab_embeddings[ref_lab]
    ref_sidx = {s: j for j, s in enumerate(ref_samples)}

    aligned_coords = {}
    for j, s in enumerate(ref_samples):
        aligned_coords[s] = [ref_coords[j]]

    for lab_i in range(num_labs):
        if lab_i == ref_lab or lab_embeddings[lab_i] is None:
            continue
        coords_i, samps_i = lab_embeddings[lab_i]
        sidx_i = {s: j for j, s in enumerate(samps_i)}

        shared_here = [s for s in shared if s in ref_sidx and s in sidx_i]
        if len(shared_here) < 4:
            shared_here = [s for s in samps_i if s in ref_sidx]
        if len(shared_here) < 3:
            for j, s in enumerate(samps_i):
                if s not in aligned_coords:
                    aligned_coords[s] = []
                aligned_coords[s].append(coords_i[j])
            continue

        src_pts = np.array([coords_i[sidx_i[s]] for s in shared_here])
        tgt_pts = np.array([ref_coords[ref_sidx[s]] for s in shared_here])

        src_mean = src_pts.mean(axis=0)
        tgt_mean = tgt_pts.mean(axis=0)
        src_c = src_pts - src_mean
        tgt_c = tgt_pts - tgt_mean

        R, _ = orthogonal_procrustes(src_c, tgt_c)
        src_scale = np.sqrt(np.sum(src_c ** 2))
        tgt_scale = np.sqrt(np.sum(tgt_c ** 2))
        s_factor = tgt_scale / src_scale if src_scale > 0 else 1.0

        for j, s in enumerate(samps_i):
            transformed = s_factor * (coords_i[j] - src_mean) @ R + tgt_mean
            if s not in aligned_coords:
                aligned_coords[s] = []
            aligned_coords[s].append(transformed)

    final_coords = {}
    for s, coord_list in aligned_coords.items():
        final_coords[s] = np.mean(coord_list, axis=0)

    coord_samples = sorted(final_coords.keys())
    coord_mat = np.array([final_coords[s] for s in coord_samples])
    coord_sidx = {s: i for i, s in enumerate(coord_samples)}

    km = known_mask.values
    calib_mds = []
    calib_actual = []
    for i_c, s1 in enumerate(coord_samples):
        i1 = sidx[s1]
        for j_c in range(i_c + 1, len(coord_samples)):
            s2 = coord_samples[j_c]
            i2 = sidx[s2]
            if km[i1, i2] and not np.isnan(mat[i1, i2]) and mat[i1, i2] > 0:
                mds_d = np.sqrt(np.sum((coord_mat[i_c] - coord_mat[j_c]) ** 2))
                calib_mds.append(mds_d)
                calib_actual.append(mat[i1, i2])

    calib_mds = np.array(calib_mds)
    calib_actual = np.array(calib_actual)
    print(f"  MDS calibration pairs: {len(calib_mds)}")

    bin_models = {}
    for lo, hi, name in [(0, 500, 'close'), (500, 1500, 'medium'), (1500, 5000, 'far')]:
        m = (calib_actual >= lo) & (calib_actual < hi)
        if m.sum() >= 20:
            A = np.vstack([calib_mds[m], np.ones(m.sum())]).T
            result = np.linalg.lstsq(A, calib_actual[m], rcond=None)
            slope, intercept = result[0]
            r2 = 1 - np.sum((calib_actual[m] - (slope * calib_mds[m] + intercept)) ** 2) / np.sum((calib_actual[m] - np.mean(calib_actual[m])) ** 2)
            bin_models[name] = (slope, intercept, r2)
            print(f"    {name}: slope={slope:.3f}, intercept={intercept:.1f}, R²={r2:.3f} (n={m.sum()})")

    A = np.vstack([calib_mds, np.ones(len(calib_mds))]).T
    result = np.linalg.lstsq(A, calib_actual, rcond=None)
    global_slope, global_intercept = result[0]
    global_r2 = 1 - np.sum((calib_actual - (global_slope * calib_mds + global_intercept)) ** 2) / np.sum((calib_actual - np.mean(calib_actual)) ** 2)
    print(f"    global: slope={global_slope:.3f}, intercept={global_intercept:.1f}, R²={global_r2:.3f}")

    mds_estimates = {}
    for i_c, s1 in enumerate(coord_samples):
        i1 = sidx[s1]
        for j_c in range(i_c + 1, len(coord_samples)):
            s2 = coord_samples[j_c]
            i2 = sidx[s2]
            if not km[i1, i2]:  
                mds_d = np.sqrt(np.sum((coord_mat[i_c] - coord_mat[j_c]) ** 2))
                est = global_slope * mds_d + global_intercept
                est = max(est, 0)
                mds_estimates[(s1, s2)] = est
                mds_estimates[(s2, s1)] = est

    print(f"  MDS estimates computed: {len(mds_estimates) // 2} cross-lab pairs")
    return mds_estimates


def impute_anchor_guided(incomplete_matrix, known_mask, anchors, sample_to_lab, mds_estimates=None):

    print("Using Enhanced Anchor-Guided Imputation")

    debug_pairs = set()
    
    matrix = incomplete_matrix.copy()
    original_known_mask = known_mask.copy()
    samples = list(matrix.index)
    known = known_mask.copy()
    
    present_anchors = [a for a in anchors if a in samples]
    
    n_labs = max(lab for labs in sample_to_lab.values() for lab in labs) + 1
    effective_anchors = [s for s in samples if len(sample_to_lab.get(s, [])) == n_labs]
    present_anchors = sorted(set(present_anchors) | set(effective_anchors))
    print(f"  Using {len(present_anchors)} effective anchors ({len(effective_anchors)} auto-detected from shared samples)")
    
    close_detected_mask = pd.DataFrame(
        np.zeros((len(samples), len(samples)), dtype=bool),
        index=samples,
        columns=samples
    )
    
    for anchor in present_anchors:
        known_to_anchor = (~matrix.loc[:, anchor].isna()).sum()
        print(f"    Anchor {anchor}: {known_to_anchor}/{len(samples)} samples have known distances")
    
    original_matrix = incomplete_matrix.copy()
    original_known = incomplete_matrix.copy()

    sidx = {s: i for i, s in enumerate(samples)}
    n_samp = len(samples)
    mat_arr = matrix.values
    orig_arr = original_matrix.values
    orig_known_arr = original_known.values
    known_arr = known.values
    close_arr = close_detected_mask.values
    orig_kmask_arr = original_known_mask.values
    anchor_idxs = np.array([sidx[a] for a in present_anchors])

    high_confidence_arr = np.zeros((n_samp, n_samp), dtype=bool)
    high_confidence_arr[orig_kmask_arr] = True

    calibration_data = {"close": [], "medium": [], "far": []}
    
    for i, a1 in enumerate(present_anchors):
        for a2 in present_anchors[i+1:]:
            a1i, a2i = sidx[a1], sidx[a2]
            if known_arr[a1i, a2i]:
                actual = mat_arr[a1i, a2i]
                d1_all = mat_arr[:, a1i]
                d2_all = mat_arr[:, a2i]
                valid = (~np.isnan(d1_all)) & (~np.isnan(d2_all)) & known_arr[:, a1i] & known_arr[:, a2i]
                valid[a1i] = False
                valid[a2i] = False
                
                if valid.any():
                    upper_all = d1_all[valid] + d2_all[valid]
                    lower_all = np.abs(d1_all[valid] - d2_all[valid])
                    range_all = upper_all - lower_all
                    valid_range = (range_all > 0)
                    
                    if valid_range.any():
                        ratios = (actual - lower_all[valid_range]) / range_all[valid_range]
                        valid_ratios = (ratios >= 0) & (ratios <= 1)
                        for r in ratios[valid_ratios]:
                            if actual < 500:
                                calibration_data["close"].append(r)
                            elif actual < 1500:
                                calibration_data["medium"].append(r)
                            else:
                                calibration_data["far"].append(r)
    
    weight_factors = {}
    for category, ratios in calibration_data.items():
        if ratios:
            weight_factors[category] = np.median(ratios)
            print(f"  Calibrated weight factor ({category}): {weight_factors[category]:.3f} (from {len(ratios)} observations)")
        else:
            if category == "close":
                weight_factors[category] = 0.15
            elif category == "medium":
                weight_factors[category] = 0.35
            else:
                weight_factors[category] = 0.45
            print(f"  Default weight factor ({category}): {weight_factors[category]:.3f}")
    
    within_lab_weights = {"very_close": [], "close": [], "mid_close": [], "medium_check": []}
    for i_cal in range(n_samp):
        d1_anc = orig_arr[i_cal, anchor_idxs]
        if np.any(np.isnan(d1_anc)):
            continue
        for j_cal in range(i_cal + 1, n_samp):
            if not orig_kmask_arr[i_cal, j_cal]:
                continue
            actual_dist = orig_arr[i_cal, j_cal]
            if np.isnan(actual_dist) or actual_dist <= 0:
                continue
            d2_anc = orig_arr[j_cal, anchor_idxs]
            valid_anc = ~np.isnan(d2_anc) & (d1_anc > 0) & (d2_anc > 0)
            if valid_anc.sum() < 2:
                continue
            lower = float(np.max(np.abs(d1_anc[valid_anc] - d2_anc[valid_anc])))
            upper = float(np.min(d1_anc[valid_anc] + d2_anc[valid_anc]))
            rng = upper - lower
            if rng <= 0:
                continue
            w = (actual_dist - lower) / rng
            if 0 <= w <= 1:
                if actual_dist < 100:
                    within_lab_weights["very_close"].append(w)
                elif actual_dist < 300:
                    within_lab_weights["close"].append(w)
                elif actual_dist < 500:
                    within_lab_weights["mid_close"].append(w)
                elif actual_dist < 1500:
                    within_lab_weights["medium_check"].append(w)
    
    sub_weights = {}
    for cat, ws in within_lab_weights.items():
        if ws:
            sub_weights[cat] = np.median(ws)
            print(f"  Within-lab weight ({cat}): {sub_weights[cat]:.3f} (from {len(ws)} pairs)")
    
    anchor_calibration = []
    print("\n  Anchor-based cross-lab calibration:")
    for i, a1 in enumerate(present_anchors):
        for a2 in present_anchors[i+1:]:
            actual_dist = original_matrix.loc[a1, a2]
            if np.isnan(actual_dist) or actual_dist <= 0:
                continue
            
            p1 = np.array([original_matrix.loc[a1, a] for a in present_anchors if a not in [a1, a2]])
            p2 = np.array([original_matrix.loc[a2, a] for a in present_anchors if a not in [a1, a2]])
            valid = ~np.isnan(p1) & ~np.isnan(p2)
            
            if valid.sum() >= 3:
                diffs = np.abs(p1[valid] - p2[valid])
                max_diff = np.max(diffs)
                
                if max_diff > 0:
                    ratio = actual_dist / max_diff
                    anchor_calibration.append({
                        'a1': a1, 'a2': a2,
                        'max_diff': max_diff,
                        'actual_dist': actual_dist,
                        'ratio': ratio
                    })
                    if actual_dist < 300:
                        print(f"    {a1} <-> {a2}: max_diff={max_diff:.0f}, actual={actual_dist:.0f}, ratio={ratio:.1f}x")
    
    close_anchor_pairs = [p for p in anchor_calibration if p['max_diff'] < 50 and p['actual_dist'] < 200]
    if close_anchor_pairs:
        very_close_multiplier = np.median([p['ratio'] for p in close_anchor_pairs])
        print(f"  Calibrated very_close_multiplier: {very_close_multiplier:.1f}x (from {len(close_anchor_pairs)} anchor pairs)")
    else:
        very_close_multiplier = 6.0
        print(f"  Default very_close_multiplier: {very_close_multiplier:.1f}x (no close anchor pairs found)")
    
    profile_arr = orig_arr[:, anchor_idxs]
    sample_profiles = {s: profile_arr[sidx[s]] for s in samples}
    
    ANCHOR_THRESHOLD = 500

    anchor_dists_for_min = orig_arr[:, anchor_idxs]
    anchor_dists_masked = np.where(np.isnan(anchor_dists_for_min) | (anchor_dists_for_min <= 0), np.inf, anchor_dists_for_min)
    min_anchor_per_sample = np.min(anchor_dists_masked, axis=1)
    sample_min_anchor_dist = {s: float(min_anchor_per_sample[sidx[s]]) for s in samples}
    
    samples_with_close_neighbors = set()
    CLOSE_NEIGHBOR_THRESHOLD = 200

    close_check = ~np.isnan(orig_arr) & (orig_arr > 0) & (orig_arr < CLOSE_NEIGHBOR_THRESHOLD)
    np.fill_diagonal(close_check, False)
    has_close = close_check.any(axis=1)
    samples_with_close_neighbors = {samples[i] for i in range(n_samp) if has_close[i]}
    
    samples_with_similar_profiles = set()
    PROFILE_SIMILARITY_THRESHOLD = 30

    if len(anchor_idxs) >= 3:
        with np.errstate(invalid='ignore'):
            pairwise_diffs = np.abs(profile_arr[:, np.newaxis, :] - profile_arr[np.newaxis, :, :])
            valid_counts = np.sum(~np.isnan(pairwise_diffs), axis=2)
            max_diffs = np.nanmax(pairwise_diffs, axis=2)
        similar = (valid_counts >= 3) & (max_diffs < PROFILE_SIMILARITY_THRESHOLD)
        np.fill_diagonal(similar, False)
        for i in range(n_samp):
            if samples[i] in samples_with_close_neighbors:
                continue
            for j in range(n_samp):
                if i == j or samples[j] in samples_with_close_neighbors:
                    continue
                if similar[i, j]:
                    samples_with_similar_profiles.add(samples[i])
                    samples_with_similar_profiles.add(samples[j])
                    break
    
    far_from_anchors = {s for s, d in sample_min_anchor_dist.items() 
                        if d > ANCHOR_THRESHOLD 
                        and s not in samples_with_close_neighbors
                        and s not in samples_with_similar_profiles}
    
    print(f"\n  Anchor-Constrained Insertion: ANCHOR_THRESHOLD = {ANCHOR_THRESHOLD}")
    print(f"  Samples with close neighbors (<{CLOSE_NEIGHBOR_THRESHOLD} SNPs): {len(samples_with_close_neighbors)}")
    print(f"  Samples with similar profiles (<{PROFILE_SIMILARITY_THRESHOLD} diff): {len(samples_with_similar_profiles)}")
    print(f"  Samples truly isolated: {len(far_from_anchors)}")
    if far_from_anchors:
        for s in list(far_from_anchors)[:10]:
            print(f"    {s}: min_anchor_dist = {sample_min_anchor_dist[s]:.0f}")
    
    all_known_arr = np.where(known_arr & ~np.isnan(orig_arr) & (orig_arr > 0), orig_arr, np.nan)
    np.fill_diagonal(all_known_arr, np.nan)

    all_known_distances = {}
    for i in range(n_samp):
        s = samples[i]
        all_known_distances[s] = {}
        valid_j = np.where(~np.isnan(all_known_arr[i]))[0]
        for j in valid_j:
            all_known_distances[s][samples[j]] = all_known_arr[i, j]
    
    sample_bias = calculate_sample_bias(samples, all_known_distances, present_anchors, original_matrix)
    
    print("\n  Building lineage groups from known distances")
    
    parent = {s: s for s in samples}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    close_pairs_mask = known_arr & ~np.isnan(orig_arr) & (orig_arr < 150) & (orig_arr > 0)
    upper_tri_mask = np.triu(np.ones((n_samp, n_samp), dtype=bool), k=1)
    close_union_i, close_union_j = np.where(close_pairs_mask & upper_tri_mask)
    for idx in range(len(close_union_i)):
        union(samples[close_union_i[idx]], samples[close_union_j[idx]])
    
    lineage_groups = {}
    for s in samples:
        root = find(s)
        if root not in lineage_groups:
            lineage_groups[root] = []
        lineage_groups[root].append(s)
    
    lineage_groups = {k: v for k, v in lineage_groups.items() if len(v) > 1}
    
    print(f"  Found {len(lineage_groups)} lineage groups with multiple members")
    for root, members in list(lineage_groups.items())[:5]:
        print(f"    Group {root}: {len(members)} members - {members[:5]}")
    
    lineage_cross_lab_distances = {}

    for root, members in lineage_groups.items():
        for s1 in members:
            s1_labs = set(sample_to_lab.get(s1, []))
            i1 = sidx[s1]
            for s2 in members:
                if s1 >= s2:
                    continue
                s2_labs = set(sample_to_lab.get(s2, []))
                if s1_labs & s2_labs:
                    continue
                i2 = sidx[s2]
                if known_arr[i1, i2]:
                    lineage_cross_lab_distances[(root, s1, s2)] = orig_arr[i1, i2]
    
    print(f"  Found {len(lineage_cross_lab_distances)} known cross-lab distances")
    
    imputation_stats = {
        "close_detection": 0, "consistency_override": 0, "parallel_branch": 0,
        "transitivity_check": 0, "anchor_direct": 0, "two_hop": 0, "fallback": 0,
        "lineage_proxy": 0, "bridge_estimate": 0, "same_lineage": 0,
        "phylo_close": 0, "anchor_triangulation": 0, "neighbor_based": 0
    }
    
    nan_mask_mp = np.isnan(mat_arr)
    upper_tri_mp = np.triu(np.ones((n_samp, n_samp), dtype=bool), k=1)
    miss_i, miss_j = np.where(nan_mask_mp & upper_tri_mp)
    missing_pairs = [(samples[miss_i[k]], samples[miss_j[k]]) for k in range(len(miss_i))]
    
    print(f"\n  Missing pairs to impute: {len(missing_pairs)}")
    
    n_anchors = len(present_anchors)
    anchor_anchor_dist = np.full((n_anchors, n_anchors), np.nan)
    for ai in range(n_anchors):
        for aj in range(n_anchors):
            if ai != aj:
                idx_ai = sidx[present_anchors[ai]]
                idx_aj = sidx[present_anchors[aj]]
                d = orig_known_arr[idx_ai, idx_aj]
                if not np.isnan(d) and d > 0:
                    anchor_anchor_dist[ai, aj] = d
    n_valid_anchor_pairs = np.sum(~np.isnan(anchor_anchor_dist)) // 2
    print(f"  Four-point anchor pairs available: {n_valid_anchor_pairs}")
    
    fp_anchor_dists = orig_known_arr[:, anchor_idxs]
    
    def compute_four_point_estimate(i1_fp, i2_fp):
        d1_fp = fp_anchor_dists[i1_fp]
        d2_fp = fp_anchor_dists[i2_fp]
        
        resolved_ests = []
        unresolved_ests = []
        for ai_fp in range(n_anchors):
            d_ac = d1_fp[ai_fp]
            d_bc = d2_fp[ai_fp]
            if np.isnan(d_ac) or d_ac <= 0 or np.isnan(d_bc) or d_bc <= 0:
                continue
            for aj_fp in range(ai_fp + 1, n_anchors):
                d_ad = d1_fp[aj_fp]
                d_bd = d2_fp[aj_fp]
                if np.isnan(d_ad) or d_ad <= 0 or np.isnan(d_bd) or d_bd <= 0:
                    continue
                d_cd = anchor_anchor_dist[ai_fp, aj_fp]
                if np.isnan(d_cd):
                    continue
                s2v = d_ac + d_bd
                s3v = d_ad + d_bc
                fp_e = max(s2v, s3v) - d_cd
                if fp_e <= 0:
                    continue
                resolution = abs(s2v - s3v)
                if resolution > 0.05 * d_cd and resolution > 20:
                    resolved_ests.append(fp_e)
                else:
                    unresolved_ests.append(fp_e)
        
        all_valid = resolved_ests + unresolved_ests
        if len(all_valid) < 3:
            return None, 0
        
        if len(resolved_ests) >= 5:
            fp_val = np.median(resolved_ests)
        elif len(resolved_ests) >= 2:
            fp_val = 0.7 * np.median(resolved_ests) + 0.3 * np.percentile(all_valid, 25)
        else:
            fp_val = np.percentile(all_valid, 25)
        
        return fp_val, len(all_valid)
    
    fp_precomputed = {}
    for s1, s2 in missing_pairs:
        i1_fp = sidx[s1]
        i2_fp = sidx[s2]
        fp_val, n_valid = compute_four_point_estimate(i1_fp, i2_fp)
        if fp_val is not None:
            fp_precomputed[(s1, s2)] = fp_val
            fp_precomputed[(s2, s1)] = fp_val
    
    print(f"  Four-point pre-computed for {len(fp_precomputed) // 2} pairs")
    
    close_pairs_to_impute = []
    processed_pairs = set()
    
    MAX_PASSES = 3
    
    for pass_num in range(MAX_PASSES):
        pairs_added_this_pass = 0
        
        for s1, s2 in missing_pairs:
            if (s1, s2) in processed_pairs or (s2, s1) in processed_pairs:
                continue
            
            should_debug = (s1, s2) in debug_pairs
            
            if should_debug and pass_num == 0:
                print(f"\n  DEBUG PASS {pass_num}: {s1} <-> {s2}")
            
            profile1 = sample_profiles[s1]
            profile2 = sample_profiles[s2]
            
            valid_mask = ~np.isnan(profile1) & ~np.isnan(profile2)
            
            if valid_mask.sum() < 3:
                continue
            
            diffs = np.abs(profile1[valid_mask] - profile2[valid_mask])
            min_diff = np.min(diffs)
            median_diff = np.median(diffs)
            mean_diff = np.mean(diffs)
            max_diff = np.max(diffs)
            std_diff = np.std(diffs)
            
            s1_labs = set(sample_to_lab.get(s1, []))
            s2_labs = set(sample_to_lab.get(s2, []))
            is_cross_lab = not bool(s1_labs & s2_labs)

            i1, i2 = sidx[s1], sidx[s2]
            d_s1_known = all_known_arr[i1].copy()
            d_s2_known = all_known_arr[i2].copy()
            d_s1_known[i1] = np.nan
            d_s1_known[i2] = np.nan
            d_s2_known[i1] = np.nan
            d_s2_known[i2] = np.nan

            both_orig = ~np.isnan(d_s1_known) & ~np.isnan(d_s2_known)
            known_consistency_lower = float(np.max(np.abs(d_s1_known[both_orig] - d_s2_known[both_orig]))) if both_orig.any() else 0

            d_s1_eff = d_s1_known.copy()
            d_s2_eff = d_s2_known.copy()
            mat_s1 = mat_arr[i1]
            mat_s2 = mat_arr[i2]
            fill_s1 = np.isnan(d_s1_eff) & ~np.isnan(mat_s1) & (mat_s1 > 0)
            fill_s2 = np.isnan(d_s2_eff) & ~np.isnan(mat_s2) & (mat_s2 > 0)
            d_s1_eff[fill_s1] = mat_s1[fill_s1]
            d_s2_eff[fill_s2] = mat_s2[fill_s2]
            d_s1_eff[i1] = np.nan
            d_s1_eff[i2] = np.nan
            d_s2_eff[i1] = np.nan
            d_s2_eff[i2] = np.nan

            both_eff = ~np.isnan(d_s1_eff) & ~np.isnan(d_s2_eff)

            consistency_lower = 0
            consistency_upper = float('inf')
            consistency_evidence = []

            if both_eff.any():
                eff_diffs = np.abs(d_s1_eff[both_eff] - d_s2_eff[both_eff])
                eff_sums = d_s1_eff[both_eff] + d_s2_eff[both_eff]
                consistency_lower = float(np.max(eff_diffs))
                consistency_upper = float(np.min(eff_sums))
                if should_debug:
                    eff_k_idxs = np.where(both_eff)[0][:3]
                    for ki in eff_k_idxs:
                        consistency_evidence.append((samples[ki], d_s1_eff[ki], d_s2_eff[ki],
                                                     abs(d_s1_eff[ki] - d_s2_eff[ki]),
                                                     d_s1_eff[ki] + d_s2_eff[ki]))

            anchor_lower = max_diff
            anchor_upper = float('inf')

            d1_anc = orig_arr[i1, anchor_idxs]
            d2_anc = orig_arr[i2, anchor_idxs]
            valid_anc = ~np.isnan(d1_anc) & ~np.isnan(d2_anc) & (d1_anc > 0) & (d2_anc > 0)
            if valid_anc.any():
                anchor_upper = float(np.min(d1_anc[valid_anc] + d2_anc[valid_anc]))
            
            effective_lower = max(consistency_lower, anchor_lower)
            effective_upper = min(consistency_upper, anchor_upper) if consistency_upper < float('inf') else anchor_upper
            
            if should_debug:
                print(f"    Pass {pass_num}: consistency_lower={consistency_lower:.0f}, known_cons_lower={known_consistency_lower:.0f}, effective_upper={effective_upper:.0f}")
                print(f"    consistency_evidence count={len(consistency_evidence)}")
                for k, d1, d2, lo, up in consistency_evidence[:3]:
                    print(f"      via {k}: d1={d1:.0f}, d2={d2:.0f}, bounds=[{lo:.0f}, {up:.0f}]")
            
            s1_root = find(s1)
            s2_root = find(s2)
            same_lineage_group = (s1_root == s2_root)
            
            mean_profile = np.nanmean(np.concatenate([profile1[valid_mask], profile2[valid_mask]]))
            has_low_consistency_bound = consistency_lower < 200
            
            s1_min_anchor = sample_min_anchor_dist.get(s1, 0)
            s2_min_anchor = sample_min_anchor_dist.get(s2, 0)
            both_deep = (s1_min_anchor > 800 and s2_min_anchor > 800)
            
            is_deep_same_lineage = (mean_profile > 1200 and max_diff < 50 and std_diff < 25 
                                    and has_low_consistency_bound and not both_deep)
            is_potentially_close = (max_diff < 20 and has_low_consistency_bound and not both_deep)
            
            if should_debug:
                print(f"    mean_profile={mean_profile:.0f}, max_diff={max_diff:.0f}, std_diff={std_diff:.2f}")
                print(f"    is_deep_same_lineage={is_deep_same_lineage}, is_potentially_close={is_potentially_close}")
            
            lineage_estimate = None
            
            if is_cross_lab and is_deep_same_lineage:
                s1_far = s1 in far_from_anchors
                s2_far = s2 in far_from_anchors
                both_far = s1_far and s2_far
                
                if both_far:
                    s1_min = sample_min_anchor_dist[s1]
                    s2_min = sample_min_anchor_dist[s2]
                    parallel_estimate = s1_min + s2_min
                    
                    if s1 in sample_bias or s2 in sample_bias:
                         correction = (sample_bias.get(s1, 0) + sample_bias.get(s2, 0)) * 0.5
                         if correction > 0:
                             parallel_estimate = max(parallel_estimate - correction, effective_lower)
                             if should_debug:
                                 print(f"    Bias correction: -{correction:.1f} (s1_bias={sample_bias.get(s1,0):.1f}, s2_bias={sample_bias.get(s2,0):.1f})")

                    if effective_upper < float('inf'):
                        parallel_estimate = min(parallel_estimate, effective_upper * 0.9)
                    
                    lineage_estimate = max(parallel_estimate, effective_lower, 500)
                    
                    if should_debug:
                        print(f"    Both far from anchors")
                        print(f"    s1_min={s1_min:.0f}, s2_min={s2_min:.0f} to parallel_estimate={lineage_estimate:.0f}")
                else:
                    calibration_distances = []
                    if same_lineage_group:
                        lineage_root = find(s1)
                        for (root, sa, sb), known_dist in lineage_cross_lab_distances.items():
                            if root == lineage_root:
                                calibration_distances.append(known_dist)
                                if should_debug:
                                    print(f"Same lineage: Found reference {sa}↔{sb} = {known_dist}")
                    
                    if calibration_distances:
                        lineage_estimate = np.median(calibration_distances)
                    elif max_diff <= 10 and std_diff < 2:
                        mult = very_close_multiplier
                        lineage_estimate = max(max_diff * mult, median_diff * (mult + 1), 30)
                        if should_debug:
                            print(f"    max_diff={max_diff:.0f} tight, ignoring effective_lower={effective_lower:.0f}")
                    elif max_diff <= 15 and std_diff < 4 and known_consistency_lower < 50:
                        mult = very_close_multiplier - 2
                        lineage_estimate = max(max_diff * mult, median_diff * (mult + 1), 50)
                        if should_debug:
                            print(f"    max_diff={max_diff:.0f}, known_cons={known_consistency_lower:.0f}")
                            print(f"      Bypassing effective_lower={effective_lower:.0f}, estimate={lineage_estimate:.0f}")
                    else:
                        lineage_estimate = max(max_diff * 6, effective_lower * 1.1, 80)
                    
                    if not (max_diff <= 10 and std_diff < 2) and not (max_diff <= 15 and std_diff < 4 and known_consistency_lower < 50):
                        lineage_estimate = max(lineage_estimate, effective_lower)
                
                if should_debug:
                    print(f"Same lineage: max_diff={max_diff:.0f}, std_diff={std_diff:.2f}, estimate={lineage_estimate:.0f}")
            
            elif is_cross_lab and is_potentially_close:
                s1_far = s1 in far_from_anchors
                s2_far = s2 in far_from_anchors
                both_far = s1_far and s2_far
                
                if both_far:
                    s1_min = sample_min_anchor_dist[s1]
                    s2_min = sample_min_anchor_dist[s2]
                    parallel_estimate = s1_min + s2_min
                    
                    if s1 in sample_bias or s2 in sample_bias:
                         correction = (sample_bias.get(s1, 0) + sample_bias.get(s2, 0)) * 0.5
                         if correction > 0:
                             parallel_estimate = max(parallel_estimate - correction, effective_lower)
                             if should_debug:
                                 print(f"    Bias correction: -{correction:.1f} (s1_bias={sample_bias.get(s1,0):.1f}, s2_bias={sample_bias.get(s2,0):.1f})")

                    if effective_upper < float('inf'):
                        parallel_estimate = min(parallel_estimate, effective_upper * 0.9)
                    lineage_estimate = max(parallel_estimate, effective_lower, 400)
                    if should_debug:
                        print(f"    POTENTIALLY_CLOSE but ANCHOR-CONSTRAINED: estimate={lineage_estimate:.0f}")
                else:
                    close_calibration = []
                    if same_lineage_group:
                        for other in samples:
                            if other == s1 or other == s2:
                                continue
                            oi = sidx[other]
                            other_labs = set(sample_to_lab.get(other, []))
                            
                            if s1_labs & other_labs and orig_kmask_arr[i1, oi]:
                                d = orig_arr[i1, oi]
                                if 0 < d < 150:
                                    other_profile = sample_profiles.get(other, np.array([]))
                                    if len(other_profile) == len(profile2):
                                        other_diffs = np.abs(profile2 - other_profile)
                                        valid = ~np.isnan(other_diffs)
                                        if valid.sum() >= 3 and np.nanmax(other_diffs) < 25:
                                            close_calibration.append(d)
                            
                            if s2_labs & other_labs and orig_kmask_arr[i2, oi]:
                                d = orig_arr[i2, oi]
                                if 0 < d < 150:
                                    other_profile = sample_profiles.get(other, np.array([]))
                                    if len(other_profile) == len(profile1):
                                        other_diffs = np.abs(profile1 - other_profile)
                                        valid = ~np.isnan(other_diffs)
                                        if valid.sum() >= 3 and np.nanmax(other_diffs) < 25:
                                            close_calibration.append(d)
                    
                    if close_calibration:
                        close_estimate = np.median(close_calibration) + max_diff * 2
                    else:
                        close_estimate = max(max_diff * 4, effective_lower * 1.05, 10)
                    
                    close_estimate = max(close_estimate, effective_lower)
                    
                    if should_debug:
                        print(f"    POTENTIALLY_CLOSE: max_diff={max_diff:.0f}, calibration={close_calibration[:3]}, estimate={close_estimate:.0f}")
                    
                    lineage_estimate = close_estimate
            
            elif is_cross_lab and max_diff < 50 and mean_profile > 1500:
                s1_far = s1 in far_from_anchors
                s2_far = s2 in far_from_anchors
                both_far = s1_far and s2_far
                
                if both_far:
                    s1_min = sample_min_anchor_dist[s1]
                    s2_min = sample_min_anchor_dist[s2]
                    parallel_estimate = s1_min + s2_min
                    if effective_upper < float('inf'):
                        parallel_estimate = min(parallel_estimate, effective_upper * 0.9)
                    lineage_estimate = max(parallel_estimate, effective_lower, 500)
                    if should_debug:
                        print(f"    MODERATELY_CLOSE but ANCHOR-CONSTRAINED: estimate={lineage_estimate:.0f}")
                else:
                    calibration_distances = []
                    
                    s1_anchor_dist = sample_min_anchor_dist.get(s1, 0)
                    s2_anchor_dist = sample_min_anchor_dist.get(s2, 0)
                    either_deep = (s1_anchor_dist > 800 or s2_anchor_dist > 800)
                    
                    if should_debug:
                        print(f"    s1_anchor_dist={s1_anchor_dist:.0f}, s2_anchor_dist={s2_anchor_dist:.0f}, either_deep={either_deep}")
                    
                    if same_lineage_group:
                        lineage_root = find(s1)
                        for (root, sa, sb), known_dist in lineage_cross_lab_distances.items():
                            if root == lineage_root:
                                calibration_distances.append(known_dist)
                                if should_debug:
                                    print(f"    Found lineage cross-lab reference: {sa}↔{sb} = {known_dist}")
                    
                    if not either_deep:
                        for other in samples:
                            if other == s1 or other == s2:
                                continue
                            oi = sidx[other]
                            other_labs = set(sample_to_lab.get(other, []))
                            if s1_labs & other_labs and known_arr[i1, oi]:
                                d_s1_other = orig_arr[i1, oi]
                                if d_s1_other > 0 and d_s1_other < 400:
                                    other_profile = sample_profiles.get(other, np.array([]))
                                    if len(other_profile) == len(profile2):
                                        other_s2_diffs = np.abs(profile2 - other_profile)
                                        valid_other = ~np.isnan(other_s2_diffs)
                                        if valid_other.sum() >= 3:
                                            other_s2_max_diff = np.nanmax(other_s2_diffs)
                                            if other_s2_max_diff < 50:
                                                calibration_distances.append(d_s1_other)
                                                if should_debug:
                                                    print(f"    Calibration via {other}: d_s1={d_s1_other:.0f}, profile_diff={other_s2_max_diff:.0f}")
                            
                            if s2_labs & other_labs and known_arr[i2, oi]:
                                d_s2_other = orig_arr[i2, oi]
                                if d_s2_other > 0 and d_s2_other < 400:
                                    other_profile = sample_profiles.get(other, np.array([]))
                                    if len(other_profile) == len(profile1):
                                        other_s1_diffs = np.abs(profile1 - other_profile)
                                        valid_other = ~np.isnan(other_s1_diffs)
                                        if valid_other.sum() >= 3:
                                            other_s1_max_diff = np.nanmax(other_s1_diffs)
                                            if other_s1_max_diff < 50:
                                                calibration_distances.append(d_s2_other)
                                                if should_debug:
                                                    print(f"    Calibration via {other}: d_s2={d_s2_other:.0f}, profile_diff={other_s1_max_diff:.0f}")
                    else:
                        if should_debug:
                            print(f"    SKIPPING calibration")
                    
                    if calibration_distances:
                        median_cal = np.median(calibration_distances)
                        lineage_estimate = median_cal + max_diff * 1.5
                        lineage_estimate = max(lineage_estimate, effective_lower)
                        
                        if should_debug:
                            print(f"    MODERATELY_CLOSE with calibration: {calibration_distances[:5]}, estimate={lineage_estimate:.0f}")
                    else:

                        both_deep_local = (s1_anchor_dist > 800 and s2_anchor_dist > 800)

                        has_loose_bounds = effective_upper > 1000 and std_diff > 2
                        is_deep_parallel = both_deep_local and (consistency_lower > 400 or has_loose_bounds)
                        
                        if should_debug:
                            print(f"    both_deep={both_deep_local}, consistency_lower={consistency_lower:.0f}")
                            print(f"      effective_upper={effective_upper:.0f}, std_diff={std_diff:.2f}, has_loose_bounds={has_loose_bounds}")
                            print(f"      is_deep_parallel={is_deep_parallel}")
                        
                        if is_deep_parallel:
                            anchor_sum_estimate = (s1_anchor_dist + s2_anchor_dist) * 0.4
                            anchor_sum_estimate = max(anchor_sum_estimate, 400)
                            lineage_estimate = max(anchor_sum_estimate, effective_lower)
                            if should_debug:
                                print(f"    MODERATELY_CLOSE (BOTH_DEEP + INCONSISTENT): anchor_sum_estimate={anchor_sum_estimate:.0f}")
                                print(f"      s1_anchor_dist={s1_anchor_dist:.0f}, s2_anchor_dist={s2_anchor_dist:.0f}")
                                print(f"      consistency_lower={consistency_lower:.0f}")
                        elif max_diff <= 10 and std_diff < 2.0 and not both_deep_local:
                            mult = very_close_multiplier  
                            lineage_estimate = max(max_diff * mult, median_diff * (mult + 1), 30)
                            if should_debug:
                                print(f"    MODERATELY_CLOSE PROFILE-PRIORITY: max_diff={max_diff:.0f}, std_diff={std_diff:.2f}")
                                print(f"      Bypassing effective_lower={effective_lower:.0f}, estimate={lineage_estimate:.0f}")
                        elif std_diff < 15:
                            lineage_estimate = max(max_diff * 4, 40)
                        else:
                            lineage_estimate = max(max_diff * 5, 50)
                        
                        if both_deep_local or max_diff > 10 or std_diff >= 2.0:
                            lineage_estimate = max(lineage_estimate, effective_lower)
                        
                        if should_debug:
                            print(f"    MODERATELY_CLOSE fallback: estimate={lineage_estimate:.0f}")
            
            elif is_cross_lab and both_deep and max_diff < 60 and std_diff < 20 and has_low_consistency_bound:
                if max_diff < 10 and std_diff < 3:
                    lineage_estimate = max(max_diff * 4, effective_lower, 10)
                elif max_diff < 30 and std_diff < 8:
                    lineage_estimate = max(max_diff * 5, effective_lower, 30)
                else:
                    lineage_estimate = max(max_diff * 6, effective_lower, 50)
                if should_debug:
                    print(f"    BOTH_DEEP_CLOSE: max_diff={max_diff:.0f}, std={std_diff:.2f}, est={lineage_estimate:.0f}")
            
            if lineage_estimate is not None:
                close_pairs_to_impute.append((s1, s2, lineage_estimate, f"LINEAGE_PROXY_PASS{pass_num}"))
                processed_pairs.add((s1, s2))
                
                mat_arr[i1, i2] = lineage_estimate
                mat_arr[i2, i1] = lineage_estimate
                close_arr[i1, i2] = True
                close_arr[i2, i1] = True
                known_arr[i1, i2] = True
                known_arr[i2, i1] = True
                
                bound_spread = effective_upper - effective_lower if effective_upper < float('inf') else float('inf')
                if bound_spread < 150:
                    high_confidence_arr[i1, i2] = True
                    high_confidence_arr[i2, i1] = True
                
                pairs_added_this_pass += 1
                imputation_stats["lineage_proxy"] += 1
                
                if should_debug:
                    print(f"    Applied estimate: {lineage_estimate:.0f}")
                continue
            
            if pass_num < MAX_PASSES - 1 and is_cross_lab and is_deep_same_lineage:
                continue
            
            if pass_num == MAX_PASSES - 1:
                small_diff_count = np.sum(diffs < 100)
                small_diff_ratio = small_diff_count / len(diffs)
                
                if consistency_lower > 100:
                    imputation_stats["consistency_override"] += 1
                    processed_pairs.add((s1, s2))
                    continue
                
                if effective_lower > 100:
                    processed_pairs.add((s1, s2))
                    continue
                
                if max_diff < 5 and small_diff_ratio >= 0.95 and effective_lower < 30:
                    mult = 2.0
                    close_estimate = max(max_diff * mult, median_diff * 2, effective_lower * 1.05, 5)
                    close_pairs_to_impute.append((s1, s2, close_estimate, "VERY_CLOSE"))
                    processed_pairs.add((s1, s2))
                    mat_arr[i1, i2] = close_estimate
                    mat_arr[i2, i1] = close_estimate
                    close_arr[i1, i2] = True
                    close_arr[i2, i1] = True
                    known_arr[i1, i2] = True
                    known_arr[i2, i1] = True
                    high_confidence_arr[i1, i2] = True
                    high_confidence_arr[i2, i1] = True
                
                elif max_diff < 15 and small_diff_ratio >= 0.90 and effective_lower < 75:
                    mult = 2.0  
                    close_estimate = max(max_diff * mult, mean_diff * 2, effective_lower * 1.02, 10)
                    close_pairs_to_impute.append((s1, s2, close_estimate, "CLOSE"))
                    processed_pairs.add((s1, s2))
                    mat_arr[i1, i2] = close_estimate
                    mat_arr[i2, i1] = close_estimate
                    close_arr[i1, i2] = True
                    close_arr[i2, i1] = True
                    known_arr[i1, i2] = True
                    known_arr[i2, i1] = True
                    high_confidence_arr[i1, i2] = True
                    high_confidence_arr[i2, i1] = True
                
                elif max_diff < 50 and small_diff_ratio >= 0.8 and effective_lower < 150:
                    mult = 4.0
                    close_estimate = max(max_diff * mult, mean_diff * 5, effective_lower * 1.05, 100)
                    close_pairs_to_impute.append((s1, s2, close_estimate, "MODERATELY_CLOSE"))
                    processed_pairs.add((s1, s2))
                    mat_arr[i1, i2] = close_estimate
                    mat_arr[i2, i1] = close_estimate
                    close_arr[i1, i2] = True
                    close_arr[i2, i1] = True
                    known_arr[i1, i2] = True
                    known_arr[i2, i1] = True
                    bound_spread_mc = effective_upper - effective_lower if effective_upper < float('inf') else float('inf')
                    if bound_spread_mc < 150:
                        high_confidence_arr[i1, i2] = True
                        high_confidence_arr[i2, i1] = True
        
        print(f"  Pass {pass_num}: added {pairs_added_this_pass} estimates")
        
        if pairs_added_this_pass == 0 and pass_num > 0:
            break
    
    print(f"\n  Close pairs detected: {len(close_pairs_to_impute)}")
    category_counts = {}
    for s1, s2, estimate, category in close_pairs_to_impute:
        category_counts[category] = category_counts.get(category, 0) + 1
    print(f"  Category breakdown: {category_counts}")

    correction_count = 0
    for i, (s1, s2, estimate, category) in enumerate(close_pairs_to_impute):
        if not category.startswith("LINEAGE_PROXY"):
            continue
        p1 = sample_profiles.get(s1, np.array([]))
        p2 = sample_profiles.get(s2, np.array([]))
        if len(p1) == 0 or len(p2) == 0 or len(p1) != len(p2):
            continue
        valid = ~np.isnan(p1) & ~np.isnan(p2)
        if valid.sum() < 3:
            continue
        diffs_arr = np.abs(p1[valid] - p2[valid])
        pair_max_diff = np.max(diffs_arr)
        
        if pair_max_diff > 15 and pair_max_diff <= 30:
            adaptive_mult = max(12, pair_max_diff * 0.5)
            new_estimate = max(pair_max_diff * adaptive_mult, estimate)
            if new_estimate > estimate:
                should_debug = (s1, s2) in debug_pairs
                if should_debug:
                    print(f"  POST-CORRECTION: {s1} and {s2}: max_diff={pair_max_diff:.0f}, {estimate:.0f} -> {new_estimate:.0f}")
                i1_pc, i2_pc = sidx[s1], sidx[s2]
                mat_arr[i1_pc, i2_pc] = new_estimate
                mat_arr[i2_pc, i1_pc] = new_estimate
                close_pairs_to_impute[i] = (s1, s2, new_estimate, category)
                correction_count += 1
    if correction_count > 0:
        print(f"  Post-lineage-proxy correction: {correction_count} pairs adjusted")

    crosslab_very_close = []
    for s1, s2 in missing_pairs:
        if (s1, s2) in processed_pairs or (s2, s1) in processed_pairs:
            continue
        
        s1_labs = set(sample_to_lab.get(s1, []))
        s2_labs = set(sample_to_lab.get(s2, []))
        is_cross_lab = not bool(s1_labs & s2_labs)
        
        if not is_cross_lab:
            continue
        
        profile1 = sample_profiles.get(s1, np.array([]))
        profile2 = sample_profiles.get(s2, np.array([]))
        if len(profile1) == 0 or len(profile2) == 0 or len(profile1) != len(profile2):
            continue
        valid_mask = ~np.isnan(profile1) & ~np.isnan(profile2)
        if valid_mask.sum() < 3:
            continue
        diffs = np.abs(profile1[valid_mask] - profile2[valid_mask])
        max_diff = np.max(diffs)
        std_diff = np.std(diffs)
        median_diff = np.median(diffs)
        
        small_diff_count = np.sum(diffs < 100)
        small_diff_ratio = small_diff_count / len(diffs)
        
        s1_min_anchor = sample_min_anchor_dist.get(s1, 0)
        s2_min_anchor = sample_min_anchor_dist.get(s2, 0)
        both_deep = (s1_min_anchor > 800 and s2_min_anchor > 800)
        
        if both_deep:
            if max_diff > 15 or std_diff >= 5.0 or small_diff_ratio < 0.85:
                continue
        else:
            if max_diff > 50 or std_diff >= 15.0 or small_diff_ratio < 0.85:
                continue
        
        known_cons_lower = 0
        i1_vc, i2_vc = sidx[s1], sidx[s2]
        d_s1_vc = all_known_arr[i1_vc].copy()
        d_s2_vc = all_known_arr[i2_vc].copy()
        d_s1_vc[i1_vc] = np.nan
        d_s1_vc[i2_vc] = np.nan
        d_s2_vc[i1_vc] = np.nan
        d_s2_vc[i2_vc] = np.nan
        both_vc = ~np.isnan(d_s1_vc) & ~np.isnan(d_s2_vc)
        known_cons_lower = float(np.max(np.abs(d_s1_vc[both_vc] - d_s2_vc[both_vc]))) if both_vc.any() else 0
        
        should_debug_vc = (s1, s2) in debug_pairs
        if should_debug_vc:
            print(f"  CROSSLAB_VERY_CLOSE check: {s1} and {s2}")
            print(f"    max_diff={max_diff:.0f}, std_diff={std_diff:.2f}, small_ratio={small_diff_ratio:.2f}, known_cons_lower={known_cons_lower:.0f}, both_deep={both_deep}")
        
        if known_cons_lower < 50:
            mult = very_close_multiplier
            close_estimate = max(max_diff * mult, median_diff * (mult + 1), known_cons_lower * 1.2, 30)
            
            crosslab_very_close.append((s1, s2, close_estimate))
            processed_pairs.add((s1, s2))
            mat_arr[i1_vc, i2_vc] = close_estimate
            mat_arr[i2_vc, i1_vc] = close_estimate
            close_arr[i1_vc, i2_vc] = True
            close_arr[i2_vc, i1_vc] = True
            known_arr[i1_vc, i2_vc] = True
            known_arr[i2_vc, i1_vc] = True
            if max_diff < 15:
                high_confidence_arr[i1_vc, i2_vc] = True
                high_confidence_arr[i2_vc, i1_vc] = True
            imputation_stats["profile_very_close"] = imputation_stats.get("profile_very_close", 0) + 1
            
            if should_debug_vc:
                print(f"    estimate={close_estimate:.0f}")
    
    print(f"  Cross-lab VERY_CLOSE detected: {len(crosslab_very_close)}")

    neighbor_bridge_count = 0
    
    for s1, s2 in missing_pairs:
        if (s1, s2) in processed_pairs or (s2, s1) in processed_pairs:
            continue
        
        s1_labs = set(sample_to_lab.get(s1, []))
        s2_labs = set(sample_to_lab.get(s2, []))
        is_cross_lab = not bool(s1_labs & s2_labs)
        if not is_cross_lab:
            continue
        
        profile1 = sample_profiles.get(s1, np.array([]))
        profile2 = sample_profiles.get(s2, np.array([]))
        if len(profile1) == 0 or len(profile2) == 0 or len(profile1) != len(profile2):
            continue
        valid_mask = ~np.isnan(profile1) & ~np.isnan(profile2)
        if valid_mask.sum() < 3:
            continue
        diffs = np.abs(profile1[valid_mask] - profile2[valid_mask])
        max_diff = np.max(diffs)
        std_diff = np.std(diffs)
        
        if max_diff > 25 or std_diff >= 2.9:
            continue
        
        should_debug_nb = (s1, s2) in debug_pairs
        
        bridge_estimates = []
        bridge_d_neighbor_other = []
        
        for target, other in [(s1, s2), (s2, s1)]:
            it, io = sidx[target], sidx[other]
            target_labs = set(sample_to_lab.get(target, []))
            for ni in range(n_samp):
                if ni == it or ni == io:
                    continue
                neighbor = samples[ni]
                neighbor_labs = set(sample_to_lab.get(neighbor, []))
                if not bool(target_labs & neighbor_labs):
                    continue
                if not orig_kmask_arr[it, ni]:
                    continue
                d_target_neighbor = orig_arr[it, ni]
                if np.isnan(d_target_neighbor) or d_target_neighbor > 150:
                    continue

                is_orig = orig_kmask_arr[ni, io]
                is_high_conf = high_confidence_arr[ni, io]
                if not (is_orig or is_high_conf):
                    continue

                if is_orig:
                    d_neighbor_other = orig_arr[ni, io]
                else:
                    d_neighbor_other = mat_arr[ni, io]

                if np.isnan(d_neighbor_other):
                    continue

                bridge_est = d_neighbor_other + d_target_neighbor  
                bridge_lower = abs(d_neighbor_other - d_target_neighbor)
                bridge_est = (bridge_lower + d_neighbor_other + d_target_neighbor) / 2 
                
                src_type = "orig" if is_orig else "close"
                bridge_estimates.append(bridge_est)
                bridge_d_neighbor_other.append(d_neighbor_other)
                
                if should_debug_nb:
                    print(f"  NEIGHBOR-BRIDGE: {target}->{neighbor}(d={d_target_neighbor:.0f})->{other}(d={d_neighbor_other:.0f} [{src_type}]), est={bridge_est:.0f}")
        
        if bridge_estimates and len(bridge_estimates) >= 2:
            close_bridges = [est for est, d_no in zip(bridge_estimates, bridge_d_neighbor_other) if d_no < 150]
            
            if len(close_bridges) >= 2:
                final_estimate = np.median(close_bridges)
                if should_debug_nb:
                    print(f"  NEIGHBOR-BRIDGE: using {len(close_bridges)}/{len(bridge_estimates)} close bridges")
            else:
                final_estimate = np.median(bridge_estimates)
            
            if np.std(bridge_estimates) > 100:
                if should_debug_nb:
                    print(f"  NEIGHBOR-BRIDGE SKIPPED: high variance {np.std(bridge_estimates):.0f}")
                continue
            
            if should_debug_nb:
                print(f"  NEIGHBOR-BRIDGE FINAL: {s1} <-> {s2}, bridges={len(bridge_estimates)}, estimate={final_estimate:.0f}")
            
            processed_pairs.add((s1, s2))
            i1_nb, i2_nb = sidx[s1], sidx[s2]
            mat_arr[i1_nb, i2_nb] = final_estimate
            mat_arr[i2_nb, i1_nb] = final_estimate
            close_arr[i1_nb, i2_nb] = True
            close_arr[i2_nb, i1_nb] = True
            known_arr[i1_nb, i2_nb] = True
            known_arr[i2_nb, i1_nb] = True
            if np.std(bridge_estimates) < 50:
                high_confidence_arr[i1_nb, i2_nb] = True
                high_confidence_arr[i2_nb, i1_nb] = True
            neighbor_bridge_count += 1
            imputation_stats["neighbor_bridge"] = imputation_stats.get("neighbor_bridge", 0) + 1
    
    print(f"  Neighbor-bridge estimated: {neighbor_bridge_count}")

    fp_post_corrections = 0
    fp_post_unprotected = 0
    fp_tier_counts = {'tier1': 0, 'tier2': 0, 'tier3': 0, 'tier4': 0}
    
    for s1, s2 in list(processed_pairs):
        fp_val = fp_precomputed.get((s1, s2), None)
        if fp_val is None:
            continue
        
        i1_fpc, i2_fpc = sidx[s1], sidx[s2]
        current_est = mat_arr[i1_fpc, i2_fpc]
        
        if current_est <= 0 or np.isnan(current_est):
            continue
        
        p1_fpc = sample_profiles.get(s1, np.array([]))
        p2_fpc = sample_profiles.get(s2, np.array([]))
        max_diff_fpc = 0
        profile_euclid = 0
        if len(p1_fpc) > 0 and len(p2_fpc) > 0 and len(p1_fpc) == len(p2_fpc):
            valid_fpc = ~np.isnan(p1_fpc) & ~np.isnan(p2_fpc)
            if valid_fpc.sum() >= 3:
                diffs_fpc = np.abs(p1_fpc[valid_fpc] - p2_fpc[valid_fpc])
                max_diff_fpc = float(np.max(diffs_fpc))
                profile_euclid = float(np.sqrt(np.sum(diffs_fpc ** 2)))
        
        if max_diff_fpc > 12 and fp_val > 1.5 * current_est and fp_val > 120:
            new_est = fp_val * 0.75
            mat_arr[i1_fpc, i2_fpc] = new_est
            mat_arr[i2_fpc, i1_fpc] = new_est
            close_arr[i1_fpc, i2_fpc] = False
            close_arr[i2_fpc, i1_fpc] = False
            high_confidence_arr[i1_fpc, i2_fpc] = False
            high_confidence_arr[i2_fpc, i1_fpc] = False
            fp_post_corrections += 1
            fp_tier_counts['tier1'] += 1
        
        elif max_diff_fpc > 6:
            mds_val = mds_estimates.get((s1, s2), None) if mds_estimates else None
            if (mds_val is not None and mds_val > 350 
                and fp_val > 2.0 * current_est and fp_val > 200):
                new_est = 0.4 * mds_val + 0.4 * fp_val * 0.7 + 0.2 * max(current_est, max_diff_fpc * 5)
                mat_arr[i1_fpc, i2_fpc] = new_est
                mat_arr[i2_fpc, i1_fpc] = new_est
                close_arr[i1_fpc, i2_fpc] = False
                close_arr[i2_fpc, i1_fpc] = False
                high_confidence_arr[i1_fpc, i2_fpc] = False
                high_confidence_arr[i2_fpc, i1_fpc] = False
                fp_post_corrections += 1
                fp_tier_counts['tier2'] += 1
            
            elif (mds_val is not None and mds_val > 250
                  and profile_euclid > 14
                  and fp_val > 1.5 * current_est and fp_val > 120):
                # Softer correction with higher current weight
                new_est = 0.3 * mds_val + 0.3 * fp_val * 0.7 + 0.4 * max(current_est, max_diff_fpc * 5)
                mat_arr[i1_fpc, i2_fpc] = new_est
                mat_arr[i2_fpc, i1_fpc] = new_est
                close_arr[i1_fpc, i2_fpc] = False
                close_arr[i2_fpc, i1_fpc] = False
                high_confidence_arr[i1_fpc, i2_fpc] = False
                high_confidence_arr[i2_fpc, i1_fpc] = False
                fp_post_corrections += 1
                fp_tier_counts['tier3'] += 1
            
            elif (mds_val is not None and mds_val > 280
                  and profile_euclid > 12
                  and fp_val > 1.2 * current_est):
                close_arr[i1_fpc, i2_fpc] = False
                close_arr[i2_fpc, i1_fpc] = False
                fp_post_unprotected += 1
        
        elif max_diff_fpc > 4:
            mds_val = mds_estimates.get((s1, s2), None) if mds_estimates else None
            if (mds_val is not None and mds_val > 300
                and profile_euclid > 16
                and fp_val > 2.0 * current_est and fp_val > 200):
                new_est = 0.25 * mds_val + 0.25 * fp_val * 0.7 + 0.5 * max(current_est, max_diff_fpc * 5)
                mat_arr[i1_fpc, i2_fpc] = new_est
                mat_arr[i2_fpc, i1_fpc] = new_est
                close_arr[i1_fpc, i2_fpc] = False
                close_arr[i2_fpc, i1_fpc] = False
                high_confidence_arr[i1_fpc, i2_fpc] = False
                high_confidence_arr[i2_fpc, i1_fpc] = False
                fp_post_corrections += 1
                fp_tier_counts['tier4'] += 1
    
    print(f"  Four-point post-correction: {fp_post_corrections} false close pairs corrected "
          f"(T1={fp_tier_counts['tier1']}, T2={fp_tier_counts['tier2']}, "
          f"T3={fp_tier_counts['tier3']}, T4={fp_tier_counts['tier4']}), "
          f"{fp_post_unprotected} unprotected")

    remaining_pairs = []
    for s1, s2 in missing_pairs:
        if (s1, s2) not in processed_pairs and (s2, s1) not in processed_pairs:
            remaining_pairs.append((s1, s2))
    
    print(f"  Remaining pairs to impute with anchor triangulation: {len(remaining_pairs)}")
    
    for s1, s2 in remaining_pairs:
        should_debug = (s1, s2) in debug_pairs
        estimates = []
        weights = []
        fp_gating_value = None
        
        profile1 = sample_profiles[s1]
        profile2 = sample_profiles[s2]
        valid_mask = ~np.isnan(profile1) & ~np.isnan(profile2)
        
        s1_root = find(s1)
        s2_root = find(s2)
        same_lineage_group = (s1_root == s2_root)
        
        known_cons_lower = 0
        cons_lower_with_imputed = 0
        i1_r, i2_r = sidx[s1], sidx[s2]

        d_s1_rk = all_known_arr[i1_r].copy()
        d_s2_rk = all_known_arr[i2_r].copy()
        d_s1_rk[i1_r] = np.nan
        d_s1_rk[i2_r] = np.nan
        d_s2_rk[i1_r] = np.nan
        d_s2_rk[i2_r] = np.nan
        both_rk = ~np.isnan(d_s1_rk) & ~np.isnan(d_s2_rk)
        known_cons_lower = float(np.max(np.abs(d_s1_rk[both_rk] - d_s2_rk[both_rk]))) if both_rk.any() else 0

        d_s1_reff = d_s1_rk.copy()
        d_s2_reff = d_s2_rk.copy()
        fill_r1 = np.isnan(d_s1_reff) & ~np.isnan(mat_arr[i1_r]) & (mat_arr[i1_r] > 0)
        fill_r2 = np.isnan(d_s2_reff) & ~np.isnan(mat_arr[i2_r]) & (mat_arr[i2_r] > 0)
        d_s1_reff[fill_r1] = mat_arr[i1_r][fill_r1]
        d_s2_reff[fill_r2] = mat_arr[i2_r][fill_r2]
        d_s1_reff[i1_r] = np.nan
        d_s1_reff[i2_r] = np.nan
        d_s2_reff[i1_r] = np.nan
        d_s2_reff[i2_r] = np.nan
        both_reff = ~np.isnan(d_s1_reff) & ~np.isnan(d_s2_reff)
        cons_lower_with_imputed = float(np.max(np.abs(d_s1_reff[both_reff] - d_s2_reff[both_reff]))) if both_reff.any() else 0
        
        consistency_lower = known_cons_lower
        
        s1_min_anchor = sample_min_anchor_dist.get(s1, 0)
        s2_min_anchor = sample_min_anchor_dist.get(s2, 0)
        both_deep = (s1_min_anchor > 800 and s2_min_anchor > 800)
        
        has_low_consistency_bound = known_cons_lower < 200
        
        lower_bounds_early = []
        upper_bounds_early = []
        max_lower_early = 0
        min_upper_early = float('inf')
        d1_anc_early = orig_arr[i1_r, anchor_idxs]
        d2_anc_early = orig_arr[i2_r, anchor_idxs]
        valid_anc_early = ~np.isnan(d1_anc_early) & ~np.isnan(d2_anc_early) & (d1_anc_early > 0) & (d2_anc_early > 0)
        if valid_anc_early.any():
            lower_bounds_early = list(np.abs(d1_anc_early[valid_anc_early] - d2_anc_early[valid_anc_early]))
            upper_bounds_early = list(d1_anc_early[valid_anc_early] + d2_anc_early[valid_anc_early])
            max_lower_early = max(lower_bounds_early)
            min_upper_early = min(upper_bounds_early)
        
        if valid_mask.sum() >= 3:
            diffs = np.abs(profile1[valid_mask] - profile2[valid_mask])
            max_diff = np.max(diffs)
            std_diff = np.std(diffs)
            mean_profile = np.nanmean(np.concatenate([profile1[valid_mask], profile2[valid_mask]]))
            is_deep_same_lineage = (mean_profile > 1200 and max_diff < 50 and std_diff < 25 
                                    and has_low_consistency_bound and not both_deep)
        else:
            is_deep_same_lineage = False
            max_diff = 0
            mean_profile = 0
        
        if valid_mask.sum() >= 3:
            diffs = np.abs(profile1[valid_mask] - profile2[valid_mask])
            min_diff = np.min(diffs)
            median_diff = np.median(diffs)
            mean_diff = np.mean(diffs)
            max_diff = np.max(diffs)
            
            small_count = np.sum(diffs < 100)
            small_ratio = small_count / len(diffs)
            
            can_be_close = has_low_consistency_bound and not both_deep
            
            if max_lower_early > 200:
                can_be_close = False
            
            if can_be_close:
                if median_diff < 50 and small_ratio >= 0.7:
                    close_estimate = min_diff + 0.1 * (median_diff - min_diff)
                    close_estimate = max(close_estimate, 20)
                    estimates.append(close_estimate)
                    weights.append(10.0)
                    imputation_stats["phylo_close"] += 1
                    
                elif median_diff < 100 and small_ratio >= 0.5:
                    close_estimate = min_diff + 0.2 * (mean_diff - min_diff)
                    close_estimate = max(close_estimate, 30)
                    estimates.append(close_estimate)
                    weights.append(5.0)
                    imputation_stats["phylo_close"] += 1
                    
                elif median_diff < 200 and small_ratio >= 0.4:
                    close_estimate = median_diff + 0.15 * (mean_diff - median_diff)
                    close_estimate = max(close_estimate, min_diff)
                    estimates.append(close_estimate)
                    weights.append(3.0)
                    imputation_stats["phylo_close"] += 1
            
            elif both_deep and max_diff < 50 and mean_profile > 800:
                if max_diff < 30:
                    profile_est = max(max_diff * 5, effective_lower, 30)
                    estimates.append(profile_est)
                    weights.append(8.0)
                    imputation_stats["deep_profile_close"] = imputation_stats.get("deep_profile_close", 0) + 1
                else:
                    parallel_estimate = s1_min_anchor + s2_min_anchor
                    
                    if s1 in sample_bias or s2 in sample_bias:
                        corr = (sample_bias.get(s1, 0) + sample_bias.get(s2, 0)) * 0.1
                        if corr > 0:
                            parallel_estimate = max(parallel_estimate - corr, effective_lower)
                            if should_debug:
                                 print(f"    Bias correction (Deep): -{corr:.1f}")

                    parallel_estimate = max(parallel_estimate * 0.3, 400)
                    estimates.append(parallel_estimate)
                    weights.append(5.0)
                    imputation_stats["deep_parallel"] = imputation_stats.get("deep_parallel", 0) + 1
        
        lineage_cons_lower = consistency_lower
        s1_in_group = find(s1) in lineage_groups


        s2_in_group = find(s2) in lineage_groups
        if s1_in_group or s2_in_group:
            for lg_root in [find(s1), find(s2)]:
                if lg_root in lineage_groups:
                    lg_members = lineage_groups[lg_root]
                    if len(lg_members) >= 2:
                        for member in lg_members:
                            if member == s1 or member == s2:
                                continue
                            mi = sidx[member]
                            d_s1_m = mat_arr[i1_r, mi]
                            d_s2_m = mat_arr[i2_r, mi]
                            if not np.isnan(d_s1_m) and not np.isnan(d_s2_m) and d_s1_m > 0 and d_s2_m > 0:
                                lower = abs(d_s1_m - d_s2_m)
                                if lower > lineage_cons_lower:
                                    lineage_cons_lower = lower
        
        
        very_close_detected = False
        lower_bounds = lower_bounds_early
        upper_bounds = upper_bounds_early
        
        if lower_bounds:
            max_lower = max_lower_early
            min_upper = min_upper_early
            
            d1_anchors_fp = orig_known_arr[i1_r, anchor_idxs]
            d2_anchors_fp = orig_known_arr[i2_r, anchor_idxs]
            
            resolved_fp_estimates = []
            unresolved_fp_estimates = []
            for ai in range(n_anchors):
                d_ac = d1_anchors_fp[ai]
                d_bc = d2_anchors_fp[ai]
                if np.isnan(d_ac) or d_ac <= 0 or np.isnan(d_bc) or d_bc <= 0:
                    continue
                for aj in range(ai + 1, n_anchors):
                    d_ad = d1_anchors_fp[aj]
                    d_bd = d2_anchors_fp[aj]
                    if np.isnan(d_ad) or d_ad <= 0 or np.isnan(d_bd) or d_bd <= 0:
                        continue
                    d_cd = anchor_anchor_dist[ai, aj]
                    if np.isnan(d_cd):
                        continue
                    
                    s2_val = d_ac + d_bd
                    s3_val = d_ad + d_bc
                    fp_est = max(s2_val, s3_val) - d_cd
                    
                    if fp_est <= 0:
                        continue

                    resolution = abs(s2_val - s3_val)
                    if resolution > 0.05 * d_cd and resolution > 20:
                        resolved_fp_estimates.append(fp_est)
                    else:
                        unresolved_fp_estimates.append(fp_est)
            
            all_valid_fp = resolved_fp_estimates + unresolved_fp_estimates
            if len(all_valid_fp) >= 3:

                if len(resolved_fp_estimates) >= 5:
                    fp_value = np.median(resolved_fp_estimates)
                elif len(resolved_fp_estimates) >= 2:
                    resolved_med = np.median(resolved_fp_estimates)
                    all_p25 = np.percentile(all_valid_fp, 25)
                    fp_value = 0.7 * resolved_med + 0.3 * all_p25
                else:
                    fp_value = np.percentile(all_valid_fp, 25)
                
                fp_value = max(fp_value, max_lower)
                fp_value = min(fp_value, min_upper)
                
                if s1 in sample_bias or s2 in sample_bias:
                    corr = (sample_bias.get(s1, 0) + sample_bias.get(s2, 0)) * 0.1
                    if corr > 0:
                        fp_value = max(fp_value - corr, max_lower)
                
                fp_weight = 10.0 + min(len(resolved_fp_estimates), 10)
                estimates.append(fp_value)
                weights.append(fp_weight)
                fp_gating_value = fp_value
                
                if should_debug:
                    print(f"    FOUR-POINT: {len(all_valid_fp)} valid quartets "
                          f"({len(resolved_fp_estimates)} resolved), value={fp_value:.0f}, "
                          f"range=[{min(all_valid_fp):.0f}, {max(all_valid_fp):.0f}], "
                          f"bounds=[{max_lower:.0f}, {min_upper:.0f}]")
                
                imputation_stats["four_point"] = imputation_stats.get("four_point", 0) + 1
            else:
                current_weight = weight_factors.get("medium", 0.487)
                if same_lineage_group or is_deep_same_lineage:
                    current_weight *= 0.3
                for anchor in present_anchors:
                    ai_fb = sidx[anchor]
                    d1 = orig_known_arr[i1_r, ai_fb]
                    d2 = orig_known_arr[i2_r, ai_fb]
                    if not np.isnan(d1) and not np.isnan(d2) and d1 > 0 and d2 > 0:
                        lower = abs(d1 - d2)
                        upper = d1 + d2
                        estimate = lower + current_weight * (upper - lower)
                        balance = 1.0 - abs(d1 - d2) / (d1 + d2)
                        weight = 0.5 + 0.5 * balance
                        estimates.append(estimate)
                        weights.append(weight)
                imputation_stats["anchor_triangulation"] += 1
            
            current_weight = weight_factors.get("close", 0.15) if max_lower < 300 else (
                weight_factors.get("medium", 0.487) if max_lower < 800 else
                weight_factors.get("far", 0.526))
        
        if mds_estimates and (s1, s2) in mds_estimates:
            mds_est = mds_estimates[(s1, s2)]
            if estimates:
                median_est = np.median(estimates)
                agreement = 1.0 / (1.0 + abs(mds_est - median_est) / max(median_est, 100))
                if 300 < mds_est < 1500:
                    mds_weight = 4.0 * agreement
                else:
                    mds_weight = 2.0 * agreement
            else:
                mds_weight = 3.0 if 300 < mds_est < 1500 else 1.5
            estimates.append(mds_est)
            weights.append(mds_weight)
            imputation_stats["mds_supplemental"] = imputation_stats.get("mds_supplemental", 0) + 1
        
        if len(estimates) < 3:
            neighbor_estimates = []
            for sk_idx in range(n_samp):
                if sk_idx == i1_r or sk_idx == i2_r:
                    continue

                d1k = orig_known_arr[i1_r, sk_idx]
                dk2 = orig_known_arr[sk_idx, i2_r]

                if (not np.isnan(d1k) and not np.isnan(dk2) and
                    known_arr[i1_r, sk_idx] and known_arr[sk_idx, i2_r] and
                    d1k > 0 and dk2 > 0):
                    
                    lower = abs(d1k - dk2)
                    upper = d1k + dk2
                    
                    proximity = 1.0 / (1.0 + (d1k + dk2) / 1000.0)
                    
                    cw = current_weight if 'current_weight' in dir() else 0.3
                    est = lower + cw * (upper - lower)
                    neighbor_estimates.append((est, proximity))
            
            if neighbor_estimates:
                total_weight = sum(p for _, p in neighbor_estimates)
                if total_weight > 0:
                    weighted_est = sum(e * p for e, p in neighbor_estimates) / total_weight
                    estimates.append(weighted_est)
                    weights.append(0.5)
                    imputation_stats["neighbor_based"] += 1
        
        effective_lower = 0
        d1_eff_anc = orig_known_arr[i1_r, anchor_idxs]
        d2_eff_anc = orig_known_arr[i2_r, anchor_idxs]
        valid_eff_anc = ~np.isnan(d1_eff_anc) & ~np.isnan(d2_eff_anc) & (d1_eff_anc > 0) & (d2_eff_anc > 0)
        if valid_eff_anc.any():
            effective_lower = float(np.max(np.abs(d1_eff_anc[valid_eff_anc] - d2_eff_anc[valid_eff_anc])))
        
        if very_close_detected and estimates:
            profile_estimate = estimates[0]
            imputed = max(profile_estimate, effective_lower)
            mat_arr[i1_r, i2_r] = imputed
            mat_arr[i2_r, i1_r] = imputed
            close_arr[i1_r, i2_r] = True
            close_arr[i2_r, i1_r] = True
            if should_debug:
                print(f"  DEBUG remaining_pairs: {s1} and {s2}")
                print(f"    VERY CLOSE DIRECT: profile_estimate={profile_estimate:.0f}, eff_lower={effective_lower:.0f}, final={imputed:.0f}")
            continue  
        
        if estimates:
            weights = np.array(weights)
            weights = weights / weights.sum()
            imputed = np.average(estimates, weights=weights)
            
            imputed = max(imputed, effective_lower)
            
            if should_debug:
                print(f"  DEBUG remaining_pairs: {s1} and {s2}")
                print(f"    s1_min_anchor={s1_min_anchor:.0f}, s2_min_anchor={s2_min_anchor:.0f}")
                print(f"    both_deep={both_deep}, imputed_before_floor={imputed:.0f}")
            
            is_very_close_candidate = (max_diff < 50 and consistency_lower < 100 and 
                                       effective_lower < 100 and cons_lower_with_imputed < 200)

            if fp_gating_value is not None and fp_gating_value > 400 and effective_lower > 100:
                is_very_close_candidate = False
            

            if (fp_gating_value is not None and fp_gating_value > 300 
                    and effective_lower > 80 and imputed < fp_gating_value * 0.4):
                imputed = 0.4 * imputed + 0.6 * fp_gating_value
                imputed = max(imputed, effective_lower)
                if should_debug:
                    print(f"    FOUR-POINT DOMINANCE: corrected to {imputed:.0f} (fp={fp_gating_value:.0f})")
            
            if is_very_close_candidate and imputed > 100:
                very_close_estimate = max(max_diff * 3 + 10, effective_lower)
                very_close_estimate = max(very_close_estimate, 30)
                if should_debug:
                    print(f"    VERY CLOSE OVERRIDE: imputed={imputed:.0f} -> {very_close_estimate:.0f} (max_diff={max_diff:.0f}, imputed_cons={cons_lower_with_imputed:.0f})")
                imputed = very_close_estimate
                imputation_stats["phylo_close"] += 1
            
            if both_deep and max_diff >= 15 and not is_very_close_candidate:
                sum_anchor_min = s1_min_anchor + s2_min_anchor
                if s1 in sample_bias or s2 in sample_bias:
                     corr = (sample_bias.get(s1, 0) + sample_bias.get(s2, 0)) * 0.1
                     if corr > 0:
                         sum_anchor_min = max(sum_anchor_min - corr, 0)
                
                parallel_floor = max(sum_anchor_min, 800) * 0.4
                parallel_floor = max(parallel_floor, 400)
                if should_debug:
                    print(f"    parallel_floor={parallel_floor:.0f}")
                if imputed < parallel_floor:
                    imputed = parallel_floor
                    if should_debug:
                        print(f"    Applied floor: imputed={imputed:.0f}")
            
            if same_lineage_group:
                lineage_root = find(s1)
                known_lineage_distances = []
                for (root, sa, sb), known_dist in lineage_cross_lab_distances.items():
                    if root == lineage_root:
                        known_lineage_distances.append(known_dist)
                
                if known_lineage_distances:
                    max_known = max(known_lineage_distances)
                    if same_lineage_group and is_deep_same_lineage and len(known_lineage_distances) >= 2:
                        min_known = min(known_lineage_distances)
                        if max_known < min_known * 2 and imputed > max_known * 4:
                            capped_val = min(imputed, max_known * 2.5)
                            imputed = max(capped_val, effective_lower)
                            if should_debug:
                                print(f"    CAPPED estimate to {imputed:.0f} based on lineage max {max_known}")
            
            mat_arr[i1_r, i2_r] = imputed
            mat_arr[i2_r, i1_r] = imputed

            if lower_bounds:
                anchor_bound_spread = min_upper - max_lower
                if anchor_bound_spread < 150:
                    high_confidence_arr[i1_r, i2_r] = True
                    high_confidence_arr[i2_r, i1_r] = True

            known_suggests_close = consistency_lower < 50
            imputed_also_suggests_close = consistency_lower < 100 and cons_lower_with_imputed < 200
            is_truly_close = (max_diff < 50 and imputed < 150 and effective_lower < 50 and 
                             (known_suggests_close or imputed_also_suggests_close))
            if is_truly_close:
                close_arr[i1_r, i2_r] = True
                close_arr[i2_r, i1_r] = True
                if should_debug:
                    print(f"    PROTECTED as close (max_diff={max_diff:.0f}, imputed={imputed:.0f}, eff_lower={effective_lower:.0f}, known_cons={consistency_lower:.0f}, imputed_cons={cons_lower_with_imputed:.0f})")
            elif should_debug and max_diff < 20:
                print(f"    NOT protected (max_diff={max_diff:.0f}, imputed={imputed:.0f}, eff_lower={effective_lower:.0f}, known_cons={consistency_lower:.0f}, imputed_cons={cons_lower_with_imputed:.0f})")
        else:
            imputation_stats["fallback"] += 1
            known_vals = original_known.values[known.values & (original_known.values > 0)]
            fallback_val = np.median(known_vals) if len(known_vals) > 0 else 1500
            mat_arr[i1_r, i2_r] = fallback_val
            mat_arr[i2_r, i1_r] = fallback_val
    
    print(f"  Imputation stats: {imputation_stats}")

    restore_mask = known_mask.values
    mat_arr[restore_mask] = orig_known_arr[restore_mask]

    n_high_conf = high_confidence_arr[np.triu(np.ones((n_samp, n_samp), dtype=bool), k=1)].sum()
    n_total_imputed = (~orig_kmask_arr & np.triu(np.ones((n_samp, n_samp), dtype=bool), k=1)).sum()
    print(f"  High-confidence imputed pairs: {n_high_conf} / {n_total_imputed}")

    return matrix, close_detected_mask, high_confidence_arr


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
    
    imputed_mask = ~known_mask.values
    imputed_vals = completed_matrix.values[imputed_mask]
    imputed_vals = imputed_vals[~np.isnan(imputed_vals)]
    unique_imputed = np.unique(np.round(imputed_vals, 2))
    
    print(f"  Number of unique imputed values: {len(unique_imputed)}")
    
    known_vals = original_matrix.values[known_mask.values & ~np.isnan(original_matrix.values)]
    known_vals = known_vals[known_vals > 0]
    imputed_vals_clean = imputed_vals[imputed_vals > 0]
    
    if len(known_vals) > 0 and len(imputed_vals_clean) > 0:
        print(f"  Known distances: mean={np.mean(known_vals):.1f}, std={np.std(known_vals):.1f}, "
              f"range=[{np.min(known_vals):.0f}, {np.max(known_vals):.0f}]")
        print(f"  Imputed distances: mean={np.mean(imputed_vals_clean):.1f}, std={np.std(imputed_vals_clean):.1f}, "
              f"range=[{np.min(imputed_vals_clean):.0f}, {np.max(imputed_vals_clean):.0f}]")
        
        known_mean = np.mean(known_vals)
        imputed_mean = np.mean(imputed_vals_clean)
        bias = (imputed_mean - known_mean) / known_mean * 100
        print(f"  Mean bias: {bias:+.1f}% (positive = overestimating)")
        
        known_std = np.std(known_vals)
        imputed_std = np.std(imputed_vals_clean)
        
        if imputed_std < known_std * 0.5:
            print("Imputed values have much lower variance")
        else:
            print("Imputed variance looks normal")


def ridge_post_correction(global_matrix, incomplete_matrix, known_mask, sample_to_lab, close_detected_mask):
    print("\n  Ridge regression post-correction")
    
    samples = list(global_matrix.index)
    n = len(samples)
    sidx = {s: i for i, s in enumerate(samples)}
    km = known_mask.values
    mat = global_matrix.values.copy()
    orig = incomplete_matrix.values
    
    n_labs = max(lab for labs in sample_to_lab.values() for lab in labs) + 1
    effective_anchors = sorted([s for s in samples if len(sample_to_lab.get(s, [])) == n_labs])
    n_anchors = len(effective_anchors)
    anchor_idxs = [sidx[a] for a in effective_anchors]
    
    print(f"    Using {n_anchors} anchor features")
    if n_anchors < 3:
        print("    Too few anchors for Ridge regression, skipping")
        return global_matrix
    
    anchor_dists = np.zeros((n, n_anchors))
    for j, ai in enumerate(anchor_idxs):
        anchor_dists[:, j] = orig[:, ai]
    
    train_features = []
    train_targets = []
    mask_upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    known_pairs = mask_upper & km & ~np.isnan(orig)
    
    pair_idxs = np.argwhere(known_pairs)
    if len(pair_idxs) > 50000:
        rng = np.random.RandomState(42)
        chosen = rng.choice(len(pair_idxs), 50000, replace=False)
        pair_idxs = pair_idxs[chosen]
    
    for idx in pair_idxs:
        i, j = idx
        if np.any(np.isnan(anchor_dists[i])) or np.any(np.isnan(anchor_dists[j])):
            continue
        diffs = np.abs(anchor_dists[i] - anchor_dists[j])
        sums = anchor_dists[i] + anchor_dists[j]
        feat = np.concatenate([
            np.sort(diffs),
            [np.min(diffs), np.max(diffs), np.mean(diffs), np.std(diffs), np.median(diffs)],
            [np.min(sums), np.max(sums), np.mean(sums)],
        ])
        train_features.append(feat)
        train_targets.append(orig[i, j])
    
    X_train = np.array(train_features)
    y_train = np.array(train_targets)
    print(f"    Training on {len(X_train)} within-lab pairs")
    
    if len(X_train) < 100:
        print("    Too few training pairs, skipping Ridge correction")
        return global_matrix
    
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std == 0] = 1
    X_norm = (X_train - X_mean) / X_std
    
    alpha = 10.0
    XtX = X_norm.T @ X_norm + alpha * np.eye(X_norm.shape[1])
    Xty = X_norm.T @ y_train
    w = np.linalg.solve(XtX, Xty)
    
    y_pred_train = X_norm @ w
    train_mae = np.mean(np.abs(y_train - y_pred_train))
    train_r2 = 1 - np.sum((y_train - y_pred_train)**2) / np.sum((y_train - np.mean(y_train))**2)
    print(f"    Train MAE={train_mae:.1f}, R2={train_r2:.4f}")
    
    cross_lab_mask = mask_upper & ~km
    cross_pairs = np.argwhere(cross_lab_mask)
    
    corrections = 0
    for idx in cross_pairs:
        i, j = idx
        if np.any(np.isnan(anchor_dists[i])) or np.any(np.isnan(anchor_dists[j])):
            continue
        
        diffs = np.abs(anchor_dists[i] - anchor_dists[j])
        sums = anchor_dists[i] + anchor_dists[j]
        feat = np.concatenate([
            np.sort(diffs),
            [np.min(diffs), np.max(diffs), np.mean(diffs), np.std(diffs), np.median(diffs)],
            [np.min(sums), np.max(sums), np.mean(sums)],
        ])
        feat_norm = (feat - X_mean) / X_std
        ridge_pred = float(feat_norm @ w)
        ridge_pred = max(ridge_pred, 0)
        
        current = mat[i, j]
        
        max_diff = np.max(diffs)
        if close_detected_mask is not None and close_detected_mask.values[i, j]:
            blend_weight = 0.1
        elif max_diff < 100:
            blend_weight = 0.4
        elif 100 <= max_diff < 400:
            blend_weight = 0.25
        else:
            blend_weight = 0.0
        
        blended = (1 - blend_weight) * current + blend_weight * ridge_pred
        blended = max(blended, np.max(diffs))
        
        mat[i, j] = blended
        mat[j, i] = blended
        corrections += 1
    
    print(f"    Corrected {corrections} cross-lab pairs")
    
    result = pd.DataFrame(mat, index=global_matrix.index, columns=global_matrix.columns)
    return result


def impute_cross_site_distances(matrices, lab_samples, anchors, primary_anchor, method='auto'):
    
    incomplete_matrix, sample_to_lab, known_mask = create_incomplete_matrix(
        matrices, lab_samples, anchors
    )
    
    can_tri, cannot_tri = diagnose_imputation_coverage(incomplete_matrix, known_mask, anchors, sample_to_lab)
    
    n_total = len(incomplete_matrix)
    n_missing = np.isnan(incomplete_matrix.values).sum()
    n_known = n_total * n_total - n_missing
    pct_known = 100 * n_known / (n_total * n_total)
    
    print(f"Matrix size: {n_total} x {n_total}")
    print(f"Known entries: {n_known} ({pct_known:.1f}%)")
    print(f"Missing entries: {n_missing} ({100-pct_known:.1f}%)")
    
    if method == 'auto':
        if can_tri > 0:
            method = 'anchor_guided'
        elif pct_known > 70:
            method = 'knn'
        elif FANCYIMPUTE_AVAILABLE:
            method = 'softimpute'
        else:
            method = 'anchor_guided'
        print(f"Auto-selected method: {method}")
    
    close_detected_mask = None
    
    mds_estimates = {}
    if method == 'anchor_guided':
        try:
            mds_estimates = mds_cross_lab_estimation(
                incomplete_matrix, known_mask, sample_to_lab
            )
        except Exception as e:
            print(f"  MDS estimation failed: {e}, continuing without it")
    
    high_confidence_arr = None
    if method == 'softimpute' and FANCYIMPUTE_AVAILABLE:
        global_matrix = impute_softimpute_improved(incomplete_matrix, known_mask)
    elif method == 'knn' and SKLEARN_KNN_AVAILABLE:
        global_matrix = impute_knn_improved(incomplete_matrix, known_mask)
    elif method == 'anchor_guided':
        global_matrix, close_detected_mask, high_confidence_arr = impute_anchor_guided(
            incomplete_matrix, known_mask, anchors, sample_to_lab, mds_estimates=mds_estimates
        )
    else:
        global_matrix, close_detected_mask, high_confidence_arr = impute_anchor_guided(
            incomplete_matrix, known_mask, anchors, sample_to_lab, mds_estimates=mds_estimates
        )
    
    print("Enforcing metric properties")
    completed_values = enforce_metric_properties_balanced(
        global_matrix.values.copy(), 
        known_mask,
        close_detected_mask,
        max_iterations=3
    )
    
    global_matrix = pd.DataFrame(
        completed_values,
        index=global_matrix.index,
        columns=global_matrix.columns
    )
    
    validate_and_report(incomplete_matrix, global_matrix, known_mask, anchors)
    
    return global_matrix, method, close_detected_mask, high_confidence_arr, mds_estimates


def merge_matrices(matrix_files, anchors, primary_anchor, imputation_method='auto'):
    matrices = [load_distance_matrix(f) for f in matrix_files]
    
    print(f"Loaded {len(matrices)} matrices")
    for i, df in enumerate(matrices):
        print(f"  Matrix {i}: {len(df)} samples")
        print(f"    Distance range: {df.values[df.values > 0].min():.0f} - {df.values.max():.0f}")
        present = [a for a in anchors if a in df.index]
        print(f"    Anchors present: {len(present)}/{len(anchors)} - {present}")
    
    correction_factors = calculate_correction_factor(matrices, anchors, primary_anchor)
    print(f"Correction factors: {correction_factors}")
    
    corrected_matrices = [
        apply_correction(df, factor) 
        for df, factor in zip(matrices, correction_factors)
    ]
    
    lab_samples = [set(df.index) for df in corrected_matrices]
    
    global_matrix, method_used, close_detected_mask, high_confidence_arr, mds_estimates = impute_cross_site_distances(
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
    
    return global_matrix, report, lab_samples, close_detected_mask, high_confidence_arr, mds_estimates


def als_matrix_completion(M, known_mask, rank=15, n_iter=50, reg=0.01):

    n = M.shape[0]
    km = known_mask.astype(bool)
    rank = min(rank, n - 1)
    
    U_init, S_init, Vt_init = np.linalg.svd(M, full_matrices=False)
    U = U_init[:, :rank] * np.sqrt(S_init[:rank])
    V = Vt_init[:rank, :].T * np.sqrt(S_init[:rank])
    
    reg_eye = reg * np.eye(rank)
    prev_rmse = float('inf')
    
    for it in range(n_iter):
        for i in range(n):
            j_idx = np.where(km[i])[0]
            if len(j_idx) < rank:
                continue
            Vj = V[j_idx]
            y = M[i, j_idx]
            U[i] = np.linalg.solve(Vj.T @ Vj + reg_eye, Vj.T @ y)
        
        for j in range(n):
            i_idx = np.where(km[:, j])[0]
            if len(i_idx) < rank:
                continue
            Ui = U[i_idx]
            y = M[i_idx, j]
            V[j] = np.linalg.solve(Ui.T @ Ui + reg_eye, Ui.T @ y)
        
        pred = U @ V.T
        rmse = np.sqrt(np.mean((pred[km] - M[km]) ** 2))
        
        if it % 10 == 0 or it == n_iter - 1:
            print(f"    ALS iter {it}: RMSE = {rmse:.2f}")
        
        if abs(prev_rmse - rmse) < 0.01:
            print(f"    ALS converged at iteration {it}")
            break
        prev_rmse = rmse
    
    completed = U @ V.T
    completed = (completed + completed.T) / 2.0
    np.fill_diagonal(completed, 0)
    completed = np.maximum(completed, 0)
    
    completed[km] = M[km]
    
    return completed


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
    
    global_matrix, report, lab_samples, close_detected_mask, high_confidence_arr, mds_estimates = merge_matrices(
        args.matrices, anchors, args.primary_anchor, args.method
    )
    
    print("\n" + "=" * 60)
    print("PCA SVD Denoising")
    print("=" * 60)
    print("Loading original matrices")
    known_mask = pd.DataFrame(False, index=global_matrix.index, columns=global_matrix.columns)
    
    for mat_file in args.matrices:
        abs_path = os.path.abspath(mat_file)
        print(f"  Processing {mat_file} (Absolute: {abs_path})")
        if not os.path.exists(mat_file):
             print(f"  ERROR: File does not exist: {mat_file}")
             continue
             
        try:
            lab_mat = pd.read_csv(mat_file, sep='\t', index_col=0)
            common_idx = global_matrix.index.intersection(lab_mat.index)
            common_col = global_matrix.columns.intersection(lab_mat.columns)
            
            lab_aligned = lab_mat.reindex(index=global_matrix.index, columns=global_matrix.columns)
            known_in_lab = lab_aligned.notnull()
            
            known_in_lab = known_in_lab.fillna(False)
            
            known_mask = known_mask | known_in_lab
            
            print(f"    Loaded {lab_mat.shape} : Aligned {known_in_lab.sum().sum()} knowns")
            
        except Exception as e:
            print(f"  Could not load {mat_file} for mask: {e}")
            import traceback
            traceback.print_exc()

    known_count = known_mask.sum().sum()
    print(f"  Preserving {known_count} known values")

    print("\n  Pre-SVD weighted median correction")
    M_pre = global_matrix.values.copy()
    km_pre = known_mask.values
    n_pre = M_pre.shape[0]
    
    upper_tri_pre = np.triu(np.ones((n_pre, n_pre), dtype=bool), k=1)
    refinable_pre = upper_tri_pre & ~km_pre
    
    cd_pre = close_detected_mask.values if close_detected_mask is not None else np.zeros((n_pre, n_pre), dtype=bool)
    hc_pre = high_confidence_arr if high_confidence_arr is not None else np.zeros((n_pre, n_pre), dtype=bool)
    
    samples_pre = list(global_matrix.index)
    anchor_list_pre = [a.strip() for a in args.anchors.split(',')]
    anchor_idx_pre = np.array([samples_pre.index(a) for a in anchor_list_pre if a in samples_pre])
    anchor_profile_pre = M_pre[:, anchor_idx_pre]  # (n_pre, n_anchors)
    n_anchors_pre = len(anchor_idx_pre)
    
    M_pre_before_wm = M_pre.copy()
    
    wm_pre_changes = 0
    wm_pre_total = 0.0
    wm_pre_capped = 0
    
    for i in range(n_pre):
        if not refinable_pre[i].any():
            continue
        
        d_i = M_pre[i, :]
        km_i = km_pre[i, :]
        
        j_indices = np.where(refinable_pre[i])[0]
        
        for j in j_indices:
            current = M_pre[i, j]
            
            if current >= 500:
                continue
            
            d_j = M_pre[j, :]
            km_j = km_pre[j, :]
            
            valid = np.ones(n_pre, dtype=bool)
            valid[i] = False
            valid[j] = False
            valid &= (d_i > 0) & (d_j > 0) & ~np.isnan(d_i) & ~np.isnan(d_j)
            
            semi_known = valid & (km_i | km_j)
            
            if semi_known.sum() < 30:
                continue
            
            dik = d_i[semi_known]
            dkj = d_j[semi_known]
            gap_sk = 2.0 * np.minimum(dik, dkj)
            midpoint_sk = np.maximum(dik, dkj)
            
            weights = 1.0 / np.maximum(gap_sk, 1.0)
            
            sorted_idx = np.argsort(midpoint_sk)
            sorted_mid = midpoint_sk[sorted_idx]
            sorted_w = weights[sorted_idx]
            cum_w = np.cumsum(sorted_w)
            half_w = cum_w[-1] / 2.0
            wm_idx = np.searchsorted(cum_w, half_w)
            wm_idx = min(wm_idx, len(sorted_mid) - 1)
            weighted_median = sorted_mid[wm_idx]
            
            if cd_pre[i, j]:
                self_anc_mask = np.array([a_idx != i and a_idx != j for a_idx in anchor_idx_pre])
                if self_anc_mask.sum() < 3:
                    continue
                prof_i_cd = anchor_profile_pre[i][self_anc_mask]
                prof_j_cd = anchor_profile_pre[j][self_anc_mask]
                pv_cd = ~np.isnan(prof_i_cd) & ~np.isnan(prof_j_cd) & (prof_i_cd > 0) & (prof_j_cd > 0)
                if pv_cd.sum() >= 3:
                    max_pd_cd = float(np.max(np.abs(prof_i_cd[pv_cd] - prof_j_cd[pv_cd])))
                    if max_pd_cd < 8:
                        continue  
                    elif max_pd_cd < 25 and weighted_median > current * 2:
                        continue  
            
            if current < 80 and weighted_median > current * 3 and weighted_median > 200:
                continue
            
            self_anc_mask_pb = np.array([a_idx != i and a_idx != j for a_idx in anchor_idx_pre])
            prof_i_wm = anchor_profile_pre[i][self_anc_mask_pb] if self_anc_mask_pb.sum() >= 3 else anchor_profile_pre[i]
            prof_j_wm = anchor_profile_pre[j][self_anc_mask_pb] if self_anc_mask_pb.sum() >= 3 else anchor_profile_pre[j]
            pv_wm = ~np.isnan(prof_i_wm) & ~np.isnan(prof_j_wm) & (prof_i_wm > 0) & (prof_j_wm > 0)
            if pv_wm.sum() >= 3:
                max_pd_wm = float(np.max(np.abs(prof_i_wm[pv_wm] - prof_j_wm[pv_wm])))
                if max_pd_wm < 80 and current < 400 and weighted_median > max(current * 2, 300):
                    continue
            
            anchor_upper_i = anchor_profile_pre[i]
            anchor_upper_j = anchor_profile_pre[j]
            pv_upper = ~np.isnan(anchor_upper_i) & ~np.isnan(anchor_upper_j) & (anchor_upper_i > 0) & (anchor_upper_j > 0)
            if pv_upper.sum() >= 2:
                hard_upper_wm = float(np.min(anchor_upper_i[pv_upper] + anchor_upper_j[pv_upper]))
                weighted_median = min(weighted_median, hard_upper_wm * 0.85)
            
            new_val = max(weighted_median, 0)
            
            if current < 150 and new_val > current * 3:
                self_anc_mask_cap = np.array([a_idx != i and a_idx != j for a_idx in anchor_idx_pre])
                prof_i = anchor_profile_pre[i][self_anc_mask_cap]
                prof_j = anchor_profile_pre[j][self_anc_mask_cap]
                pv = ~np.isnan(prof_i) & ~np.isnan(prof_j) & (prof_i > 0) & (prof_j > 0)
                if pv.sum() >= 2:
                    max_pd = float(np.max(np.abs(prof_i[pv] - prof_j[pv])))
                    if max_pd < 20:
                        cap = max(current * 3, 400)
                        new_val = min(new_val, cap)
                        wm_pre_capped += 1
            
            change = abs(new_val - current)
            if change > 1.0:
                M_pre[i, j] = new_val
                M_pre[j, i] = new_val
                wm_pre_changes += 1
                wm_pre_total += change
    
    with np.errstate(invalid='ignore'):
        prof_diffs_3d = np.abs(anchor_profile_pre[:, np.newaxis, :] - anchor_profile_pre[np.newaxis, :, :])
        n_close_anchors = np.nansum(prof_diffs_3d < 10, axis=2)
        max_prof_diff_2d = np.nanmax(prof_diffs_3d, axis=2)
    
    very_close_mult = 6.0
    n_corrected = 0
    
    for i in range(n_pre):
        for j in range(i + 1, n_pre):
            if km_pre[i, j]:
                continue
            prewm = M_pre_before_wm[i, j]
            postwm = M_pre[i, j]
            if prewm <= 0 or np.isnan(prewm) or postwm <= 0:
                continue
            
            max_pd = max_prof_diff_2d[i, j]
            n_close = n_close_anchors[i, j]
            inflation_ratio = postwm / max(prewm, 1)
            
            if n_close >= min(7, n_anchors_pre) and inflation_ratio > 4 and prewm < 150 and max_pd < 15:
                profile_est = max(max_pd * very_close_mult, 30)
                M_pre[i, j] = profile_est
                M_pre[j, i] = profile_est
                n_corrected += 1
    n_restored = 0
    
    M_pre[km_pre] = global_matrix.values[km_pre]
    M_pre = (M_pre + M_pre.T) / 2.0
    np.fill_diagonal(M_pre, 0)
    M_pre = np.maximum(M_pre, 0)
    
    avg_ch = wm_pre_total / max(wm_pre_changes, 1)
    print(f"    Pre-SVD WM: {wm_pre_changes} pairs corrected, {wm_pre_capped} capped, avg_change={avg_ch:.1f}")
    print(f"    Post-WM profile correction: {n_corrected} extreme close pairs corrected")
    
    n_rescued = 0
    pre_svd_rescued_mask = np.zeros((n_pre, n_pre), dtype=bool)
    for i in range(n_pre):
        for j in range(i + 1, n_pre):
            if km_pre[i, j]:
                continue
            pred_ij = M_pre[i, j]
            if pred_ij < 200:
                continue
            
            prof_i_r = anchor_profile_pre[i]
            prof_j_r = anchor_profile_pre[j]
            pv_r = ~np.isnan(prof_i_r) & ~np.isnan(prof_j_r) & (prof_i_r > 0) & (prof_j_r > 0)
            if pv_r.sum() < 3:
                continue
            diffs_r = np.abs(prof_i_r[pv_r] - prof_j_r[pv_r])
            max_anchor_diff = float(np.max(diffs_r))
            median_anchor_diff = float(np.median(diffs_r))
            
            if max_anchor_diff < 30 and pred_ij > max_anchor_diff * 5 and pred_ij > 200:
                if cd_pre[i, j]:
                    s1_r = samples_pre[i]
                    s2_r = samples_pre[j]
                    mds_r = mds_estimates.get((s1_r, s2_r), None) or mds_estimates.get((s2_r, s1_r), None)
                    if mds_r is not None and mds_r > 350:
                        continue
                
                profile_est = max(max_anchor_diff * 4, median_anchor_diff * 6, 30)
                profile_est = max(profile_est, float(np.max(diffs_r))) 
                if profile_est < pred_ij * 0.5:
                    M_pre[i, j] = profile_est
                    M_pre[j, i] = profile_est
                    pre_svd_rescued_mask[i, j] = True
                    pre_svd_rescued_mask[j, i] = True
                    n_rescued += 1
    
    print(f"    Close-pair rescue: {n_rescued} pairs corrected")
    
    M_pre[km_pre] = global_matrix.values[km_pre]
    M_pre = (M_pre + M_pre.T) / 2.0
    np.fill_diagonal(M_pre, 0)
    M_pre = np.maximum(M_pre, 0)
    
    global_matrix = pd.DataFrame(M_pre, index=global_matrix.index, columns=global_matrix.columns)

    pre_svd_path = args.output.replace('.tsv', '_pre_svd.tsv')
    global_matrix.to_csv(pre_svd_path, sep='\t')
    print(f"  Pre-SVD matrix saved to {pre_svd_path}")

    if global_matrix.isnull().values.any():
        global_matrix = global_matrix.fillna(global_matrix.mean().mean())

    n_iterations = 5
    min_dim = min(global_matrix.shape)
    rank = min(10, int(min_dim * 0.5))
    rank = max(2, rank)
    print(f"  Truncating to rank={rank}, {n_iterations} iterations")
    
    km = known_mask.values
    
    medium_mask = (global_matrix.values >= 400) & (global_matrix.values <= 1000)
    close_det = close_detected_mask.values if close_detected_mask is not None else np.zeros_like(km, dtype=bool)
    protect_mask = km | medium_mask | close_det | pre_svd_rescued_mask
    
    M_original = global_matrix.values.copy()
    M_work = global_matrix.values.copy()
    
    for iteration in range(n_iterations):
        U, S, Vt = np.linalg.svd(M_work, full_matrices=False)
        S_trunc = np.zeros_like(S)
        S_trunc[:rank] = S[:rank]
        M_reconst = np.dot(U * S_trunc, Vt)
        
        M_reconst = (M_reconst + M_reconst.T) / 2.0
        np.fill_diagonal(M_reconst, 0)
        M_reconst = np.maximum(M_reconst, 0)
        
        M_reconst[protect_mask] = M_work[protect_mask]
        
        diff = np.abs(M_reconst[~protect_mask] - M_work[~protect_mask])
        max_change = np.max(diff) if diff.size > 0 else 0
        mean_change = np.mean(diff) if diff.size > 0 else 0
        print(f"    Iteration {iteration+1}: max_change={max_change:.2f}, mean_change={mean_change:.2f}")
        
        M_work = M_reconst
        
        if max_change < 1.0:
            print(f"    Converged at iteration {iteration+1}")
            break
    
    M_final = pd.DataFrame(M_work, index=global_matrix.index, columns=global_matrix.columns)
    
    print("  Final reset of known values")
    M_final = M_final.mask(known_mask, global_matrix)
    
    M_final = (M_final + M_final.T) / 2.0
    np.fill_diagonal(M_final.values, 0)
    M_final[M_final < 0] = 0
    
    global_matrix = M_final
    print("  SVD denoising complete.\n")

    print("=" * 60)
    print("Post-Imputation Refinement (confidence-weighted)")
    print("=" * 60)
    
    M = global_matrix.values.copy()
    n_ref = M.shape[0]
    km_ref = known_mask.values
    close_det_ref = close_detected_mask.values if close_detected_mask is not None else np.zeros_like(km_ref, dtype=bool)
    hc_ref = high_confidence_arr if high_confidence_arr is not None else np.zeros_like(km_ref, dtype=bool)
    protected_ref = km_ref  
    
    cd_ceiling = 100
    cd_only_ref = close_det_ref & ~km_ref
    
    quality_ref = np.zeros((n_ref, n_ref), dtype=int)
    quality_ref[hc_ref] = 1
    quality_ref[km_ref] = 2
    
    upper_tri_ref = np.triu(np.ones((n_ref, n_ref), dtype=bool), k=1)
    refinable = upper_tri_ref & ~protected_ref
    
    n_refinable = refinable.sum()
    n_cd_refinable = (refinable & cd_only_ref).sum()
    print(f"  Refinable pairs: {n_refinable} (incl {n_cd_refinable} CD hard-lower-only)")
    n_known_inter = km_ref[upper_tri_ref].sum()
    n_hc_inter = (hc_ref & ~km_ref)[upper_tri_ref].sum()
    print(f"  Quality tiers: known={n_known_inter}, high_conf={n_hc_inter}, low_conf={n_refinable - n_hc_inter}")
    
    n_refine_iters = 15
    for ref_iter in range(n_refine_iters):
        changes = 0
        total_change = 0.0
        
        for i in range(n_ref):
            if not refinable[i].any():
                continue
            
            d_i = M[i, :]
            km_i = km_ref[i, :]
            
            j_indices = np.where(refinable[i])[0]
            if len(j_indices) == 0:
                continue
            
            for j in j_indices:
                d_j = M[j, :]
                km_j = km_ref[j, :]
                
                valid = np.ones(n_ref, dtype=bool)
                valid[i] = False
                valid[j] = False
                valid &= (d_i > 0) & (d_j > 0) & ~np.isnan(d_i) & ~np.isnan(d_j)
                
                if valid.sum() < 10:
                    continue
                
                dik = d_i[valid]
                dkj = d_j[valid]
                lower_all = np.abs(dik - dkj)
                upper_all = dik + dkj
                
                both_known = valid & km_i & km_j
                
                if both_known.sum() >= 2:
                    hard_lower = np.max(np.abs(d_i[both_known] - d_j[both_known]))
                    hard_upper = np.min(d_i[both_known] + d_j[both_known])
                else:
                    hard_lower = 0
                    hard_upper = float('inf')
                
                semi_known = valid & (km_i | km_j)
                sk_mask = semi_known[valid]
                
                if sk_mask.sum() >= 30:
                    soft_lower = np.percentile(lower_all[sk_mask], 95)
                    soft_upper = np.percentile(upper_all[sk_mask], 5)
                else:
                    soft_lower = np.percentile(lower_all, 95)
                    soft_upper = np.percentile(upper_all, 5)
                
                current = M[i, j]
                
                max_lower = max(hard_lower, soft_lower)
                min_upper = min(hard_upper, soft_upper) if hard_upper < float('inf') else soft_upper
                
                if max_lower >= min_upper:
                    new_val = max_lower
                elif current < max_lower:
                    new_val = max_lower
                elif current > min_upper:
                    new_val = min_upper
                else:
                    tight_range = min_upper - max_lower
                    midpoint = (max_lower + min_upper) / 2.0
                    relative_range = tight_range / max(midpoint, 1.0)
                    
                    if relative_range < 2.0:
                        alpha = min(0.4, 0.15 / max(relative_range, 0.01))
                        new_val = (1 - alpha) * current + alpha * midpoint
                    else:
                        continue
                
                new_val = max(new_val, 0)
                
                if cd_only_ref[i, j] and new_val > M[i, j]:
                    new_val = min(new_val, cd_ceiling)
                
                change = abs(new_val - current)
                if change > 0.1:
                    M[i, j] = new_val
                    M[j, i] = new_val
                    changes += 1
                    total_change += change
        
        avg_change = total_change / max(changes, 1)
        print(f"  Refinement iter {ref_iter+1}: {changes} pairs adjusted, avg_change={avg_change:.1f}")
        
        if changes == 0:
            break
    
    M[km_ref] = global_matrix.values[km_ref]
    M = (M + M.T) / 2.0
    np.fill_diagonal(M, 0)
    M = np.maximum(M, 0)
    
    global_matrix = pd.DataFrame(M, index=global_matrix.index, columns=global_matrix.columns)
    print("  Refinement complete.\n")

    print("=" * 60)
    print("Post-Refinement Close-Pair Verification")
    print("=" * 60)
    
    M_verify = global_matrix.values.copy()
    n_v = M_verify.shape[0]
    km_v = known_mask.values
    cd_v = close_detected_mask.values if close_detected_mask is not None else np.zeros((n_v, n_v), dtype=bool)
    upper_tri_v = np.triu(np.ones((n_v, n_v), dtype=bool), k=1)
    
    candidate_mask = upper_tri_v & cd_v & (M_verify < 100) & ~km_v
    candidate_i, candidate_j = np.where(candidate_mask)
    print(f"  Close-detected pairs with pred < 100: {len(candidate_i)}")
    
    samples_v = list(global_matrix.index)
    sample_lab_map_v = {}
    for lab_idx_v, samples_set_v in enumerate(lab_samples):
        for s in samples_set_v:
            if s in samples_v:
                sample_lab_map_v[s] = lab_idx_v
    lab_sample_indices_v = {}
    for idx_s, s in enumerate(samples_v):
        lab = sample_lab_map_v.get(s, -1)
        if lab not in lab_sample_indices_v:
            lab_sample_indices_v[lab] = []
        lab_sample_indices_v[lab].append(idx_s)
    
    anchor_list_v = [a.strip() for a in args.anchors.split(',')]
    anchor_idx_v = np.array([samples_v.index(a) for a in anchor_list_v if a in samples_v])
    anchor_profile_v = M_verify[:, anchor_idx_v]
    
    n_verified_close = 0
    n_corrected_up = 0
    n_corrected_nbr = 0
    n_corrected_p75 = 0
    corrections_detail = []
    corrected_samples = {}
    
    for idx in range(len(candidate_i)):
        i, j = candidate_i[idx], candidate_j[idx]
        current_pred = M_verify[i, j]
        
        lab_i_v = sample_lab_map_v.get(samples_v[i], -1)
        lab_j_v = sample_lab_map_v.get(samples_v[j], -1)
        j_lab_idx = lab_sample_indices_v.get(lab_j_v, [])
        close_j_nbrs = [k for k in j_lab_idx if k != j and km_v[j, k] and M_verify[j, k] < 100]
        uncontam_j = [k for k in close_j_nbrs if not cd_v[i, k]]
        i_lab_idx = lab_sample_indices_v.get(lab_i_v, [])
        close_i_nbrs = [k for k in i_lab_idx if k != i and km_v[i, k] and M_verify[i, k] < 100]
        uncontam_i = [k for k in close_i_nbrs if not cd_v[j, k]]
        nbr_dists = [M_verify[i, k] for k in uncontam_j] + [M_verify[j, k] for k in uncontam_i]
        
        nbr_corrected = False
        if len(nbr_dists) >= 3:
            nbr_arr = np.array(nbr_dists)
            median_nbr = float(np.median(nbr_arr))
            frac_far_nbr = float(np.mean(nbr_arr > 400))
            if median_nbr > 400 and frac_far_nbr > 0.5:
                new_est = median_nbr
                M_verify[i, j] = new_est
                M_verify[j, i] = new_est
                cd_v[i, j] = False
                cd_v[j, i] = False
                n_corrected_up += 1
                n_corrected_nbr += 1
                corrected_samples[i] = corrected_samples.get(i, 0) + 1
                corrected_samples[j] = corrected_samples.get(j, 0) + 1
                corrections_detail.append(('NBR', median_nbr, current_pred, new_est, len(nbr_dists)))
                nbr_corrected = True
        if nbr_corrected:
            continue
        
        prof_i_v = anchor_profile_v[i]
        prof_j_v = anchor_profile_v[j]
        pv_mask = ~np.isnan(prof_i_v) & ~np.isnan(prof_j_v) & (prof_i_v > 0) & (prof_j_v > 0)
        if pv_mask.sum() >= 3:
            prof_diffs_v = np.abs(prof_i_v[pv_mask] - prof_j_v[pv_mask])
            profile_euclid_v = float(np.sqrt(np.sum(prof_diffs_v ** 2)))
        else:
            profile_euclid_v = 0
        
        d_i = M_verify[i, :]
        d_j = M_verify[j, :]
        
        valid_k = np.ones(n_v, dtype=bool)
        valid_k[i] = False
        valid_k[j] = False
        valid_k &= (d_i > 0) & (d_j > 0) & ~np.isnan(d_i) & ~np.isnan(d_j)
        known_side = valid_k & (km_v[i, :] | km_v[j, :])
        
        if known_side.sum() < 30:
            n_verified_close += 1
            continue
        
        dik = d_i[known_side]
        dkj = d_j[known_side]
        
        lower_bounds_ks = np.abs(dik - dkj)
        p75_lower = float(np.percentile(lower_bounds_ks, 75))
        p90_lower = float(np.percentile(lower_bounds_ks, 90))
        frac_far = float(np.mean(lower_bounds_ks > 200))
        
        midpoints = np.maximum(dik, dkj)
        
        s1_v = samples_v[i]
        s2_v = samples_v[j]
        mds_v = mds_estimates.get((s1_v, s2_v), None) if mds_estimates else None
        mds_score = max(0, (mds_v - 250) / 300) if mds_v is not None else 0
        
        p75_score = min(p75_lower / 300, 1.0)
        prof_score = min(profile_euclid_v / 25, 1.0)
        combined = p75_score + prof_score + mds_score
        
        if pv_mask.sum() >= 3 and float(np.median(prof_diffs_v)) < 10 and p75_lower < 200:
            n_verified_close += 1
            continue
        
        if combined > 1.5 and p75_lower > 150 and frac_far > 0.25:
            p50_midpoint = float(np.percentile(midpoints, 50))
            new_est = max(p75_lower, min(p50_midpoint, p90_lower * 1.5))
            new_est = max(new_est, p75_lower)
            
            M_verify[i, j] = new_est
            M_verify[j, i] = new_est
            cd_v[i, j] = False
            cd_v[j, i] = False
            n_corrected_up += 1
            n_corrected_p75 += 1
            corrected_samples[i] = corrected_samples.get(i, 0) + 1
            corrected_samples[j] = corrected_samples.get(j, 0) + 1
            corrections_detail.append(('P75', p75_lower, current_pred, new_est, combined))
        
        else:
            n_verified_close += 1
    
    suspicious_samples = {s for s, c in corrected_samples.items() if c >= 3}
    n_prop_corrected = 0
    if suspicious_samples:
        prop_mask = upper_tri_v & cd_v & (M_verify < 100) & ~km_v
        prop_i, prop_j = np.where(prop_mask)
        for pidx in range(len(prop_i)):
            pi, pj = prop_i[pidx], prop_j[pidx]
            if pi not in suspicious_samples and pj not in suspicious_samples:
                continue
            lab_pi = sample_lab_map_v.get(samples_v[pi], -1)
            lab_pj = sample_lab_map_v.get(samples_v[pj], -1)
            pj_nbrs = [k for k in lab_sample_indices_v.get(lab_pj, [])
                        if k != pj and km_v[pj, k] and M_verify[pj, k] < 100 and not cd_v[pi, k]]
            pi_nbrs = [k for k in lab_sample_indices_v.get(lab_pi, [])
                        if k != pi and km_v[pi, k] and M_verify[pi, k] < 100 and not cd_v[pj, k]]
            nbr_d = [M_verify[pi, k] for k in pj_nbrs] + [M_verify[pj, k] for k in pi_nbrs]
            if len(nbr_d) >= 2:
                nbr_a = np.array(nbr_d)
                med_n = float(np.median(nbr_a))
                frac_f = float(np.mean(nbr_a > 300))
                if med_n > 400 and frac_f > 0.5:
                    old_pred_prop = float(M_verify[pi, pj])
                    M_verify[pi, pj] = med_n
                    M_verify[pj, pi] = med_n
                    cd_v[pi, pj] = False
                    cd_v[pj, pi] = False
                    n_corrected_up += 1
                    n_prop_corrected += 1
                    corrections_detail.append(('PROP', med_n, old_pred_prop, med_n, len(nbr_d)))

    n_cp_rescued = 0
    n_cp_rescued_extended = 0
    rescued_mask = np.zeros((n_v, n_v), dtype=bool)  
    rescue_mask = upper_tri_v & (M_verify > 60) & (M_verify < 1000) & ~km_v
    rescue_i, rescue_j = np.where(rescue_mask)
    for ridx in range(len(rescue_i)):
        ri, rj = rescue_i[ridx], rescue_j[ridx]
        pred_rij = M_verify[ri, rj]
        self_anc_r = np.array([a_idx != ri and a_idx != rj for a_idx in anchor_idx_v])
        if self_anc_r.sum() < 3:
            continue
        prof_ri = anchor_profile_v[ri][self_anc_r]
        prof_rj = anchor_profile_v[rj][self_anc_r]
        pv_r = ~np.isnan(prof_ri) & ~np.isnan(prof_rj) & (prof_ri > 0) & (prof_rj > 0)
        if pv_r.sum() >= 3:
            diffs_r = np.abs(prof_ri[pv_r] - prof_rj[pv_r])
            max_pd_r = float(np.max(diffs_r))
            median_pd_r = float(np.median(diffs_r))
            if median_pd_r < 12:
                rescue_est = max(max_pd_r * 4, median_pd_r * 6, 30)
                if rescue_est < pred_rij * 0.5:

                    if pred_rij >= 300:
                        d_ri = M_verify[ri, :]
                        d_rj = M_verify[rj, :]
                        v_r = np.ones(n_v, dtype=bool)
                        v_r[ri] = False
                        v_r[rj] = False
                        v_r &= (d_ri > 0) & (d_rj > 0) & ~np.isnan(d_ri) & ~np.isnan(d_rj)
                        sk_r = v_r & (km_v[ri, :] | km_v[rj, :])
                        if sk_r.sum() < 30:
                            continue
                        lb_r = np.abs(d_ri[sk_r] - d_rj[sk_r])
                        p95_r = float(np.percentile(lb_r, 95))
                        if p95_r >= 200 or pred_rij < 4 * p95_r:
                            continue
                        n_cp_rescued_extended += 1
                    M_verify[ri, rj] = rescue_est
                    M_verify[rj, ri] = rescue_est
                    cd_v[ri, rj] = False
                    cd_v[rj, ri] = False
                    rescued_mask[ri, rj] = True
                    rescued_mask[rj, ri] = True
                    n_cp_rescued += 1

    M_verify[km_v] = global_matrix.values[km_v]
    M_verify = (M_verify + M_verify.T) / 2.0
    np.fill_diagonal(M_verify, 0)
    M_verify = np.maximum(M_verify, 0)
    
    global_matrix = pd.DataFrame(M_verify, index=global_matrix.index, columns=global_matrix.columns)
    close_detected_mask = pd.DataFrame(cd_v, index=global_matrix.index, columns=global_matrix.columns)
    
    if corrections_detail:
        corrections_detail.sort(key=lambda x: x[3], reverse=True)
        avg_old = np.mean([c[2] for c in corrections_detail])
        avg_new = np.mean([c[3] for c in corrections_detail])
        print(f"  Verified as close: {n_verified_close}")
        print(f"  Corrected upward: {n_corrected_up} (NBR={n_corrected_nbr}, P75={n_corrected_p75}, PROP={n_prop_corrected})")
        print(f"    Close-pair rescue: {n_cp_rescued} pairs restored ({n_cp_rescued_extended} from extended range)")
        print(f"    Suspicious samples (>=3 corrections): {len(suspicious_samples)}")
        print(f"    Avg old pred: {avg_old:.0f}, avg new pred: {avg_new:.0f}")
        print(f"    Top 5 corrections:")
        for c in corrections_detail[:5]:
            print(f"      type={c[0]}, signal={c[1]:.0f}, old={c[2]:.0f} -> new={c[3]:.0f}")
    else:
        print(f"  Verified as close: {n_verified_close}")
        print(f"  No corrections needed")
        print(f"  Close-pair rescue: {n_cp_rescued} pairs restored ({n_cp_rescued_extended} from extended range)")
    print()

    print("=" * 60)
    print("Post-Verification Refinement")
    print("=" * 60)
    
    M_pv = global_matrix.values.copy()
    km_pv = known_mask.values
    cd_pv = close_detected_mask.values if close_detected_mask is not None else np.zeros_like(km_pv, dtype=bool)
    protected_pv = km_pv 
    cd_only_pv = cd_pv & ~km_pv
    upper_tri_pv = np.triu(np.ones((n_v, n_v), dtype=bool), k=1)
    refinable_pv = upper_tri_pv & ~protected_pv
    n_pv_iters = 10
    
    for pv_iter in range(n_pv_iters):
        changes_pv = 0
        total_change_pv = 0.0
        for i_pv in range(n_v):
            if not refinable_pv[i_pv].any():
                continue
            d_ipv = M_pv[i_pv, :]
            j_indices_pv = np.where(refinable_pv[i_pv])[0]
            for j_pv in j_indices_pv:
                d_jpv = M_pv[j_pv, :]
                valid_pv = np.ones(n_v, dtype=bool)
                valid_pv[i_pv] = False
                valid_pv[j_pv] = False
                valid_pv &= (d_ipv > 0) & (d_jpv > 0) & ~np.isnan(d_ipv) & ~np.isnan(d_jpv)
                if valid_pv.sum() < 10:
                    continue
                dik_pv = d_ipv[valid_pv]
                dkj_pv = d_jpv[valid_pv]
                lower_pv = np.abs(dik_pv - dkj_pv)
                upper_pv = dik_pv + dkj_pv
                bk_pv = valid_pv & km_pv[i_pv, :] & km_pv[j_pv, :]
                if bk_pv.sum() >= 2:
                    hard_lo = np.max(np.abs(d_ipv[bk_pv] - d_jpv[bk_pv]))
                    hard_hi = np.min(d_ipv[bk_pv] + d_jpv[bk_pv])
                else:
                    hard_lo = 0
                    hard_hi = float('inf')
                sk_pv = valid_pv & (km_pv[i_pv, :] | km_pv[j_pv, :])
                sk_m = sk_pv[valid_pv]
                if sk_m.sum() >= 30:
                    soft_lo = np.percentile(lower_pv[sk_m], 95)
                    soft_hi = np.percentile(upper_pv[sk_m], 5)
                else:
                    soft_lo = np.percentile(lower_pv, 95)
                    soft_hi = np.percentile(upper_pv, 5)
                cur_pv = M_pv[i_pv, j_pv]
                mlo = max(hard_lo, soft_lo)
                mhi = min(hard_hi, soft_hi) if hard_hi < float('inf') else soft_hi
                if mlo >= mhi:
                    new_pv = mlo
                elif cur_pv < mlo:
                    new_pv = mlo
                elif cur_pv > mhi:
                    new_pv = mhi
                else:
                    tr = mhi - mlo
                    mp = (mlo + mhi) / 2.0
                    rr = tr / max(mp, 1.0)
                    if rr < 2.0:
                        al = min(0.4, 0.15 / max(rr, 0.01))
                        new_pv = (1 - al) * cur_pv + al * mp
                    else:
                        continue
                new_pv = max(new_pv, 0)
                if cd_only_pv[i_pv, j_pv] and new_pv > M_pv[i_pv, j_pv]:
                    new_pv = min(new_pv, cd_ceiling)
                ch = abs(new_pv - cur_pv)
                if ch > 0.1:
                    M_pv[i_pv, j_pv] = new_pv
                    M_pv[j_pv, i_pv] = new_pv
                    changes_pv += 1
                    total_change_pv += ch
        avg_ch_pv = total_change_pv / max(changes_pv, 1)
        print(f"  Iter {pv_iter+1}: {changes_pv} pairs adjusted, avg_change={avg_ch_pv:.1f}")
        if changes_pv == 0:
            break
    M_pv[km_pv] = global_matrix.values[km_pv]
    M_pv = (M_pv + M_pv.T) / 2.0
    np.fill_diagonal(M_pv, 0)
    M_pv = np.maximum(M_pv, 0)
    global_matrix = pd.DataFrame(M_pv, index=global_matrix.index, columns=global_matrix.columns)
    print("  Post-verification refinement complete.\n")

    print("=" * 60)
    print("Stable False-Close Correction (post-refinement)")
    print("=" * 60)

    M_fc = global_matrix.values.copy()
    n_fc = M_fc.shape[0]
    km_fc = known_mask.values
    cd_fc = close_detected_mask.values if close_detected_mask is not None else np.zeros((n_fc, n_fc), dtype=bool)
    upper_tri_fc = np.triu(np.ones((n_fc, n_fc), dtype=bool), k=1)
    samples_fc = list(global_matrix.index)

    fc_candidates = upper_tri_fc & cd_fc & ~km_fc
    fc_i, fc_j = np.where(fc_candidates)
    print(f"  CD pairs to evaluate: {len(fc_i)}")

    n_fc_corrected = 0
    n_fc_checked = 0
    for idx in range(len(fc_i)):
        i_fc, j_fc = fc_i[idx], fc_j[idx]
        current_fc = M_fc[i_fc, j_fc]

        d_i_fc = M_fc[i_fc, :]
        d_j_fc = M_fc[j_fc, :]

        valid_fc = np.ones(n_fc, dtype=bool)
        valid_fc[i_fc] = False
        valid_fc[j_fc] = False
        valid_fc &= (d_i_fc > 0) & (d_j_fc > 0) & ~np.isnan(d_i_fc) & ~np.isnan(d_j_fc)

        semi_known_fc = valid_fc & (km_fc[i_fc, :] | km_fc[j_fc, :])
        if semi_known_fc.sum() < 30:
            continue

        n_fc_checked += 1
        lower_bounds_fc = np.abs(d_i_fc[semi_known_fc] - d_j_fc[semi_known_fc])
        p95_lower_fc = float(np.percentile(lower_bounds_fc, 95))
        p75_lower_fc = float(np.percentile(lower_bounds_fc, 75))
        median_lower_fc = float(np.median(lower_bounds_fc))

        if p95_lower_fc > 250:
            new_est_fc = max(p75_lower_fc, median_lower_fc * 2)
            new_est_fc = min(new_est_fc, p95_lower_fc)  
            new_est_fc = max(new_est_fc, 100)  

            if new_est_fc > current_fc * 2:  
                M_fc[i_fc, j_fc] = new_est_fc
                M_fc[j_fc, i_fc] = new_est_fc
                cd_fc[i_fc, j_fc] = False
                cd_fc[j_fc, i_fc] = False
                n_fc_corrected += 1

    M_fc[km_fc] = global_matrix.values[km_fc]
    M_fc = (M_fc + M_fc.T) / 2.0
    np.fill_diagonal(M_fc, 0)
    M_fc = np.maximum(M_fc, 0)
    global_matrix = pd.DataFrame(M_fc, index=global_matrix.index, columns=global_matrix.columns)
    close_detected_mask = pd.DataFrame(cd_fc, index=global_matrix.index, columns=global_matrix.columns)
    print(f"  Evaluated {n_fc_checked} CD pairs")
    print(f"  Corrected {n_fc_corrected} false-close pairs upward")
    print()

    print("=" * 60)
    print("Close-Pair Deflation (post-stable-correction)")
    print("=" * 60)

    M_def = global_matrix.values.copy()
    n_def = M_def.shape[0]
    km_def = known_mask.values
    cd_def = close_detected_mask.values
    samples_def = list(global_matrix.index)
    upper_tri_def = np.triu(np.ones((n_def, n_def), dtype=bool), k=1)

    anchor_list_def = [a.strip() for a in args.anchors.split(',')]
    anchor_idx_def = np.array([samples_def.index(a) for a in anchor_list_def if a in samples_def])

    sample_lab_def = {}
    lab_indices_def = {}
    for lab_idx_d, ss in enumerate(lab_samples):
        for s in ss:
            if s in samples_def:
                sidx = samples_def.index(s)
                sample_lab_def[sidx] = lab_idx_d
                if lab_idx_d not in lab_indices_def:
                    lab_indices_def[lab_idx_d] = []
                lab_indices_def[lab_idx_d].append(sidx)

    n_deflated = 0
    n_nbr_blocked = 0
    deflation_mask = upper_tri_def & cd_def & ~km_def & (M_def >= 50) & (M_def <= 150)
    def_i, def_j = np.where(deflation_mask)
    print(f"  CD pairs in deflation range (50-150): {len(def_i)}")

    for didx in range(len(def_i)):
        i_d, j_d = def_i[didx], def_j[didx]
        current_d = M_def[i_d, j_d]

        self_anc_d = np.array([a != i_d and a != j_d for a in anchor_idx_def])
        if self_anc_d.sum() < 3:
            continue
        prof_id = M_def[i_d, anchor_idx_def[self_anc_d]]
        prof_jd = M_def[j_d, anchor_idx_def[self_anc_d]]
        pv_d = ~np.isnan(prof_id) & ~np.isnan(prof_jd) & (prof_id > 0) & (prof_jd > 0)
        if pv_d.sum() < 3:
            continue
        diffs_d = np.abs(prof_id[pv_d] - prof_jd[pv_d])
        median_pd_d = float(np.median(diffs_d))
        max_pd_d = float(np.max(diffs_d))

        if median_pd_d >= 10:
            continue

        d_id = M_def[i_d, :]
        d_jd = M_def[j_d, :]
        valid_d = np.ones(n_def, dtype=bool)
        valid_d[i_d] = False
        valid_d[j_d] = False
        valid_d &= (d_id > 0) & (d_jd > 0) & ~np.isnan(d_id) & ~np.isnan(d_jd)
        semi_known_d = valid_d & (km_def[i_d, :] | km_def[j_d, :])
        if semi_known_d.sum() < 30:
            continue
        lower_bounds_d = np.abs(d_id[semi_known_d] - d_jd[semi_known_d])
        p95_d = float(np.percentile(lower_bounds_d, 95))

        if p95_d >= 60:
            continue

        lab_id = sample_lab_def.get(i_d, -1)
        lab_jd = sample_lab_def.get(j_d, -1)
        i_nbrs = [k for k in lab_indices_def.get(lab_id, [])
                   if k != i_d and km_def[i_d, k] and M_def[i_d, k] < 100]
        j_nbrs = [k for k in lab_indices_def.get(lab_jd, [])
                   if k != j_d and km_def[j_d, k] and M_def[j_d, k] < 100]
        nbr_dists = [M_def[k, j_d] for k in i_nbrs] + [M_def[i_d, k] for k in j_nbrs]
        if len(nbr_dists) >= 2:
            nbr_med = float(np.median(nbr_dists))
            if nbr_med > 200:
                n_nbr_blocked += 1
                continue

        deflated = max(max_pd_d * 4, median_pd_d * 6, 30)
        deflated = min(deflated, current_d)  

        M_def[i_d, j_d] = deflated
        M_def[j_d, i_d] = deflated
        n_deflated += 1

    M_def[km_def] = global_matrix.values[km_def]
    M_def = (M_def + M_def.T) / 2.0
    np.fill_diagonal(M_def, 0)
    M_def = np.maximum(M_def, 0)
    global_matrix = pd.DataFrame(M_def, index=global_matrix.index, columns=global_matrix.columns)
    print(f"  Deflated {n_deflated} genuine close CD pairs")
    print(f"  Blocked by NBR check: {n_nbr_blocked}")
    print()

    print("=" * 60)
    print("Post-Pipeline Neighbor-Consensus Correction")
    print("=" * 60)

    M_nc = global_matrix.values.copy()
    n_nc = M_nc.shape[0]
    km_nc = known_mask.values
    samples_nc = list(global_matrix.index)

    anchor_list_nc = [a.strip() for a in args.anchors.split(',')]
    anchor_idx_nc = np.array([samples_nc.index(a) for a in anchor_list_nc if a in samples_nc])

    sample_lab_nc = {}
    lab_indices_nc = {}
    for lab_idx_nc, ss in enumerate(lab_samples):
        for s in ss:
            if s in samples_nc:
                sample_lab_nc[s] = lab_idx_nc
                if lab_idx_nc not in lab_indices_nc:
                    lab_indices_nc[lab_idx_nc] = []
                lab_indices_nc[lab_idx_nc].append(samples_nc.index(s))

    n_nc_corrected = 0
    n_nc_checked = 0
    for i in range(n_nc):
        lab_i_nc = sample_lab_nc.get(samples_nc[i], -1)
        i_nbrs = lab_indices_nc.get(lab_i_nc, [])
        for j in range(i + 1, n_nc):
            if km_nc[i, j]:
                continue
            if rescued_mask[i, j] or pre_svd_rescued_mask[i, j]:
                continue
            lab_j_nc = sample_lab_nc.get(samples_nc[j], -1)
            if lab_i_nc == lab_j_nc:
                continue
            pred_nc = M_nc[i, j]
            if pred_nc >= 200:
                continue
            # Check anchor profile tightness
            prof_i_nc = M_nc[i, anchor_idx_nc]
            prof_j_nc = M_nc[j, anchor_idx_nc]
            pv_nc = (prof_i_nc > 0) & (prof_j_nc > 0)
            if pv_nc.sum() < 3:
                continue
            diffs_nc = np.abs(prof_i_nc[pv_nc] - prof_j_nc[pv_nc])
            if np.max(diffs_nc) >= 30:
                continue
            n_nc_checked += 1

            j_nbrs = lab_indices_nc.get(lab_j_nc, [])
            close_j = [k for k in j_nbrs if k != j and km_nc[j, k] and M_nc[j, k] < 100]
            close_i = [k for k in i_nbrs if k != i and km_nc[i, k] and M_nc[i, k] < 100]
            nbr_d = [M_nc[i, k] for k in close_j] + [M_nc[j, k] for k in close_i]

            if len(nbr_d) >= 3:
                nbr_a = np.array(nbr_d)
                med_nc = float(np.median(nbr_a))
                frac_far_nc = float(np.mean(nbr_a > 200))
                if med_nc > 200 and frac_far_nc > 0.5:
                    M_nc[i, j] = med_nc
                    M_nc[j, i] = med_nc
                    n_nc_corrected += 1

    M_nc[km_nc] = global_matrix.values[km_nc]
    M_nc = (M_nc + M_nc.T) / 2.0
    np.fill_diagonal(M_nc, 0)
    M_nc = np.maximum(M_nc, 0)
    global_matrix = pd.DataFrame(M_nc, index=global_matrix.index, columns=global_matrix.columns)
    print(f"  Checked {n_nc_checked} tight-profile cross-lab pairs (pred < 200)")
    print(f"  Corrected {n_nc_corrected} false-close pairs upward")
    print()

    print("  Four-Point Constraint Enforcement: SKIPPED\n")

    print("  Profile-Inconsistency Correction: SKIPPED (FP rate too high)\n")

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
