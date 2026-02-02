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
                    val = df.loc[s1, s2]
                    if not np.isnan(val) and val >= 0:
                        values.append(val)
            
            if values:
                global_matrix.loc[s1, s2] = np.mean(values)
                known_mask.loc[s1, s2] = True
    
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
    
    print(f"\nAnchor coverage:")
    for anchor in present_anchors:
        known_count = known_mask.loc[:, anchor].sum()
        non_nan_count = (~incomplete_matrix.loc[:, anchor].isna()).sum()
        print(f"  {anchor}: known_mask={known_count}, non_nan={non_nan_count}/{len(samples)}")
    
    missing_pairs = []
    can_triangulate = 0
    cannot_triangulate = 0
    
    for i, s1 in enumerate(samples):
        for j, s2 in enumerate(samples):
            if i >= j:
                continue
            if np.isnan(incomplete_matrix.loc[s1, s2]):
                missing_pairs.append((s1, s2))
                
                has_common_anchor = False
                for anchor in present_anchors:
                    d1 = incomplete_matrix.loc[s1, anchor]
                    d2 = incomplete_matrix.loc[s2, anchor]
                    if not np.isnan(d1) and not np.isnan(d2) and d1 > 0 and d2 > 0:
                        has_common_anchor = True
                        break
                
                if has_common_anchor:
                    can_triangulate += 1
                else:
                    cannot_triangulate += 1
                    if cannot_triangulate <= 5:
                        print(f"\n  Cannot triangulate: {s1} <-> {s2}")
                        for anchor in present_anchors:
                            d1 = incomplete_matrix.loc[s1, anchor]
                            d2 = incomplete_matrix.loc[s2, anchor]
                            print(f"    {anchor}: {s1}={d1}, {s2}={d2}")
    
    print(f"\nMissing pairs: {len(missing_pairs)}")
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
    
    for iteration in range(max_iterations):
        changes = 0
        total_adjustment = 0.0
        
        for i in range(n):
            for j in range(i + 1, n):
                if protected[i, j]:
                    continue
                
                upper_bounds = []
                lower_bounds = []
                
                for k in range(n):
                    if k == i or k == j:
                        continue
                    
                    d_ik = matrix[i, k]
                    d_kj = matrix[k, j]
                    
                    if d_ik > 0 and d_kj > 0:
                        upper_bounds.append(d_ik + d_kj)
                        lower_bounds.append(abs(d_ik - d_kj))
                
                if not upper_bounds:
                    continue
                
                upper = min(upper_bounds)
                lower = max(lower_bounds)
                current = matrix[i, j]
                
                if current > upper:
                    new_val = current * 0.3 + upper * 0.7
                    changes += 1
                    total_adjustment += abs(new_val - current)
                    matrix[i, j] = new_val
                    matrix[j, i] = new_val
                elif current < lower:
                    new_val = current * 0.3 + lower * 0.7
                    changes += 1
                    total_adjustment += abs(new_val - current)
                    matrix[i, j] = new_val
                    matrix[j, i] = new_val
        
        avg_adjustment = total_adjustment / max(changes, 1)
        print(f"  Iteration {iteration + 1}: {changes} adjustments, avg change: {avg_adjustment:.2f}")
        
        if changes == 0 or avg_adjustment < 0.5:
            break
    
    for i in range(n):
        for j in range(i + 1, n):
            if protected[i, j]:
                matrix[i, j] = original_protected[i, j]
                matrix[j, i] = original_protected[i, j]
    
    return matrix


def impute_anchor_guided(incomplete_matrix, known_mask, anchors, sample_to_lab):

    print("Using Enhanced Anchor-Guided Imputation")
    
    matrix = incomplete_matrix.copy()
    samples = list(matrix.index)
    known = known_mask.copy()
    
    present_anchors = [a for a in anchors if a in samples]
    print(f"  Using {len(present_anchors)} anchors for guided imputation")
    
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
    
    calibration_data = {"close": [], "medium": [], "far": []}
    
    for i, a1 in enumerate(present_anchors):
        for a2 in present_anchors[i+1:]:
            if known.loc[a1, a2]:
                actual = matrix.loc[a1, a2]
                for s in samples:
                    if s == a1 or s == a2:
                        continue
                    d1 = matrix.loc[s, a1]
                    d2 = matrix.loc[s, a2]
                    if not np.isnan(d1) and not np.isnan(d2) and known.loc[s, a1] and known.loc[s, a2]:
                        upper = d1 + d2
                        lower = abs(d1 - d2)
                        if upper > lower and upper > 0:
                            ratio = (actual - lower) / (upper - lower)
                            if 0 <= ratio <= 1:
                                if actual < 500:
                                    calibration_data["close"].append(ratio)
                                elif actual < 1500:
                                    calibration_data["medium"].append(ratio)
                                else:
                                    calibration_data["far"].append(ratio)
    
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
    
    sample_profiles = {}
    for s in samples:
        profile = []
        for anchor in present_anchors:
            d = original_matrix.loc[s, anchor]
            if not np.isnan(d):
                profile.append(d)
            else:
                profile.append(np.nan)
        sample_profiles[s] = np.array(profile)
    
    all_known_distances = {}
    for s1 in samples:
        all_known_distances[s1] = {}
        for s2 in samples:
            if s1 == s2:
                continue
            if known.loc[s1, s2]:
                d = original_matrix.loc[s1, s2]
                if d > 0:
                    all_known_distances[s1][s2] = d
    
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
    
    for s1 in samples:
        for s2 in samples:
            if s1 >= s2:
                continue
            if known.loc[s1, s2]:
                d = original_matrix.loc[s1, s2]
                if d < 150:
                    union(s1, s2)
    
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
            for s2 in members:
                if s1 >= s2:
                    continue
                s2_labs = set(sample_to_lab.get(s2, []))
                
                if s1_labs & s2_labs:
                    continue  
                
                if known.loc[s1, s2]:
                    lineage_cross_lab_distances[(root, s1, s2)] = original_matrix.loc[s1, s2]
    
    print(f"  Found {len(lineage_cross_lab_distances)} known cross-lab distances")
    
    imputation_stats = {
        "close_detection": 0, "consistency_override": 0, "parallel_branch": 0,
        "transitivity_check": 0, "anchor_direct": 0, "two_hop": 0, "fallback": 0,
        "lineage_proxy": 0, "bridge_estimate": 0, "same_lineage": 0,
        "phylo_close": 0, "anchor_triangulation": 0, "neighbor_based": 0
    }
    
    missing_pairs = []
    for i, s1 in enumerate(samples):
        for j, s2 in enumerate(samples):
            if i >= j:
                continue
            if np.isnan(matrix.loc[s1, s2]):
                missing_pairs.append((s1, s2))
    
    print(f"\n  Missing pairs to impute: {len(missing_pairs)}")
    
    close_pairs_to_impute = []
    processed_pairs = set()
    
    # Multiple passes because some pairs depend on estimates from others
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
            
            consistency_lower = 0
            consistency_upper = float('inf')
            consistency_evidence = []
            
            for k in samples:
                if k == s1 or k == s2:
                    continue
                
                d_s1_k = all_known_distances.get(s1, {}).get(k)
                d_s2_k = all_known_distances.get(s2, {}).get(k)
                
                if d_s1_k is None and not np.isnan(matrix.loc[s1, k]) and matrix.loc[s1, k] > 0:
                    d_s1_k = matrix.loc[s1, k]
                if d_s2_k is None and not np.isnan(matrix.loc[s2, k]) and matrix.loc[s2, k] > 0:
                    d_s2_k = matrix.loc[s2, k]
                
                if d_s1_k is not None and d_s2_k is not None:
                    lower = abs(d_s1_k - d_s2_k)
                    upper = d_s1_k + d_s2_k
                    consistency_evidence.append((k, d_s1_k, d_s2_k, lower, upper))
                    if lower > consistency_lower:
                        consistency_lower = lower
                    if upper < consistency_upper:
                        consistency_upper = upper
            
            anchor_lower = max_diff
            anchor_upper = float('inf')
            
            for anchor in present_anchors:
                d1 = original_matrix.loc[s1, anchor]
                d2 = original_matrix.loc[s2, anchor]
                if not np.isnan(d1) and not np.isnan(d2) and d1 > 0 and d2 > 0:
                    upper = d1 + d2
                    if upper < anchor_upper:
                        anchor_upper = upper
            
            effective_lower = max(consistency_lower, anchor_lower)
            effective_upper = min(consistency_upper, anchor_upper) if consistency_upper < float('inf') else anchor_upper
            
            if should_debug:
                print(f"    Pass {pass_num}: consistency_lower={consistency_lower:.0f}, effective_upper={effective_upper:.0f}")
                print(f"    consistency_evidence count={len(consistency_evidence)}")
                for k, d1, d2, lo, up in consistency_evidence[:3]:
                    print(f"      via {k}: d1={d1:.0f}, d2={d2:.0f}, bounds=[{lo:.0f}, {up:.0f}]")
            
            s1_root = find(s1)
            s2_root = find(s2)
            same_lineage_group = (s1_root == s2_root)
            
            mean_profile = np.nanmean(np.concatenate([profile1[valid_mask], profile2[valid_mask]]))
            
            is_deep_same_lineage = (mean_profile > 1200 and max_diff < 50 and std_diff < 25)
            is_potentially_close = (max_diff < 20)
            
            if should_debug:
                print(f"    mean_profile={mean_profile:.0f}, max_diff={max_diff:.0f}, std_diff={std_diff:.2f}")
                print(f"    is_deep_same_lineage={is_deep_same_lineage}, is_potentially_close={is_potentially_close}")
            
            lineage_estimate = None
            
            if is_cross_lab and is_deep_same_lineage:

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
                else:

                    lineage_estimate = max(max_diff * 6, effective_lower * 1.1, 80)
                
                lineage_estimate = max(lineage_estimate, effective_lower)
                
                if should_debug:
                    print(f"Same lineage: max_diff={max_diff:.0f}, std_diff={std_diff:.2f}, estimate={lineage_estimate:.0f}")
            
            elif is_cross_lab and is_potentially_close:

                close_calibration = []
                if same_lineage_group:
                    for other in samples:
                        if other == s1 or other == s2:
                            continue
                        other_labs = set(sample_to_lab.get(other, []))
                        
                        if s1_labs & other_labs and known.loc[s1, other]:
                            d = original_matrix.loc[s1, other]
                            if 0 < d < 150:
                                other_profile = sample_profiles.get(other, np.array([]))
                                if len(other_profile) == len(profile2):
                                    other_diffs = np.abs(profile2 - other_profile)
                                    valid = ~np.isnan(other_diffs)
                                    if valid.sum() >= 3 and np.nanmax(other_diffs) < 25:
                                        close_calibration.append(d)
                        
                        if s2_labs & other_labs and known.loc[s2, other]:
                            d = original_matrix.loc[s2, other]
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
                    print(f"    POTENTIALLY_CLOSE detected: max_diff={max_diff:.0f}, calibration={close_calibration[:3]}, estimate={close_estimate:.0f}")
                
                lineage_estimate = close_estimate
            
            elif is_cross_lab and max_diff < 50 and mean_profile > 1500:
                
                calibration_distances = []
                
                if same_lineage_group:
                    lineage_root = find(s1)
                    for (root, sa, sb), known_dist in lineage_cross_lab_distances.items():
                        if root == lineage_root:
                            calibration_distances.append(known_dist)
                            if should_debug:
                                print(f"    Found lineage cross-lab reference: {sa}↔{sb} = {known_dist}")
                
                for other in samples:
                    if other == s1 or other == s2:
                        continue
                    
                    other_labs = set(sample_to_lab.get(other, []))
                    if s1_labs & other_labs and known.loc[s1, other]:
                        d_s1_other = original_matrix.loc[s1, other]
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
                    
                    if s2_labs & other_labs and known.loc[s2, other]:
                        d_s2_other = original_matrix.loc[s2, other]
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
                
                if calibration_distances:
                    median_cal = np.median(calibration_distances)
                    lineage_estimate = median_cal + max_diff * 1.5
                    lineage_estimate = max(lineage_estimate, effective_lower)
                    
                    if should_debug:
                        print(f"    MODERATELY_CLOSE with calibration: {calibration_distances[:5]}, estimate={lineage_estimate:.0f}")
                else:
                    if std_diff < 15:
                        lineage_estimate = max(max_diff * 4, 80)
                    else:
                        lineage_estimate = max(max_diff * 5, 100)
                    lineage_estimate = max(lineage_estimate, effective_lower)
                    
                    if should_debug:
                        print(f"    MODERATELY_CLOSE fallback: estimate={lineage_estimate:.0f}")
            
            if lineage_estimate is not None:
                close_pairs_to_impute.append((s1, s2, lineage_estimate, f"LINEAGE_PROXY_PASS{pass_num}"))
                processed_pairs.add((s1, s2))
                
                matrix.loc[s1, s2] = lineage_estimate
                matrix.loc[s2, s1] = lineage_estimate
                close_detected_mask.loc[s1, s2] = True
                close_detected_mask.loc[s2, s1] = True
                known.loc[s1, s2] = True
                known.loc[s2, s1] = True
                
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
                    mult = 6.0
                    close_estimate = max(max_diff * mult, median_diff * 7, effective_lower * 1.2, 30)
                    close_pairs_to_impute.append((s1, s2, close_estimate, "VERY_CLOSE"))
                    processed_pairs.add((s1, s2))
                    matrix.loc[s1, s2] = close_estimate
                    matrix.loc[s2, s1] = close_estimate
                    close_detected_mask.loc[s1, s2] = True
                    close_detected_mask.loc[s2, s1] = True
                    known.loc[s1, s2] = True
                    known.loc[s2, s1] = True
                
                elif max_diff < 15 and small_diff_ratio >= 0.90 and effective_lower < 75:
                    mult = 5.0
                    close_estimate = max(max_diff * mult, mean_diff * 6, effective_lower * 1.1, 50)
                    close_pairs_to_impute.append((s1, s2, close_estimate, "CLOSE"))
                    processed_pairs.add((s1, s2))
                    matrix.loc[s1, s2] = close_estimate
                    matrix.loc[s2, s1] = close_estimate
                    close_detected_mask.loc[s1, s2] = True
                    close_detected_mask.loc[s2, s1] = True
                    known.loc[s1, s2] = True
                    known.loc[s2, s1] = True
                
                elif max_diff < 50 and small_diff_ratio >= 0.8 and effective_lower < 150:
                    mult = 4.0
                    close_estimate = max(max_diff * mult, mean_diff * 5, effective_lower * 1.05, 100)
                    close_pairs_to_impute.append((s1, s2, close_estimate, "MODERATELY_CLOSE"))
                    processed_pairs.add((s1, s2))
                    matrix.loc[s1, s2] = close_estimate
                    matrix.loc[s2, s1] = close_estimate
                    close_detected_mask.loc[s1, s2] = True
                    close_detected_mask.loc[s2, s1] = True
                    known.loc[s1, s2] = True
                    known.loc[s2, s1] = True
        
        print(f"  Pass {pass_num}: added {pairs_added_this_pass} estimates")
        
        if pairs_added_this_pass == 0 and pass_num > 0:
            break
    
    print(f"\n  Close pairs detected: {len(close_pairs_to_impute)}")
    category_counts = {}
    for s1, s2, estimate, category in close_pairs_to_impute:
        category_counts[category] = category_counts.get(category, 0) + 1
    print(f"  Category breakdown: {category_counts}")

    remaining_pairs = []
    for s1, s2 in missing_pairs:
        if (s1, s2) not in processed_pairs and (s2, s1) not in processed_pairs:
            remaining_pairs.append((s1, s2))
    
    print(f"  Remaining pairs to impute with anchor triangulation: {len(remaining_pairs)}")
    
    for s1, s2 in remaining_pairs:
        should_debug = (s1, s2) in debug_pairs
        estimates = []
        weights = []
        
        profile1 = sample_profiles[s1]
        profile2 = sample_profiles[s2]
        valid_mask = ~np.isnan(profile1) & ~np.isnan(profile2)
        
        s1_root = find(s1)
        s2_root = find(s2)
        same_lineage_group = (s1_root == s2_root)
        
        if valid_mask.sum() >= 3:
            diffs = np.abs(profile1[valid_mask] - profile2[valid_mask])
            max_diff = np.max(diffs)
            std_diff = np.std(diffs)
            mean_profile = np.nanmean(np.concatenate([profile1[valid_mask], profile2[valid_mask]]))
            is_deep_same_lineage = (mean_profile > 1200 and max_diff < 50 and std_diff < 25)
        else:
            is_deep_same_lineage = False
            max_diff = 0
        
        if valid_mask.sum() >= 3:
            diffs = np.abs(profile1[valid_mask] - profile2[valid_mask])
            min_diff = np.min(diffs)
            median_diff = np.median(diffs)
            mean_diff = np.mean(diffs)
            max_diff = np.max(diffs)
            
            small_count = np.sum(diffs < 100)
            small_ratio = small_count / len(diffs)
            
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
        
        lower_bounds = []
        upper_bounds = []
        
        for anchor in present_anchors:
            d1 = original_known.loc[s1, anchor]
            d2 = original_known.loc[s2, anchor]
            
            if not np.isnan(d1) and not np.isnan(d2) and d1 > 0 and d2 > 0:
                lower = abs(d1 - d2)
                upper = d1 + d2
                lower_bounds.append(lower)
                upper_bounds.append(upper)
        
        if lower_bounds:
            max_lower = max(lower_bounds)
            min_upper = min(upper_bounds)
            
            if max_lower < 100:
                current_weight = weight_factors.get("close", 0.15)
            elif max_lower < 300:
                current_weight = weight_factors.get("close", 0.15) * 0.7 + weight_factors.get("medium", 0.35) * 0.3
            elif max_lower < 800:
                current_weight = weight_factors.get("medium", 0.35)
            else:
                current_weight = weight_factors.get("far", 0.45)
            
            if same_lineage_group or is_deep_same_lineage:
                current_weight *= 0.3
            
            for anchor in present_anchors:
                d1 = original_known.loc[s1, anchor]
                d2 = original_known.loc[s2, anchor]
                
                if not np.isnan(d1) and not np.isnan(d2) and d1 > 0 and d2 > 0:
                    lower = abs(d1 - d2)
                    upper = d1 + d2
                    
                    estimate = lower + current_weight * (upper - lower)
                    
                    balance = 1.0 - abs(d1 - d2) / (d1 + d2)
                    weight = 0.5 + 0.5 * balance
                    
                    if len(estimates) > 0 and estimates[0] < 300:
                        weight *= 0.1
                    
                    estimates.append(estimate)
                    weights.append(weight)
            
            imputation_stats["anchor_triangulation"] += 1
        
        if len(estimates) < 3:
            neighbor_estimates = []
            for sk in samples:
                if sk == s1 or sk == s2:
                    continue
                
                d1k = original_known.loc[s1, sk]
                dk2 = original_known.loc[sk, s2]
                
                if (not np.isnan(d1k) and not np.isnan(dk2) and 
                    known.loc[s1, sk] and known.loc[sk, s2] and
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
        for anchor in present_anchors:
            d1 = original_known.loc[s1, anchor]
            d2 = original_known.loc[s2, anchor]
            if not np.isnan(d1) and not np.isnan(d2) and d1 > 0 and d2 > 0:
                lower = abs(d1 - d2)
                if lower > effective_lower:
                    effective_lower = lower
        
        if estimates:
            weights = np.array(weights)
            weights = weights / weights.sum()
            imputed = np.average(estimates, weights=weights)
            
            imputed = max(imputed, effective_lower)
            
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
            
            matrix.loc[s1, s2] = imputed
            matrix.loc[s2, s1] = imputed
        else:
            imputation_stats["fallback"] += 1
            known_vals = original_known.values[known.values & (original_known.values > 0)]
            matrix.loc[s1, s2] = np.median(known_vals) if len(known_vals) > 0 else 1500
            matrix.loc[s2, s1] = matrix.loc[s1, s2]
    
    print(f"  Imputation stats: {imputation_stats}")
    
    for s1 in samples:
        for s2 in samples:
            if known_mask.loc[s1, s2]:
                matrix.loc[s1, s2] = original_known.loc[s1, s2]
    
    return matrix, close_detected_mask


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
        if can_tri > 0 and len(anchors) >= 3:
            method = 'anchor_guided'
        elif pct_known > 70:
            method = 'knn'
        elif FANCYIMPUTE_AVAILABLE:
            method = 'softimpute'
        else:
            method = 'anchor_guided'
        print(f"Auto-selected method: {method}")
    
    close_detected_mask = None
    
    if method == 'softimpute' and FANCYIMPUTE_AVAILABLE:
        global_matrix = impute_softimpute_improved(incomplete_matrix, known_mask)
    elif method == 'knn' and SKLEARN_KNN_AVAILABLE:
        global_matrix = impute_knn_improved(incomplete_matrix, known_mask)
    elif method == 'anchor_guided':
        global_matrix, close_detected_mask = impute_anchor_guided(
            incomplete_matrix, known_mask, anchors, sample_to_lab
        )
    else:
        global_matrix, close_detected_mask = impute_anchor_guided(
            incomplete_matrix, known_mask, anchors, sample_to_lab
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
    
    return global_matrix, method


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
