#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import pandas as pd
import numpy as np

try:
    import dendropy
    from dendropy.calculate import treecompare
    DENDROPY_AVAILABLE = True
except ImportError:
    DENDROPY_AVAILABLE = False


def load_distance_matrix(filepath):
    return pd.read_csv(filepath, sep='\t', index_col=0)


def normalize_distances(distance_matrix):

    max_dist = distance_matrix.values.max()
    if max_dist > 0:
        return distance_matrix / max_dist
    return distance_matrix


def count_tree_tips(tree_file):

    with open(tree_file, 'r') as f:
        tree_str = f.read().strip()
    
    tips = re.findall(r'[(,]([A-Za-z0-9_]+):', tree_str)
    if not tips:
        tips = re.findall(r'[(,]([A-Za-z0-9_]+)[,)]', tree_str)
    
    return len(tips) if tips else 0


def get_tree_tips(tree_file):

    with open(tree_file, 'r') as f:
        tree_str = f.read().strip()
    
    tips = re.findall(r'[(,]([A-Za-z0-9_]+):', tree_str)
    if not tips:
        tips = re.findall(r'[(,]([A-Za-z0-9_]+)[,)]', tree_str)
    
    return list(set(tips))


def get_leaf_set(tree):
    return set([leaf.taxon.label for leaf in tree.leaf_node_iter()])


def load_trees_dendropy(tree_files):
    taxa = dendropy.TaxonNamespace()
    trees = []
    
    for tf in tree_files:
        try:
            tree = dendropy.Tree.get(path=tf, schema="newick", taxon_namespace=taxa)
            trees.append(tree)
            print(f"Loaded tree from {tf} with {len(list(tree.leaf_node_iter()))} tips")
        except Exception as e:
            print(f"Could not load tree {tf}: {e}")
    
    return trees, taxa


def build_pdm_from_matrix(distance_matrix, taxa):

    for sample in distance_matrix.index:
        taxa.require_taxon(label=sample)
    
    pdm = dendropy.PhylogeneticDistanceMatrix(taxon_namespace=taxa)
    
    for t1 in taxa:
        if t1.label not in distance_matrix.index:
            continue
        for t2 in taxa:
            if t2.label not in distance_matrix.index:
                continue
            pdm[t1, t2] = float(distance_matrix.loc[t1.label, t2.label])
    
    return pdm


def constrained_neighbor_joining(distance_matrix, constraint_trees=None):

    samples = list(distance_matrix.index)
    n = len(samples)
    
    if n < 3:
        if n == 2:
            d = distance_matrix.iloc[0, 1]
            return f"({samples[0]}:{d/2:.6f},{samples[1]}:{d/2:.6f});"
        elif n == 1:
            return f"({samples[0]}:0);"
        else:
            return "();"
    
    D = distance_matrix.values.astype(float).copy()
    active = list(range(n))
    tree_strs = {i: samples[i] for i in range(n)}
    
    constraint_pairs = set()
    if constraint_trees:
        for tree in constraint_trees:
            for node in tree.postorder_internal_node_iter():
                children = node.child_nodes()
                if len(children) >= 2:
                    for i, c1 in enumerate(children):
                        for c2 in children[i+1:]:
                            leaves1 = set(l.taxon.label for l in c1.leaf_iter())
                            leaves2 = set(l.taxon.label for l in c2.leaf_iter())
                            for l1 in leaves1:
                                for l2 in leaves1:
                                    if l1 < l2:
                                        constraint_pairs.add((l1, l2))
                            for l1 in leaves2:
                                for l2 in leaves2:
                                    if l1 < l2:
                                        constraint_pairs.add((l1, l2))
    
    while len(active) > 2:
        m = len(active)
        
        row_sums = {}
        for i, idx_i in enumerate(active):
            row_sums[i] = sum(D[idx_i, idx_j] for j, idx_j in enumerate(active) if i != j)
        
        min_q = float('inf')
        min_i, min_j = 0, 1
        
        for i in range(m):
            for j in range(i + 1, m):
                idx_i = active[i]
                idx_j = active[j]
                
                q_val = (m - 2) * D[idx_i, idx_j] - row_sums[i] - row_sums[j]
                
                sample_i = samples[idx_i] if idx_i < len(samples) else None
                sample_j = samples[idx_j] if idx_j < len(samples) else None
                
                if sample_i and sample_j:
                    pair = (min(sample_i, sample_j), max(sample_i, sample_j))
                    if pair in constraint_pairs:
                        q_val -= 0.001
                
                if q_val < min_q:
                    min_q = q_val
                    min_i, min_j = i, j
        
        idx_i = active[min_i]
        idx_j = active[min_j]
        
        if m > 2:
            dist_i = 0.5 * D[idx_i, idx_j] + (row_sums[min_i] - row_sums[min_j]) / (2 * (m - 2))
            dist_j = D[idx_i, idx_j] - dist_i
        else:
            dist_i = D[idx_i, idx_j] / 2
            dist_j = D[idx_i, idx_j] / 2
        
        dist_i = max(0, dist_i)
        dist_j = max(0, dist_j)
        
        new_node_str = f"({tree_strs[idx_i]}:{dist_i:.6f},{tree_strs[idx_j]}:{dist_j:.6f})"
        tree_strs[idx_i] = new_node_str
        
        for k, idx_k in enumerate(active):
            if idx_k != idx_i and idx_k != idx_j:
                new_dist = 0.5 * (D[idx_i, idx_k] + D[idx_j, idx_k] - D[idx_i, idx_j])
                new_dist = max(0, new_dist)
                D[idx_i, idx_k] = new_dist
                D[idx_k, idx_i] = new_dist
        
        active.remove(idx_j)
    
    if len(active) == 2:
        idx_i, idx_j = active
        dist = max(0, D[idx_i, idx_j] / 2)
        return f"({tree_strs[idx_i]}:{dist:.6f},{tree_strs[idx_j]}:{dist:.6f});"
    else:
        return f"({tree_strs[active[0]]}:0);"


def merge_trees_improved(tree_files, global_matrix, anchors, output_file):

    print("=" * 60)
    print("Improved Tree Merging Algorithm")
    print("=" * 60)
    
    max_dist = global_matrix.values.max()
    print(f"Maximum distance in matrix: {max_dist}")
    
    normalized_matrix = global_matrix / max_dist if max_dist > 0 else global_matrix
    print(f"Normalized distance range: {normalized_matrix.values.min():.4f} - {normalized_matrix.values.max():.4f}")
    
    all_samples = list(global_matrix.index)
    print(f"Total samples: {len(all_samples)}")
    
    constraint_trees = None
    
    if DENDROPY_AVAILABLE:
        try:
            trees, taxa = load_trees_dendropy(tree_files)
            if trees:
                constraint_trees = trees
                print(f"Using {len(trees)} input trees as topological constraints")
        except Exception as e:
            print(f"Could not load constraint trees: {e}")
    
    newick_tree = constrained_neighbor_joining(normalized_matrix, constraint_trees)
    
    with open(output_file, 'w') as f:
        f.write(newick_tree)
    
    print(f"Merged tree written to {output_file}")
    return newick_tree


def merge_trees_dendropy_nj(tree_files, global_matrix, anchors, output_file):

    print("Building merged tree using DendroPy NJ")
    
    max_dist = global_matrix.values.max()
    normalized_matrix = global_matrix / max_dist if max_dist > 0 else global_matrix
    
    all_samples = list(normalized_matrix.index)
    print(f"Total samples: {len(all_samples)}")
    print(f"Distance range: {normalized_matrix.values.min():.4f} - {normalized_matrix.values.max():.4f}")
    
    taxa = dendropy.TaxonNamespace(all_samples)
    pdm = dendropy.PhylogeneticDistanceMatrix(taxon_namespace=taxa)
    
    for i, t1 in enumerate(taxa):
        for j, t2 in enumerate(taxa):
            pdm[t1, t2] = float(normalized_matrix.iloc[i, j])
    
    tree = pdm.nj_tree()
    tree.write(path=output_file, schema="newick")
    
    print(f"Merged tree written to {output_file}")
    return tree.as_string(schema="newick")


def calculate_merge_statistics(tree_files, output_file, global_matrix):
    stats = {
        "num_input_trees": len(tree_files),
        "input_tree_sizes": [count_tree_tips(f) for f in tree_files],
        "merged_tree_size": count_tree_tips(output_file),
        "method": "constrained_neighbor_joining",
        "matrix_size": len(global_matrix),
        "max_distance": float(global_matrix.values.max()),
        "min_distance": float(global_matrix.values[global_matrix.values > 0].min()) if (global_matrix.values > 0).any() else 0,
        "dendropy_available": DENDROPY_AVAILABLE
    }
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Merge phylogenetic trees using NJ algorithm"
    )
    parser.add_argument('--trees', nargs='+', required=True, help="Input tree files")
    parser.add_argument('--matrix', type=str, required=True, help="Global distance matrix")
    parser.add_argument('--mapping', type=str, required=True, help="Sample mapping JSON")
    parser.add_argument('--anchors', type=str, required=True, help="Comma-separated anchor samples")
    parser.add_argument('--output', type=str, default="global_tree.nwk", help="Output tree file")
    parser.add_argument('--stats', type=str, default="merge_stats.json", help="Output stats file")
    args = parser.parse_args()
    
    anchors = [a.strip() for a in args.anchors.split(',')]
    global_matrix = load_distance_matrix(args.matrix)
    
    with open(args.mapping, 'r') as f:
        mapping = json.load(f)
    
    print(f"Loaded {len(args.trees)} trees")
    print(f"Global matrix: {len(global_matrix)} samples")
    print(f"Anchors: {anchors}")
    print(f"DendroPy available: {DENDROPY_AVAILABLE}")
    
    if DENDROPY_AVAILABLE:
        try:
            merge_trees_dendropy_nj(args.trees, global_matrix, anchors, args.output)
        except Exception as e:
            print(f"DendroPy NJ failed: {e}, using custom implementation")
            merge_trees_improved(args.trees, global_matrix, anchors, args.output)
    else:
        merge_trees_improved(args.trees, global_matrix, anchors, args.output)
    
    stats = calculate_merge_statistics(args.trees, args.output, global_matrix)
    stats["anchors"] = anchors
    stats["input_files"] = [os.path.basename(f) for f in args.trees]
    
    with open(args.stats, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Statistics saved to {args.stats}")


if __name__ == "__main__":
    main()