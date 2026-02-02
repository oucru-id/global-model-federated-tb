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


def root_tree_by_outgroup(tree, outgroup_name="ERR4872250"):

    try:
        outgroup_node = None
        for leaf in tree.leaf_node_iter():
            if leaf.taxon and leaf.taxon.label == outgroup_name:
                outgroup_node = leaf
                break
        
        if outgroup_node is None:
            print(f"Warning: Outgroup '{outgroup_name}' not found in tree")
            return tree, False
        
        tree.reroot_at_edge(
            outgroup_node.edge,
            update_bipartitions=True,
            length1=outgroup_node.edge_length / 2 if outgroup_node.edge_length else 0,
            length2=outgroup_node.edge_length / 2 if outgroup_node.edge_length else 0
        )
        print(f"Rooted tree using outgroup: {outgroup_name}")
        return tree, True
        
    except Exception as e:
        print(f"Could not root by outgroup {outgroup_name}: {e}")
        return tree, False


def root_tree_midpoint(tree):
    try:
        tree.reroot_at_midpoint(update_bipartitions=True)
        print("Rooted tree using midpoint rooting")
        return tree, True
    except Exception as e:
        print(f"Could not perform midpoint rooting: {e}")
        return tree, False


def normalize_for_tree(distance_matrix, method='max'):
    """
    Normalize distances for tree visualization.
    
    Methods:
    - 'max': Divide by maximum distance (gives 0-1 scale)
    - 'genome': Divide by approximate TB genome size (~4.4M bp)
    - 'none': No normalization (raw SNP counts)
    """
    if method == 'none':
        return distance_matrix.copy(), 1.0
    elif method == 'max':
        max_dist = distance_matrix.values.max()
        if max_dist > 0:
            return distance_matrix / max_dist, max_dist
        return distance_matrix.copy(), 1.0
    elif method == 'genome':
        # TB genome is ~4.4 million bp, but for SNP distances
        # we typically normalize by max observed or a reference
        genome_size = 4411532  # H37Rv genome size
        return distance_matrix / genome_size, genome_size
    else:
        return distance_matrix.copy(), 1.0


def constrained_neighbor_joining(distance_matrix, constraint_trees=None, outgroup="ERR4872250"):
    """
    NJ with cleaner branch length formatting.
    """
    samples = list(distance_matrix.index)
    n = len(samples)
    
    if n < 3:
        if n == 2:
            d = distance_matrix.iloc[0, 1]
            return f"({samples[0]}:{d:.4f},{samples[1]}:{d:.4f});"
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
        
        # Use 3 decimal places for cleaner output
        new_node_str = f"({tree_strs[idx_i]}:{dist_i:.3f},{tree_strs[idx_j]}:{dist_j:.3f})"
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
        final_tree = f"({tree_strs[idx_i]}:{dist:.3f},{tree_strs[idx_j]}:{dist:.3f});"
    else:
        final_tree = f"({tree_strs[active[0]]}:0);"
    
    if DENDROPY_AVAILABLE and outgroup in samples:
        try:
            taxa = dendropy.TaxonNamespace(samples)
            tree = dendropy.Tree.get(data=final_tree, schema="newick", taxon_namespace=taxa)
            tree, rooted = root_tree_by_outgroup(tree, outgroup)
            if rooted:
                return tree.as_string(schema="newick")
        except Exception as e:
            print(f"Could not root NJ tree: {e}")
    
    return final_tree


def merge_trees_improved(tree_files, global_matrix, anchors, output_file, outgroup="ERR4872250", normalize='max'):
    print("=" * 60)
    print("Tree Merging Algorithm")
    print("=" * 60)
    
    # Normalize distances for tree building
    working_matrix, norm_factor = normalize_for_tree(global_matrix, method=normalize)
    
    print(f"Normalization: {normalize} (factor: {norm_factor:.2f})")
    print(f"Normalized distance range: {working_matrix.values.min():.6f} - {working_matrix.values.max():.6f}")
    
    all_samples = list(global_matrix.index)
    print(f"Total samples: {len(all_samples)}")
    print(f"Outgroup for rooting: {outgroup}")
    
    constraint_trees = None
    
    if DENDROPY_AVAILABLE:
        try:
            trees, taxa = load_trees_dendropy(tree_files)
            if trees:
                constraint_trees = trees
                print(f"Using {len(trees)} input trees as topological constraints")
        except Exception as e:
            print(f"Could not load constraint trees: {e}")
    
    newick_tree = constrained_neighbor_joining(working_matrix, constraint_trees, outgroup)
    
    with open(output_file, 'w') as f:
        f.write(newick_tree)
    
    print(f"Merged tree saved to {output_file}")
    return newick_tree, norm_factor


def merge_trees_dendropy_nj(tree_files, global_matrix, anchors, output_file, outgroup="ERR4872250", normalize='max'):
    print("Building merged tree using DendroPy NJ")
    print(f"Outgroup for rooting: {outgroup}")
    
    # Normalize distances for tree building
    working_matrix, norm_factor = normalize_for_tree(global_matrix, method=normalize)
    
    all_samples = list(working_matrix.index)
    print(f"Total samples: {len(all_samples)}")
    print(f"Normalization: {normalize} (factor: {norm_factor:.2f})")
    print(f"Normalized distance range: {working_matrix.values.min():.6f} - {working_matrix.values.max():.6f}")
    
    taxa = dendropy.TaxonNamespace(all_samples)
    pdm = dendropy.PhylogeneticDistanceMatrix(taxon_namespace=taxa)
    
    for i, t1 in enumerate(taxa):
        for j, t2 in enumerate(taxa):
            pdm[t1, t2] = float(working_matrix.iloc[i, j])
    
    tree = pdm.nj_tree()
    
    if outgroup in all_samples:
        tree, rooted = root_tree_by_outgroup(tree, outgroup)
        if not rooted:
            print("Using midpoint rooting")
            tree, _ = root_tree_midpoint(tree)
    else:
        print(f"Outgroup {outgroup} not in samples, using midpoint rooting")
        tree, _ = root_tree_midpoint(tree)
    
    tree.write(path=output_file, schema="newick")
    
    print(f"Merged tree saved to {output_file}")
    return tree.as_string(schema="newick"), norm_factor


def calculate_merge_statistics(tree_files, output_file, global_matrix, outgroup):
    stats = {
        "num_input_trees": len(tree_files),
        "input_tree_sizes": [count_tree_tips(f) for f in tree_files],
        "merged_tree_size": count_tree_tips(output_file),
        "method": "neighbor_joining",
        "rooting_method": "outgroup",
        "outgroup": outgroup,
        "matrix_size": len(global_matrix),
        "max_distance": float(global_matrix.values.max()),
        "min_distance": float(global_matrix.values[global_matrix.values > 0].min()) if (global_matrix.values > 0).any() else 0,
        "dendropy_available": DENDROPY_AVAILABLE
    }
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Merge phylogenetic trees using NJ algorithm with outgroup rooting"
    )
    parser.add_argument('--trees', nargs='+', required=True, help="Input tree files")
    parser.add_argument('--matrix', type=str, required=True, help="Global distance matrix")
    parser.add_argument('--mapping', type=str, required=True, help="Sample mapping JSON")
    parser.add_argument('--anchors', type=str, required=True, help="Comma-separated anchor samples")
    parser.add_argument('--output', type=str, default="global_tree.nwk", help="Output tree file")
    parser.add_argument('--stats', type=str, default="merge_stats.json", help="Output stats file")
    parser.add_argument('--outgroup', type=str, default="ERR4872250", 
                        help="Outgroup for rooting (default: ERR4872250, Lineage 5)")
    parser.add_argument('--normalize', type=str, default='max', choices=['none', 'max', 'genome'],
                        help="Distance normalization method (default: max)")
    args = parser.parse_args()
    
    anchors = [a.strip() for a in args.anchors.split(',')]
    global_matrix = load_distance_matrix(args.matrix)
    
    with open(args.mapping, 'r') as f:
        mapping = json.load(f)
    
    print(f"Loaded {len(args.trees)} trees")
    print(f"Global matrix: {len(global_matrix)} samples")
    print(f"Anchors: {anchors}")
    print(f"Outgroup: {args.outgroup}")
    print(f"Normalization: {args.normalize}")
    print(f"DendroPy available: {DENDROPY_AVAILABLE}")
    
    norm_factor = 1.0
    
    if DENDROPY_AVAILABLE:
        try:
            _, norm_factor = merge_trees_dendropy_nj(
                args.trees, global_matrix, anchors, args.output, args.outgroup, args.normalize
            )
        except Exception as e:
            print(f"DendroPy NJ failed: {e}, using custom implementation")
            _, norm_factor = merge_trees_improved(
                args.trees, global_matrix, anchors, args.output, args.outgroup, args.normalize
            )
    else:
        _, norm_factor = merge_trees_improved(
            args.trees, global_matrix, anchors, args.output, args.outgroup, args.normalize
        )
    
    stats = calculate_merge_statistics(args.trees, args.output, global_matrix, args.outgroup)
    stats["anchors"] = anchors
    stats["input_files"] = [os.path.basename(f) for f in args.trees]
    stats["normalization_method"] = args.normalize
    stats["normalization_factor"] = norm_factor
    
    with open(args.stats, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Statistics saved to {args.stats}")


if __name__ == "__main__":
    main()
