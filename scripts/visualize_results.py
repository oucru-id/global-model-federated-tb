#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np
import math
from Bio import Phylo


def generate_network(df, meta_df, threshold, output_file):
    G = nx.Graph()
    
    for sample in df.index:
        meta = meta_df[meta_df['sample_id'] == sample]
        
        if not meta.empty:
            m = meta.iloc[0]
            title_html = f"<b>ID:</b> {sample}"
        else:
            title_html = sample

        G.add_node(sample, title=title_html, label=sample)
    
    samples = df.index.tolist()
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            s1 = samples[i]
            s2 = samples[j]
            dist = df.iloc[i, j]
            
            if dist <= threshold:
                weight = 1.0 / (dist + 1)
                G.add_edge(s1, s2, weight=weight, title=f"{dist:.1f} SNPs", label=f"{dist:.0f}")

    net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black")
    net.from_nx(G)
    
    net.set_options("""
    var options = {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -8000,
          "springLength": 250,
          "springConstant": 0.04
        }
      }
    }
    """)
    
    net.write_html(output_file)


def generate_plots(df, output_prefix):
    mask = np.triu(np.ones_like(df, dtype=bool), k=1)
    distances = df.where(mask).stack().values
    
    # Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(distances, bins=20, kde=True, color="skyblue")
    plt.title("Distribution of SNP Distances")
    plt.xlabel("SNP Distance")
    plt.ylabel("Frequency")
    plt.savefig(f"{output_prefix}_histogram.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Heatmap
    n_samples = df.shape[0]
    fig_dim = max(12, n_samples * 0.6)
    plt.figure(figsize=(fig_dim, fig_dim))
    font_size = 10 if n_samples < 20 else 8
    
    sns.heatmap(
        df, 
        cmap="viridis", 
        annot=True, 
        fmt=".0f",
        square=True,
        cbar_kws={"shrink": 0.8, "label": "SNP Distance"},
        annot_kws={"size": font_size}
    )
    
    plt.title("SNP Distance Heatmap")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Violin Plot
    plt.figure(figsize=(8, 6))
    sns.violinplot(y=distances, color="lightblue", inner="quartile")
    plt.title("Distribution of SNP Distances")
    plt.ylabel("SNP Distance")
    plt.savefig(f"{output_prefix}_violin.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_rectangular_tree(tree, output_file):
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_visible(False)

    def get_branch_label(clade):
        if clade.branch_length and clade.branch_length > 0.001:
            return f"{clade.branch_length:.3f}"
        return None

    Phylo.draw(
        tree, 
        axes=ax, 
        do_show=False, 
        show_confidence=False,
        label_func=lambda x: x.name if x.is_terminal() else "",
        branch_labels=get_branch_label,
    )
    
    terminals = tree.get_terminals()
    for i, clade in enumerate(terminals):
        y_pos = i + 1
        x_pos = tree.distance(tree.root, clade)
        ax.scatter(x_pos, y_pos, color='steelblue', s=80, zorder=10, edgecolors='white', linewidth=0.5)
    
    plt.title("Phylogenetic Tree (Rectangular)", fontsize=14)
    plt.xlabel("Genetic Distance", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def get_coords(tree):
    coords = {}
    leaves = tree.get_terminals()
    total_leaves = len(leaves)
    
    for i, leaf in enumerate(leaves):
        angle = (2 * math.pi * i) / total_leaves
        coords[leaf] = {'theta': angle}

    for clade in tree.get_nonterminals(order='postorder'):
        children_angles = [coords[c]['theta'] for c in clade.clades]
        if children_angles:
            avg_angle = sum(children_angles) / len(children_angles)
            coords[clade] = {'theta': avg_angle}

    coords[tree.root]['r'] = 0
    for clade in tree.get_nonterminals(order='preorder'):
        parent_r = coords[clade]['r']
        for child in clade.clades:
            length = child.branch_length if child.branch_length else 0.01
            coords[child] = coords.get(child, {})
            coords[child]['r'] = parent_r + length
            
    return {k: (v['r'], v['theta']) for k, v in coords.items()}


def plot_circular_tree(tree, output_file):
    coords = get_coords(tree)
    
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='polar')
    
    ax.set_frame_on(False)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    max_r = max(r for r, t in coords.values())
    parents = {c: p for p in tree.find_clades() for c in p.clades}

    for clade in tree.find_clades(order='level'):
        if clade == tree.root:
            continue
        parent = parents.get(clade)
        if not parent:
            continue

        r1, t1 = coords[parent]
        r2, t2 = coords[clade]
        
        if abs(t1 - t2) > 0:
            theta_range = np.linspace(t1, t2, num=20)
            r_range = [r1] * len(theta_range)
            ax.plot(theta_range, r_range, color='gray', linewidth=0.5)
            
        ax.plot([t2, t2], [r1, r2], color='gray', linewidth=0.5)

    for clade, (r, theta) in coords.items():
        if clade.is_terminal():
            ax.scatter(theta, r, color='steelblue', s=40, zorder=10, edgecolors='white', linewidth=0.5)
            
            rot = math.degrees(theta)
            if 90 < rot < 270:
                rot += 180
                ha = 'right'
                label_r = r + (max_r * 0.02)
            else:
                ha = 'left'
                label_r = r + (max_r * 0.01)
                
            ax.text(theta, label_r, clade.name, rotation=rot, ha=ha, va='center', fontsize=8)
    
    plt.title("Phylogenetic Tree (Circular)", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def plot_unrooted_tree(tree, output_file):
    coords = get_coords(tree)
    
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_aspect('equal')
    ax.axis('off')
    
    cart_coords = {}
    max_r = 0
    for clade, (r, theta) in coords.items():
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        cart_coords[clade] = (x, y)
        if r > max_r:
            max_r = r

    parents = {c: p for p in tree.find_clades() for c in p.clades}

    for clade in tree.find_clades(order='level'):
        if clade == tree.root:
            continue
        parent = parents.get(clade)
        if not parent:
            continue

        x1, y1 = cart_coords[parent]
        x2, y2 = cart_coords[clade]
        ax.plot([x1, x2], [y1, y2], color='gray', linewidth=0.5)

    for clade, (x, y) in cart_coords.items():
        if clade.is_terminal():
            ax.scatter(x, y, color='steelblue', s=40, zorder=10, edgecolors='white', linewidth=0.5)
            
            r, theta = coords[clade]
            rot = math.degrees(theta)
            
            if 90 < rot < 270:
                rot += 180
                ha = 'right'
                lx = x + (max_r * 0.02) * math.cos(theta)
                ly = y + (max_r * 0.02) * math.sin(theta)
            else:
                ha = 'left'
                lx = x + (max_r * 0.01) * math.cos(theta)
                ly = y + (max_r * 0.01) * math.sin(theta)
            
            ax.text(lx, ly, clade.name, rotation=rot, ha=ha, va='center', fontsize=8, rotation_mode='anchor')
    
    plt.title("Phylogenetic Tree (Radial)", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def generate_phylo_trees(tree_file, output_prefix):
    try:
        tree = Phylo.read(tree_file, "newick")
    except Exception as e:
        print(f"Error reading tree file: {e}")
        return

    plot_rectangular_tree(tree, f"{output_prefix}_rectangular.png")
    plot_unrooted_tree(tree, f"{output_prefix}_unrooted.png")
    plot_circular_tree(tree, f"{output_prefix}_circular.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix', required=True, help="Path to distance_matrix.tsv")
    parser.add_argument('--metadata', required=True, help="Path to metadata.tsv")
    parser.add_argument('--tree', required=False, help="Path to phylo_tree.nwk")
    parser.add_argument('--threshold', type=int, default=12, help="SNP threshold")
    args = parser.parse_args()

    df = pd.read_csv(args.matrix, sep='\t', index_col=0)
    df.index.name = "Sample"
    
    meta_df = pd.read_csv(args.metadata, sep='\t')

    generate_network(df, meta_df, args.threshold, "transmission_network.html")
    generate_plots(df, "stats")
    
    if args.tree:
        generate_phylo_trees(args.tree, "phylo_tree")


if __name__ == "__main__":
    main()
