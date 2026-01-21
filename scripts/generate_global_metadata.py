#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import json
import re
import pandas as pd


def get_tree_tips(tree_file):
    with open(tree_file, 'r') as f:
        tree_str = f.read().strip()
    
    tips = re.findall(r'[(,]([A-Za-z0-9_]+)[):,]', tree_str)
    
    if not tips:
        tips = re.findall(r'([A-Za-z0-9_]+):', tree_str)
    
    return list(set(tips))


def main():
    parser = argparse.ArgumentParser(
        description="Generate global metadata for federated analysis"
    )
    parser.add_argument(
        '--mapping',
        type=str,
        required=True,
        help="Sample mapping JSON file"
    )
    parser.add_argument(
        '--tree',
        type=str,
        required=True,
        help="Merged tree file"
    )
    parser.add_argument(
        '--output',
        type=str,
        default="global_metadata.tsv",
        help="Output metadata file"
    )
    args = parser.parse_args()
    
    with open(args.mapping, 'r') as f:
        mapping = json.load(f)
    
    tree_samples = get_tree_tips(args.tree)
    
    records = []
    anchors = set(mapping.get('anchors', []))
    
    for sample in mapping.get('all_samples', tree_samples):
        source_lab = "Unknown"
        for lab_name, lab_samples in mapping.get('lab_samples', {}).items():
            if sample in lab_samples:
                source_lab = lab_name
                break
        
        is_anchor = sample in anchors
        
        records.append({
            'sample_id': sample,
            'source_lab': source_lab,
            'is_anchor': is_anchor,
            'patient_id': 'Reference' if is_anchor else 'NA',
            'latitude': 'NA',
            'longitude': 'NA',
            'conclusion': 'Anchor Reference' if is_anchor else 'Federated Sample'
        })
    
    df = pd.DataFrame(records)
    df.to_csv(args.output, sep='\t', index=False)
    print("Global metadata saved to {0}".format(args.output))
    print("Total samples: {0}".format(len(df)))
    print("Anchor samples: {0}".format(sum(1 for r in records if r['is_anchor'])))


if __name__ == "__main__":
    main()