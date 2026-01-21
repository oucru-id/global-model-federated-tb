#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

log.info """
    Mycobacterium tuberculosis Federated Phylogeny (Central Node) Version: ${params.version}
    Developed by SPHERES Lab Team
"""

include { FEDERATED_MERGE }  from './workflows/federated_merge.nf'
include { VISUALIZATION }    from './workflows/visualization.nf'
include { VERSIONS }         from './workflows/utils.nf'

workflow {
    matrix_ch = Channel.fromPath("${params.matrix_dir}/*.tsv", checkIfExists: true)
        .collect()
    
    tree_ch = Channel.fromPath("${params.tree_dir}/*.nwk", checkIfExists: true)
        .collect()
    
    FEDERATED_MERGE(
        matrix_ch,
        tree_ch,
        params.anchor_samples,
        params.primary_anchor
    )
    
    VISUALIZATION(
        FEDERATED_MERGE.out.matrix,
        FEDERATED_MERGE.out.metadata,
        FEDERATED_MERGE.out.tree
    )
    
    VERSIONS()
}