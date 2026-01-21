nextflow.enable.dsl = 2

process MERGE_MATRICES {
    publishDir "${params.results_dir}/federated", mode: 'copy'
    
    input:
    path matrix_files
    val anchors
    val primary_anchor

    output:
    path "global_distance_matrix.tsv", emit: global_matrix
    path "correction_report.json",     emit: report
    path "samples_mapping.json",       emit: mapping

    script:
    """
    ${params.python} $baseDir/scripts/merge_matrices.py \\
        --matrices ${matrix_files} \\
        --anchors "${anchors}" \\
        --primary-anchor "${primary_anchor}" \\
        --output global_distance_matrix.tsv \\
        --report correction_report.json \\
        --mapping samples_mapping.json
    """
}

process MERGE_TREES {
    publishDir "${params.results_dir}/federated", mode: 'copy'
    
    input:
    path tree_files
    path global_matrix
    path mapping
    val anchors

    output:
    path "global_tree.nwk",        emit: tree
    path "merge_stats.json",       emit: stats

    script:
    """
    ${params.python} $baseDir/scripts/merge_trees.py \\
        --trees ${tree_files} \\
        --matrix ${global_matrix} \\
        --mapping ${mapping} \\
        --anchors "${anchors}" \\
        --output global_tree.nwk \\
        --stats merge_stats.json
    """
}

process GENERATE_GLOBAL_METADATA {
    publishDir "${params.results_dir}/federated", mode: 'copy'
    
    input:
    path mapping
    path tree

    output:
    path "global_metadata.tsv", emit: metadata

    script:
    """
    ${params.python} $baseDir/scripts/generate_global_metadata.py \\
        --mapping ${mapping} \\
        --tree ${tree} \\
        --output global_metadata.tsv
    """
}

workflow FEDERATED_MERGE {
    take:
    matrix_files
    tree_files
    anchors
    primary_anchor

    main:
    MERGE_MATRICES(matrix_files, anchors, primary_anchor)
    MERGE_TREES(
        tree_files, 
        MERGE_MATRICES.out.global_matrix, 
        MERGE_MATRICES.out.mapping, 
        anchors
    )
    GENERATE_GLOBAL_METADATA(MERGE_MATRICES.out.mapping, MERGE_TREES.out.tree)

    emit:
    matrix   = MERGE_MATRICES.out.global_matrix
    tree     = MERGE_TREES.out.tree
    metadata = GENERATE_GLOBAL_METADATA.out.metadata
    report   = MERGE_MATRICES.out.report
    stats    = MERGE_TREES.out.stats
}