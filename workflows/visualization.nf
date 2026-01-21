nextflow.enable.dsl = 2

process VISUALIZE_REPORT {
    publishDir "${params.results_dir}/visualization", mode: 'copy'
    
    input:
    path matrix
    path metadata
    path tree

    output:
    path "transmission_network.html", emit: network
    path "stats_histogram.png",       emit: hist
    path "stats_heatmap.png",         emit: heatmap
    path "stats_violin.png",          emit: violin
    path "phylo_tree_rectangular.png",    emit: tree_rect
    path "phylo_tree_unrooted.png", emit: tree_unrooted
    path "phylo_tree_circular.png", emit: tree_circ

    script:
    """
    ${params.python} $baseDir/scripts/visualize_results.py \\
        --matrix ${matrix} \\
        --metadata ${metadata} \\
        --tree ${tree} \\
        --threshold ${params.snp_threshold}
    """
}

workflow VISUALIZATION {
    take:
    matrix
    metadata
    tree

    main:
    VISUALIZE_REPORT(matrix, metadata, tree)
}