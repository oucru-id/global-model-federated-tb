nextflow.enable.dsl = 2

process VERSION_INFO {
    publishDir "${params.results_dir}", mode: 'copy'
    
    output:
    path "versions.txt"

    script:
    """
    echo "Pipeline: TB Federated Phylogeny - Central Node" > versions.txt
    echo "Version: ${params.version}" >> versions.txt
    echo "Run Date: \$(date)" >> versions.txt
    echo "" >> versions.txt
    echo "=== Tool Versions ===" >> versions.txt
    echo "Python: \$(${params.python} --version 2>&1)" >> versions.txt
    echo "Nextflow: ${nextflow.version}" >> versions.txt
    echo "" >> versions.txt
    echo "=== Python Packages ===" >> versions.txt
    ${params.python} -c "import pandas; print('pandas:', pandas.__version__)" >> versions.txt
    ${params.python} -c "import numpy; print('numpy:', numpy.__version__)" >> versions.txt
    ${params.python} -c "import dendropy; print('dendropy:', dendropy.__version__)" >> versions.txt 2>/dev/null || echo "dendropy: not installed" >> versions.txt
    ${params.python} -c "import networkx; print('networkx:', networkx.__version__)" >> versions.txt
    ${params.python} -c "import Bio; print('biopython:', Bio.__version__)" >> versions.txt
    """
}

workflow VERSIONS {
    VERSION_INFO()
}
