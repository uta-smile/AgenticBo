# Ground-truth structures

Place reference `.cif`, `.mmcif`, or `.pdb` files here, and record the path and
reference chain ID in `data/targets/manifest.csv`. File names should correspond
to the FASTA files in `inputs/`.

Use single-chain proteins with complete Cα coordinates and a sequence matching
the FASTA. Reference coordinates are reserved for the scoring oracle; they must
never be supplied as Boltz templates or generation conditioning.
