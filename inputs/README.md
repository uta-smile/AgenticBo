
# Protein sequence inputs

Put one single-chain protein per FASTA file here, for example `target_01.fasta`:

```fasta
>target_01
YOUR_AMINO_ACID_SEQUENCE
```

Use actual standard one-letter amino acid codes (the line above is a placeholder).
The FASTA header need not use Boltz's special syntax. The preparation command
converts it to a Boltz2 YAML input with a fixed chain ID.

Register each file in `data/targets/manifest.csv`. Reference structures go in
`reference/`. Development targets calibrate the radius; test targets remain held
out. The POC calls for five test proteins of 100–200 residues, plus separate
development data. No benchmark targets have been supplied yet.

An optional local `.a3m` path in the manifest fixes MSA conditioning. A blank MSA
field explicitly selects single-sequence mode; no sequence is sent to a server.
