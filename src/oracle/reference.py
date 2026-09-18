"""Explicit mmCIF polymer mapping for references with unresolved residues."""
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.SeqUtils import seq1

from oracle.validation import ProteinStructure


def read_resolved_reference(path: Path, chain_id: str, sequence: str):
    if path.suffix.lower() not in {".cif", ".mmcif"}:
        raise ValueError("Resolved-reference mode requires mmCIF with polymer sequence numbering")
    data=MMCIF2Dict(str(path))
    entities=data.get("_entity_poly.entity_id",[])
    chains=data.get("_entity_poly.pdbx_strand_id",[])
    sequences=data.get("_entity_poly.pdbx_seq_one_letter_code_can",[])
    matches=[i for i,c in enumerate(chains) if chain_id in [v.strip() for v in c.split(",")]]
    if len(matches)!=1 or len(entities)!=len(chains) or len(sequences)!=len(chains):
        raise ValueError("Reference needs an unambiguous declared polymer sequence for this chain")
    declared="".join(sequences[matches[0]].split())
    if declared!=sequence:
        raise ValueError("Declared reference polymer sequence differs from FASTA")
    # label_seq_id is the position in the declared polymer, unlike author numbering.
    structure=MMCIFParser(QUIET=True,auth_residues=False).get_structure("reference",str(path))
    models=list(structure)
    if len(models)!=1 or chain_id not in models[0]:
        raise ValueError("Reference must have one model with the requested chain")
    atoms,ca,indices,names,alternates={},[],[],[],{}
    for residue in models[0][chain_id]:
        if residue.id[0]!=" ":
            continue
        if residue.is_disordered()==2 or residue.id[2].strip():
            raise ValueError("Ambiguous polymer residue identity")
        position=residue.id[1]-1
        if not 0<=position<len(sequence) or seq1(residue.resname)!=sequence[position]:
            raise ValueError("Resolved residue differs from its declared polymer position")
        if "CA" not in residue:
            continue
        if indices and position<=indices[-1]:
            raise ValueError("Resolved polymer positions must be unique and increasing")
        labels=sorted({child.altloc for atom in residue if atom.is_disordered()
                       for child in atom.disordered_get_list() if child.altloc.strip()})
        # Choose one conformer per residue, avoiding a mixture of A and B atoms.
        def conformer_score(label):
            values=[atom.child_dict.get(label,atom.child_dict.get(" "))
                    for atom in residue if atom.is_disordered()]
            return (sum(v is not None for v in values),sum((v.occupancy or 0.) for v in values if v is not None))
        chosen=max(labels,key=conformer_score) if labels else None
        index=len(indices)
        for atom in residue:
            if atom.is_disordered():
                atom=atom.child_dict.get(chosen,atom.child_dict.get(" "))
                if atom is None:
                    raise ValueError("Selected alternate conformer lacks a reference atom")
            coordinates=np.asarray(atom.coord,dtype=np.float64)
            if not np.isfinite(coordinates).all():
                raise ValueError("Nonfinite reference coordinates")
            if atom.element not in {"H","D"}:
                atoms[index,atom.name]=coordinates
        ca.append(atoms[index,"CA"])
        indices.append(position)
        names.append(residue.resname)
        if chosen is not None:
            alternates[str(position+1)]=chosen
    if len(indices)<3:
        raise ValueError("Reference requires at least three resolved C-alpha positions")
    subset="".join(sequence[i] for i in indices)
    info={"mode":"resolved","full_sequence_length":len(sequence),"resolved_residues":len(indices),
          "coverage":len(indices)/len(sequence),"tm_normalization_length":len(indices),
          "scored_sequence_positions":[i+1 for i in indices],
          "unscored_sequence_positions":[i+1 for i in range(len(sequence)) if i not in set(indices)],
          "alternate_conformers":alternates,
          "alternate_policy":"maximum atom coverage then summed occupancy per residue; alphabetical tie break"}
    return ProteinStructure(subset,np.asarray(ca),atoms,tuple(names)),tuple(indices),info


def select_residues(model: ProteinStructure, indices: tuple[int,...]):
    reverse={old:new for new,old in enumerate(indices)}
    return ProteinStructure("".join(model.sequence[i] for i in indices),model.ca[list(indices)],
        {(reverse[i],name):coord for (i,name),coord in model.atoms.items() if i in reverse},
        tuple(model.residue_names[i] for i in indices))
