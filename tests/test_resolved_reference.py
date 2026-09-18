from pathlib import Path
import numpy as np
import pytest
from oracle.reference import read_resolved_reference,select_residues
from oracle.validation import ProteinStructure
from oracle.oracle import StructuralOracle


def test_22pe_full_sequence_mapping_and_alternates(monkeypatch):
    sequence=''.join(line.strip() for line in Path('inputs/22PE.fasta').read_text().splitlines() if not line.startswith('>'))
    reference,indices,info=read_resolved_reference(Path('reference/22PE.cif'),'A',sequence)
    assert len(sequence)==349 and indices==tuple(range(2,344))
    assert reference.sequence==sequence[2:344]
    assert info['unscored_sequence_positions']==[1,2,345,346,347,348,349]
    assert info['tm_normalization_length']==342
    assert len(info['alternate_conformers'])==15
    # Unresolved positions are excluded by polymer index, not sequence truncation.
    ca=np.full((349,3),1e6)
    ca[list(indices)]=reference.ca
    atoms={(indices[i],name):coord for (i,name),coord in reference.atoms.items()}
    names=['UNK']*349
    for i,name in zip(indices,reference.residue_names):
        names[i]=name
    full=ProteinStructure(sequence,ca,atoms,tuple(names))
    selected=select_residues(full,indices)
    np.testing.assert_array_equal(selected.ca,reference.ca)
    assert selected.sequence==reference.sequence and selected.atoms.keys()==reference.atoms.keys()
    oracle=StructuralOracle(Path('reference/22PE.cif'),sequence,'A','A',objective='tm',reference_mode='resolved')
    # A partial reference still cannot masquerade as a complete generated protein.
    assert not oracle.score(Path('reference/22PE.cif'))['valid']
    monkeypatch.setattr('oracle.oracle.read_protein',lambda *args: full)
    score=oracle.score(Path('generated_fixture.pdb'))
    assert score['valid'] and score['tm']==pytest.approx(1,abs=1e-6)
    assert score['objective']==score['tm'] and score['lddt']==1
    with pytest.raises(ValueError,match='polymer sequence differs'):
        read_resolved_reference(Path('reference/22PE.cif'),'A','A'+sequence[1:])
