from torchmetrics import BinnedPrecisionRecallCurve
from typing import Any, Dict, List, Tuple, Union
from torch import Tensor
import torch
from collections import defaultdict


DEFAULT_JSON = '''{
"training_set": "Scop3P/ST/PF",
"test_set": "default",
"test_fold": 0,
"save_model": false,

"representation":"ProtTransT5_XL_UniRef50",
"freeze_representation": true,

"receptive_field": 65,
"conv_depth":3,
"conv_width":9,
"conv_channels": 200,
"final_fc_neurons":64,
"dropout":0.2,
"max_pool_size":1,
"batch_norm":false,

"batch_size":4,
"learning_rate":1e-4,
"warm_up_epochs":1.5,
"pos_weight":1,
"max_epochs":15
}'''

def _precision_at_recall(
    precision: Tensor,
    recall: Tensor,
    thresholds: Tensor,
    min_recall: float,
) -> Tuple[Tensor, Tensor]:
    """
    Quick but suboptimal solution to get precision-at-fixed-recall. Adapted from TorchMetrics code for
    BinnedRecallAtFixedPrecision. See TorchMetrics documentation for more details
    """
    try:
        max_precision, _, best_threshold = max(
            (p, r, t) for p, r, t in zip(precision, recall, thresholds) if r >= min_recall
        )

    except ValueError:
        max_precision = torch.tensor(0.0, device=precision.device, dtype=precision.dtype)
        best_threshold = torch.tensor(0)

    if max_precision == 0.0:
        best_threshold = torch.tensor(1e6, device=thresholds.device, dtype=thresholds.dtype)

    return max_precision, best_threshold

class BinnedPrecisionAtFixedRecall(BinnedPrecisionRecallCurve):
    """
    Quick but suboptimal solution to get precision-at-fixed-recall. Adapted from TorchMetrics code for
    BinnedRecallAtFixedPrecision. See TorchMetrics documentation for more details
    """
    def __init__(
        self,
        num_classes: int,
        min_recall: float,
        thresholds: Union[int, Tensor, List[float]] = 100,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(num_classes=num_classes, thresholds=thresholds, **kwargs)
        self.min_recall = min_recall

    def compute(self) -> Tuple[Tensor, Tensor]:  # type: ignore
        """Returns float tensor of size n_classes."""
        precisions, recalls, thresholds = super().compute()

        if self.num_classes == 1:
            return _precision_at_recall(precisions, recalls, thresholds, self.min_recall)

        precisions_at_p = torch.zeros(self.num_classes, device=recalls[0].device, dtype=recalls[0].dtype)
        thresholds_at_p = torch.zeros(self.num_classes, device=thresholds[0].device, dtype=thresholds[0].dtype)
        for i in range(self.num_classes):
            precisions_at_p[i], thresholds_at_p[i] = _precision_at_recall(
                precisions[i], recalls[i], thresholds[i], self.min_recall
            )
        return precisions_at_p, thresholds_at_p

def get_gpu_max_batchsize(representation: str, freeze_representation: bool):
    """
    Manually set an upper limit to the batch size to be used on your system. This depends on the representation type.
    This is not optimized, so please modify this to fit your own system.

    Parameters
    ----------
    representation : str
        The used protein language model

    freeze_representation : bool
        If True, the language model weights will not be fine-tuned, and thus less GPU memory is needed
    """
    # just a fixed size, change this function at will
    return 16

def digest_protein(sequence, missed_cleavages=0, min_length=0, max_length=None):
    """
    Performs trypsin digestion on a sinlge protein sequence.

    **Input**:
    PEPTIDERPRRPEPTIDE
    **Output**:
    [('PEPTIDER', 0), ('PEPTIDERR', 0),..., ('RPEPTIDE', 9)]
    """
    cleavage_sites = []
    for i in range(len(sequence) - 1):
        if sequence[i] in ('K', 'R'):
            cleavage_sites.append(i+1)
    cleavage_sites = [0] + cleavage_sites + [len(sequence)]
    peptides = []
    for start_idx in range(len(cleavage_sites)-1):
        for end_idx in range(start_idx+1, min(len(cleavage_sites), start_idx+missed_cleavages+2)):
            pep_start = cleavage_sites[start_idx]
            pep_end = cleavage_sites[end_idx]
            peptide = sequence[pep_start:pep_end]
            if min_length <= len(peptide) <= (max_length if max_length else len(peptide)):
                peptides.append((peptide, pep_start))
    return peptides # [(peptide, start_pos), ...]

def trypsin_digest(sequences, missed_cleavages, min_length, max_length):
    """
    Performs trypsin digestion on all protein sequences.
    
    **Input**:
    {prot1: 'PEPTIDERRRPEPTIDE...', ...}
    **Output**:
    [{'prot_id': prot1, 'sequence': PEPTIDERPR ,'miscleavages': 0,'peptide_length': 10,'start_pos': 0}, ...]
    """
    digested_peptides = []  # List of dicts with peptide info
    for prot_id, seq in sequences.items():
        peptides = digest_protein(seq, missed_cleavages, min_length, max_length)
        for peptide, start_pos in peptides:
            digested_peptides.append({
                'prot_id': prot_id,
                'sequence': peptide,
                'peptide_length': len(peptide),
                'start_pos': start_pos
            })
    return digested_peptides

def annotate_proteome_with_target_residues(
    peptides: List[Dict],
    proteomes: Dict[str, str],
    target_residues: str,
    PTM: str
) -> Dict[str, str]:
    """
    Annotates proteome sequences with '#' after target residues that appear in peptides.
    Only returns proteins that contain at least one annotated target residue.
    
    Parameters:
    -----------
    peptides : List[Dict]
        List of peptide dictionaries containing:
        - prot_id: protein identifier
        - sequence: peptide sequence
        - start_pos: start position in protein
        - peptide_length: length of peptide
    proteomes : Dict[str, str]
        Dictionary mapping protein IDs to their sequences
    target_residues : str
        String containing target amino acid residues (e.g., 'ST' for Serine and Threonine)
    
    Returns:
    --------
    Dict[str, str]
        Dictionary mapping protein IDs to their annotated sequences,
        containing only proteins with at least one annotated target residue
    
    Example:
    --------
    peptides = [{'prot_id': 'A0A075B706', 'sequence': 'TDKLIFGK', 'start_pos': 0}]
    proteomes = {
        'A0A075B706': 'TDKLIFGKGTRVTVEP',
        'OTHER_PROT': 'MKLNPYAAA'  # No target residues from peptides
    }
    target_residues = 'ST'
    
    Returns:
    {'A0A075B706': 'T#DKLIFGKGT#RVT#VEP'}  # OTHER_PROT is not included
    """
    # Dictionary to track which positions need to be annotated for each protein
    positions_to_annotate = defaultdict(set)
    
    # First pass: identify all positions that need annotation
    for peptide in peptides:
        prot_id = peptide['prot_id']
        if prot_id not in proteomes:
            continue
            
        peptide_seq = peptide['sequence']
        start_pos = peptide['start_pos']
        
        # Find all target residues in the peptide
        for i, residue in enumerate(peptide_seq):
            if residue in target_residues:
                absolute_pos = start_pos + i
                positions_to_annotate[prot_id].add(absolute_pos)
    
    # Create annotated sequences only for proteins with target residues
    annotated_proteomes = {}
    for prot_id, positions in positions_to_annotate.items():
        # Skip if no positions to annotate
        if not positions:
            continue
            
        original_seq = proteomes[prot_id]
        # Convert sequence to list for easier manipulation
        seq_list = list(original_seq)
        
        # Insert annotations after all marked positions
        # Sort positions in reverse order to avoid affecting subsequent insertions
        for pos in sorted(positions, reverse=True):
            # Verify the position exists and contains a target residue
            if pos < len(seq_list) and seq_list[pos] in target_residues:
                seq_list.insert(pos + 1, '#')
        
        # Only add to annotated_proteomes if we actually added annotations
        annotated_seq = ''.join(seq_list)
        if '#' in annotated_seq:  # Double-check that annotations were actually added
            annotated_proteomes[prot_id] = annotated_seq
    
    # Log statistics
    total_proteins = len(proteomes)
    annotated_proteins = len(annotated_proteomes)
    print(f"Found target residues in {annotated_proteins} out of {total_proteins} proteins for {PTM}")
    print(f"Total annotated positions across all proteins for {PTM}: {sum(len(pos) for pos in positions_to_annotate.values())}")
    
    return annotated_proteomes

def annotate_full_proteome(sequences: Dict[str, str], target_residues: str,  PTM: str) -> Tuple[Dict[str, str], int]:
   """
   Annotates proteome sequences with '#' after target residues.
   
   Parameters
   ----------
   sequences : Dict[str, str]
       Dictionary mapping protein IDs to their sequences
   target_residues : str
       String containing target amino acid residues (e.g., 'ST' for Serine and Threonine)
       
   Returns
   -------
   Tuple[Dict[str, str], int]
       - Dictionary mapping protein IDs to their annotated sequences
       - Total number of target sites annotated
   """
   annotated_sequences = {}
   total_target_sites = 0
   for prot_id, seq in sequences.items():
       seq_list = list(seq)
       target_sites_in_protein = 0
       for i, residue in enumerate(seq_list):
           if residue in target_residues:
               seq_list.insert(i + 1, '#')
               target_sites_in_protein += 1
       total_target_sites += target_sites_in_protein
       annotated_sequences[prot_id] = ''.join(seq_list)
   
   print(f"Total number of {PTM} sites annotated: {total_target_sites}")
   return annotated_sequences

def read_proteome(canonical_proteome):
    """
    Reads the proteome sequences from the FASTA file, filters X and U aa from proteome

    Parameters:
    canonical_proteome : str
        Path to the FASTA file containing the proteome sequences

    Returns:
    dict
        A dictionary where keys are protein IDs and values are sequences
        {prot1: 'PEPTIDERRRPEPTIDE...', ...}
    """
    sequences = {}  # {prot_id: sequence}
    with open(canonical_proteome, 'r') as f:
        prot_id = None
        seq_lines = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if prot_id:
                    sequence = ''.join(seq_lines)
                    if 'U' not in sequence and 'X' not in sequence:
                        sequences[prot_id] = sequence
                    seq_lines = []
                prot_id = line[1:].split()[0]
            else:
                seq_lines.append(line)
        if prot_id:
            sequence = ''.join(seq_lines)
            if 'U' not in sequence and 'X' not in sequence:
                sequences[prot_id] = sequence
    return sequences
