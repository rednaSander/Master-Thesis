import os 
# Suppress UserWarnings from the pyopenms module
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:pyopenms"

import multiprocessing
#multiprocessing.set_start_method('forkserver', force=True) # Needed is you want to use mutltiprocessing on the server

from ms2pip.core import predict_batch, read_psms, _Parallelized
from ms2pip.core import _into_batches
from ms2pip._utils.retention_time import RetentionTime
from ms2pip._utils.ion_mobility import IonMobility
from ms2pip._utils.xgb_models import validate_requested_xgb_model
from ms2pip._utils.encoder import Encoder
import xgboost as xgb
from ms2pip.result import ProcessingResult
from psm_utils import PSMList, Peptidoform, PSM
from ms2pip.spectrum_output import write_spectra
from typing import Any, Callable, Generator, Iterable, List, Optional, Tuple, Union, Dict
from pathlib import Path
from rich.progress import track
from itertools import combinations 
import pandas as pd
import re
from collections import defaultdict
from tqdm import tqdm
import random
import argparse
import json


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
        if sequence[i] in ('K', 'R') and sequence[i+1] != 'P':
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

def calculate_miscleavages(peptide_seq, protein_seq, start_pos):
    """
    Calculates the number of missed cleavages in a single peptide.

    **Input**:
    (peptide_seq = 'PEPTIDERPR', protein_seq = 'PEPTIDERPRRPEPTIDE', start_pos=0)
    **Output**:
    0
    """
    miscleavages = 0
    for i in range(1, len(peptide_seq)):
        abs_pos = start_pos + i -1  # Position in protein sequence
        if abs_pos+1 >= len(protein_seq):
            continue
        if protein_seq[abs_pos] in ('K', 'R') and protein_seq[abs_pos+1] != 'P':
            miscleavages +=1
    return miscleavages

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
            miscleavages = calculate_miscleavages(peptide, seq, start_pos)
            digested_peptides.append({
                'prot_id': prot_id,
                'sequence': peptide,
                'miscleavages': miscleavages,
                'peptide_length': len(peptide),
                'start_pos': start_pos
            })
    return digested_peptides

def filter_prediction(csv_file, PTM, threshold, control=False):
    """
    Filters predictions from a CSV file based on a threshold and adds 'residue' and 'label' fields.
    If control is True, returns random entries instead of those filtered by the threshold.
    
    Parameters:
    - csv_file (str): Path to the input CSV file.
    - PTM (str): The PTM type to extract the residue and label information.
    - threshold (float): The threshold value for filtering the 'pred' column.
    - control (bool): If True, returns random entries instead of filtered ones.
    
    Returns:
    - List[dict]: A list of dictionaries with keys 'prot_id', 'position', 'pred', 'residue', and 'label'.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    original_row_count = len(df)
    
    # Filter the DataFrame based on the threshold
    filtered_df = df[df['pred'] >= threshold].copy()
    filtered_row_count = len(filtered_df)
    
    # Parse the PTM to get 'residue' and 'label'
    if '[' in PTM and ']' in PTM:
        residue, label = PTM.split('[')
        label = label.replace(']', '')
    else:
        raise ValueError(f"Invalid PTM format: {PTM}. Expected format 'residue[label]'.")
    
    # Create the list of dictionaries to return
    result = []
    
    if control:
        # If control is True, randomly sample from the entire dataframe
        random_sample = df.sample(n=filtered_row_count, random_state=42)  # For reproducibility, random_state can be set
        for _, row in random_sample.iterrows():
            entry = {
                'prot_id': row['prot_id'],
                'position': row['position'],
                'pred': row['pred'],
                'residue': residue,
                'label': label
            }
            result.append(entry)
        print(f"Control Library: {filtered_row_count} random entries sampled from {original_row_count} rows.")
    else:
        # Regular filtering and return
        for _, row in filtered_df.iterrows():
            entry = {
                'prot_id': row['prot_id'],
                'position': row['position'],
                'pred': row['pred'],
                'residue': residue,
                'label': label
            }
            result.append(entry)
        print(f"{filtered_row_count} / {original_row_count} rows retained after filtering for {PTM}")
    
    return result

def add_target_ptms(peptides, sequences, target_ptms):
    updated_peptides = []
    # Create a dictionary for quick lookup of PTMs by (prot_id, position)
    ptm_dict = {}  # {(prot_id, position): [{'residue': 'Y', 'label': '[Phospho]'}, ...]}
    for ptm in target_ptms:
        key = (ptm['prot_id'], ptm['position'] - 1)  # -1 to correct for zero-based indexing
        ptm_info = {'residue': ptm['residue'], 'label': ptm['label']}
        ptm_dict.setdefault(key, []).append(ptm_info)

    # Keep track of PTMs that could not be mapped
    unmapped_ptms = set(ptm_dict.keys())
    total_ptms = len(unmapped_ptms)

    # Iterate over each peptide
    for peptide_info in peptides:
        prot_id = peptide_info['prot_id']
        seq = peptide_info['sequence']
        start_pos = peptide_info['start_pos']  # Start position in the protein sequence (0-based index)
        peptide_length = peptide_info['peptide_length']
        # Initialize residues with their amino acids and empty mods list
        residues = [{'residue': aa, 'mods': []} for aa in seq]
        
        # Retrieve the full protein sequence for cross-reference checking of residues at absolute positions
        protein_seq = sequences[prot_id]
        for i, res in enumerate(residues):
            abs_pos = start_pos + i 
            key = (prot_id, abs_pos)
            if key in ptm_dict:
                # Retrieve the PTM information
                ptm_infos = ptm_dict[key]

                # Cross-reference checking: Verify that the residue matches the one in the protein sequence
                protein_residue = protein_seq[abs_pos]
                if res['residue'] == protein_residue:
                    # Iterate over all PTMs at this position
                    for ptm_info in ptm_infos:
                        # Check if the residue matches any of the specified residues in ptm_info['residue']
                        if res['residue'] in ptm_info['residue']:
                            # Add the PTM label to the mods list
                            res['mods'].append(ptm_info['label'])
                            # Remove this PTM from unmapped_ptms
                            unmapped_ptms.discard(key)
                else:
                    # Residue mismatch detected
                    print(f"Residue mismatch at position {abs_pos} in protein {prot_id}: "
                          f"expected '{protein_residue}', found '{res['residue']}' in peptide.")

        # Add the updated peptide to the list
        updated_peptides.append({
            'prot_id': prot_id,
            'residues': residues
        })

    # Log how many PTMs could not be mapped
    num_unmapped_ptms = len(unmapped_ptms)
    print(f"{num_unmapped_ptms}/{total_ptms} PTMs could not be mapped to peptides due to size boundaries")

    return updated_peptides

def add_variable_ptms(peptides, modifications):
    """
    Combines all variable modifications (including predicted PTMs) to the residues in peptides.

    Args:
        peptides (list): List of peptides with residues and predicted PTMs in 'mods'.
        modifications (list): List of modification dictionaries.

    Returns:
        list: Peptides with all potential variable modifications listed in 'mods'.
    """
    # Create a dictionary of variable modifications for quick lookup
    variable_mods = {}
    for mod in modifications:
        if not mod['fixed']:
            aa = mod['amino_acid']
            label = mod['label']
            variable_mods.setdefault(aa, set()).add(label)

    # Iterate over each peptide
    for peptide in tqdm(peptides, desc = "Adding variable PTMs", unit="peptide"):
        # Ensure 'residues' key exists in peptide
        if 'residues' in peptide:
            for res in peptide['residues']:
                aa = res['residue']
                # Initialize 'mods' list if not present
                if 'mods' not in res:
                    res['mods'] = []
                existing_mods = set(res['mods'])
                # Add variable modifications applicable to this amino acid
                if aa in variable_mods:
                    for mod_label in variable_mods[aa]:
                        if mod_label not in existing_mods:
                            res['mods'].append(mod_label)
        else:
            # Handle cases where 'residues' key is missing
            print(f"Warning: 'residues' key not found in peptide {peptide.get('prot_id', 'unknown')}")
    return peptides

def build_peptidoforms_improved(peptides, max_variable_mods=1):
    """
    Generates peptidoforms by considering all possible combinations of variable modifications,
    up to the specified maximum number of modifications per peptide.
    
    Args:
        peptides (list): List of peptides with potential variable modifications in 'mods'.
        max_variable_mods (int): Maximum number of variable modifications allowed in a peptidoform.
    
    Returns:
        list: List of tuples (peptidoform_sequence, protein_id)
    
    Example input peptide structure:
    {
        'prot_id': 'PROT1',
        'residues': [
            {'residue': 'M', 'mods': ['Oxidation']},
            {'residue': 'S', 'mods': ['Phospho']},
            {'residue': 'Y', 'mods': ['Phospho']}
        ]
    }
    """
    peptidoform_protein_pairs = set()
    
    for peptide in tqdm(peptides, desc="Building peptidoforms", unit="peptidoform"):
        residues = peptide['residues']
        prot_id = peptide['prot_id']
        
        # Collect positions that have potential modifications
        mod_positions = []
        for idx, res in enumerate(residues):
            if res.get('mods'):
                mod_positions.append((idx, res['residue'], res['mods']))
        
        # Generate the unmodified version first
        base_sequence = ''.join(res['residue'] for res in residues)
        peptidoform_protein_pairs.add((base_sequence, prot_id))
        
        # If no modifications possible, continue to next peptide
        if not mod_positions:
            continue
            
        # Generate combinations for different numbers of modifications
        for num_mods in range(1, min(len(mod_positions) + 1, max_variable_mods + 1)):
            # Get all possible combinations of positions for this number of modifications
            for pos_combo in combinations(mod_positions, num_mods):
                # Generate all possible modification combinations for these positions
                mod_choices = [res[2] for res in pos_combo]  # Get modifications for each position
                
                # Generate cartesian product of modifications
                from itertools import product
                for mod_combo in product(*mod_choices):
                    # Create new sequence with these modifications
                    new_sequence = list(base_sequence)
                    for (idx, res, _), mod in zip(pos_combo, mod_combo):
                        # Insert modification after the residue
                        new_sequence[idx] = f"{res}[{mod}]"
                    
                    peptidoform = ''.join(new_sequence)
                    peptidoform_protein_pairs.add((peptidoform, prot_id))
    
    result = list(peptidoform_protein_pairs)
    print(f"Search space consists of {len(result)} peptidoform-protein pairs.\n")
    return result

def add_fixed_ptms(peptidoforms, modifications):
    """
    Adds fixed modifications to peptide sequences while preserving existing modifications
    and protein ID associations.

    Args:
        peptidoforms (list): List of tuples (peptidoform_sequence, protein_id).
        modifications (list): List of modification dictionaries.

    Returns:
        list: List of tuples (modified_peptidoform_sequence, protein_id) with fixed modifications added,
              maintaining existing modifications and protein ID associations.
    """
    # Extract fixed modifications into a dictionary for quick lookup
    fixed_mods = {mod['amino_acid']: mod['label'] for mod in modifications if mod['fixed']}

    modified_peptidoforms = []

    for peptide, protein_id in tqdm(peptidoforms, desc="Adding fixed PTMs", unit="peptidoform"):
        # Tokenize the peptide sequence into amino acids with optional modifications
        tokens = re.findall(r'([A-Z](?:\[[^\]]+\])?)', peptide)
        modified_tokens = []

        for token in tokens:
            aa = token[0]  # The amino acid is the first character
            existing_mods = re.findall(r'\[([^\]]+)\]', token) if '[' in token else []

            # Add fixed modification if necessary and not already present
            if aa in fixed_mods and fixed_mods[aa] not in existing_mods:
                existing_mods.append(fixed_mods[aa])

            # Reconstruct the token with all modifications
            if existing_mods:
                # Sort modifications for consistency (optional)
                existing_mods_sorted = sorted(existing_mods)
                mod_str = ''.join(f'[{mod}]' for mod in existing_mods_sorted)
                modified_token = f'{aa}{mod_str}'
            else:
                modified_token = aa

            modified_tokens.append(modified_token)

        # Reconstruct the modified peptide sequence
        modified_peptide = ''.join(modified_tokens)

        # Add the modified peptide to the new list, preserving protein ID association
        modified_peptidoforms.append((modified_peptide, protein_id))

    return modified_peptidoforms

def build_precursors(search_space, charges=[1, 2, 3, 4], mz_range=[300, 1800]):
    """
    Build precursors from a search space, considering different charge states and m/z range.
    Ensures unique precursor-protein pairs.

    Arguments: 
    - search_space (list): A list of tuples where each tuple contains (peptidoform_sequence, protein_id).
    - charges (list): A list of charge states to consider.
    - mz_range (list): A list containing the minimum and maximum m/z values to consider.

    Returns: 
    - precursors (list): A list of tuples where each tuple contains (precursor, protein_id).
    - Prints the number of unique precursor-protein pairs in the search space.
    """
    # Initialize ModificationsDB
    mz_min, mz_max = mz_range
    precursors = set()  # Using a set to ensure uniqueness

    for peptidoform, protein_id in tqdm(search_space, desc="Building Precursors", unit="peptidoform"):
        peptide = Peptidoform(peptidoform)
        theoretical_mass = peptide.theoretical_mass

        for charge in charges:
            mz = (theoretical_mass + charge * 1.007276466) / charge  # Adding proton mass for each charge
            if mz_min <= mz <= mz_max:
                precursor_key = f"{peptidoform}/{charge}"
                precursors.add((precursor_key, protein_id))

    precursors = list(precursors)  # Convert set back to list
    print(f"Number of unique Precursor-Protein pairs in the search space: {len(precursors)}\n")
    return precursors

def Precursors2Speclib(
    search_space: List[Tuple[str, str]],
    add_retention_time: bool = False,
    add_ion_mobility: bool = False,
    model: Optional[str] = "HCD",
    model_dir: Optional[Union[str, Path]] = None,
    batch_size: int = 100000,
    processes: Optional[int] = 1,
    output_path: str = '/home/sander/apps/speclib/data/',
    output_name: str = 'speclib_test',
    frag_mass_lower: float = 300.0,
    frag_mass_upper: float = 1800.0,
    ) -> None:
    """
    Converts a custom search space to a PSMList of PSM objects, makes predictions in batches,
    and writes the predictions directly to an .msp file at the specified location.

    Args:
        - search_space (List[Tuple[str, str]]): A list of tuples, where each tuple contains a peptide sequence with charge (str) and a protein ID (str).
            => [('HQRWAAR/1', 'A0A087X1C5'), ('HQRWAAR/2', 'A0A087X1C5'), ...]
        - add_retention_time (bool, optional): Whether to add retention time predictions. Defaults to False.
        - add_ion_mobility (bool, optional): Whether to add ion mobility predictions. Defaults to False.
        - model (Optional[str], optional): The model to use for predictions. Defaults to "HCD".
        - model_dir (Optional[Union[str, Path]], optional): The directory containing the model. Defaults to None.
        - batch_size (int, optional): The size of each batch for processing. Defaults to 100000.
        - processes (Optional[int], optional): The number of processes to use. Defaults to 1.
        - output_path (str, optional): The directory where the .msp file will be saved. Defaults to '/home/sander/apps/speclib/data/'.
        - output_name (str, optional): The name of the output .msp file. Defaults to 'speclib_test'.
        - frag_mass_lower (float, optional): The lower m/z limit for fragment ions. Defaults to 0.0.
        - frag_mass_upper (float, optional): The upper m/z limit for fragment ions. Defaults to 1800.0.

    Returns:
        - None: Writes the spectral library to the specified .msp file.
    """
    def _into_batches(search_space, batch_size):
        """Split the search space into batches."""
        batch = []
        for item in search_space:
            batch.append(item)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _create_psm_list(batch):
        """Create a PSMList from a batch of peptides."""
        psm_list = []
        for spectrum_id, (peptide, protein_id) in enumerate(batch):
            peptidoform = Peptidoform(peptide)
            psm = PSM(
                spectrum_id=str(spectrum_id),
                peptidoform=peptidoform,
                is_decoy=False,
                protein_list=[protein_id],
            )
            psm_list.append(psm)
        return PSMList(psm_list=psm_list)

    # Create the output directory if it doesn't exist
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / output_name

    # Process batches
    batches = list(_into_batches(search_space, batch_size))
    total_batches = len(batches)
    
    # Inform the user about the total number of batches
    print(f"Processing {total_batches} batch{'es' if total_batches != 1 else ''}...")

    for i, batch in enumerate(tqdm(batches, desc='Processing Batches', unit='batch')):
        psm_list = _create_psm_list(batch)
        
        #logging.disable(logging.CRITICAL)
        predictions = predict_batch(
            psm_list,
            add_retention_time=add_retention_time,
            add_ion_mobility=add_ion_mobility,
            model=model,
            model_dir=model_dir,
            processes=processes,
        )
        #logging.disable(logging.NOTSET)
        
        # Filter predictions to the specified m/z range
        for pred in predictions:
            # For each ion type (e.g., 'b', 'y')
            for ion_type in pred.theoretical_mz.keys():
                mz_array = pred.theoretical_mz[ion_type]
                intensity_array = pred.predicted_intensity[ion_type]
                
                # Filter based on frag_mass_lower and frag_mass_upper
                valid_indices = (mz_array >= frag_mass_lower) & (mz_array <= frag_mass_upper)
                
                # Update the mz and intensity arrays
                pred.theoretical_mz[ion_type] = mz_array[valid_indices]
                pred.predicted_intensity[ion_type] = intensity_array[valid_indices]
        
        # Write predictions to the MSP file
        write_mode = "w" if i == 0 else "a"
        write_spectra(str(output_file), predictions, 'spectronaut', write_mode=write_mode)

    print(f"Spectral library saved to {output_file}")

# Define function that put the pipeline together: 
def Peptidoforms2Speclib(
    peptides,
    sequences,
    predictions,
    modifications: List[Dict],
    control_library: bool = False,
    max_variable_mods: int = 3 ,
    charges: List = [1, 2, 3, 4],
    add_retention_time: bool = True ,
    add_ion_mobility: bool = True,
    mz_precursor_range: list = [300, 1800],
    model_dir: Optional[Union[str, Path]] = None,
    model: Optional[str] = "HCD",
    batch_size: int = 100000,
    processes: Optional[int] = 1,
    frag_mass_lower: float = 300.0,
    frag_mass_upper: float = 1800.0,
    out_path: str = '/home/sander/apps/speclib/data/',
    out_name: str = 'speclib'
    ):
    """
    **Input**: 
        - peptides = [{'prot_id': 'A0A087X1C5','sequence': 'MGLEALVPLAMIVAIFLLLVDLMHR','miscleavages': 0,'peptide_length': 25,'start_pos': 0}, ...]
        - fasta file of the proteome(s) where the peptides originate from
        - predictions = [{'pred_csv:, PTM: 'Y[Phospho]', threshold:},...] 
        - modifications = [
                            {"label": "Oxidation", "amino_acid": "M", "fixed": False},
                            {"label": "Carbamidomethyl", "amino_acid": "C", "fixed": True}
                        ]
        - max_mods_peptide: (int)
        - charges: (list, e.g.: [1, 2, 3, 4])
        - add_retention_times (bool)
        - add_ion_mobility (bool)
        - mz_range (list, e.g.: [0, 1800]) 

    **Output**:
    speclib.msp
    """
    target_ptms = []
    if control_library:
        print('peptidoform2speclib is running in control mode. Random entries will be sampled and added instead of threshold-filtered predictions.')
    # Read & Filter all the predictions of different PTM types
    for ptm in predictions:
        prediction_list = filter_prediction(csv_file = ptm['pred_csv'], PTM = ptm['PTM'], threshold = ptm['threshold'], control = control_library)
        target_ptms.extend(prediction_list)
    # Add ptm annotations on peptides
    peptides_target_ptms = add_target_ptms(peptides, sequences, target_ptms)
    peptides_variable_mods = add_variable_ptms(peptides_target_ptms, modifications)
    # Generate all unique peptidoforms from peptides
    search_space = build_peptidoforms_improved(peptides_variable_mods, max_variable_mods = max_variable_mods)
    # Add fixed modifications to search space
    search_space = add_fixed_ptms(search_space, modifications)
    # Generate all Precursors that fall within specifications
    precursors = build_precursors(search_space, charges = charges, mz_range = mz_precursor_range)
    # Split the precursors in batches and predict their spectra + save the speclib to path
    Precursors2Speclib(search_space = precursors,
                        add_retention_time = add_retention_time,
                        add_ion_mobility = add_ion_mobility,
                        model = model,
                        model_dir = model_dir,
                        batch_size = batch_size,
                        processes = processes,
                        frag_mass_lower = frag_mass_lower,
                        frag_mass_upper = frag_mass_upper,
                        output_path = out_path,
                        output_name = out_name,
                        )

if __name__ == "__main__": 
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate spectral library from proteome and PTM predictions.')
    parser.add_argument('--config', required=True, help='Path to the configuration JSON file.')

    args = parser.parse_args()

     # Read configuration file
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Extract parameters from config
    proteome_fastas = config['proteome_fastas']
    # Digestion parameters
    missed_cleavages = config.get('missed_cleavages', 1)
    min_length = config.get('min_length', 7)
    max_length = config.get('max_length', 30)
    # Predictions
    predictions = config['predictions']  # List of dictionaries with keys: pred_csv, PTM, threshold
    modifications = config['modifications']  # List of modifications
    control_library = config.get('control_library', False)
    max_variable_mods = config.get('max_variable_mods', 1)
    charges = config.get('charges', [1, 2, 3, 4])
    add_retention_time = config.get('add_retention_time', True)
    add_ion_mobility = config.get('add_ion_mobility', True)
    mz_precursor_range = config.get('mz_precursor_range', [300, 1800])
    model = config.get('ms2pip_model', 'HCD')
    model_dir = config.get('model_dir', None)
    batch_size = config.get('batch_size', 100000)
    processes = config.get('processes', 1)
    frag_mass_lower = config.get('frag_mass_lower', 300.0)
    frag_mass_upper = config.get('frag_mass_upper', 1800.0)
    out_path = config.get('out_path', '.')
    out_name = config.get('out_name', 'speclib')
  
    # for debugging 
    """
    test_peptides = [{'prot_id': 'A0A0B4J2F2',
        'sequence': 'TRLDSSNLEKIYR',d
        'miscleavages': 2,
        'peptide_length': 13,
        'start_pos': 60},
        {'prot_id': 'A0A0B4J2F2',
        'sequence': 'MVIMSEFSADPAGQGQGQQKPLR',
        'miscleavages': 0,
        'peptide_length': 23,
        'start_pos': 0},
        {'prot_id': 'A0A087X1C5',
        'sequence': 'YPPGPLPLPGLGNLLHVDFQNTPYCFDQLR',
        'miscleavages': 0,
        'peptide_length': 30,
        'start_pos': 32}]
    """
    # make the peptides: 
    sequences = {}
    for proteome in proteome_fastas:
        prot = read_proteome(proteome)
        sequences.update(prot)
    peptides = trypsin_digest(sequences = sequences, missed_cleavages = missed_cleavages, min_length = min_length, max_length = max_length)
    # Call Peptidoforms2Speclib function
    Peptidoforms2Speclib(
        peptides=peptides,
        sequences = sequences,
        predictions=predictions,
        modifications=modifications,
        control_library=control_library,
        max_variable_mods=max_variable_mods,
        charges=charges,
        add_retention_time=add_retention_time,
        add_ion_mobility=add_ion_mobility,
        mz_precursor_range=mz_precursor_range,
        model_dir=model_dir,
        model=model,
        batch_size=batch_size,
        processes=processes,
        frag_mass_lower=frag_mass_lower,
        frag_mass_upper=frag_mass_upper,
        out_path=out_path,
        out_name=out_name
        )