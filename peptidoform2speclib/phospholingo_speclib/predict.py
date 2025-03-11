import input_reader as ir
import torch.nn
from torch.utils.data.dataloader import DataLoader
from lightning_module import LightningModule
import utils
from tqdm import tqdm
import os
import json
from Bio import SeqIO
from utils import trypsin_digest, annotate_proteome_with_target_residues, annotate_full_proteome, read_proteome

def run_predict(json_file) -> None:
    """
    Runs predictions on a FASTA file using an already existing model

    Parameters
    ----------
    model : str
        The file location of an already trained model

    dataset_fasta : str
        The location of the FASTA file for which predictions are to be made. Predicted residues should be succeeded in
        the file by either a '#' or '@' symbol

    output_file : str
        The output csv file to which predictions are written
    """
    # read the json
    config = json.load(open(json_file))

    # Read in the proteome fastas and put them all together

    predictions_to_run = []
    for pred in config['predictions']:
        if os.path.exists(pred['pred_csv']):
            print(f"Prediciton file {pred['pred_csv']} already exists, will skip")
        else: 
            predictions_to_run.append(pred)
        
    if not predictions_to_run:
        print("All prediction files already exist. Nothing to do :)")
        return
    
    # Read in the proteome fastas and put them all together
    sequences = {}
    for fasta_file in config["proteome_fastas"]:
        proteome = read_proteome(fasta_file)
        sequences.update(proteome)
    

    if config['peptides']:
        peptides = trypsin_digest(sequences, 
                                    config['missed_cleavages'],
                                    config['min_length'],
                                    config['max_length']
                                     )

    for pred in predictions_to_run:
        # Get target residues from PTM type (e.g., 'Y' from 'Y[Phospho]' or 'ST' from 'ST[Phospho]')
        target_residues = pred['PTM'].split('[')[0]
        print(f"\nProcessing predictions for target residues: {pred['PTM']}")

        # Annotate the fasta based on additional config arguments
        if config['peptides']:
            print("Peptides == True: Using peptide-based annotation")
            annotated_sequences = annotate_proteome_with_target_residues(peptides,
                                                                            sequences,
                                                                            target_residues,
                                                                            pred['PTM']
                                                                        )
        else:
            # annotate full proteome 
            print("Peptides == False:Using full proteome annotation")
            annotated_sequences = annotate_full_proteome(sequences = sequences, 
                                                         target_residues = target_residues,
                                                         PTM = pred['PTM']
                                                         )

       # Run Predict using the annotated sequences        
        model_loc = pred['model']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_d = torch.load(model_loc, map_location=device)
        model_config = model_d['hyper_parameters']['config']
        model = LightningModule(model_config, 0, model_d['hyper_parameters']['tokenizer'])
        model.load_state_dict(model_d['state_dict'])
        model.to(device)
        model.eval()

        # use  a custom Fasta dataset reader: so it accepts the dict format already => look in claude chat history
        test_set = ir.AnnotatedDictDataset(sequences= annotated_sequences, tokenizer=model.tokenizer)
        gpu_batch_size = utils.get_gpu_max_batchsize(model_config['representation'], True)
        test_loader = DataLoader(test_set, gpu_batch_size, shuffle=False, pin_memory=True, collate_fn=test_set.collate_fn)

        with open(pred['pred_csv'], 'w') as write_to:
            print('prot_id,position,pred', file=write_to)
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Predicting"):
                    (
                        prot_ids,
                        site_positions,
                        logit_outputs,
                        predicted_probs,
                        targets,
                    ) = model.process_batch(batch)

                    for id, pos, prob in zip(prot_ids, site_positions, predicted_probs):
                        print(','.join([id, str(int(pos) + 1), '{:.4f}'.format(float(prob))]), file=write_to)