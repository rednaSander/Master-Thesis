import torch
import pytorch_lightning as pl
from torch import nn
from torch.optim import AdamW
from torchmetrics import Metric, AUROC, AveragePrecision, Precision
from torchmetrics.classification import BinaryCalibrationError
from utils import BinnedPrecisionAtFixedRecall
from typing import Any
from input_tokenizers import TokenAlphabet
import network_architectures
from torch.optim.lr_scheduler import LinearLR
from sklearn.metrics import precision_recall_curve, roc_curve
import wandb
import numpy as np

# Define costum evaluation metrics:
class RecallAtProbabilityK(Metric):
    def __init__(self, k, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.k = k
        self.add_state("true_positives", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("actual_positives", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds = preds.detach()
        targets = targets.detach()
        selected = preds >= self.k
        true_positives = (selected & (targets == 1)).sum().float()
        actual_positives = (targets == 1).sum().float()
        self.true_positives += true_positives
        self.actual_positives += actual_positives

    def compute(self):
        if self.actual_positives > 0:
            return self.true_positives / self.actual_positives
        else:
            return torch.tensor(0.0)

# PyTorch-Lightning Module class; takes care of training, batch organization, metrics, logging, evaluation
class LightningModule(pl.LightningModule):
    def __init__(self, config: dict[str,Any], steps_per_training_epoch: int, tokenizer: TokenAlphabet) -> None:
        """
        Pytorch-Lightning class that takes care of training, batch organization, metrics, logging, and evaluation

        Parameters
        ----------
        config : dict[str,Any]
            The full prediction model

        tokenizer : TokenAlphabet
            The tokenizer used for the selected protein representation

        Attributes
        ----------
        lr : float
            The learning rate to be used in training

        tokenizer : TokenAlphabet
            The tokenizer used for the selected protein representation

        train/valid/test_cross_entropy_loss : torch.nn.BCEWithLogitsLoss
            Loss functions

        train/valid/test_metrics : nn.ModuleList
            Metrics to be run at training, validation and test time

        metric_names : list[str]
            The names of the respective metrics, for logging purposes

        model : torch.nn.Module
            The full prediction model

        encoding : nn.Module
            The one-hot encoding or PLM encoding module in the model architecture
        """
        # (1) initialize all the objects in self(~backpack): 
        super().__init__() # calls the constructor of the parent class (pl.lightningModule)
        self.lr = config['learning_rate'] # store the learning rate
        self.tokenizer = tokenizer # store the tokenizer 
        self.save_hyperparameters() # saves all pytorch lightning arguments passed to __init__() as hyperparameters
        self.model = network_architectures.get_architecture(config=config) # creates the actual neural network model based on the provided config
        self.warm_up_steps = int(config['warm_up_epochs'] * steps_per_training_epoch) # calculates the number of warm-up steps of the learning rate scheduler
        # setup the loss functions for training/validation/test
        self.train_cross_entropy_loss = torch.nn.BCEWithLogitsLoss(
            weight=torch.Tensor([config['pos_weight']])
        )
        self.valid_cross_entropy_loss = torch.nn.BCEWithLogitsLoss(
            weight=torch.Tensor([config['pos_weight']])
        )
        self.test_cross_entropy_loss = torch.nn.BCEWithLogitsLoss(
            weight=torch.Tensor([config['pos_weight']])
        )
        # initialize metrics for training/validation/test
        self.metric_names, self.train_metrics = self._init_metrics() # self.metric_names gets assigned a list of metric names, self.train_metrics gets actual metric objects assigned
        _, self.valid_metrics = self._init_metrics() # '_' discards the metric_names as they are the same as in train_metrics, but the self.valid_metrics are saved() 
        _, self.test_metrics = self._init_metrics() # same here as mentioned above
        self.encoding = self.model.encoding # stores the protein encodings, is probably used for feature visualization
        # Initialize accumulators
        self.validation_predictions = []
        self.validation_targets = []
        self.test_predictions = []
        self.test_targets = []
    
    
    
    # (2) define forward which pases instructions to the model in self, allows lightning module to handle any input structure the underlaying model expects. 
    def forward(self, *args, **kwargs):# *args: positional argument, *kwargs: number/keyword arguments
        return self.model(*args, **kwargs)



    # (3) Setup the optimizer and learning rate scheduler for training, returns a list of optimizers and schedulers for pl to use
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.warm_up_steps
        )
        print(self.warm_up_steps)
        scheduler = {
            'scheduler': scheduler,
            'name': 'actual_learning_rate',
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    # (4) initializes the evaluation metrics
    def _init_metrics(self) -> tuple[list[str], nn.ModuleList]:
        """
        Initializes Metric objects

        Returns
        -------
        names : list[str]
            The names of the metrics created in this function

        : nn.ModuleList
            The Metric objects
        """
        # define the list of metric names
        names = ["AUPRC", "AUROC", "ECE", "Rec@Prob0.5", "Rec@Prob0.6", "Rec@Prob0.7", "Rec@Prob0.8", "Rec@Prob0.9"]
        return (
        # function returns both the name and the corresponding objects
            names,
            nn.ModuleList(
                [
                    AveragePrecision(compute_on_step=False),
                    AUROC(compute_on_step=False),
                    BinaryCalibrationError(n_bins=10, compute_on_step=False),
                    RecallAtProbabilityK(k=0.5),
                    RecallAtProbabilityK(k=0.6),
                    RecallAtProbabilityK(k=0.7),
                    RecallAtProbabilityK(k=0.8),
                    RecallAtProbabilityK(k=0.9),
                ]
            ),
        )



    # (5) this function takes a batch as input and returns a tuple of 5 elements    
    def process_batch(
        self, batch: dict[str, Any]
    ) -> tuple[list[str], torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        """
        Important function to preprocess, predict, and postprocess data. Different steps in the code are further
        described in detail below

        Parameters
        ----------
        batch : dict[str, np.ndarray]
            Batch data, as composed by the collate function by the DataLoader

        Returns
        -------
        prot_id_per_tar : list[str]
            Protein ids per P-site candidate within the batch

        actual_pos_per_tar : torch.tensor
            The actual positions within the full protein of all P-site candidates in the batch

        logit_per_tar : torch.tensor
            Predicted logits for all P-site candidates in the batch

        probability_per_tar : torch.tensor
            Predicted probabilities for all P-site candidates in the batch

        annot_per_tar : torch.tensor
            Labels for all P-site candidates in the batch
        """
        # extract different components from the input batch and move them to the appropiate device
        prot_ids_per_seq = batch["prot_id"]
        tokens_per_seq = batch["prot_token_ids"].to(self.device)
        offsets_per_seq = batch["prot_offsets"].to(self.device)
        mask_per_seq = batch["prot_input_mask"].to(self.device)
        mask_no_extra_per_seq = batch["prot_input_mask_without_added_tokens"].to(self.device)
        targets_per_seq = batch["targets"]

        # create a mask of targets where 1 indicates a valid target and 0 indicates padding
        target_mask_per_seq = torch.zeros_like(targets_per_seq, device=self.device)
        target_mask_per_seq[targets_per_seq != -1] = 1
        # runs the forward pass of the model with the preprocessed inputs
        forward_output = self(
            tokens_per_seq,
            mask_per_seq,
            mask_no_extra_per_seq,
            target_mask_per_seq,
        )
        batch_size = len(prot_ids_per_seq) # calculate the batch size
        seq_len = max(mask_no_extra_per_seq.sum(dim=1)) # calculate the maximum sequence length in the batch
        # create an index tensor for each target in the batch
        protein_idx_per_tar = (
            torch.tensor(range(batch_size), dtype=torch.int32)
            .to(self.device)
            .expand(targets_per_seq.shape[::-1])
            .T[targets_per_seq != -1]
        )
        # gather the offsets for all targets
        offset_per_tar = offsets_per_seq.expand(
            targets_per_seq.shape[::-1]
        ).T[targets_per_seq != -1]
        # calculates the actual position of each target in the full protein sequence
        pos_in_fragment_per_tar = (
            torch.tensor(range(seq_len), dtype=torch.int32)
            .to(self.device)
            .expand((batch_size, seq_len))[targets_per_seq != -1]
        )
        # calculate the actual positions for each target, by adding up the position + the offset
        actual_pos_per_tar = (
            pos_in_fragment_per_tar + offset_per_tar
        )
        # gather the protein ids for all targets
        prot_id_per_tar = [
            prot_ids_per_seq[id_index] for id_index in protein_idx_per_tar
        ]
        # gather the annotations for all targets
        annot_per_tar = targets_per_seq[targets_per_seq != -1]
        # gather the predicted probabilities for all targets
        logit_per_tar = forward_output
        probability_per_tar = torch.sigmoid(
            logit_per_tar
        )

        return (
            prot_id_per_tar, # = the protein IDs
            actual_pos_per_tar,# = the actual position of the protein target
            logit_per_tar, # = the logit value of each target
            probability_per_tar, # = sigmoid activation of the logits, = prediction value per target
            annot_per_tar, # = label of each target
        )
       
       
        
    # (6) define what the process of each train/validation/test step.
    def training_step(self, batch, batch_idx):
        (
            prot_ids,
            site_positions,
            logit_outputs,
            predicted_probs,
            targets,
        ) = self.process_batch(batch) # forward pass thourgh the network returing the tensors defined in process_batch
        loss = self.train_cross_entropy_loss(logit_outputs, targets.float()) # calculates the loss of the current batch 
        self.log("training_loss", loss, on_step=False, on_epoch=True) # calcuates training loss on epoch
        # calls all training metrics with predicted values and targets to calcuate the predefined metrics
        for metric in self.train_metrics:
            metric(predicted_probs, targets) # metrics can accumulate the preds and targets
        return loss # retruns the loss


    def validation_step(self, batch, batch_idx):
        (
            prot_ids,
            site_positions,
            logit_outputs,
            predicted_probs,
            targets,
        ) = self.process_batch(batch)
        loss = self.valid_cross_entropy_loss(logit_outputs, targets.float())
        self.log("validation_loss", loss, on_step=False, on_epoch=True) # calculates validation loss on epoch
        for metric in self.valid_metrics:
            metric(predicted_probs, targets)
        # Accumulate predictions and targets
        self.validation_predictions.append(predicted_probs.detach().cpu())
        self.validation_targets.append(targets.detach().cpu())
        return loss
        
        
    def test_step(self, batch, batch_idx):
        (
            prot_ids,
            site_positions,
            logit_outputs,
            predicted_probs,
            targets,
        ) = self.process_batch(batch)
        loss = self.test_cross_entropy_loss(logit_outputs, targets.float())
        self.log("test_set_loss", loss, on_step=False, on_epoch=True)
        for metric in self.test_metrics:
            metric(predicted_probs, targets)
        # Accumulate predictions and targets
        self.test_predictions.append(predicted_probs.detach().cpu())
        self.test_targets.append(targets.detach().cpu())
        return loss    
   
   
            
    # (7) fucntions which are called after every train/validation/test epoch
    def training_epoch_end(self, outputs):
        for metric_name, metric in zip(self.metric_names, self.train_metrics):
            result = metric.compute() # calls the metrics to compute their values
            self.log(f"train_{metric_name}", result) # logs the metric names and its values to the logger. 
            metric.reset() # states of all metrics are reset


    def validation_epoch_end(self, outputs):
        for metric_name, metric in zip(self.metric_names, self.valid_metrics):
            result = metric.compute()
            self.log(f"valid_{metric_name}", result)
            metric.reset()
            
        # MAKE CUSTOM PLOTS IN WandB:
        # (6A)Get predictions and targets
        all_preds = torch.cat(self.validation_predictions).numpy()
        all_targets = torch.cat(self.validation_targets).numpy()
        predictions = np.array(all_preds).astype(float)  # Ensure predictions are floats
        ground_truth = np.array(all_targets).astype(int)  # ground truth labels (0 or 1)
                
        # (6B) Prepare data for WandB table for Histogram
        valid_data = [[pred, target] for pred, target in zip(predictions, ground_truth)]
        valid_table = wandb.Table(data=valid_data, columns=["pred", "target"])
                # Create histogram
        histogram_1 = wandb.plot_table(
            vega_spec_name='sander-heyndrickx-universiteit-gent/stacked_his_bytar',
            data_table=valid_table,
            fields={"x": "pred", "color": "target"}
            )
        histogram_2 = wandb.plot_table(
            vega_spec_name='sander-heyndrickx-universiteit-gent/his_pred_tar0',
            data_table=valid_table,
            fields={"x": "pred", "color": "target"}
            )
        histogram_3 = wandb.plot_table(
            vega_spec_name='sander-heyndrickx-universiteit-gent/his_pred_tar1',
            data_table=valid_table,
            fields={"x": "pred", "color": "target"}
            )
        
        # (6C) Create PR curve:
        # Create PR and ROC curves
        # Calculate precision, recall, and thresholds using Scikit-learn
        precision, recall, thresholds = precision_recall_curve(ground_truth, predictions)
        # Append None to thresholds to match length of precision and recall
        
        # Downsize ROC curve if necessary
        num_thresholds = len(thresholds)
        num_points = 100
        if num_thresholds > num_points:
            indices = np.linspace(0, len(thresholds) - 1, num=num_points, dtype=int)
            thresholds_downsampled = thresholds[indices]
            precision_downsampled = precision[indices + 1]  # shift by 1 due to the way precision_recall_curve works
            recall_downsampled = recall[indices + 1]
            thresholds_downsampled = np.append(thresholds_downsampled, None)
            precision_downsampled = np.append(precision_downsampled, precision[-1])
            recall_downsampled = np.append(recall_downsampled, recall[-1])
            prc_data_downsampled = [[p, r, t] for p, r, t in zip(precision_downsampled, recall_downsampled, thresholds_downsampled)]
            prc_table_downsampled = wandb.Table(data=prc_data_downsampled, columns=["Precision", "Recall", "Threshold"])
            prc_curve = wandb.plot_table(
                    vega_spec_name="sander-heyndrickx-universiteit-gent/prc_",
                    data_table=prc_table_downsampled,
                    fields={"x": "Recall", "y": "Precision"}
                    )
        else: 
            prc_data = [[p, r, t] for p, r, t in zip(precision, recall, np.append(thresholds, None))]
            prc_table = wandb.Table(data=prc_data, columns=["Precision", "Recall", "Threshold"])
            prc_curve = wandb.plot_table(
            vega_spec_name="sander-heyndrickx-universiteit-gent/prc_",
            data_table=prc_table,
            fields={"x": "Recall", "y": "Precision"}
            )
        
        
        # (6D) Create ROC curve:
        # Calculate FPR, TPR, and thresholds using Scikit-learn
        fpr, tpr, thresholds_roc = roc_curve(ground_truth, predictions)
        
        # Downsize ROC curve if necessary
        num_thresholds_roc = len(thresholds_roc)
        if num_thresholds_roc > num_points:
            indices = np.linspace(0, num_thresholds_roc - 1, num=num_points, dtype=int)
            thresholds_roc_downsampled = thresholds_roc[indices]
            fpr_downsampled = fpr[indices]
            tpr_downsampled = tpr[indices]
            roc_data_downsampled = [
                [fpr_val, tpr_val, thr] for fpr_val, tpr_val, thr in zip(fpr_downsampled, tpr_downsampled, thresholds_roc_downsampled)
            ]
            roc_table_downsampled = wandb.Table(
                data=roc_data_downsampled, columns=["FPR", "TPR", "Threshold"]
            )
            roc_curve_plot = wandb.plot_table(
                vega_spec_name="sander-heyndrickx-universiteit-gent/roc_",
                data_table=roc_table_downsampled,
                fields={"x": "FPR", "y": "TPR"}
            )
        else:
            roc_data = [
                [fpr_val, tpr_val, thr] for fpr_val, tpr_val, thr in zip(fpr, tpr, thresholds_roc)
            ]
            roc_table = wandb.Table(data=roc_data, columns=["FPR", "TPR", "Threshold"])
            roc_curve_plot = wandb.plot_table(
                vega_spec_name="sander-heyndrickx-universiteit-gent/roc_",
                data_table=roc_table,
                fields={"x": "FPR", "y": "TPR"}
            )

        # Log plots to WandB
        self.logger.experiment.log({'histogram_1a': histogram_1,
                    'histogram_2a': histogram_2,
                    'histogram_3a': histogram_3,
                    "prc_curvea": prc_curve,
                    "roc_curvea": roc_curve_plot
                    })
        # Reset accumulators
        self.validation_predictions = []
        self.validation_targets = []


    def test_epoch_end(self, step_outputs):
        for metric_name, metric in zip(self.metric_names, self.test_metrics):
            result = metric.compute()
            self.log(f"test_{metric_name}", result)
            metric.reset()
        
        # MAKE CUSTOM PLOTS IN WandB:        
        # (6A)Get predictions and targets
        all_preds = torch.cat(self.test_predictions).numpy()
        all_targets = torch.cat(self.test_targets).numpy()
        predictions = np.array(all_preds).astype(float)  # Ensure predictions are floats
        ground_truth = np.array(all_targets).astype(int)  # ground truth labels (0 or 1)
        
        test_data = [[pred, target] for pred, target in zip(predictions, ground_truth)]
        test_table = wandb.Table(data=test_data, columns=["pred", "target"])
        # Create histogram
        histogram_1_test = wandb.plot_table(
            vega_spec_name='sander-heyndrickx-universiteit-gent/his1_test',
            data_table=test_table,
            fields={"x": "pred", "color": "target"}
            )
        histogram_2_test = wandb.plot_table(
            vega_spec_name='sander-heyndrickx-universiteit-gent/his2_test',
            data_table=test_table,
            fields={"x": "pred", "color": "target"}
            )
        histogram_3_test = wandb.plot_table(
            vega_spec_name='sander-heyndrickx-universiteit-gent/his3_test',
            data_table=test_table,
            fields={"x": "pred", "color": "target"}
            )
        
        # (6C) Create PR curve:
        # Calculate precision, recall, and thresholds using Scikit-learn
        precision, recall, thresholds_pr = precision_recall_curve(ground_truth, predictions)
        
        # Downsize PR curve if necessary
        num_thresholds = len(thresholds_pr)
        num_points = 100  # Maximum number of points to plot
        if num_thresholds > num_points:
            indices = np.linspace(0, num_thresholds - 1, num=num_points, dtype=int)
            thresholds_downsampled = thresholds_pr[indices]
            # Shift indices by 1 due to the way precision_recall_curve works
            precision_downsampled = precision[indices + 1]
            recall_downsampled = recall[indices + 1]
            # Append None to thresholds to match lengths
            thresholds_downsampled = np.append(thresholds_downsampled, None)
            precision_downsampled = np.append(precision_downsampled, precision[-1])
            recall_downsampled = np.append(recall_downsampled, recall[-1])
            prc_data_downsampled = [[p, r, t] for p, r, t in zip(precision_downsampled, recall_downsampled, thresholds_downsampled)]
            prc_table_downsampled = wandb.Table(
                data=prc_data_downsampled, columns=["Precision", "Recall", "Threshold"]
            )
            prc_curve_test = wandb.plot_table(
                vega_spec_name="sander-heyndrickx-universiteit-gent/prc_test",
                data_table=prc_table_downsampled,
                fields={"x": "Recall", "y": "Precision"}
            )
        else:
            thresholds_full = np.append(thresholds_pr, None)
            prc_data = [
                [p, r, t] for p, r, t in zip(precision[1:], recall[1:], thresholds_full)
            ]
            prc_table = wandb.Table(data=prc_data, columns=["Precision", "Recall", "Threshold"])
            prc_curve_test = wandb.plot_table(
                vega_spec_name="sander-heyndrickx-universiteit-gent/prc_test",
                data_table=prc_table,
                fields={"x": "Recall", "y": "Precision"}
            )
        
        # Create ROC curve:
        # Calculate FPR, TPR, and thresholds using Scikit-learn
        fpr, tpr, thresholds_roc = roc_curve(ground_truth, predictions)
        
        # Downsize ROC curve if necessary
        num_thresholds_roc = len(thresholds_roc)
        if num_thresholds_roc > num_points:
            indices = np.linspace(0, num_thresholds_roc - 1, num=num_points, dtype=int)
            thresholds_roc_downsampled = thresholds_roc[indices]
            fpr_downsampled = fpr[indices]
            tpr_downsampled = tpr[indices]
            roc_data_downsampled = [
                [fpr_val, tpr_val, thr] for fpr_val, tpr_val, thr in zip(fpr_downsampled, tpr_downsampled, thresholds_roc_downsampled)
            ]
            roc_table_downsampled = wandb.Table(
                data=roc_data_downsampled, columns=["FPR", "TPR", "Threshold"]
            )
            roc_curve_test = wandb.plot_table(
                vega_spec_name="sander-heyndrickx-universiteit-gent/roc_test",
                data_table=roc_table_downsampled,
                fields={"x": "FPR", "y": "TPR"}
            )
        else:
            roc_data = [
                [fpr_val, tpr_val, thr] for fpr_val, tpr_val, thr in zip(fpr, tpr, thresholds_roc)
            ]
            roc_table = wandb.Table(data=roc_data, columns=["FPR", "TPR", "Threshold"])
            roc_curve_test = wandb.plot_table(
                vega_spec_name="sander-heyndrickx-universiteit-gent/roc_test",
                data_table=roc_table,
                fields={"x": "FPR", "y": "TPR"}
            )
            
        # Log plots to WandB
        self.logger.experiment.log({'histogram_1b': histogram_1_test,
                    'histogram_2b': histogram_2_test,
                    'histogram_3b': histogram_3_test,
                    "prc_curveb": prc_curve_test,
                    "roc_curveb": roc_curve_test
                    })
        # Reset accumulators
        self.test_predictions = []
        self.test_targets = []
            
