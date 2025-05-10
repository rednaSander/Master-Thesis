import json
import os.path
import wandb
import pytorch_lightning as pl
from datetime import datetime

from lightning_module import LightningModule
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
import input_reader as ir
import utils
from torch.utils.data.dataloader import DataLoader
from input_tokenizers import tokenizers
from pytorch_lightning.loggers import WandbLogger #wandb

def run_training(json_file=None, wandb_config=None):
    """
    Main function to run training, and to evaluate and save the resulting model

    Parameters
    ----------
    json_file : str
        file location of the configuration file.
    wandb_config : dict
        Configuration provided by W&B sweep.
    """
    
    # (1a) Configuration Loading and Processing
    default_config = json.loads(utils.DEFAULT_JSON) # load the default config file
    loaded_config = json.load(open(json_file)) # load the users config file
    
    # Merge configs and remove 'config/' prefixes, overriding the users config
    config = {**default_config, **loaded_config}
    config = {k.replace('config/', ''): v for k, v in config.items()}
    config['json_file'] = json_file
    
    # (1b) Override config with W&B sweep parameters if provided
    if wandb_config:
        config.update(wandb_config)

    assert config['conv_depth'] * config['max_pool_size'] <= config['receptive_field'], \
        'The receptive field should be large enough to accommodate the specified pooling size'
    
    
        
    # (2) Set up logging with Weights & Biases
    json_name = os.path.basename(json_file[:-5])
    wandb_logger = WandbLogger(project="PhosphoLingo_test", name=json_name, log_model=False)
    
    # Log hyperparameters to W&B
    wandb_logger.log_hyperparams(config)


    # (3) Set the batch size to run on the GPU, as well as the gradient accumulation parameter in case this exceeds
    #     the maximum GPU batch size for the specified representation
    batch_size = config['batch_size']
    gpu_batch_size = utils.get_gpu_max_batchsize(config['representation'], config['freeze_representation'])
    while batch_size % gpu_batch_size != 0:
        gpu_batch_size-=1 # make sure that the batch_size can be split in equal parts <= max gpu batch size
    grad_accumulation = batch_size // gpu_batch_size


    # (4) Read in datasets, handles different dataset loading senarios
    tokenizer_class = tokenizers[config['representation']]
    tokenizer = tokenizer_class()
    print()
    if config['test_set'] == 'default':
        print('Loading training data...')
        train_set = ir.SingleFastaDataset(dataset_loc=config['training_set'], tokenizer=tokenizer, train_valid_test='train')
        print('Loading validation data...')
        valid_set = ir.SingleFastaDataset(dataset_loc=config['training_set'], tokenizer=tokenizer, train_valid_test='valid')
        print('Loading test data...')
        test_set = ir.SingleFastaDataset(dataset_loc=config['training_set'], tokenizer=tokenizer, train_valid_test='test')
    elif config['test_fold'] >= 0:
        test_folds = {config['test_fold']}
        valid_folds = {(config['test_fold']+1)%10}
        train_folds = set(range(10)) - valid_folds # test proteins are excluded via the exclude_proteins set
        print('Loading test data...')
        test_set = ir.MultiFoldDataset(dataset_loc=config['test_set'],
                                        tokenizer=tokenizer,
                                        exclude_proteins=[],
                                        folds=test_folds)
        exclude_proteins = test_set.get_proteins_in_dataset()
        print('Loading validation data...')
        valid_set = ir.MultiFoldDataset(dataset_loc=config['training_set'],
                                        tokenizer=tokenizer,
                                        exclude_proteins=exclude_proteins,
                                        folds=valid_folds)
        print('Loading training data...')
        train_set = ir.MultiFoldDataset(dataset_loc=config['training_set'],
                                        tokenizer=tokenizer,
                                        exclude_proteins=exclude_proteins,
                                        folds=train_folds)
    else:
        raise AttributeError('If the test set is not set to "default", test_fold needs to be set')
    train_loader = DataLoader(train_set, gpu_batch_size, shuffle=True, pin_memory=True, collate_fn=train_set.collate_fn, num_workers=1)
    valid_loader = DataLoader(valid_set, gpu_batch_size, shuffle=False, pin_memory=True, collate_fn=valid_set.collate_fn, num_workers=1)
    test_loader = DataLoader(test_set, gpu_batch_size, shuffle=False, pin_memory=True, collate_fn=test_set.collate_fn, num_workers=1)


    # (5) Model initialization. Create network architecture, construct pytorch lightning module
    lightning_module = LightningModule(config=config,
                                       steps_per_training_epoch=len(train_loader),
                                       tokenizer=tokenizer
    )

    # (6) Set up training
    checkpoint_dir = '/data/gent/476/vsc47680/PhosphoLingo/models/'
    
    # Define the list of callbacks (early stopping, learning rate monitor)
    early_stopping = EarlyStopping(monitor='validation_loss', patience=18, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [early_stopping, lr_monitor]
    # Conditionally add ModelCheckpoint callback based on 'save_model'
    if config.get('save_model', True):
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='best_model',
            save_top_k=1,
            monitor='validation_loss',
            mode='min',
            save_last=False,
            verbose=True,
        )
        callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        accumulate_grad_batches=grad_accumulation,
        logger=[wandb_logger],
        max_epochs=config['max_epochs'],
        callbacks=callbacks,
        default_root_dir=None,
        val_check_interval=0.25,
        enable_checkpointing=config.get('save_model', True),  # Disable checkpointing when save_model is False
    )


    # (7) train the model
    trainer.fit(lightning_module, train_loader, valid_loader)


    # (8) Evaluate the model
    if config.get('save_model', True):
        # If save_model is True, test using the best checkpoint
        trainer.test(dataloaders=test_loader, ckpt_path='best')
    else:
        # If save_model is False, test using the current state of the model
        trainer.test(model=lightning_module, dataloaders=test_loader)
    
    
    # (9) Log the best model to W&B if it was saved
    if config.get('save_model', True) and 'checkpoint_callback' in locals() and checkpoint_callback.best_model_path:
        best_model_path = checkpoint_callback.best_model_path
        artifact = wandb.Artifact('best_model', type='model')
        artifact.add_file(best_model_path)
        wandb_logger.experiment.log_artifact(artifact)
    
    wandb.finish()
