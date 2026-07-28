import os
import gin
import fire
import wandb
import torch

from data import load_dataset, load_dataset_from_huggingface, load_mixed_datasets_from_huggingface, batch_preparation_ctc
from seed_utils import seed_everything
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from ModelManager import get_model, LighntingE2EModelUnfolding
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


@gin.configurable
def main(train_path=None, val_path=None, test_path=None, encoding=None, model_name=None, use_huggingface=False, use_mixed_datasets=False, dataset_name="PRAIG/grandstaff", dataset_names=None, wandb_run_id=None, resume_from=None, random_seed=42):
    # Seed everything for reproducibility
    seed_everything(random_seed)
    
    outpath = f"out/GrandStaff_{encoding}/{model_name}"
    os.makedirs(outpath, exist_ok=True)
    os.makedirs(f"{outpath}/hyp", exist_ok=True)
    os.makedirs(f"{outpath}/gt", exist_ok=True)

    # Load dataset from HuggingFace, mixed datasets, or local files
    if use_mixed_datasets:
        train_dataset, val_dataset, test_dataset = load_mixed_datasets_from_huggingface(
            dataset_names=dataset_names,
            corpus_name=f"GrandStaff_{encoding}"
        )
        dataset_label = ",".join([ds.split("/")[-1] for ds in dataset_names]) if dataset_names else "mixed"
    elif use_huggingface:
        train_dataset, val_dataset, test_dataset = load_dataset_from_huggingface(
            dataset_name=dataset_name,
            corpus_name=f"GrandStaff_{encoding}"
        )
        dataset_label = dataset_name.split("/")[-1]
    else:
        train_dataset, val_dataset, test_dataset = load_dataset(
            train_path, val_path, test_path,
            corpus_name=f"GrandStaff_{encoding}"
        )
        dataset_label = os.path.basename(train_path) if train_path else "local"

    _, i2w = train_dataset.get_dictionaries()

    train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=8, collate_fn=batch_preparation_ctc)
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=8, collate_fn=batch_preparation_ctc)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=8, collate_fn=batch_preparation_ctc)

    maxheight, maxwidth = train_dataset.get_max_hw()

    model, torchmodel = get_model(maxwidth=maxwidth, maxheight=maxheight, in_channels=1, blank_idx=len(i2w), out_size=train_dataset.vocab_size()+1, i2w=i2w, model_name=model_name, output_path=outpath)

    # Récupère le run_id wandb depuis l'argument ou variable d'environnement
    run_id = wandb_run_id or os.environ.get('WANDB_RUN_ID', None)
    wandb_logger = WandbLogger(
        project='E2E_Pianoform',
        name=f"{model_name}_{dataset_label}",
        id=run_id,
        resume="allow" if run_id else None
    )
    wandb_logger.experiment.config.update({"dataset": dataset_label})
    
    early_stopping = EarlyStopping(monitor='val_SER', min_delta=0.01, patience=5, mode="min", verbose=True)
    
    checkpointer = ModelCheckpoint(dirpath=f"weights/{encoding}/{model_name}", filename=f"{model_name}", 
                                   monitor="val_SER", mode='min',
                                   save_top_k=1, verbose=True)

    trainer = Trainer(max_epochs=10000, logger=wandb_logger, callbacks=[checkpointer, early_stopping],
                      enable_progress_bar=False)  # Disable verbose progress bar
    
    # Pour reprendre depuis un checkpoint précis si précisé
    checkpoint_path = resume_from or os.environ.get('RESUME_FROM', None)
    
    # Start fresh training with the initialized model (no checkpoint loading unless resuming)
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=checkpoint_path)
    trainer.test(model, test_dataloader)
    wandb.finish()

def launch(config):
    gin.parse_config_file(config)
    main()

if __name__ == "__main__":
    fire.Fire(launch)