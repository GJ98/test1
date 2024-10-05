import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import module.data as module_data
from module.retrieval import DenseRetrieval
import module.encoder as module_encoder
import hydra

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

@hydra.main(config_path="./config", config_name="retrieval", version_base=None)
def main(config):

    # 0. logger
    logger = WandbLogger(project=config.wandb.project) if config.wandb.enable else None

    # 1. set data_module(=pl.LightningDataModule class)
    data_module = getattr(module_data, config.data.data_module)(config)

    # 2. set model
    # 2.1. set encoder model
    p_encoder = getattr(module_encoder, config.model.encoder).from_pretrained(config.model.plm_name)
    q_encoder = getattr(module_encoder, config.model.encoder).from_pretrained(config.model.plm_name)
    # 2.2. set retrieval module(=pl.LightningModule class)
    retrieval = DenseRetrieval(config, q_encoder, p_encoder)

    # 3. set trainer(=pl.Trainer) & train
    trainer = pl.Trainer(num_sanity_val_steps=0, 
                         accelerator="gpu", 
                         devices=1, 
                         max_epochs=config.train.num_train_epochs, 
                         log_every_n_steps=1,
                         logger=logger)
    trainer.fit(model=retrieval, datamodule=data_module)

    # 4. test by validation dataset
    retrieval.create_embedding_vector()
    print(retrieval.search("우리나라 대통령은 누구야?"))

if __name__ == "__main__":
    main()
