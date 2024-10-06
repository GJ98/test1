import pytorch_lightning as pl
import module.data as module_data
from module.mrc import *
from pytorch_lightning.loggers import WandbLogger
import hydra

# fix random seeds for reproducibility
'''
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
'''

@hydra.main(config_path="./config", config_name="mrc", version_base=None)
def main(config):
    config = config.mrc

    # 0. logger
    logger = WandbLogger(project=config.wandb.project) if config.wandb.enable else None

    # 1. set data_module(=pl.LightningDataModule class)
    data_module = getattr(module_data, config.data.data_module)(config)
    data_module.setup()
    # 1.1. create eval & test dataset
    eval_dataset = data_module.eval_dataset
    test_dataset = data_module.test_dataset
    eval_examples = data_module.eval_examples
    test_examples = data_module.test_examples
 
    # 2. set mrc module(=pl.LightningModule class)
    model_module = MrcLightningModule(config, eval_dataset, test_dataset, eval_examples, test_examples)

    # 3. set trainer(=pl.Trainer) & train
    trainer = pl.Trainer(num_sanity_val_steps=0, 
                         accelerator="gpu", 
                         devices=1, 
                         max_epochs=config.train.num_train_epochs, 
                         log_every_n_steps=1,
                         logger=logger)
    trainer.fit(model=model_module, datamodule=data_module)

if __name__ == "__main__":
    main()
